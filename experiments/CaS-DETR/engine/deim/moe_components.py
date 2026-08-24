"""Universal Token-Level MoE Components for CaS_DETR - Vectorized Implementation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

class MoELayer(nn.Module):
    """Universal Token-Level MoE Layer - 向量化实现（无 for 循环）
    
    Token级别的MoE实现。
    输入: [B, N, C] (N可以是动态的剪枝后的数量)
    输出: [B, N, C] (保持输入形状)
    
    关键优化：使用批量矩阵运算替代循环，充分利用 GPU 并行计算能力。
    """
    
    def __init__(self, d_model: int, dim_feedforward: int, num_experts: int = 4,
                 top_k: int = 2, dropout: float = 0.1, activation: str = 'gelu',
                 noise_std: float = 0.1, router_init_std: float = 0.02,
                 pg_enabled: bool = False, pg_hidden_dim: int = 32,
                 pg_scale: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.noise_std = noise_std
        self.dropout_rate = dropout
        self.pg_enabled = pg_enabled
        self.pg_scale = pg_scale
        
        # [修复] 提高初始化 std 并支持配置化
        self.router = nn.Linear(d_model, num_experts, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=router_init_std)
        if self.pg_enabled:
            self.spatial_router = nn.Sequential(
                nn.Linear(2, pg_hidden_dim),
                nn.SiLU(),
                nn.Linear(pg_hidden_dim, num_experts),
            )
        
        # 向量化专家权重：每个专家的两层 MLP
        self.expert_w1 = nn.Parameter(torch.empty(num_experts, dim_feedforward, d_model))
        self.expert_b1 = nn.Parameter(torch.zeros(num_experts, dim_feedforward))
        self.expert_w2 = nn.Parameter(torch.empty(num_experts, d_model, dim_feedforward))
        self.expert_b2 = nn.Parameter(torch.zeros(num_experts, d_model))
        
        # 初始化专家权重
        for i in range(num_experts):
            nn.init.xavier_uniform_(self.expert_w1[i])
            nn.init.xavier_uniform_(self.expert_w2[i])
        
        self.dropout = nn.Dropout(dropout)
        
        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
        
        # [修复] 缓存改为列表，以支持共享层多次 forward 的记录
        self.router_logits_cache = []
        self.expert_indices_cache = []
        self.reference_points_cache = []
        # Routing statistics are detached; only the scalar auxiliary loss
        # retains the graph until it is consumed after this forward.
        self._balance_loss_cache = []

    def reset_cache(self):
        """用于在共享层模式下，每个 Batch 开始前清空记录"""
        self.router_logits_cache = []
        self.expert_indices_cache = []
        self.reference_points_cache = []
        self._balance_loss_cache = []
    
    def forward(self, x: torch.Tensor, spatial_shape: Optional[Tuple[int, int]] = None,
                dynamic_top_k: Optional[int] = None,
                reference_points: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Memory-Efficient Token-Grouping MoE - Scales with Token Count, not Weight-Token Product.
        Optimized to prevent OOM on large batches.
        
        Args:
            x: [B, N, C] Token features
            dynamic_top_k: Optional override for activated experts; ``None`` uses fixed ``top_k``.
        Returns:
            output: [B, N, C]
        """
        B, N, C = x.shape
        E = self.num_experts
        K = dynamic_top_k if (dynamic_top_k is not None and 1 <= dynamic_top_k <= E) else self.top_k
        
        # 1. Router Logic
        router_logits = self.router(x)  # [B, N, E]
        if self.pg_enabled:
            if reference_points is None:
                raise ValueError('PG-MoE requires normalized decoder reference points.')
            router_logits = router_logits + self.pg_scale * self.spatial_router(reference_points[..., :2])
        # [修复] Noisy Top-K: 仅在训练阶段加入探索噪声
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise
        if torch.onnx.is_in_onnx_export():
            # TopK is invariant under softmax. Normalizing only the selected logits
            # removes the full-router softmax and the subsequent renormalization.
            expert_logits, expert_indices = torch.topk(router_logits, K, dim=-1)
            expert_weights = F.softmax(expert_logits, dim=-1)
        else:
            router_probs = F.softmax(router_logits, dim=-1)  # [B, N, E]
            expert_weights, expert_indices = torch.topk(router_probs, K, dim=-1)  # [B, N, K]

            # Renormalize expert weights
            expert_weights = expert_weights / (expert_weights.sum(dim=-1, keepdim=True) + 1e-9)
        
        if self.training and not torch.onnx.is_in_onnx_export():
            # Preserve gradients for the auxiliary loss without retaining the
            # large router tensors in the statistics cache.
            self._balance_loss_cache.append(compute_moe_balance_loss(
                [router_logits.reshape(-1, E)],
                num_experts=E,
                expert_indices_list=[expert_indices.reshape(-1, K)],
                top_k=K,
            ))
        elif not torch.onnx.is_in_onnx_export():
            # These tensors are needed only by eval-time router diagnostics.
            # Never retain them during training: even detached GPU tensors can
            # raise the peak memory substantially on large token batches.
            self.router_logits_cache.append(router_logits.detach().reshape(-1, E))
            self.expert_indices_cache.append(expert_indices.detach().reshape(-1, K))
            self.reference_points_cache.append(reference_points.detach()[..., :2] if reference_points is not None else None)

        x_flat = x.view(-1, C)
        flat_expert_indices = expert_indices.view(-1, K)
        flat_expert_weights = expert_weights.view(-1, K)
        if torch.onnx.is_in_onnx_export():
            h = torch.matmul(x_flat.unsqueeze(0), self.expert_w1.transpose(1, 2))
            h = self.dropout(self.activation(h + self.expert_b1.unsqueeze(1)))
            expert_out = torch.matmul(h, self.expert_w2.transpose(1, 2))
            expert_out = expert_out + self.expert_b2.unsqueeze(1)
            gate = (
                F.one_hot(flat_expert_indices, E).to(flat_expert_weights.dtype)
                * flat_expert_weights.unsqueeze(-1)
            ).sum(dim=1)
            out_flat = (expert_out.permute(1, 0, 2) * gate.unsqueeze(-1)).sum(dim=1)
            return out_flat.view(B, N, C)

        out_flat = torch.zeros_like(x_flat)
        
        # 2. Key Fix: Loop over experts and process assigned tokens as a group
        for i in range(E):
            # Find tokens assigned to current expert i
            token_indices, slot_indices = torch.where(flat_expert_indices == i)
            # Extract only the tokens that need this expert
            temp_x = x_flat[token_indices]
            
            # Compute: Single expert pass
            h = F.linear(temp_x, self.expert_w1[i], self.expert_b1[i])
            h = self.activation(h)
            h = self.dropout(h)
            temp_out = F.linear(h, self.expert_w2[i], self.expert_b2[i])
            
            # Apply routing weights and accumulate back to output
            weights = flat_expert_weights[token_indices, slot_indices].unsqueeze(-1)
            out_flat.index_add_(0, token_indices, temp_out * weights)
            
        return out_flat.view(B, N, C)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        if self.pg_enabled:
            for name, tensor in self.spatial_router.state_dict().items():
                state_dict.setdefault(prefix + 'spatial_router.' + name, tensor)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


# =========================================================================
# 统一的 MoE 负载均衡损失函数
# =========================================================================

def compute_moe_balance_loss(router_logits_list: List[torch.Tensor], 
                             num_experts: int,
                             expert_indices_list: List[torch.Tensor] = None,
                             top_k: int = 2) -> torch.Tensor:
    """
    Switch Transformer 风格的 MoE 负载均衡损失。
    
    使用公式: Loss = E × Σ(f_i × P_i)
    其中:
        f_i: 实际路由到专家 i 的 token 比例 (通过 bincount 统计)
        P_i: 专家 i 的平均 router 概率
    
    Args:
        router_logits_list: Router logits 列表，每个元素形状为 [N, E]
        num_experts: 专家数量
        expert_indices_list: 专家索引列表，用于计算实际路由比例
        top_k: Top-K 路由的 K 值（保持接口兼容）
    
    Returns:
        负载均衡损失标量
    """
    if not router_logits_list: 
        return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    total_loss = 0.0
    num_layers = 0
    
    for i, logits in enumerate(router_logits_list):
        if logits is None or logits.numel() == 0:
            continue
        
        # 计算 Softmax 概率
        probs = F.softmax(logits.float(), dim=-1)  # [N, E]
        expert_probs = probs.mean(dim=0)  # [E] 每个专家的平均概率
        
        # 计算实际路由比例 f_i
        if expert_indices_list is not None and i < len(expert_indices_list) and expert_indices_list[i] is not None:
            indices = expert_indices_list[i]
            flat_indices = indices.view(-1)
            usage_counts = torch.bincount(flat_indices, minlength=num_experts).float()
            total_dispatches = flat_indices.size(0)
            expert_usage = usage_counts / total_dispatches  # [E] 实际选择比例
        else:
            # 如果没有 indices，退化为使用概率作为 usage
            expert_usage = expert_probs
        
        # Switch Transformer 损失: Loss = E × Σ(f_i × P_i)
        loss = num_experts * torch.sum(expert_usage * expert_probs)
        
        total_loss += loss
        num_layers += 1
    
    return total_loss / num_layers if num_layers > 0 else torch.tensor(0.0)
