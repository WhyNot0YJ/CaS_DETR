# CaS_DETR

## 项目概述

CaS_DETR 是一个基于 DEIM 的目标检测实验项目，主实现位于 `experiments/CaS-DETR/`。项目在 DEIM 训练和配置体系上集成稀疏 Token 建模与轻量 MoE，用于评估精度、参数量和部署速度之间的权衡。

当前主线由以下四部分构成：

- **Token 剪枝**：由编码器预测 Token 重要性，降低后续计算量；
- **CAIP**：将全局复杂度信息用于 Token 排序；
- **CASS**：以软监督训练 Token 重要性预测；
- **Decoder MoE**：在 D-FINE 解码器的 FFN 中使用稀疏专家路由，并加入负载均衡损失。

完整的实验目标、比较边界、停止条件和交付物见 [MODIFICATION_PLAN.md](MODIFICATION_PLAN.md)。

## 当前主线配置

当前主线统一使用 **CaS-DETR** 作为名称。其 DAIR-V2X 配置为 `experiments/CaS-DETR/configs/deim_dfine/cas_detr_hgnetv2_s_dairv2x.yml`：

```yaml
HybridEncoder:
  token_keep_ratio: 0.3
  use_caip: true
  use_cass: true
  caip_complexity_alpha: 1.0

DFINETransformer:
  use_moe: true
  num_experts: 4
  moe_top_k: 2
  dim_feedforward: 128
```

其中 `dim_feedforward: 128` 对应 **0.5x MoE capacity**：4 个专家的 FFN 总参数量约为普通单 FFN 的一半。当前主结果建议只比较下列模型：

| ID | 模型 | 用途 |
| --- | --- | --- |
| M0 | DEIM | 基线 |
| M1 | CaS-DETR（CASS + CAIP + 0.5x MoE） | 当前主线配置 |

容量为 1x、2x、4x 的 MoE 仅用于容量分析，不能作为主表结果混入。

## 项目结构

- `experiments/CaS-DETR/train.py`：训练、断点恢复、微调和仅评测入口。
- `experiments/CaS-DETR/engine/deim/hybrid_encoder.py`：编码器、Token 剪枝与 CAIP。
- `experiments/CaS-DETR/engine/deim/token_level_pruning.py`：重要性预测和 CASS 损失。
- `experiments/CaS-DETR/engine/deim/moe_components.py`：MoE 层和路由逻辑。
- `experiments/CaS-DETR/engine/deim/dfine_decoder.py`：支持 MoE 的 D-FINE 解码器。
- `experiments/CaS-DETR/engine/deim/deim_criterion.py`：检测、CASS 和 MoE 负载均衡损失。
- `experiments/CaS-DETR/configs/base/cas_deim.yml`：CaS 基础默认值。
- `experiments/CaS-DETR/configs/deim_dfine/`：主实验和部署配置。
- `experiments/CaS-DETR/configs/dataset/ablation/`：组件、容量与专家数消融配置。
- `experiments/CaS-DETR/tools/experiments/`：实验矩阵执行、种子汇总和容量分析。
- `experiments/CaS-DETR/tools/deployment/`：ONNX 导出、TensorRT engine 构建和一致性验证。
- `experiments/CaS-DETR/tools/benchmark/`：TensorRT 模型端与端到端性能测试。
- `experiments/reports/`：跨实验的评测和性能 CSV。
- `artifacts/tensorrt_benchmark/`：已归档的 ONNX、TensorRT engine 与原始测速记录。

## 配置机制

YAML 使用 `__include__` 组合基础模型、数据集和 CaS 参数。新增实验应新建派生配置，避免直接修改基线配置。核心配置项如下：

| 模块 | 配置项 | 作用 |
| --- | --- | --- |
| Token 剪枝 | `enable_cas_predictor`、`token_keep_ratio` | 是否启用预测器及目标保留比例 |
| CAIP | `use_caip`、`caip_complexity_alpha` | 是否启用复杂度感知排序及其权重 |
| CASS | `use_cass`、`cass_loss_type`、`cass_loss_weight` | 是否启用软监督及其损失形式、权重 |
| MoE | `use_moe`、`num_experts`、`moe_top_k`、`dim_feedforward` | 是否启用 MoE、专家池规模、激活专家数和容量 |
| MoE 损失 | `decoder_moe_balance_weight` | 路由负载均衡损失权重 |

主要消融分组如下：

| 分组 | 配置/ID | 比较内容 |
| --- | --- | --- |
| 组件基线 | `M0` / `cas_deim_all_off_*` | 关闭 CaS 与 MoE 的 DEIM 基线 |
| 主线配置 | `M1` / `cas_detr_hgnetv2_s_dairv2x.yml` | CASS、CAIP 与 0.5x MoE |
| 仅 MoE | `A2` / `cas_deim_moe4_only_*` | 隔离轻量 MoE 的影响 |
| 容量扫描 | `C05`、`C1`、`C2`、`C4` | 固定 4 专家、`top_k=2`，扫描 0.5x 至 4x 容量 |
| 专家数扫描 | `E3`、`C05`、`E6` | 固定 `dim_feedforward=128`，扫描 3、4、6 专家 |

## 数据集与环境

当前配置使用 COCO 格式标注：

- DAIR-V2X：`/root/autodl-fs/datasets/DAIR-V2X`，8 类；
- UA-DETRAC：`/root/autodl-fs/datasets/UA-DETRAC_COCO`，4 类。

训练、导出和性能测试优先使用：

```bash
/root/autodl-tmp/cas_trt_env/bin/python
```

GPU 或 TensorRT 任务开始前可检查环境：

```bash
/root/autodl-tmp/cas_trt_env/bin/python -c "import torch, tensorrt; print(torch.cuda.is_available(), tensorrt.__version__)"
```

## 训练与评测

在 `experiments/CaS-DETR/` 下运行单个主配置：

```bash
/root/autodl-tmp/cas_trt_env/bin/python train.py \
  -c configs/deim_dfine/cas_detr_hgnetv2_s_dairv2x.yml
```

恢复训练和微调分别使用 `-r` 与 `-t`，两者不能同时使用：

```bash
python train.py -c <配置文件> -r <checkpoint>
python train.py -c <配置文件> -t <checkpoint>
```

仅运行验证：

```bash
python train.py -c <配置文件> -r <checkpoint> --test-only
```

批量脚本提供常用快捷入口：

```bash
# 运行 CaS-DETR 消融队列
./experiments/run_batch_experiments.sh --cas_detr

# 仅 DAIR-V2X；快速模式会将每个实验限制为 2 个 epoch
./experiments/run_batch_experiments.sh --test --dairv2x --cas_detr
```

## 可复现实验协议

`tools/experiments/run_modification_plan.py` 将固定实验矩阵串联为训练、验证、路由统计、ONNX 导出、TensorRT 构建和测速流程。默认主实验使用种子 `0 1 2`，并拒绝向已有实验目录混写结果。

```bash
cd experiments/CaS-DETR
/root/autodl-tmp/cas_trt_env/bin/python \
  tools/experiments/run_modification_plan.py --group main
```

可用分组为 `main`、`moe`、`moe_capacity`、`moe_experts`、`moe_scan` 与 `all`。先查看实际命令而不执行时，可加 `--dry-run`；不需要 TensorRT 收尾时，可加 `--no-trt-benchmark`。

每个 seed 的输出位于 `experiments/CaS-DETR/outputs/modification_plan/<模型 ID>/seed_<种子>/`，包含解析后的配置、运行环境、checkpoint、预测结果、评测指标和路由统计。完成主实验后汇总结果：

```bash
/root/autodl-tmp/cas_trt_env/bin/python \
  experiments/CaS-DETR/tools/experiments/summarize_seeds.py \
  experiments/CaS-DETR/outputs/modification_plan
```

汇总会生成 `main_3seed_summary.csv`、`paired_deltas.csv` 和 `moe_ablation.csv`。主表应报告 `mean +/- std`，并同时关注 `mAP50:95`、`AP50`、`AP75`、`APs50:95`、`APs50`、总参数量、激活参数量和 GFLOPs。

## 部署与性能测试

TensorRT FP16 是正式速度口径，PyTorch FPS 只用于开发诊断。测速协议固定为 batch size 1、640 x 640、预热 100 次、同步计时 1000 次，并分别记录：

- `model`：固定设备输入的模型端延迟，不含图像 I/O、预处理和数据传输；
- `end-to-end`：包含读图、解码、缩放、传输、TensorRT 图执行和结果传回。

部署配置 `cas_detr_hgnetv2_s_dairv2x_deploy.yml` 固定 `token_keep_ratio: 0.3`，并启用 `caip_static_keep_eval`，使部署图保持静态 Token 数。部署性能必须与该静态图对应的精度共同报告。

标准流程为：导出 ONNX，构建 FP16 engine，比较 PyTorch 与 ONNX 的预测/COCO 指标，再运行 TensorRT 测速。相关脚本分别是：

```text
tools/deployment/export_onnx_protocol.py
tools/deployment/build_trt_engine.py
tools/deployment/validate_onnx_coco.py
tools/benchmark/benchmark_trt_protocol.py
```

归档 engine 的复测命令：

```bash
/root/autodl-tmp/cas_trt_env/bin/python \
  experiments/CaS-DETR/tools/benchmark/benchmark_artifact_engines.py \
  --warmup 100 --iterations 1000
```

测速记录写入 `experiments/reports/benchmark.csv`；归档模型的原始测量记录位于 `artifacts/tensorrt_benchmark/benchmark.csv`。两者均保留 GPU、驱动、CUDA、TensorRT、延迟分位数、显存和测量模式，禁止将 `model` 与 `end-to-end` 数据混合比较。

## 验证要求

- 配置改动至少验证 YAML 可解析，并确认 `__include__` 解析后的配置符合预期。
- 模型改动至少进行导入、配置加载和可用范围内的短训或评测。
- ONNX/TensorRT 改动必须保留导出日志、构建环境和 ONNX 一致性报告。
- 无法执行完整 GPU、数据集或 TensorRT 验证时，应明确记录未验证部分，不以单次最高指标替代多种子结论。
