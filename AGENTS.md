# CaS_DETR 的 AGENTS.md

这份文件面向本仓库中的 AI 编码助手，既说明项目的关键背景、结构和技术口径，也说明在这个仓库里应该如何改动代码。

项目相关的更完整背景、架构、配置和工作流，请优先查看 `.github/copilot-instructions.md`。
关于注释、文档字符串和数据集约定，请查看 `.cursor/rules/`。

---

## 项目背景

CaS_DETR 是一个基于 DEIM 的目标检测实验项目，主实现位于 `experiments/CaS-DETR/`。当前工作重点不是重做整个检测框架，而是在 DEIM 的训练和配置体系上，验证稀疏 Token 建模与轻量 MoE 的组合效果，重点关注精度、参数量、激活参数量和部署速度之间的权衡。

当前主线由四个部分构成：

- Token 剪枝：由编码器预测 Token 重要性，降低后续计算量
- CAIP：将全局复杂度信息用于 Token 排序
- CASS：以软监督训练 Token 重要性预测
- Decoder MoE：在 D-FINE 解码器的 FFN 中使用稀疏专家路由，并加入负载均衡损失

完整的实验目标、比较边界、停止条件和交付物见 [MODIFICATION_PLAN.md](MODIFICATION_PLAN.md)。

---

## 关键结构

主配置文件：

- `experiments/CaS-DETR/configs/deim_dfine/cas_detr_hgnetv2_s_dairv2x.yml`
- 部署配置：`experiments/CaS-DETR/configs/deim_dfine/cas_detr_hgnetv2_s_dairv2x_deploy.yml`

当前主线配置统一使用 `CaS-DETR` 作为名称，核心参数如下：

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

其中 `dim_feedforward: 128` 对应 0.5x MoE capacity。当前主结果口径只建议对比 `M0 = DEIM` 和 `M1 = CaS-DETR`，1x、2x、4x MoE 仅用于容量分析，不要混进主表。

主要代码结构：

- `experiments/CaS-DETR/train.py`：训练、恢复、微调和仅评测入口
- `experiments/CaS-DETR/engine/deim/hybrid_encoder.py`：编码器、Token 剪枝和 CAIP
- `experiments/CaS-DETR/engine/deim/token_level_pruning.py`：重要性预测与 CASS 损失
- `experiments/CaS-DETR/engine/deim/moe_components.py`：MoE 层与路由逻辑
- `experiments/CaS-DETR/engine/deim/dfine_decoder.py`：支持 MoE 的 D-FINE 解码器
- `experiments/CaS-DETR/engine/deim/deim_criterion.py`：检测、CASS 和 MoE 负载均衡损失
- `experiments/CaS-DETR/configs/base/cas_deim.yml`：CaS 基础默认值
- `experiments/CaS-DETR/configs/dataset/ablation/`：组件、容量和专家数消融配置
- `experiments/CaS-DETR/tools/experiments/`：实验矩阵执行、种子汇总和容量分析
- `experiments/CaS-DETR/tools/deployment/`：ONNX 导出、TensorRT 构建和一致性验证
- `experiments/CaS-DETR/tools/benchmark/`：TensorRT 端到端和模型端测速
- `experiments/reports/`：跨实验的评测和性能 CSV
- `artifacts/tensorrt_benchmark/`：已归档的 ONNX、TensorRT engine 和原始测速记录

---

## 技术栈与实验口径

- 训练、导出和基准测试优先使用 `/root/autodl-tmp/cas_trt_env/bin/python`
- GPU 或 TensorRT 工作前先检查环境：

  ```bash
  /root/autodl-tmp/cas_trt_env/bin/python -c "import torch, tensorrt; print(torch.cuda.is_available(), tensorrt.__version__)"
  ```

- 当前数据集以 COCO 格式标注为主：
  - DAIR-V2X：`/root/autodl-fs/datasets/DAIR-V2X`，8 类
  - UA-DETRAC：`/root/autodl-fs/datasets/UA-DETRAC_COCO`，4 类
- YAML 使用 `__include__` 组合基础模型、数据集和 CaS 参数。新增实验优先新建派生配置，不要直接改基线配置
- 主实验默认使用 3 个随机种子
- 结果汇总要保留 `mean +/- std`
- 组件消融、容量扫描和专家数扫描要分表，不要互相混写
- TensorRT FP16 是正式速度口径，PyTorch FPS 只用于开发诊断
- 测速时必须区分 `model` 和 `end-to-end`，禁止混用
- 部署配置 `cas_detr_hgnetv2_s_dairv2x_deploy.yml` 固定 `token_keep_ratio: 0.3`，并启用 `caip_static_keep_eval`

常用入口：

- 训练主配置：`experiments/CaS-DETR/train.py -c configs/deim_dfine/cas_detr_hgnetv2_s_dairv2x.yml`
- 可复现实验协议：`experiments/CaS-DETR/tools/experiments/run_modification_plan.py`
- 多 seed 汇总：`experiments/CaS-DETR/tools/experiments/summarize_seeds.py`
- ONNX / TensorRT 导出与验证：`experiments/CaS-DETR/tools/deployment/`
- TensorRT 测速：`experiments/CaS-DETR/tools/benchmark/`
- 批量实验脚本：`experiments/run_batch_experiments.sh`

当前实验边界：

- 当前主线统一写作 `CaS-DETR`
- 目前讨论的是“主线配置 / 候选主模型”，不要写成已经完全定稿的最终模型
- 不在主表中展示 1x、2x、4x MoE
- 主实验、消融、容量分析和部署测速要各自独立记录
- 如果结果没有完成多种子验证，不要用单次最高值下结论
- 如果无法完成完整 GPU、数据集或 TensorRT 验证，要明确记录未验证部分

---

## 运行环境

- 训练、导出和基准测试优先使用 `/root/autodl-tmp/cas_trt_env/bin/python`
- 在进行 GPU 或 TensorRT 工作前，先用下面的命令检查环境：

  ```bash
  /root/autodl-tmp/cas_trt_env/bin/python -c "import torch, tensorrt; print(torch.cuda.is_available(), tensorrt.__version__)"
  ```

- 期望的检查结果包括：
  - CUDA 可用
  - TensorRT 可以成功导入

---

## Karpathy 风格编码规则

这些规则偏向谨慎和正确性，而不是速度。

对于非常简单的一行修改，可以酌情处理；但对于任何非平凡改动，都应遵守下面这些规则。

### 1. 先思考，再编码

- 明确写出你的假设。如果你不确定意图、路径、命名规则或框架细节，就先问，或者先在代码库里搜索。
- 如果存在多种解释，要把它们说出来，不要悄悄选一个。
- 如果有更简单的方案，要直接指出来，避免过度设计。
- 如果有地方不清楚，就停下来说明哪里模糊，不要靠猜。

### 2. 先追求简单

- 用最少的代码解决问题，不要多做无关功能。
- 不要为了单次使用的代码去抽象，不要过早考虑“扩展性”。
- 对不可能发生的情况，不要写额外错误处理。
- 如果 200 行能写成 50 行，就重写成更简洁的版本。
- 你可以用这个标准判断：资深工程师会不会觉得这段代码过度复杂。

### 3. 只做局部、精准的改动

- 只改必须改的地方，不要顺手优化旁边的代码、注释或格式。
- 不要重构没有坏掉的东西。如果你发现了无关的死代码，可以提出来，但不要擅自删除。
- 尽量保持现有风格，包括缩进、命名和导入顺序，即使你自己的写法不同。
- 只清理你自己引入的无用导入或无用变量，不要删除原本就存在的死代码，除非用户要求。
- 每一行改动都应该能直接对应用户请求。

### 4. 目标驱动执行

- 在编码前先定义成功标准，把请求转成可验证的目标：
  - “加校验” -> “先写非法输入测试，再让它通过”
  - “修 bug” -> “先写一个能复现问题的测试，再修到通过”
  - “重构 X” -> “重构前后都要确保测试通过”
- 对多步骤任务，先给出一个简短计划：

  ```text
  1. [步骤] -> 验证：[检查项]
  2. [步骤] -> 验证：[检查项]
  ```

- 清晰的目标能让你独立推进，模糊的目标则需要更频繁地确认。

### 5. 修改后要测试

- 如果有测试，就运行测试。
- 如果测试无法运行，比如需要 GPU 或数据集没挂载，要说明原因，并给出建议的手工验证方式。
- 不要在没有证据的情况下说“已经可以了”。
- 用测试输出、lint 结果，或者明确的推理来支撑结论。

### 6. 不做无关重构

- 不要改名、重排或重构与当前请求无关的代码。
- 不要批量格式化文件，除非用户明确要求。
- 不要删除文件或目录，除非用户明确要求。

### 7. 不要伪造验证

- 不要声称运行过没有真正执行的命令或测试。
- 不要伪造文件路径、行号或输出。
- 如果无法验证，要明确说出你能验证的部分和不能验证的部分。

### 8. 清楚解释修改

- 修改前，说明你准备改什么、为什么改。
- 修改后，说明改了什么、验证了什么、有哪些限制。
- 说明时要给出具体文件路径和函数名，不要只说“更新了代码”。

---

## Karpathy 风格编码技能

本项目包含一个正式的 `.agents/skills/karpathy-skill/SKILL.md` 技能文件，内容基于 [multica-ai/andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills)。

在修改代码前，请先阅读并遵守：

- `.agents/skills/karpathy-skill/SKILL.md`

这个项目希望代理做小而明确、能验证的改动，并避免不相关的重构。
