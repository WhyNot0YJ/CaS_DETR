# Agent instructions

本文件只约束在本仓库中工作的编码代理。项目使用说明见
[CaS_DETR_readme.md](CaS_DETR_readme.md)；更具体的目录规则以代码和测试为准。

## 主线边界

- 主线名称统一写作 `CaS-DETR`，核心实现位于 `experiments/CaS-DETR/`。
- CaS 组件及源码位置：Token pruning 与 CASS 在
  `engine/deim/hybrid_encoder.py`、`engine/deim/token_level_pruning.py`，MoE 在
  `engine/deim/moe_components.py` 和 `engine/deim/dfine_decoder.py`，相关损失在
  `engine/deim/deim_criterion.py`。
- 只在用户明确要求时扩展主线；不要把容量、专家数或部署扫描写成主结果。

## 数据协议

正式协议只有两个：

| 数据集 | 协议 | 类别数 | 说明 |
| --- | --- | ---: | --- |
| DAIR-V2X | `dairv2x` | 8 | 默认 |
| UA-DETRAC | `uadetrac` | 4 | 默认 |

类别映射使用原生标注：DAIR-V2X 保留八类，UA-DETRAC 保留四类。

默认路由来自 `experiments/common/dataset_protocol.py`。checkpoint、数据集、日志和
报告协议必须完全一致，禁止跨协议恢复、评测或覆盖结果。

DAWN 跨域评测只允许使用 DAIR-V2X 八类 checkpoint。

## 配置与输出

- YAML 通过 `__include__` 继承基础模型、数据集和实验组件。优先修改共享层或新建
  派生配置，不要逐个复制消融 YAML。
- 新配置名应包含模型、组件/容量、数据集和协议后缀，例如
  `..._dairv2x.yml`、`..._uadetrac.yml`。
- 训练输出目录按协议隔离为 `<outputs-or-logs>/<protocol>/<experiment>`，例如
  `outputs/dairv2x/main/...`；不要再把协议后缀拼到实验目录名末尾。
- 评测、训练结果和测速结果分开保存；不得把不同协议或不同实验类型写入同一 CSV。
- 统一结果路径只能经 `experiments/common/result_paths.py` 获取，禁止在新评测器中
  手写 `experiments/reports/...`。
- 报告目录固定为：
  `experiments/reports/dairv2x/`、`uadetrac/`。
- 主实验、组件消融、容量扫描、专家扫描分别建表。默认采用单 seed，记录 seed
  以便追溯；只有用户明确要求重复实验时才汇总多 seed，不把 `std` 作为主线交付要求。

## 训练、评测与测速口径

- 运行、验证和 GPU/TensorRT 相关操作按以下顺序选择环境：先检查仓库上一层的
  `/root/autodl-tmp/cas_trt_env/bin/python`；该环境可用时，训练、评测、YOLO 批处理、
  TensorRT 和邮件通知统一使用它，禁止改用系统 Python。只有该环境不可用或缺少所需
  依赖时，才检查 Docker，并通过仓库根目录的 `./run_rtdetr.sh` 进入 `rtdetr_dev`
  容器，使用容器内的 `python`。容器内工作目录固定为 `/root/autodl-tmp/CaS_DETR`。
- 新增训练入口必须支持两协议路由，并将协议传递到数据集、输出和报告路径。
- TensorRT FP16 是正式速度口径；PyTorch FPS 仅作诊断。
- `model` FPS 只表示固定设备输入到模型输出；`end-to-end` 必须包含图像读取、预处理、
  H2D、推理和 D2H，二者不得混写。
- 数据生成器只能生成派生标注/目录，不能修改原始数据；生成后必须检查类别范围、框数
  和 image-to-label 路径。
- GPU、数据集或 TensorRT 无法使用时，交付中必须明确写出未验证项，不得补写推测结果。

## 修改与回归要求

- 修改前检查：目标配置及其 `__include__` 链、协议注册表、入口脚本、
  `result_paths.py` 和对应测试。
- 做局部、最小改动；禁止无关重构、批量格式化和未经请求的删除。
- 修改配置/路由后至少运行：Python 编译、shell `bash -n`、协议/数据生成测试、
  dry-run，并检查 `git diff --check`。
- 完成定义：代码、配置、输出和报告协议一致；回归命令有实际输出；未运行的 GPU 或
  长时实验明确标记为未完成。
- 不得伪造实验数值、速度、checkpoint 或验证记录。

常用入口：`experiments/run_batch_experiments.sh`、各框架的 `train.py`、
`experiments/common/prepare_*`、`experiments/common/evaluate_*`，以及
`experiments/CaS-DETR/tools/deployment/` 和 `tools/benchmark/`。
