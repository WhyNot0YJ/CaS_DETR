# Agent instructions

本文件只约束在本仓库中工作的编码代理。项目使用说明见
[CaS_DETR_readme.md](CaS_DETR_readme.md)；更具体的目录规则以代码和测试为准。

## 主线边界

- 主线名称统一写作 `CaS-DETR`，核心实现位于 `experiments/CaS-DETR/`。
- CaS 组件及源码位置：Token pruning/CAIP 在
  `engine/deim/hybrid_encoder.py`，CASS 在
  `engine/deim/token_level_pruning.py`，MoE 在
  `engine/deim/moe_components.py` 和 `engine/deim/dfine_decoder.py`，相关损失在
  `engine/deim/deim_criterion.py`。
- 只在用户明确要求时扩展主线；不要把容量、专家数或部署扫描写成主结果。

## 数据协议

正式协议只有四个：

| 数据集 | 协议 | 类别数 | 说明 |
| --- | --- | ---: | --- |
| DAIR-V2X | `dairv2x_vehicle8` | 8 | 默认 |
| DAIR-V2X | `dairv2x_vehicle5` | 5 | 历史协议，必须显式指定 |
| UA-DETRAC | `uadetrac_vehicle1` | 1 | 默认 |
| UA-DETRAC | `uadetrac_vehicle4` | 4 | 历史协议，必须显式指定 |

类别映射：

- DAIR Vehicle5：`Car/Truck/Van/Bus -> vehicle`；其余为
  `Pedestrian`、`Cyclist`、`Motorcyclist`、`Trafficcone`。
- DAIR Vehicle8：保留原始八类。
- UA Vehicle1：`car/van/bus/others -> vehicle`。
- UA Vehicle4：保留原始四类。

默认路由来自 `experiments/common/dataset_protocol.py`。不传协议参数时使用
Vehicle8/Vehicle1；旧协议必须使用对应的 `--dairv2x-vehicle5` 或
`--uadetrac-vehicle4`。checkpoint、数据集、日志和报告协议必须完全一致，禁止
跨协议恢复、评测或覆盖结果。

DAWN 跨域评测只允许使用 DAIR Vehicle8 checkpoint；不得把 Vehicle5 checkpoint
用于 DAWN 8 类口径。UA Vehicle4 checkpoint 若只需无重训验证 Vehicle1，必须通过
`experiments/common/evaluate_uadetrac_vehicle1.py` 合并预测类别并执行
class-agnostic NMS，结果写入 Vehicle1 目录。

## 配置与输出

- YAML 通过 `__include__` 继承基础模型、数据集和实验组件。优先修改共享层或新建
  派生配置，不要逐个复制消融 YAML。
- 新配置名应包含模型、组件/容量、数据集和协议后缀，例如
  `..._dairv2x_vehicle5.yml`、`..._uadetrac_vehicle1.yml`。
- 训练输出目录按协议隔离为 `<outputs-or-logs>/<protocol>/<experiment>`，例如
  `outputs/dairv2x_vehicle5/main/...`；不要再把协议后缀拼到实验目录名末尾。
- 评测、训练结果和测速结果分开保存；不得把不同协议或不同实验类型写入同一 CSV。
- 统一结果路径只能经 `experiments/common/result_paths.py` 获取，禁止在新评测器中
  手写 `experiments/reports/...`。
- 报告目录固定为：
  `experiments/reports/dairv2x_vehicle5/`、`dairv2x_vehicle8/`、
  `uadetrac_vehicle1/`、`uadetrac_vehicle4/`。
- 主实验、组件消融、容量扫描、专家扫描分别建表。默认采用单 seed，记录 seed
  以便追溯；只有用户明确要求重复实验时才汇总多 seed，不把 `std` 作为主线交付要求。

## 训练、评测与测速口径

- 新增训练入口必须支持四协议路由，并将协议传递到数据集、输出和报告路径。
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
