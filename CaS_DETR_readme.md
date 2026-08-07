# CaS-DETR

CaS-DETR 是一个基于 DEIM/D-FINE 的轻量目标检测实验实现，研究 Token pruning、复杂度感知排序（CAIP）、软监督重要性学习（CASS）与 Decoder MoE 的组合。

## 方法主线

- Token pruning：减少后续编码器计算量。
- CAIP：利用图像复杂度辅助 Token 排序。
- CASS：监督 Token 重要性预测。
- Decoder MoE：在 D-FINE 解码器 FFN 中进行稀疏专家路由。

主实现位于 `experiments/CaS-DETR/engine/deim/`；训练与评测入口为
`experiments/CaS-DETR/train.py`。

## 支持范围

仓库包含 CaS-DETR、DEIM、D-FINE、RT-DETR、DQM-DETR，以及 YOLOv5/v8/v12、YOLOX 和 Faster R-CNN 的统一实验入口。主线默认比较 DEIM 基线与 CaS-DETR；其他容量和组件配置属于扩展实验。

## 目录

```text
experiments/
├── CaS-DETR/          # CaS-DETR 主实现、配置与工具
├── DEIM/ D-FINE/ ...   # 其他检测框架
├── yolo/              # YOLO/YOLOX/Faster R-CNN 入口
├── common/            # 协议、数据生成、评测和结果路径
├── reports/            # 按协议隔离的 CSV
└── run_batch_experiments.sh
```

## 环境

建议使用 `/root/autodl-tmp/cas_trt_env/bin/python`。开始 GPU/TensorRT 工作前检查：

```bash
/root/autodl-tmp/cas_trt_env/bin/python -c \
  "import torch, tensorrt; print(torch.cuda.is_available(), tensorrt.__version__)"
```

## 数据协议

项目正式支持四个协议：

| 数据集 | 默认协议 | 旧协议 |
| --- | --- | --- |
| DAIR-V2X | `dairv2x_vehicle8` | `dairv2x_vehicle5` |
| UA-DETRAC | `uadetrac_vehicle1` | `uadetrac_vehicle4` |

DAIR Vehicle5 将 `Car/Truck/Van/Bus` 合并为 `vehicle`，保留
`Pedestrian`、`Cyclist`、`Motorcyclist`、`Trafficcone`；UA Vehicle1 将
`car/van/bus/others` 合并为 `vehicle`。派生协议保留原始框，只改变类别 ID，不修改原始数据。

合并类别是为了避免单一小目标类别主导 COCO `AP_small`，并使跨模型比较的类别定义一致。`AP_small` 仍按 COCO 面积阈值统计，不能解释为某一个具体类别的 AP；small 目标过少时应谨慎解读。

数据准备：

```bash
/root/autodl-tmp/cas_trt_env/bin/python experiments/common/prepare_dairv2x_vehicle5.py
/root/autodl-tmp/cas_trt_env/bin/python experiments/common/prepare_uadetrac_vehicle1.py
```

派生数据位于 `/root/autodl-fs/datasets/` 下的 `DAIR-V2X-Vehicle5`、
`DAIR-V2X_YOLO_Vehicle5`、`UA-DETRAC-Vehicle1` 和
`UA-DETRAC_YOLO_Vehicle1`。

## 快速开始

CaS-DETR 主线：

```bash
cd experiments/CaS-DETR
/root/autodl-tmp/cas_trt_env/bin/python train.py \
  -c configs/deim_dfine/cas_detr_hgnetv2_s_dairv2x.yml
```

统一批量入口默认使用 Vehicle8/Vehicle1：

```bash
cd experiments
./run_batch_experiments.sh --dry-run --yolo --s
./run_batch_experiments.sh --yes --yolo --s
```

旧协议必须显式指定：

```bash
./run_batch_experiments.sh --dairv2x-vehicle5 --dairv2x --yolo --s
./run_batch_experiments.sh --uadetrac-vehicle4 --uadetrac --yolo --s
```

配置通过 `__include__` 组合基础模型、数据集和组件。新增实验应新建派生 YAML，不要复制整套消融配置。

训练输出按协议隔离，格式为 `outputs/<protocol>/<group>/<experiment>`；例如
`DQM-DETR/outputs/dairv2x_vehicle5/main/...`。YOLO 使用同样的协议优先目录规则。

## Checkpoint 与评测

checkpoint 必须与训练协议一致，Vehicle5/Vehicle1 不能与 Vehicle8/Vehicle4 混用。评测结果由 `experiments/common/result_paths.py` 路由到：

```text
experiments/reports/
├── dairv2x_vehicle5/
├── dairv2x_vehicle8/
├── uadetrac_vehicle1/
└── uadetrac_vehicle4/
```

训练、评测、测速和不同实验分组不写入同一张表。主线默认使用单 seed，并在结果中保留
seed 信息；多 seed 汇总仅在确有需要时执行。

UA Vehicle4 权重无须重训即可按 Vehicle1 评测：

```bash
/root/autodl-tmp/cas_trt_env/bin/python \
  experiments/common/evaluate_uadetrac_vehicle1.py \
  --ground-truth /root/autodl-fs/datasets/UA-DETRAC_COCO/annotations/instances_val.json \
  --predictions <vehicle4_predictions.json> \
  --model-name <model>
```

该评测会将四类预测合并为 `vehicle`，执行 class-agnostic NMS，再写入 `uadetrac_vehicle1` 报告目录。

## 部署测速

正式速度使用 TensorRT FP16，并区分：

- `model`：固定设备输入到模型输出；
- `end-to-end`：图像读取、预处理、H2D、推理和 D2H 全流程。

ONNX 导出、TensorRT engine 构建和 benchmark 工具分别位于
`experiments/CaS-DETR/tools/deployment/` 与 `tools/benchmark/`。历史归档和报告见
`artifacts/tensorrt_benchmark/`。

## 复现与限制

开始完整实验前先运行 `--dry-run` 和对应协议测试；长时 GPU、数据集或 TensorRT 任务若未完成，不能把未验证项写入论文结果。不同数据集、类别协议和 small/medium/large 统计不应直接横向比较。

## 许可证与引用

本仓库整合的各框架保留其原始许可证和版权声明。发表或再分发前，请同时遵循 CaS-DETR、DEIM、D-FINE 及所用数据集的引用要求。
