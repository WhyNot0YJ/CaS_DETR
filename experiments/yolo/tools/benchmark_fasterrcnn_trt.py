#!/usr/bin/env python3
"""Export a trained Faster R-CNN checkpoint and benchmark its TensorRT engine."""

import argparse
import gc
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--model", default="fasterrcnn_resnet50_fpn")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--seed", default="")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    return parser.parse_args()


def build_model(num_classes: int, imgsz: int, max_det: int) -> nn.Module:
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        min_size=imgsz,
        max_size=imgsz,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    model.roi_heads.detections_per_img = max_det
    return model


class FasterRCNNExportModel(nn.Module):
    """Give TensorRT a fixed batch-1 tensor input and a fixed detection output."""

    def __init__(self, model: nn.Module, max_det: int, nms_threshold: float):
        super().__init__()
        self.model = model
        self.max_det = max_det
        self.nms_threshold = nms_threshold

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        from torchvision.ops import batched_nms

        prediction = self.model([images[0]])[0]
        keep = batched_nms(
            prediction["boxes"],
            prediction["scores"],
            prediction["labels"],
            self.nms_threshold,
        )[: self.max_det]
        detections = torch.cat(
            (
                prediction["boxes"][keep],
                prediction["scores"][keep].unsqueeze(1),
                (prediction["labels"][keep] - 1).to(torch.float32).unsqueeze(1),
            ),
            dim=1,
        )
        detections = F.pad(
            detections, (0, 0, 0, self.max_det - detections.shape[0]),
        )
        return detections.reshape(1, self.max_det, 6)


def export_onnx(args, output: Path) -> None:
    try:
        checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.weights, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model = build_model(args.num_classes, args.imgsz, args.max_det)
    model.load_state_dict(state)
    wrapper = FasterRCNNExportModel(
        model.eval(), args.max_det, model.roi_heads.nms_thresh,
    ).eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        torch.zeros(1, 3, args.imgsz, args.imgsz),
        str(output),
        input_names=["images"],
        output_names=["detections"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )


def main():
    args = parse_args()
    args.weights = args.weights.resolve()
    if not args.weights.is_file():
        raise FileNotFoundError(f"weights not found: {args.weights}")

    trt_dir = args.output_dir.resolve() / "tensorrt"
    onnx = trt_dir / f"{args.model}_nms.onnx"
    engine = trt_dir / f"{args.model}_nms_fp16.engine"
    experiments_dir = Path(__file__).resolve().parents[2]
    if str(experiments_dir) not in sys.path:
        sys.path.insert(0, str(experiments_dir))
    from common.result_paths import result_csv

    export_onnx(args, onnx)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cas_tools = experiments_dir / "CaS-DETR" / "tools"
    subprocess.run(
        [
            sys.executable,
            str(cas_tools / "deployment" / "build_trt_engine_python.py"),
            "--onnx", str(onnx),
            "--engine", str(engine),
            "--init-plugins",
        ],
        check=True,
    )
    if not engine.is_file():
        raise RuntimeError(f"TensorRT builder completed without creating engine: {engine}")
    subprocess.run(
        [
            sys.executable,
            str(cas_tools / "benchmark" / "benchmark_trt_protocol.py"),
            "--engine", str(engine),
            "--model", args.model,
            "--output-csv", str(result_csv("benchmark")),
            "--eval-csv", str(result_csv("eval_metrics")),
            "--run-id", args.run_id or args.output_dir.resolve().name,
            "--framework", "fasterrcnn",
            "--dataset", args.dataset,
            "--seed", str(args.seed),
            "--images", str(args.images.resolve()),
            "--imgsz", str(args.imgsz),
            "--preprocess", "letterbox",
            "--warmup", str(args.warmup),
            "--iterations", str(args.iterations),
        ],
        check=True,
    )
    print(f"ONNX: {onnx}")
    print(f"Engine: {engine}")
    print(f"Benchmark: {result_csv('benchmark')}")


if __name__ == "__main__":
    main()
