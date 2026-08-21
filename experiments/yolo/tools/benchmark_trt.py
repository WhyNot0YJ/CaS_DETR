#!/usr/bin/env python3
"""Export a YOLO/YOLOX detector with NMS and benchmark both speed scopes.

The model branch times the deployment graph; the end-to-end branch additionally
includes image reading, letterbox, host-to-device, and result copies.
"""

import argparse
import gc
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--model", help="Name recorded in the benchmark CSV")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--seed", default="")
    parser.add_argument("--training-taxonomy", default="")
    parser.add_argument("--evaluation-taxonomy", default="")
    parser.add_argument("--postprocess", default="")
    parser.add_argument("--yolox-exp", type=Path)
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--trtexec", default="trtexec")
    parser.add_argument("--builder", choices=("auto", "trtexec", "python"), default="auto")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    return parser.parse_args()


def export_ultralytics_onnx(weights, output, args):
    yolo_dir = Path(__file__).resolve().parents[1]
    external_dir = yolo_dir / "external"
    sys.path.insert(0, str(external_dir))
    from ultralytics import YOLO

    model = YOLO(str(weights))
    if model.task != "detect":
        raise ValueError(f"only detection weights are supported, got task={model.task!r}")
    exported = Path(model.export(
        format="onnx",
        imgsz=args.imgsz,
        batch=1,
        dynamic=False,
        half=False,
        nms=True,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=0,
    ))
    output.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != output.resolve():
        shutil.move(str(exported), output)


class YOLOXNMSModel(nn.Module):
    """YOLOX decode plus class-aware NMS with a fixed TensorRT output shape."""

    def __init__(self, model, conf, iou, max_det, max_wh=4096):
        super().__init__()
        self.model = model
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.max_wh = max_wh

    def forward(self, images):
        from torchvision.ops import nms

        pred = self.model(images)
        xy = pred[0, :, :2]
        wh = pred[0, :, 2:4]
        boxes = torch.cat((xy - wh / 2, xy + wh / 2), dim=1)
        scores, classes = (pred[0, :, 4:5] * pred[0, :, 5:]).max(dim=1)
        mask = scores > self.conf
        boxes, scores, classes = boxes[mask], scores[mask], classes[mask]
        nms_boxes = boxes + classes.to(boxes.dtype).unsqueeze(1) * self.max_wh
        keep = nms(nms_boxes, scores, self.iou)[: self.max_det]
        detections = torch.cat(
            (
                boxes[keep],
                scores[keep].unsqueeze(1),
                classes[keep].to(boxes.dtype).unsqueeze(1),
            ),
            dim=1,
        )
        padded = torch.nn.functional.pad(
            detections,
            (0, 0, 0, self.max_det - detections.shape[0]),
        )
        return padded.reshape(1, self.max_det, 6)


def export_yolox_onnx(weights, output, args):
    if args.yolox_exp is None:
        raise ValueError("--yolox-exp is required for YOLOX checkpoints")
    yolo_dir = Path(__file__).resolve().parents[1]
    yolox_dir = yolo_dir / "external" / "YOLOX"
    for path in (yolo_dir, yolox_dir, args.yolox_exp.resolve().parent):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from yolox.exp import get_exp
    from yolox.models.network_blocks import SiLU
    from yolox.utils import fuse_model, replace_module

    exp = get_exp(exp_file=str(args.yolox_exp.resolve()))
    if args.num_classes is None:
        raise ValueError("--num-classes is required for YOLOX export")
    exp.num_classes = args.num_classes
    model = exp.get_model()
    try:
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(weights, map_location="cpu")
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    model = replace_module(fuse_model(model).eval(), nn.SiLU, SiLU)
    model.head.decode_in_inference = True
    wrapper = YOLOXNMSModel(model, args.conf, args.iou, args.max_det).eval()
    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        str(output),
        input_names=["images"],
        output_names=["detections"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )


def export_onnx(weights, output, args):
    if args.yolox_exp is not None:
        export_yolox_onnx(weights, output, args)
    else:
        export_ultralytics_onnx(weights, output, args)


def main():
    args = parse_args()
    args.weights = args.weights.resolve()
    if not args.weights.is_file():
        raise FileNotFoundError(f"weights not found: {args.weights}")

    output_dir = args.output_dir.resolve()
    trt_dir = output_dir / "tensorrt"
    trt_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.model or args.weights.stem
    run_id = args.run_id or output_dir.name
    experiments_dir = Path(__file__).resolve().parents[2]
    if str(experiments_dir) not in sys.path:
        sys.path.insert(0, str(experiments_dir))
    from common.result_paths import result_csv
    from common.trt_provenance import (
        artifact_hash_suffix,
        build_engine_provenance,
        engine_is_reusable,
        sha256_file,
        write_engine_provenance,
    )

    config_path = (args.config or (output_dir / "config.yaml")).resolve()
    export_options = {
        "imgsz": args.imgsz,
        "batch": 1,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "native_nms": True,
        "num_classes": args.num_classes,
        "yolox_exp": str(args.yolox_exp.resolve()) if args.yolox_exp else "",
        "yolox_exp_sha256": (
            sha256_file(args.yolox_exp.resolve()) if args.yolox_exp else ""
        ),
    }
    provenance = build_engine_provenance(
        checkpoint=args.weights,
        config=config_path,
        framework="yolox" if args.yolox_exp else "yolo",
        export_options=export_options,
    )
    artifact_stem = (
        f"{model_name}_nms_fp16_{artifact_hash_suffix(provenance)}"
    )
    onnx = trt_dir / f"{artifact_stem}.onnx"
    engine = trt_dir / f"{artifact_stem}.engine"
    build_log = trt_dir / f"{artifact_stem}.build.log"

    benchmark_csv = result_csv("benchmark")
    eval_csv = result_csv("eval_metrics")

    shared_dir = experiments_dir / "CaS-DETR" / "tools"
    use_trtexec = args.builder == "trtexec" or (
        args.builder == "auto" and shutil.which(args.trtexec) is not None
    )
    build_script = "build_trt_engine.py" if use_trtexec else "build_trt_engine_python.py"
    build_command = [
        sys.executable, str(shared_dir / "deployment" / build_script),
        "--onnx", str(onnx), "--engine", str(engine),
    ]
    if use_trtexec:
        build_command.extend(("--log", str(build_log), "--trtexec", args.trtexec))
    benchmark_command = [
            sys.executable, str(shared_dir / "benchmark" / "benchmark_trt_protocol.py"),
            "--engine", str(engine), "--model", model_name, "--output-csv", str(benchmark_csv),
            "--eval-csv", str(eval_csv),
            "--run-id", run_id, "--framework", "yolox" if args.yolox_exp else "yolo",
            "--dataset", args.dataset, "--seed", args.seed,
            "--images", str(args.images.resolve()), "--imgsz", str(args.imgsz),
            "--preprocess", "letterbox", "--warmup", str(args.warmup),
            "--iterations", str(args.iterations),
            "--checkpoint-sha256", provenance["checkpoint_sha256"],
            "--config-sha256", provenance["config_sha256"],
            "--training-taxonomy", args.training_taxonomy,
            "--evaluation-taxonomy", args.evaluation_taxonomy,
            "--postprocess", args.postprocess,
    ]
    if engine_is_reusable(engine, provenance):
        print(f"Reusing hash-matched engine: {engine}")
    else:
        export_onnx(args.weights, onnx, args)
        # ONNX export may leave the PyTorch model/context cached on CUDA.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        subprocess.run(build_command, check=True)
        if not engine.is_file():
            raise RuntimeError(f"TensorRT builder completed without creating engine: {engine}")
        write_engine_provenance(engine, provenance)
    subprocess.run(benchmark_command, check=True)

    print(f"ONNX: {onnx}")
    print(f"Engine: {engine}")
    print(f"Benchmark: {benchmark_csv}")


if __name__ == "__main__":
    main()
