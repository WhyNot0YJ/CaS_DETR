#!/usr/bin/env python3
"""Benchmark deploy-mode PyTorch with the same inputs and outputs as ONNX/TensorRT."""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

EXPERIMENTS_DIR = Path(__file__).resolve().parents[3]
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))
from common.result_paths import upsert_csv_rows
from benchmark_runtime import command_output, ensure_gpu_idle, summarize


FRAMEWORK_DIRS = {
    "casdeim": "CaS-DETR",
    "deim": "DEIM",
    "dfine": "D-FINE",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=FRAMEWORK_DIRS, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--seed", default="")
    parser.add_argument("--precision", choices=("fp32", "fp16", "both"), default="both")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    return parser.parse_args()


def load_state(checkpoint):
    state = checkpoint.get("ema", {}).get("module")
    return state if state is not None else checkpoint["model"]


def build_model(framework, config_path, checkpoint_path):
    experiments_dir = Path(__file__).resolve().parents[3]
    framework_dir = experiments_dir / FRAMEWORK_DIRS[framework]
    sys.path.insert(0, str(framework_dir))
    os.chdir(framework_dir)

    if framework == "dfine":
        from src.core import YAMLConfig
    else:
        from engine.core import YAMLConfig

    cfg = YAMLConfig(str(config_path))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(load_state(checkpoint), strict=True)
    if framework == "casdeim" and hasattr(model.encoder, "caip_static_keep_eval"):
        model.encoder.caip_static_keep_eval = True
    return model.deploy(), cfg.postprocessor.deploy()


def benchmark(args, precision):
    model, postprocessor = build_model(
        args.framework, args.config, args.checkpoint
    )
    model = model.to(device="cuda", dtype=torch.float32).eval()
    postprocessor = postprocessor.to("cuda").eval()
    images = torch.rand(1, 3, 640, 640, device="cuda", dtype=torch.float32)
    sizes = torch.tensor([[640, 640]], device="cuda")

    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=precision == "fp16"
    ):
        for _ in range(args.warmup):
            postprocessor(model(images), sizes)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(args.iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            postprocessor(model(images), sizes)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
    return {
        "run_id": args.run_id or args.model,
        "framework": args.framework,
        "dataset": args.dataset,
        "seed": args.seed,
        "result_type": "benchmark",
        "model": args.model,
        "engine": "",
        "backend": "pytorch",
        "precision": precision,
        "mode": "model",
        "batch_size": 1,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "execution": "cuda_event",
        "aux_streams": "",
        **summarize(times),
        "peak_memory_mib": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "gpu": torch.cuda.get_device_name(0),
        "driver": command_output([
            "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
        ]),
        "tensorrt": "",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "image_root": "",
        "preprocess": "none",
        "measured_at": datetime.now(timezone.utc).isoformat(),
    }


def main():
    args = parse_args()
    ensure_gpu_idle()
    args.config = args.config.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_csv = args.output_csv.resolve()
    precisions = ("fp32", "fp16") if args.precision == "both" else (args.precision,)
    rows = [benchmark(args, precision) for precision in precisions]
    upsert_csv_rows(
        args.output_csv,
        rows,
        key_fields=("run_id", "framework", "backend", "precision", "mode"),
    )
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
