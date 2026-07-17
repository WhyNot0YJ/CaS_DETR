#!/usr/bin/env python3
"""Build and benchmark one deployment engine into the shared benchmark.csv."""

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--framework", choices=("casdeim", "deim", "dfine", "rtdetr"), required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--trtexec", default="trtexec")
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--iterations", type=int, default=1000)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[3]
    deploy = root / "CaS-DETR" / "tools" / "deployment"
    benchmark = root / "CaS-DETR" / "tools" / "benchmark"
    trt_dir = args.output_dir / "tensorrt"
    onnx = trt_dir / f"{args.model}.onnx"
    engine = trt_dir / f"{args.model}.engine"
    build_log = trt_dir / f"{args.model}.build.log"
    benchmark_csv = root / "reports" / "benchmark.csv"
    commands = [
        [sys.executable, str(deploy / "export_onnx_protocol.py"), "--framework", args.framework,
         "--config", str(args.config), "--checkpoint", str(args.checkpoint), "--output", str(onnx)],
        [sys.executable, str(deploy / "build_trt_engine.py"), "--onnx", str(onnx),
         "--engine", str(engine), "--log", str(build_log), "--trtexec", args.trtexec],
        [sys.executable, str(benchmark / "benchmark_trt_protocol.py"), "--engine", str(engine),
         "--model", args.model, "--output-csv", str(benchmark_csv), "--run-id", args.output_dir.name,
         "--framework", args.framework, "--images", str(args.images), "--warmup", str(args.warmup),
         "--iterations", str(args.iterations)],
    ]
    for command in commands:
        subprocess.run(command, cwd=root, check=True)
    with benchmark_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    run_id = args.output_dir.name
    run_rows = {
        row["mode"]: row for row in rows if row.get("run_id") == run_id
    }
    print(
        f"[TensorRT] {args.model}: model={float(run_rows['model']['fps']):.2f} FPS, "
        f"end-to-end={float(run_rows['end-to-end']['fps']):.2f} FPS",
        flush=True,
    )


if __name__ == "__main__":
    main()
