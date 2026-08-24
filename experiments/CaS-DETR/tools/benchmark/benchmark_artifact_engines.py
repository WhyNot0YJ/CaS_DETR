#!/usr/bin/env python3
"""Benchmark every archived TensorRT engine into one normalized CSV."""

import argparse
import subprocess
import sys
from pathlib import Path


SPECS = {
    "deim_ua": (
        "deim", "UA-DETRAC", "deim_ua_fp16.engine",
        "/root/autodl-fs/datasets/UA-DETRAC_COCO/test",
    ),
    "dfine_ua": (
        "dfine", "UA-DETRAC", "dfine_ua_fp16.engine",
        "/root/autodl-fs/datasets/UA-DETRAC_COCO/test",
    ),
    "cas_dair_base03": (
        "casdeim", "DAIR-V2X", "cas_dair_base03_fp16.engine",
        "/root/autodl-fs/datasets/DAIR-V2X_YOLO/images/eval",
    ),
    "cas_ua_base05": (
        "casdeim", "UA-DETRAC", "cas_ua_base05_fp16.engine",
        "/root/autodl-fs/datasets/UA-DETRAC_COCO/test",
    ),
    "cas_05x_dair_base05": (
        "casdeim", "DAIR-V2X", "cas_05x_dair_base05_fp16.engine",
        "/root/autodl-fs/datasets/DAIR-V2X_YOLO/images/eval",
    ),
    "cas_05x_dair_base05_dense_experts": (
        "casdeim", "DAIR-V2X", "cas_05x_dair_base05_dense_experts_fp16.engine",
        "/root/autodl-fs/datasets/DAIR-V2X_YOLO/images/eval",
    ),
}
REPO_ROOT = Path(__file__).resolve().parents[4]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "tensorrt_benchmark",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--models", nargs="+", choices=SPECS, default=list(SPECS))
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    artifacts_dir = args.artifacts_dir.resolve()
    output = args.output.resolve() if args.output else artifacts_dir / "benchmark.csv"
    protocol = Path(__file__).with_name("benchmark_trt_protocol.py")
    for model in args.models:
        _, _, engine_name, images = SPECS[model]
        if not (artifacts_dir / engine_name).is_file():
            raise FileNotFoundError(artifacts_dir / engine_name)
        if not Path(images).is_dir():
            raise FileNotFoundError(images)
    for model in args.models:
        framework, dataset, engine_name, images = SPECS[model]
        command = [
            sys.executable,
            str(protocol),
            "--engine", str(artifacts_dir / engine_name),
            "--model", model,
            "--output-csv", str(output),
            "--run-id", model,
            "--framework", framework,
            "--dataset", dataset,
            "--images", images,
            "--preprocess", "resize",
            "--warmup", str(args.warmup),
            "--iterations", str(args.iterations),
        ]
        subprocess.run(command, check=True)
    print(output)


if __name__ == "__main__":
    main()
