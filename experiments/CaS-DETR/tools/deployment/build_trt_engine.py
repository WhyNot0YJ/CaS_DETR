#!/usr/bin/env python3
"""Build a reproducible TensorRT FP16 engine with trtexec."""

import argparse
import json
import platform
import shutil
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--trtexec", default="trtexec")
    parser.add_argument("--workspace-mib", type=int, default=4096)
    parser.add_argument("--max-aux-streams", type=int)
    parser.add_argument("--builder-optimization-level", type=int, choices=range(0, 6), default=5)
    return parser.parse_args()


def output(command):
    try:
        return subprocess.run(command, check=False, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def main():
    args = parse_args()
    executable = shutil.which(args.trtexec)
    if executable is None:
        raise FileNotFoundError(f"cannot find trtexec: {args.trtexec}")
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        f"--onnx={args.onnx.resolve()}",
        f"--saveEngine={args.engine.resolve()}",
        "--fp16",
        f"--memPoolSize=workspace:{args.workspace_mib}",
        f"--builderOptimizationLevel={args.builder_optimization_level}",
        "--skipInference",
    ]
    if args.max_aux_streams is not None:
        command.append(f"--maxAuxStreams={args.max_aux_streams}")
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    args.log.write_text(result.stdout + result.stderr, encoding="utf-8")
    environment = {
        "platform": platform.platform(),
        "driver": output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "gpu": output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]),
        "trtexec_version": output([executable, "--version"]),
        "command": command,
        "returncode": result.returncode,
    }
    args.engine.with_suffix(".environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
