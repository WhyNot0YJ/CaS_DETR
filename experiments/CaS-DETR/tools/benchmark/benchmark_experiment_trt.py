#!/usr/bin/env python3
"""Build and benchmark one deployment engine into the shared benchmark.csv."""

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(EXPERIMENTS_DIR))

from common.dataset_protocol import PROTOCOLS, set_report_protocol
from common.result_paths import result_csv


DATASET_NAMES = {
    "dairv2x_vehicle5": "DAIR-V2X-Vehicle5",
    "dairv2x_vehicle8": "DAIR-V2X-Vehicle8",
    "uadetrac_vehicle1": "UA-DETRAC-Vehicle1",
    "uadetrac_vehicle4": "UA-DETRAC-Vehicle4",
}


def sync_fps_to_eval_metrics(eval_csv: Path, run_id: str, run_rows: dict) -> int:
    """Copy the TensorRT model and pipeline measurements into matching eval rows."""
    if not eval_csv.is_file():
        return 0

    model = run_rows.get("model")
    end_to_end = run_rows.get("end-to-end")
    if not model or not end_to_end:
        raise RuntimeError(f"missing TensorRT benchmark rows for run_id={run_id}")

    with eval_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    updates = {
        "Inference_FPS": f"{float(model['fps']):.2f}",
        "Inference_Latency_ms": f"{float(model['mean_latency_ms']):.2f}",
        "EndToEnd_FPS": f"{float(end_to_end['fps']):.2f}",
        "EndToEnd_Latency_ms": f"{float(end_to_end['mean_latency_ms']):.2f}",
    }
    for field in updates:
        if field not in fields:
            fields.append(field)

    count = 0
    for row in rows:
        if row.get("run_id") == run_id:
            row.update(updates)
            count += 1
    if count:
        with eval_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    return count


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--framework", choices=("casdeim", "deim", "dqmdeim", "dfine", "rtdetr"), required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--run-id", default="")
    p.add_argument("--dataset-protocol", choices=PROTOCOLS, required=True)
    p.add_argument("--seed", default="")
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--trtexec", default="trtexec")
    p.add_argument("--builder", choices=("auto", "trtexec", "python"), default="auto")
    p.add_argument("--caip-static-keep-eval", action="store_true")
    p.add_argument("--caip-eval-keep-ratio", type=float)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--iterations", type=int, default=1000)
    return p.parse_args()


def main():
    args = parse_args()
    set_report_protocol(args.dataset_protocol)
    experiments_dir = EXPERIMENTS_DIR
    deploy = experiments_dir / "CaS-DETR" / "tools" / "deployment"
    benchmark = experiments_dir / "CaS-DETR" / "tools" / "benchmark"
    trt_dir = args.output_dir / "tensorrt"
    onnx = trt_dir / f"{args.model}.onnx"
    engine = trt_dir / f"{args.model}.engine"
    build_log = trt_dir / f"{args.model}.build.log"
    benchmark_csv = result_csv("benchmark")
    eval_csv = result_csv("eval_metrics")
    run_id = args.run_id or args.output_dir.name
    export_command = [
        sys.executable, str(deploy / "export_onnx_protocol.py"), "--framework", args.framework,
        "--config", str(args.config), "--checkpoint", str(args.checkpoint), "--output", str(onnx),
    ]
    export_command.extend(("--dataset-protocol", args.dataset_protocol))
    if args.caip_static_keep_eval:
        export_command.append("--caip-static-keep-eval")
    if args.caip_eval_keep_ratio is not None:
        export_command.extend(("--caip-eval-keep-ratio", str(args.caip_eval_keep_ratio)))
    use_trtexec = args.builder == "trtexec" or (
        args.builder == "auto" and shutil.which(args.trtexec) is not None
    )
    if use_trtexec:
        build_command = [
            sys.executable, str(deploy / "build_trt_engine.py"), "--onnx", str(onnx),
            "--engine", str(engine), "--log", str(build_log), "--trtexec", args.trtexec,
        ]
    else:
        build_command = [
            sys.executable, str(deploy / "build_trt_engine_python.py"), "--onnx", str(onnx),
            "--engine", str(engine),
        ]
    commands = [
        export_command,
        build_command,
        [sys.executable, str(benchmark / "benchmark_trt_protocol.py"), "--engine", str(engine),
         "--model", args.model, "--output-csv", str(benchmark_csv), "--run-id", run_id,
         "--framework", args.framework, "--images", str(args.images), "--warmup", str(args.warmup),
         "--iterations", str(args.iterations), "--preprocess", "letterbox",
         "--dataset", DATASET_NAMES[args.dataset_protocol], "--seed", str(args.seed)],
    ]
    for command in commands:
        subprocess.run(command, cwd=experiments_dir, check=True)
        if command is build_command and not engine.is_file():
            raise RuntimeError(f"TensorRT builder completed without creating engine: {engine}")
    with benchmark_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    run_rows = {
        row["mode"]: row for row in rows if row.get("run_id") == run_id
    }
    updated_rows = sync_fps_to_eval_metrics(eval_csv, run_id, run_rows)
    if not updated_rows:
        raise RuntimeError(
            f"TensorRT rows were written but no eval_metrics rows matched run_id={run_id}"
        )
    print(
        f"[TensorRT] {args.model}: model={float(run_rows['model']['fps']):.2f} FPS, "
        f"end-to-end={float(run_rows['end-to-end']['fps']):.2f} FPS",
        flush=True,
    )


if __name__ == "__main__":
    main()
