#!/usr/bin/env python3
"""Run the fixed CaS-DETR modification-plan experiment matrix."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


FRAMEWORK_DIR = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = FRAMEWORK_DIR.parent
CONFIGS = {
    "M0": "configs/dataset/ablation/cas_deim_all_off_hgnetv2_s_dairv2x.yml",
    "M1": "configs/deim_dfine/cas_detr_hgnetv2_s_dairv2x.yml",
    "A2": "configs/dataset/ablation/cas_deim_moe4_only_hgnetv2_s_dairv2x.yml",
    "C05": "configs/dataset/ablation/cas_deim_moe4_cap05x_cass_caip_base03_a10_hgnetv2_s_dairv2x.yml",
    "C1": "configs/dataset/ablation/cas_deim_moe4_cap1x_cass_caip_base03_a10_hgnetv2_s_dairv2x.yml",
    "C2": "configs/dataset/ablation/cas_deim_moe4_cap2x_cass_caip_base03_a10_hgnetv2_s_dairv2x.yml",
    "C4": "configs/dataset/ablation/cas_deim_moe4_cap4x_cass_caip_base03_a10_hgnetv2_s_dairv2x.yml",
    "E3": "configs/dataset/ablation/cas_deim_moe3_dim128_cass_caip_base03_a10_hgnetv2_s_dairv2x.yml",
    "E6": "configs/dataset/ablation/cas_deim_moe6_dim128_cass_caip_base03_a10_hgnetv2_s_dairv2x.yml",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        choices=("main", "moe", "moe_capacity", "moe_experts", "moe_scan", "all"),
        default="main",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--ablation-seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--output-root", type=Path, default=FRAMEWORK_DIR / "outputs" / "modification_plan")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-fs/datasets/DAIR-V2X"))
    parser.add_argument("--tuning", type=Path, default=FRAMEWORK_DIR / "pretrained" / "deim_dfine_hgnetv2_s_coco_120e.pth")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--trt-benchmark",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After each experiment, build and benchmark its TensorRT engine (default: enabled).",
    )
    parser.add_argument("--trt-exec", default="trtexec")
    parser.add_argument("--trt-warmup", type=int, default=100)
    parser.add_argument("--trt-iterations", type=int, default=1000)
    return parser.parse_args()


def experiment_matrix(args):
    matrix = []
    if args.group in ("main", "all"):
        matrix.extend((model, seed) for model in ("M0", "M1") for seed in args.seeds)
    if args.group == "moe":
        matrix.extend((model, seed) for model in ("M0", "A2") for seed in args.ablation_seeds)
    elif args.group == "moe_capacity":
        matrix.extend((model, seed) for model in ("C05", "C1", "C2", "C4") for seed in args.ablation_seeds)
    elif args.group == "moe_experts":
        matrix.extend((model, seed) for model in ("E3", "C05", "E6") for seed in args.ablation_seeds)
    elif args.group == "moe_scan":
        matrix.extend((model, seed) for model in ("C05", "C1", "C2", "C4", "E3", "E6") for seed in args.ablation_seeds)
    elif args.group == "all":
        matrix.extend(("A2", seed) for seed in args.ablation_seeds)
    return list(dict.fromkeys(matrix))


def run(command, cwd, dry_run):
    print("+", " ".join(str(part) for part in command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def benchmark_tensorrt(args, model, output_dir, config, checkpoint):
    """Export, build, and measure the batch-1 TensorRT deployment graph."""
    onnx_path = output_dir / "tensorrt" / f"{model}.onnx"
    engine_path = output_dir / "tensorrt" / f"{model}.engine"
    build_log = output_dir / "tensorrt" / f"{model}.build.log"
    benchmark_csv = Path(__file__).resolve().parents[3] / "reports" / "benchmark.csv"
    eval_csv = Path(__file__).resolve().parents[3] / "reports" / "eval_metrics.csv"
    tools_dir = FRAMEWORK_DIR / "tools"

    commands = [
        [
            args.python, str(tools_dir / "deployment/export_onnx_protocol.py"),
            "--framework", "casdeim", "--config", str(config),
            "--checkpoint", str(checkpoint), "--output", str(onnx_path),
        ],
        [
            args.python, str(tools_dir / "deployment/build_trt_engine.py"),
            "--onnx", str(onnx_path), "--engine", str(engine_path),
            "--log", str(build_log), "--trtexec", args.trt_exec,
        ],
        [
            args.python, str(tools_dir / "benchmark/benchmark_trt_protocol.py"),
            "--engine", str(engine_path), "--model", model,
            "--output-csv", str(benchmark_csv), "--eval-csv", str(eval_csv),
            "--warmup", str(args.trt_warmup),
            "--iterations", str(args.trt_iterations), "--run-id", output_dir.name,
            "--framework", "casdeim", "--images", str(args.data_root / "image"),
        ],
    ]
    try:
        for command in commands:
            run(command, EXPERIMENTS_DIR, args.dry_run)
        if args.dry_run:
            return
        with benchmark_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        run_rows = {
            row["mode"]: row for row in rows if row.get("run_id") == output_dir.name
        }
        print(
            f"[TensorRT] {model}: model={float(run_rows['model']['fps']):.2f} FPS, "
            f"end-to-end={float(run_rows['end-to-end']['fps']):.2f} FPS",
            flush=True,
        )
    except Exception as exc:
        print(f"[TensorRT] {model}: benchmark skipped: {exc}", file=sys.stderr, flush=True)


def preflight(args):
    if args.dry_run:
        return
    subprocess.run([args.python, "-c", "import torch, torchvision, yaml"], check=True)
    for path in (
        args.data_root / "annotations" / "instances_train.json",
        args.data_root / "annotations" / "instances_val.json",
        args.tuning,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)


def main():
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.data_root = args.data_root.resolve()
    args.tuning = args.tuning.resolve()
    preflight(args)
    matrix = experiment_matrix(args)
    manifest = {
        "group": args.group,
        "main_seeds": args.seeds,
        "ablation_seeds": args.ablation_seeds,
        "runs": [],
    }

    for model, seed in matrix:
        output_dir = args.output_root / model / f"seed_{seed}"
        config = CONFIGS[model]
        train_command = [
            args.python, "train.py", "-c", config, "--seed", str(seed),
            "--output-dir", str(output_dir), "--tuning", str(args.tuning),
            "-u",
            f"train_dataloader.dataset.img_folder={args.data_root}",
            f"train_dataloader.dataset.ann_file={args.data_root / 'annotations' / 'instances_train.json'}",
            f"val_dataloader.dataset.img_folder={args.data_root}",
            f"val_dataloader.dataset.ann_file={args.data_root / 'annotations' / 'instances_val.json'}",
        ]
        eval_command = [
            args.python, "common/eval_deim_dfine.py",
            "--framework", "casdeim",
            "--config", str(output_dir / "resolved_config.yml"),
            "--resume", str(output_dir / "best.pth"),
            "--model-name", model,
            "--dataset-name", "DAIR-V2X",
            "--output-csv", str(Path(__file__).resolve().parents[3] / "reports" / "eval_metrics.csv"),
            "--predictions-dir", str(output_dir),
            "--router-stats", str(output_dir / "router_stats.json"),
            "--splits", "val",
        ]
        manifest["runs"].append({
            "model": model, "seed": seed, "config": config,
            "output_dir": str(output_dir),
        })
        if not args.dry_run and output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"refusing to mix an existing run: {output_dir}")
        run(train_command, FRAMEWORK_DIR, args.dry_run)
        run(eval_command, EXPERIMENTS_DIR, args.dry_run)
        if args.trt_benchmark:
            benchmark_tensorrt(
                args, model, output_dir, output_dir / "resolved_config.yml",
                output_dir / "best.pth",
            )

    manifest_path = args.output_root / "experiment_manifest.json"
    print("manifest:", manifest_path)
    if not args.dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
