#!/usr/bin/env python3
"""Aggregate per-seed evaluation CSV files for the modification plan."""

import argparse
import csv
import statistics
from pathlib import Path
import sys

EXPERIMENTS_DIR = Path(__file__).resolve().parents[3]
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))
from common.result_paths import result_csv


METRICS = (
    "mAP_5095", "mAP_50", "mAP_75", "AP_small_5095", "AP_small_50",
    "Params_M", "Active_Params_M", "GFLOPs",
)


def _is_official_eval(row):
    protocol = " ".join(str(row.get(key, "")) for key in ("dataset", "training_taxonomy")).lower()
    return row.get("eval_split") == ("test" if "ua-detrac" in protocol or "uadetrac" in protocol else "eval")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    return parser.parse_args()


def read_seed(root, model, seed):
    path = root / model / f"seed_{seed}" / "eval_metrics.csv"
    if not path.exists():
        path = result_csv("eval_metrics")
    with path.open(newline="", encoding="utf-8") as f:
        rows = [
            row for row in csv.DictReader(f)
            if _is_official_eval(row)
            and (path != result_csv("eval_metrics") or (
                row.get("model") == model and str(row.get("seed", "")) == str(seed)
            ))
        ]
    if len(rows) != 1:
        raise ValueError(f"expected one official eval row in {path}, found {len(rows)}")
    return {metric: float(rows[0][metric]) for metric in METRICS}


def mean_std(values):
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def write_summary(path, models, data):
    fields = ["model", "num_seeds"]
    for metric in METRICS:
        fields.extend((f"{metric}_mean", f"{metric}_std", f"{metric}_mean_std"))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for model in models:
            row = {"model": model, "num_seeds": len(data[model])}
            for metric in METRICS:
                mean, std = mean_std([seed_data[metric] for seed_data in data[model].values()])
                row[f"{metric}_mean"] = f"{mean:.6f}"
                row[f"{metric}_std"] = f"{std:.6f}"
                row[f"{metric}_mean_std"] = f"{mean:.4f} +/- {std:.4f}"
            writer.writerow(row)


def main():
    args = parse_args()
    root = args.root.resolve()
    data = {
        model: {seed: read_seed(root, model, seed) for seed in args.seeds}
        for model in ("M0", "M1")
    }
    write_summary(root / "main_3seed_summary.csv", ("M0", "M1"), data)

    with (root / "paired_deltas.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["seed"] + [f"delta_{metric}" for metric in METRICS]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for seed in args.seeds:
            row = {"seed": seed}
            for metric in METRICS:
                row[f"delta_{metric}"] = f"{data['M1'][seed][metric] - data['M0'][seed][metric]:.6f}"
            writer.writerow(row)

    a2 = {}
    for seed in args.seeds:
        path = root / "A2" / f"seed_{seed}" / "eval_metrics.csv"
        if path.exists():
            a2[seed] = read_seed(root, "A2", seed)
    if not a2:
        raise FileNotFoundError(f"no seed results found for A2 under {root}")
    moe_ablation = {"M0": data["M0"], "A2": a2}
    write_summary(root / "moe_ablation.csv", ("M0", "A2"), moe_ablation)


if __name__ == "__main__":
    main()
