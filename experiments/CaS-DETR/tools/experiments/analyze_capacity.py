#!/usr/bin/env python3
"""Offline analysis for existing 0.5x/1x/2x/4x MoE artifacts."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


EXPERIMENTS_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(EXPERIMENTS_DIR))

from common.det_eval_metrics import run_coco_bbox_eval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, help="CSV with capacity,predictions,eval_metrics,log_file")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def area_ap_at_iou(coco_eval, area_index):
    precision = coco_eval.eval["precision"]
    values = precision[0, :, :, area_index, -1]
    values = values[values > -1]
    return float(np.mean(values)) if values.size else 0.0


def compute_iou_curves(annotations, predictions):
    gt = json.loads(annotations.read_text(encoding="utf-8"))
    preds = json.loads(predictions.read_text(encoding="utf-8"))
    rows = []
    for threshold in np.arange(0.50, 0.951, 0.05):
        evaluator = run_coco_bbox_eval(gt, preds)
        evaluator.params.iouThrs = np.array([threshold])
        evaluator.evaluate()
        evaluator.accumulate()
        rows.append({
            "iou": float(threshold),
            "AP_all": area_ap_at_iou(evaluator, 0),
            "AP_small": area_ap_at_iou(evaluator, 1),
            "AP_medium": area_ap_at_iou(evaluator, 2),
            "AP_large": area_ap_at_iou(evaluator, 3),
        })
    return rows


def extract_log(capacity, path):
    curves = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            coco = record.get("test_coco_eval_bbox", [])
            curves.append({
                "capacity": capacity,
                "epoch": int(record["epoch"]),
                "train_loss": float(record.get("train_loss", 0.0)),
                "val_mAP_5095": float(coco[0]) if coco else 0.0,
                "val_AP50": float(coco[1]) if len(coco) > 1 else 0.0,
                "val_AP75": float(coco[2]) if len(coco) > 2 else 0.0,
            })
    return curves


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_row(path):
    rows = [row for row in read_rows(path) if row.get("eval_split") == "val"]
    if len(rows) != 1:
        raise ValueError(f"expected one val row in {path}, found {len(rows)}")
    return rows[0]


def plot_pareto(rows, output_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    pairs = (
        ("Params_M", "AP_small_50"),
        ("FPS", "AP_small_50"),
        ("Params_M", "AP_small_5095"),
        ("FPS", "AP_small_5095"),
    )
    for ax, (x_key, y_key) in zip(axes.flat, pairs):
        for row in rows:
            x, y = float(row[x_key]), float(row[y_key])
            ax.scatter(x, y)
            ax.annotate(row["capacity"], (x, y))
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_rows(args.manifest)
    all_curves = []
    all_training = []
    summary = []

    for item in manifest:
        capacity = item["capacity"]
        for row in compute_iou_curves(args.annotations, Path(item["predictions"])):
            all_curves.append({"capacity": capacity, **row})
        training = extract_log(capacity, Path(item["log_file"]))
        all_training.extend(training)
        best = max(training, key=lambda row: row["val_mAP_5095"])
        metrics = metric_row(Path(item["eval_metrics"]))
        summary.append({
            "capacity": capacity,
            "best_epoch": best["epoch"],
            "best_val_mAP_5095": best["val_mAP_5095"],
            "Params_M": metrics["Params_M"],
            "Active_Params_M": metrics["Active_Params_M"],
            "FPS": metrics["FPS"],
            "AP_small_50": metrics["AP_small_50"],
            "AP_small_5095": metrics["AP_small_5095"],
        })

    write_csv(args.output_dir / "capacity_iou_ap_curves.csv", all_curves)
    write_csv(args.output_dir / "capacity_train_val_curves.csv", all_training)
    write_csv(args.output_dir / "capacity_summary.csv", summary)
    plot_pareto(summary, args.output_dir / "capacity_pareto.png")


if __name__ == "__main__":
    main()
