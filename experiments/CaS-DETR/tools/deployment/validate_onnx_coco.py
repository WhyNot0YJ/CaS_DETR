#!/usr/bin/env python3
"""Compare deploy-mode PyTorch and ONNX predictions on the same COCO split."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(EXPERIMENTS_DIR))

from common.eval_deim_dfine import (
    _dataset_label2category_map,
    _setup_deim,
    compute_cas_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def append_predictions(targets, labels, boxes, scores, label2category, output):
    for index, target in enumerate(targets):
        image_id = int(target["image_id"].flatten()[0])
        for label, box, score in zip(labels[index], boxes[index], scores[index]):
            label = int(label)
            category_id = int(label2category[label]) if label2category is not None else label
            x1, y1, x2, y2 = (float(value) for value in box)
            output.append({
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            })


def metric_delta(torch_metrics, onnx_metrics):
    return {
        key: float(onnx_metrics[key]) - float(value)
        for key, value in torch_metrics.items()
        if isinstance(value, (int, float)) and key in onnx_metrics
    }


def main():
    args = parse_args()
    config = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    onnx_path = args.onnx.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    solver, cfg, saved_cwd = _setup_deim(str(config), str(checkpoint), framework_dir="CaS-DETR")
    try:
        solver.eval()
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        model = solver.ema.module if solver.ema else solver.model
        model = model.to(device).deploy()
        postprocessor = solver.postprocessor.to(device).deploy()
        if hasattr(getattr(model, "encoder", None), "set_epoch"):
            model.encoder.set_epoch(int(getattr(solver, "last_epoch", 0)))

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
        session = ort.InferenceSession(str(onnx_path), providers=providers)
        output_names = [item.name for item in session.get_outputs()]
        label2category = _dataset_label2category_map(solver.val_dataloader)
        torch_predictions = []
        onnx_predictions = []
        error_sum = {name: 0.0 for name in output_names}
        error_max = {name: 0.0 for name in output_names}
        error_batches = 0

        for samples, targets in solver.val_dataloader:
            orig_sizes = torch.stack([target["orig_size"] for target in targets])
            with torch.no_grad():
                outputs = model(samples.to(device))
                torch_outputs = postprocessor(outputs, orig_sizes.to(device))
            torch_arrays = [value.detach().cpu().numpy() for value in torch_outputs]
            ort_outputs = session.run(None, {
                "images": samples.cpu().numpy(),
                "orig_target_sizes": orig_sizes.cpu().numpy(),
            })
            for name, torch_value, onnx_value in zip(output_names, torch_arrays, ort_outputs):
                error = np.abs(torch_value.astype(np.float64) - onnx_value.astype(np.float64))
                error_sum[name] += float(error.mean())
                error_max[name] = max(error_max[name], float(error.max()) if error.size else 0.0)
            error_batches += 1
            append_predictions(targets, *torch_arrays, label2category, torch_predictions)
            append_predictions(targets, *ort_outputs, label2category, onnx_predictions)

        ann_file = cfg.yaml_cfg["val_dataloader"]["dataset"]["ann_file"]
        torch_metrics, _, _ = compute_cas_metrics(ann_file, torch_predictions, "DAIR-V2X")
        onnx_metrics, _, _ = compute_cas_metrics(ann_file, onnx_predictions, "DAIR-V2X")
        report = {
            "config": str(config),
            "checkpoint": str(checkpoint),
            "onnx": str(onnx_path),
            "providers": session.get_providers(),
            "output_error": {
                name: {
                    "mean_absolute_error": error_sum[name] / max(1, error_batches),
                    "max_absolute_error": error_max[name],
                }
                for name in output_names
            },
            "pytorch_metrics": torch_metrics,
            "onnx_metrics": onnx_metrics,
            "onnx_minus_pytorch": metric_delta(torch_metrics, onnx_metrics),
        }
        (output_dir / "predictions_pytorch.json").write_text(json.dumps(torch_predictions), encoding="utf-8")
        (output_dir / "predictions_onnx.json").write_text(json.dumps(onnx_predictions), encoding="utf-8")
        (output_dir / "onnx_consistency_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    finally:
        os.chdir(saved_cwd)


if __name__ == "__main__":
    main()
