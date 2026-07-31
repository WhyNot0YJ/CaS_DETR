#!/usr/bin/env python3
"""Evaluate UA-DETRAC Vehicle4 predictions as one-class vehicle detection.

This is the framework-independent, no-retraining path. Predictions are first
collapsed to category 1 and then receive class-agnostic NMS before COCOeval.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

from common.dataset_protocol import set_report_protocol
from common.det_eval_metrics import (
    coco_area_ap_at_iou50,
    extract_per_category_ap_from_coco_eval,
    run_coco_bbox_eval,
    write_eval_csv,
)
from common.result_paths import result_csv


def _iou_xywh(left: List[float], right: List[float]) -> float:
    lx1, ly1, lw, lh = map(float, left)
    rx1, ry1, rw, rh = map(float, right)
    lx2, ly2 = lx1 + lw, ly1 + lh
    rx2, ry2 = rx1 + rw, ry1 + rh
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0 else 0.0


def collapse_predictions(
    predictions: Iterable[Dict[str, Any]], nms_iou: float
) -> List[Dict[str, Any]]:
    by_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for source in predictions:
        prediction = dict(source)
        prediction["category_id"] = 1
        by_image[int(prediction["image_id"])].append(prediction)

    kept = []
    for image_predictions in by_image.values():
        candidates = sorted(
            image_predictions, key=lambda item: float(item["score"]), reverse=True
        )
        while candidates:
            best = candidates.pop(0)
            kept.append(best)
            candidates = [
                item
                for item in candidates
                if _iou_xywh(best["bbox"], item["bbox"]) <= nms_iou
            ]
    return kept


def collapse_ground_truth(source: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(source)
    result["images"] = list(source.get("images", []))
    result["annotations"] = []
    for source_annotation in source.get("annotations", []):
        annotation = dict(source_annotation)
        annotation["category_id"] = 1
        result["annotations"].append(annotation)
    result["categories"] = [
        {"id": 1, "name": "vehicle", "supercategory": "vehicle"}
    ]
    return result


def compute_metrics(coco_gt: Dict[str, Any], predictions: List[Dict[str, Any]]) -> Dict[str, float]:
    coco_eval = run_coco_bbox_eval(coco_gt, predictions)
    if coco_eval is None:
        raise RuntimeError("COCOeval failed")
    small50, medium50, large50 = coco_area_ap_at_iou50(coco_eval)
    per50, per5095 = extract_per_category_ap_from_coco_eval(
        coco_eval, coco_gt["categories"]
    )
    return {
        "mAP_0.5": float(coco_eval.stats[1]),
        "mAP_0.75": float(coco_eval.stats[2]),
        "mAP_0.5_0.95": float(coco_eval.stats[0]),
        "AP_small": float(coco_eval.stats[3]),
        "AP_medium": float(coco_eval.stats[4]),
        "AP_large": float(coco_eval.stats[5]),
        "AP_small_50": small50,
        "AP_medium_50": medium50,
        "AP_large_50": large50,
        "AP50_vehicle": per50["vehicle"],
        "AP5095_vehicle": per5095["vehicle"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--output-predictions", type=Path)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_gt = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    source_predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    coco_gt = collapse_ground_truth(source_gt)
    predictions = collapse_predictions(source_predictions, args.nms_iou)
    metrics = compute_metrics(coco_gt, predictions)

    if args.output_predictions:
        args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
        args.output_predictions.write_text(
            json.dumps(predictions, ensure_ascii=False), encoding="utf-8"
        )
    set_report_protocol("uadetrac_vehicle1")
    output_csv = args.output_csv or result_csv("eval_metrics")
    write_eval_csv(
        output_csv,
        model=args.model_name,
        dataset="UA-DETRAC-Vehicle1",
        eval_split=args.split,
        metrics=metrics,
        class_names=["vehicle"],
        append=output_csv.exists(),
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
