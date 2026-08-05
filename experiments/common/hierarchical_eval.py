"""Label-collapse-only evaluation for fine-grained training taxonomies.

The detector's native predictions are preserved exactly.  Only ``category_id``
is remapped; no NMS, score fusion, thresholding, or box filtering is applied.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from common.det_eval_metrics import (
    canonical_category_metric_name,
    coco_ap_at_iou50_all,
    coco_area_ap_at_iou50,
    extract_per_category_ap_from_coco_eval,
    run_coco_bbox_eval,
)


HIERARCHICAL_EVAL_SPECS: Dict[str, Dict[str, Any]] = {
    "dairv2x_vehicle8_to_vehicle5": {
        "training_protocol": "dairv2x_vehicle8",
        "evaluation_taxonomy": "dairv2x_vehicle5_label_collapse_only",
        "category_map": {1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5},
        "categories": [
            {"id": 1, "name": "vehicle", "supercategory": "vehicle"},
            {"id": 2, "name": "Pedestrian", "supercategory": "Pedestrian"},
            {"id": 3, "name": "Cyclist", "supercategory": "Cyclist"},
            {"id": 4, "name": "Motorcyclist", "supercategory": "Motorcyclist"},
            {"id": 5, "name": "Trafficcone", "supercategory": "Trafficcone"},
        ],
    },
    "uadetrac_vehicle4_to_vehicle1": {
        "training_protocol": "uadetrac_vehicle4",
        "evaluation_taxonomy": "uadetrac_vehicle1_label_collapse_only",
        "category_map": {1: 1, 2: 1, 3: 1, 4: 1},
        "categories": [
            {"id": 1, "name": "vehicle", "supercategory": "vehicle"},
        ],
    },
}

HIERARCHICAL_EVAL_CHOICES: Tuple[str, ...] = tuple(HIERARCHICAL_EVAL_SPECS)


def hierarchical_eval_spec(mode: str) -> Dict[str, Any]:
    try:
        return HIERARCHICAL_EVAL_SPECS[str(mode)]
    except KeyError as exc:
        raise ValueError(f"unsupported hierarchical evaluation mode: {mode!r}") from exc


def _mapped_category_id(category_id: object, category_map: Mapping[int, int]) -> int:
    source_id = int(category_id)
    try:
        return int(category_map[source_id])
    except KeyError as exc:
        raise ValueError(f"source category_id {source_id} is outside the declared taxonomy") from exc


def collapse_predictions(
    predictions: Iterable[Mapping[str, Any]], mode: str
) -> List[Dict[str, Any]]:
    """Copy predictions and remap labels without suppressing or modifying boxes."""
    category_map = hierarchical_eval_spec(mode)["category_map"]
    collapsed: List[Dict[str, Any]] = []
    for source in predictions:
        prediction = dict(source)
        source_id = int(prediction["category_id"])
        prediction["source_category_id"] = source_id
        prediction["category_id"] = _mapped_category_id(source_id, category_map)
        collapsed.append(prediction)
    return collapsed


def collapse_ground_truth(source: Mapping[str, Any], mode: str) -> Dict[str, Any]:
    """Copy a COCO ground truth dictionary and remap annotation labels only."""
    spec = hierarchical_eval_spec(mode)
    result = dict(source)
    result["images"] = [dict(image) for image in source.get("images", [])]
    result["annotations"] = []
    for source_annotation in source.get("annotations", []):
        annotation = dict(source_annotation)
        source_id = int(annotation["category_id"])
        annotation["source_category_id"] = source_id
        annotation["category_id"] = _mapped_category_id(
            source_id, spec["category_map"]
        )
        result["annotations"].append(annotation)
    result["categories"] = [dict(category) for category in spec["categories"]]
    return result


def compute_coco_metrics(
    coco_gt: Dict[str, Any], predictions: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Compute the shared AP columns from one COCO ground truth/prediction pair."""
    coco_eval = run_coco_bbox_eval(coco_gt, predictions)
    if coco_eval is None:
        raise RuntimeError("COCOeval failed for hierarchical evaluation input")
    small50, medium50, large50 = coco_area_ap_at_iou50(coco_eval)
    per50, per5095 = extract_per_category_ap_from_coco_eval(
        coco_eval, list(coco_gt.get("categories", []))
    )
    metrics = {
        "mAP_50": coco_ap_at_iou50_all(coco_eval),
        "mAP_5095": max(0.0, float(coco_eval.stats[0])),
        "AP_small_50": small50,
        "AP_medium_50": medium50,
        "AP_large_50": large50,
        "AP_small_5095": max(0.0, float(coco_eval.stats[3])),
        "AP_medium_5095": max(0.0, float(coco_eval.stats[4])),
        "AP_large_5095": max(0.0, float(coco_eval.stats[5])),
    }
    for category in coco_gt.get("categories", []):
        suffix = canonical_category_metric_name(category["name"])
        metrics[f"AP50_{suffix}"] = per50[suffix]
        metrics[f"AP5095_{suffix}"] = per5095[suffix]
    return metrics


def hierarchical_metadata(mode: str, checkpoint_sha256: str = "") -> Dict[str, object]:
    spec = hierarchical_eval_spec(mode)
    return {
        "training_taxonomy": spec["training_protocol"],
        "evaluation_taxonomy": spec["evaluation_taxonomy"],
        "postprocess": "native_label_collapse_only",
        "checkpoint_sha256": checkpoint_sha256,
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

