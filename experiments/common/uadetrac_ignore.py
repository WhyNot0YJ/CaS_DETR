"""Shared UA-DETRAC ignore-region handling for training and COCO evaluation."""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Mapping, Sequence


IGNORE_LABEL = -1
IGNORE_IOA_THRESHOLD = 0.5


def ignore_regions_xyxy(
    image: Mapping[str, Any],
    *,
    width: float | None = None,
    height: float | None = None,
) -> List[List[float]]:
    """Return valid ``images[].ignore_regions`` rectangles as clipped xyxy boxes."""
    width = float(image.get("width", width)) if image.get("width", width) is not None else None
    height = float(image.get("height", height)) if image.get("height", height) is not None else None
    boxes: List[List[float]] = []
    for region in image.get("ignore_regions", []) or []:
        bbox = region.get("bbox") if isinstance(region, Mapping) else region
        if not isinstance(bbox, Sequence) or len(bbox) != 4:
            continue
        x, y, w, h = (float(value) for value in bbox)
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = x + max(0.0, w), y + max(0.0, h)
        if width is not None:
            x1, x2 = min(x1, width), min(x2, width)
        if height is not None:
            y1, y2 = min(y1, height), min(y2, height)
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
    return boxes


def prediction_ioa_xyxy(boxes: Any, ignore_boxes: Any) -> Any:
    """Return max intersection-over-prediction-area for xyxy boxes."""
    import torch

    if boxes.numel() == 0 or ignore_boxes.numel() == 0:
        return torch.zeros(boxes.shape[0], dtype=boxes.dtype, device=boxes.device)
    ignore_boxes = ignore_boxes.to(device=boxes.device, dtype=boxes.dtype)
    left_top = torch.maximum(boxes[:, None, :2], ignore_boxes[None, :, :2])
    right_bottom = torch.minimum(boxes[:, None, 2:], ignore_boxes[None, :, 2:])
    intersection = (right_bottom - left_top).clamp(min=0).prod(-1)
    area = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(-1).clamp_min(1e-12)
    return (intersection / area[:, None]).amax(dim=1)


def filter_tensor_predictions_by_ignore(
    predictions: Mapping[Any, Mapping[str, Any]],
    coco_dataset: Mapping[str, Any],
    threshold: float = IGNORE_IOA_THRESHOLD,
) -> Dict[Any, Dict[str, Any]]:
    """Apply the official UA fixed-IoA rule to batched tensor predictions."""
    import torch

    images = {int(image["id"]): image for image in coco_dataset.get("images", []) or []}
    output: Dict[Any, Dict[str, Any]] = {}
    for image_id, prediction in predictions.items():
        prediction = dict(prediction)
        boxes = prediction.get("boxes")
        image = images.get(int(image_id), {})
        ignores = ignore_regions_xyxy(image)
        if isinstance(boxes, torch.Tensor) and boxes.numel() and ignores:
            ignore_tensor = boxes.new_tensor(ignores)
            keep = prediction_ioa_xyxy(boxes, ignore_tensor) < threshold
            for key, value in tuple(prediction.items()):
                if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == keep.shape[0]:
                    prediction[key] = value[keep]
        output[image_id] = prediction
    return output


def _ignores_by_image(coco_dataset: Mapping[str, Any]) -> Dict[int, List[List[float]]]:
    """Map image id -> clipped ignore rectangles, skipping images without any."""
    return {
        int(image["id"]): ignore_regions_xyxy(image)
        for image in coco_dataset.get("images", []) or []
        if image.get("ignore_regions")
    }


def _is_ignored_xywh(bbox: Any, ignores: Sequence[Sequence[float]], threshold: float) -> bool:
    """Apply the official fixed-IoA rule to a single COCO xywh box."""
    x, y, width, height = (float(value) for value in bbox)
    area = max(0.0, width) * max(0.0, height)
    if area <= 0 or not ignores:
        return False
    x2, y2 = x + max(0.0, width), y + max(0.0, height)
    return any(
        max(0.0, min(x2, ix2) - max(x, ix1)) * max(0.0, min(y2, iy2) - max(y, iy1)) / area >= threshold
        for ix1, iy1, ix2, iy2 in ignores
    )


def filter_coco_predictions_by_ignore(
    coco_dataset: Mapping[str, Any],
    predictions: Iterable[Mapping[str, Any]],
    threshold: float = IGNORE_IOA_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Filter COCO xywh predictions with the same fixed IoA rule as training."""
    ignores_by_image = _ignores_by_image(coco_dataset)
    return [
        dict(prediction)
        for prediction in predictions
        if not _is_ignored_xywh(
            prediction["bbox"], ignores_by_image.get(int(prediction["image_id"]), ()), threshold
        )
    ]


def filter_gt_annotations_by_ignore(
    coco_dataset: Mapping[str, Any],
    threshold: float = IGNORE_IOA_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Drop non-crowd GT annotations whose IoA with an ignore region reaches the threshold.

    Keeps the recall denominator symmetric with prediction filtering: a GT inside an
    ignore region can never be matched because its prediction would have been dropped.
    """
    ignores_by_image = _ignores_by_image(coco_dataset)
    return [
        dict(ann)
        for ann in coco_dataset.get("annotations", []) or []
        if ann.get("iscrowd")
        or not _is_ignored_xywh(
            ann.get("bbox", (0, 0, 0, 0)),
            ignores_by_image.get(int(ann.get("image_id", -1)), ()),
            threshold,
        )
    ]


def drop_ignored_gt_from_coco(coco_gt, threshold: float = IGNORE_IOA_THRESHOLD):
    """Return a COCO gt object without ignore-region GTs; copies only when needed."""
    anns = filter_gt_annotations_by_ignore(coco_gt.dataset, threshold)
    if len(anns) == len(coco_gt.dataset.get("annotations", []) or []):
        return coco_gt
    coco_gt = copy.deepcopy(coco_gt)
    coco_gt.dataset["annotations"] = anns
    coco_gt.createIndex()
    return coco_gt


def unmatched_query_keep_mask(
    pred_boxes: Any,
    targets: Iterable[Mapping[str, Any]],
    indices: Iterable[tuple[Any, Any]],
    threshold: float = IGNORE_IOA_THRESHOLD,
) -> Any:
    """Mask unmatched cxcywh queries whose area lies mainly inside ignore boxes."""
    import torch

    keep = torch.ones(pred_boxes.shape[:2], dtype=torch.bool, device=pred_boxes.device)
    for batch_index, (target, (matched_queries, _)) in enumerate(zip(targets, indices)):
        ignore_boxes = target.get("ignore_boxes")
        if ignore_boxes is None or ignore_boxes.numel() == 0:
            continue
        boxes = pred_boxes[batch_index]
        boxes_xyxy = torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), -1)
        ignores_xyxy = torch.cat(
            (ignore_boxes[:, :2] - ignore_boxes[:, 2:] / 2, ignore_boxes[:, :2] + ignore_boxes[:, 2:] / 2),
            -1,
        ).to(device=boxes.device, dtype=boxes.dtype)
        ignored = prediction_ioa_xyxy(boxes_xyxy, ignores_xyxy) >= threshold
        if matched_queries.numel():
            ignored[matched_queries] = False
        keep[batch_index, ignored] = False
    return keep


def canonical_ignore_boxes(ignore_boxes: Any, *, height: int, width: int) -> Any:
    """Return ignore boxes as normalized cxcywh, accepting transformed pixel xyxy."""
    import torch

    if ignore_boxes.numel() == 0 or float(ignore_boxes.detach().max()) <= 1.01:
        return ignore_boxes
    x1, y1, x2, y2 = ignore_boxes.unbind(-1)
    boxes = ignore_boxes.new_tensor([width, height, width, height])
    return torch.stack(
        ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), dim=-1
    ) / boxes


def apply_query_keep_mask(loss: Any, pred_boxes: Any, targets: Any, indices: Any) -> Any:
    """Zero per-class loss for ignored unmatched queries, preserving matched queries."""
    keep = unmatched_query_keep_mask(pred_boxes, targets, indices)
    if loss.ndim == keep.ndim + 1:
        keep = keep.unsqueeze(-1)
    return loss * keep.to(dtype=loss.dtype)
