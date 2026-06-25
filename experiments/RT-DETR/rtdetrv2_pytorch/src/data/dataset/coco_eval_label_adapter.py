"""Map train label indices 0..N-1 to COCO category_id for validation."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import torch

from ...core import register
from .coco_eval import CocoEvaluator


class CocoEvaluatorTrainLabelAdapter:
    """Delegates to an inner evaluator after remapping prediction labels."""

    def __init__(self, inner: CocoEvaluator):
        self._inner = inner
        cats = inner.coco_gt.dataset.get("categories", [])
        self._label_to_category_id = {i: c["id"] for i, c in enumerate(cats)}

    def _map_labels(self, labels: torch.Tensor) -> torch.Tensor:
        flat = labels.flatten().tolist()
        mapped = [self._label_to_category_id.get(int(x), int(x)) for x in flat]
        return torch.tensor(mapped, device=labels.device, dtype=labels.dtype).reshape(labels.shape)

    def _remap_predictions(self, predictions: Dict[Any, Dict]) -> Dict[Any, Dict]:
        out: Dict[Any, Dict] = {}
        for image_id, pred in predictions.items():
            pred = dict(pred)
            if "labels" in pred:
                pred["labels"] = self._map_labels(pred["labels"])
            out[image_id] = pred
        return out

    @property
    def coco_gt(self):
        return self._inner.coco_gt

    @property
    def coco_eval(self):
        return self._inner.coco_eval

    @property
    def iou_types(self):
        return self._inner.iou_types

    def cleanup(self):
        return self._inner.cleanup()

    def update(self, predictions):
        return self._inner.update(self._remap_predictions(predictions))

    def synchronize_between_processes(self):
        return self._inner.synchronize_between_processes()

    def accumulate(self):
        return self._inner.accumulate()

    def summarize(self):
        return self._inner.summarize()


@register()
class CocoEvaluatorTrainLabelMapping:
    """Like CocoEvaluator, but maps prediction label indices to category ids in update.

    Cross-domain eval kwargs (DAWN val with DAIR-trained head, etc.):
        cross_domain_label_map: {src_cat_id: dst_cat_id} rewriting GT category_id
            from val-domain ids to train-domain ids before scoring.
        drop_unmapped_gt: if True, GT annotations whose category_id is not in
            cross_domain_label_map are removed (val-domain classes the train head
            cannot predict).
        override_categories: full categories list (each {"id": int, "name": str})
            to install on coco_gt. Required when cross_domain_label_map is given,
            because predictions' label index 0..N-1 maps to category id via the
            order of categories.
    """

    def __init__(
        self,
        coco_gt,
        iou_types,
        cross_domain_label_map: Optional[Dict[int, int]] = None,
        drop_unmapped_gt: bool = False,
        override_categories: Optional[List[Dict[str, Any]]] = None,
    ):
        if cross_domain_label_map is not None:
            if override_categories is None:
                raise ValueError(
                    "cross_domain_label_map requires override_categories so prediction "
                    "label indices map to the train-domain category ids."
                )
            coco_gt = self._apply_cross_domain_remap(
                coco_gt,
                {int(k): int(v) for k, v in cross_domain_label_map.items()},
                drop_unmapped_gt,
                override_categories,
            )
        self._impl = CocoEvaluatorTrainLabelAdapter(CocoEvaluator(coco_gt, iou_types))

    @staticmethod
    def _apply_cross_domain_remap(coco_gt, label_map, drop_unmapped, override_categories):
        coco_gt = copy.deepcopy(coco_gt)
        ds = coco_gt.dataset

        kept_anns = []
        for ann in ds.get("annotations", []):
            src = int(ann["category_id"])
            if src in label_map:
                ann["category_id"] = label_map[src]
                kept_anns.append(ann)
            elif not drop_unmapped:
                kept_anns.append(ann)
        ds["annotations"] = kept_anns

        ds["categories"] = [dict(c) for c in override_categories]

        coco_gt.createIndex()
        return coco_gt

    def __getattr__(self, name):
        return getattr(self._impl, name)
