"""Regression tests for the UA-DETRAC ignore-region protocol."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_protocol import _dataset_update
from uadetrac_ignore import (
    apply_query_keep_mask,
    canonical_ignore_boxes,
    filter_coco_predictions_by_ignore,
    filter_tensor_predictions_by_ignore,
    ignore_regions_xyxy,
)


ROOT = Path(__file__).resolve().parents[2]


class IgnoreAnnotationTest(unittest.TestCase):
    def test_region_clipping_and_protocol_scope(self):
        self.assertEqual(
            ignore_regions_xyxy({"width": 100, "height": 80, "ignore_regions": [{"bbox": [90, 70, 20, 20]}]}),
            [[90.0, 70.0, 100.0, 80.0]],
        )
        self.assertFalse(_dataset_update(Path("/dair"), "dairv2x", "eval", rtdetr_layout=False)["use_ignore_regions"])
        self.assertTrue(_dataset_update(Path("/ua"), "uadetrac", "test", rtdetr_layout=False)["use_ignore_regions"])
        self.assertNotIn(
            "max_truncation_ratio",
            _dataset_update(Path("/ua"), "uadetrac", "train", rtdetr_layout=False),
        )


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
class IgnoreTrainingTest(unittest.TestCase):
    def test_yolo_ignore_anchor_mask_uses_fixed_ioa(self):
        import torch

        sys.path.insert(0, str(ROOT / "experiments" / "yolo"))
        from uadetrac_ultralytics import ignored_prediction_mask

        boxes = torch.tensor([[0, 0, 20, 10], [0, 0, 21, 10]], dtype=torch.float32)
        ignore = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
        self.assertEqual(ignored_prediction_mask(boxes, ignore).tolist(), [True, False])

    def test_yolo_profile_carries_coco_ignore_source(self):
        from dataset_registry import apply_yolo_dataset_profile

        config = apply_yolo_dataset_profile(
            {"data": {}}, {"data_yaml": "/dataset/data.yaml", "coco_data_root": "/dataset/coco"}
        )
        self.assertEqual(config["data"]["coco_data_root"], "/dataset/coco")

    def test_ignore_query_has_no_background_classification_loss(self):
        import torch
        import torchvision

        logits = torch.zeros((1, 2, 1), requires_grad=True)
        outputs = {
            "pred_logits": logits,
            "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]]]),
        }
        targets = [{"ignore_boxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]])}]
        indices = [(torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64))]
        raw = torchvision.ops.sigmoid_focal_loss(
            logits, torch.zeros_like(logits), alpha=0.25, gamma=2.0, reduction="none"
        )
        masked = apply_query_keep_mask(raw, outputs["pred_boxes"], targets, indices)
        self.assertEqual(masked[0, 0].sum().item(), 0.0)
        self.assertGreater(masked[0, 1].sum().item(), 0.0)

    def test_online_predictions_use_fixed_half_ioa_threshold(self):
        import torch

        dataset = {
            "images": [{"id": 1, "ignore_regions": [{"bbox": [0, 0, 10, 10]}]}]
        }
        predictions = {
            1: {
                "boxes": torch.tensor([[0, 0, 20, 10], [0, 0, 21, 10]], dtype=torch.float32),
                "scores": torch.tensor([0.9, 0.8]),
                "labels": torch.tensor([0, 0]),
            }
        }
        filtered = filter_tensor_predictions_by_ignore(predictions, dataset)
        self.assertEqual(filtered[1]["boxes"].shape[0], 1)
        self.assertAlmostEqual(filtered[1]["scores"].item(), 0.8)

    def test_rtdetr_criterion_uses_ignore_mask(self):
        import torch

        sys.path.insert(0, str(ROOT / "experiments" / "RT-DETR" / "rtdetrv2_pytorch"))
        from src.zoo.rtdetr.rtdetrv2_criterion import RTDETRCriterionv2

        criterion = RTDETRCriterionv2(
            matcher=None, weight_dict={}, losses=[], num_classes=1
        )
        outputs = {
            "pred_logits": torch.zeros((1, 2, 1)),
            "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]]]),
        }
        indices = [(torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64))]
        base_target = {"labels": torch.empty(0, dtype=torch.int64)}
        ignored_target = {
            **base_target, "ignore_boxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]])
        }
        values = torch.empty(0)
        base = criterion.loss_labels_vfl(
            outputs, [base_target], indices, 1, values=values
        )["loss_vfl"]
        masked = criterion.loss_labels_vfl(
            outputs, [ignored_target], indices, 1, values=values
        )["loss_vfl"]
        self.assertAlmostEqual(masked.item() * 2, base.item(), places=6)

    def test_resize_flip_crop_and_mosaic_keep_ignore_boxes_synchronized(self):
        import torch
        import torchvision.transforms.v2 as transforms
        from PIL import Image
        from torchvision.tv_tensors import BoundingBoxes

        sys.path.insert(0, str(ROOT / "experiments" / "CaS-DETR"))
        from engine.data.transforms.mosaic import Mosaic

        image = Image.new("RGB", (100, 80))
        target = {
            "boxes": BoundingBoxes(
                torch.tensor([[10, 10, 30, 30], [60, 20, 90, 50]], dtype=torch.float32),
                format="XYXY", canvas_size=(80, 100),
            ),
            "labels": torch.tensor([0, -1]),
        }
        image, target = transforms.Compose([
            transforms.Resize((40, 50)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomCrop((35, 45)),
        ])(image, target)
        self.assertEqual(target["boxes"].shape[0], target["labels"].shape[0])
        self.assertEqual((target["labels"] == -1).sum().item(), 1)

        samples = [image] * 4
        targets = [{"boxes": target["boxes"].clone(), "labels": target["labels"].clone()} for _ in range(4)]
        _, merged = Mosaic(use_cache=False).create_mosaic_from_dataset(samples, targets, 35, 45)
        self.assertEqual((merged["labels"] == -1).sum().item(), 4)
        self.assertEqual(merged["boxes"].shape[0], merged["labels"].shape[0])
        normalized = canonical_ignore_boxes(
            merged["boxes"][merged["labels"] == -1], height=70, width=90
        )
        self.assertLessEqual(float(normalized.max()), 1.0)


@unittest.skipUnless(importlib.util.find_spec("pycocotools"), "pycocotools is not installed")
class IgnoreEvaluationTest(unittest.TestCase):
    def test_prediction_inside_ignore_region_is_not_false_positive(self):
        from common.det_eval_metrics import run_coco_bbox_eval

        gt = {
            "images": [{
                "id": 1, "width": 100, "height": 100,
                "ignore_regions": [{"bbox": [0, 0, 40, 40]}],
            }],
            "categories": [{"id": 1, "name": "car"}],
            "annotations": [{
                "id": 1, "image_id": 1, "category_id": 1,
                "bbox": [60, 60, 20, 20], "area": 400, "iscrowd": 0,
            }],
        }
        predictions = [
            {"image_id": 1, "category_id": 1, "bbox": [5, 5, 20, 20], "score": 0.99},
            {"image_id": 1, "category_id": 1, "bbox": [60, 60, 20, 20], "score": 0.90},
        ]
        self.assertEqual(len(filter_coco_predictions_by_ignore(gt, predictions)), 1)
        coco_eval = run_coco_bbox_eval(gt, predictions)
        self.assertIsNotNone(coco_eval)
        self.assertAlmostEqual(float(coco_eval.stats[1]), 1.0, places=6)

    def test_cas_style_metrics_reuse_fixed_ioa_filter(self):
        from common.cas_style_map_metrics import compute_cas_style_map_metrics

        predictions = [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 20, 10], "score": 0.99},
            {"image_id": 1, "category_id": 1, "bbox": [60, 60, 20, 20], "score": 0.90},
        ]
        targets = [
            {
                "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10],
                "area": 100, "iscrowd": 1, "uadetrac_ignore": 1,
            },
            {
                "image_id": 1, "category_id": 1, "bbox": [60, 60, 20, 20],
                "area": 400, "iscrowd": 0,
            },
        ]
        metrics = compute_cas_style_map_metrics(
            predictions, targets, [{"id": 1, "name": "car"}],
            image_id_to_size={1: (100, 100)}, dataset_name="UA-DETRAC",
        )
        self.assertAlmostEqual(metrics["mAP_0.5_0.95"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
