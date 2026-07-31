from __future__ import annotations

import unittest

from evaluate_uadetrac_vehicle1 import (
    collapse_ground_truth,
    collapse_predictions,
    compute_metrics,
)


class UADETRACVehicle1EvaluationTest(unittest.TestCase):
    def test_cross_class_duplicates_receive_class_agnostic_nms(self):
        predictions = [
            {
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "score": 0.9,
            },
            {
                "image_id": 1,
                "category_id": 3,
                "bbox": [0, 0, 10, 10],
                "score": 0.8,
            },
            {
                "image_id": 1,
                "category_id": 4,
                "bbox": [20, 20, 5, 5],
                "score": 0.7,
            },
        ]

        collapsed = collapse_predictions(predictions, nms_iou=0.7)

        self.assertEqual(len(collapsed), 2)
        self.assertTrue(all(item["category_id"] == 1 for item in collapsed))
        self.assertEqual(collapsed[0]["score"], 0.9)

    def test_ground_truth_retains_all_annotations(self):
        source = {
            "images": [{"id": 1}],
            "categories": [
                {"id": 1, "name": "car"},
                {"id": 2, "name": "van"},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1},
                {"id": 2, "image_id": 1, "category_id": 2},
            ],
        }

        collapsed = collapse_ground_truth(source)

        self.assertEqual(len(collapsed["annotations"]), 2)
        self.assertEqual(
            [item["category_id"] for item in collapsed["annotations"]], [1, 1]
        )

    def test_collapsed_predictions_run_through_coco_eval(self):
        ground_truth = {
            "images": [{"id": 1, "width": 100, "height": 100}],
            "categories": [{"id": 1, "name": "vehicle"}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 10, 20, 20],
                    "area": 400,
                    "iscrowd": 0,
                }
            ],
        }
        predictions = [
            {
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 20, 20],
                "score": 0.9,
            }
        ]

        metrics = compute_metrics(ground_truth, predictions)

        self.assertAlmostEqual(metrics["mAP_0.5_0.95"], 1.0)
        self.assertAlmostEqual(metrics["AP_small"], 1.0)


if __name__ == "__main__":
    unittest.main()
