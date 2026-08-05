import json
import unittest
from pathlib import Path

from common.hierarchical_eval import collapse_ground_truth, collapse_predictions


class HierarchicalEvalTest(unittest.TestCase):
    def test_dair_mapping_changes_only_category_fields(self):
        source = [
            {"image_id": 7, "category_id": 1, "bbox": [1, 2, 3, 4], "score": 0.9},
            {"image_id": 7, "category_id": 4, "bbox": [1, 2, 3, 4], "score": 0.8},
            {"image_id": 8, "category_id": 8, "bbox": [5, 6, 7, 8], "score": 0.7},
        ]
        collapsed = collapse_predictions(source, "dairv2x_vehicle8_to_vehicle5")

        self.assertEqual(len(collapsed), len(source))
        self.assertEqual([item["category_id"] for item in collapsed], [1, 1, 5])
        self.assertEqual([item["source_category_id"] for item in collapsed], [1, 4, 8])
        for original, mapped in zip(source, collapsed):
            self.assertEqual(mapped["image_id"], original["image_id"])
            self.assertEqual(mapped["bbox"], original["bbox"])
            self.assertEqual(mapped["score"], original["score"])
        self.assertNotIn("source_category_id", source[0])

    def test_overlapping_cross_class_predictions_are_not_suppressed(self):
        source = [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
            {"image_id": 1, "category_id": 2, "bbox": [0, 0, 10, 10], "score": 0.8},
            {"image_id": 1, "category_id": 3, "bbox": [0, 0, 10, 10], "score": 0.7},
            {"image_id": 1, "category_id": 4, "bbox": [0, 0, 10, 10], "score": 0.6},
        ]
        collapsed = collapse_predictions(source, "dairv2x_vehicle8_to_vehicle5")
        self.assertEqual(len(collapsed), 4)
        self.assertEqual([item["category_id"] for item in collapsed], [1, 1, 1, 1])

    def test_uadetrac_mapping_keeps_every_prediction(self):
        source = [
            {"image_id": index, "category_id": index, "bbox": [0, 0, 1, 1], "score": 0.5}
            for index in range(1, 5)
        ]
        collapsed = collapse_predictions(source, "uadetrac_vehicle4_to_vehicle1")
        self.assertEqual(len(collapsed), 4)
        self.assertTrue(all(item["category_id"] == 1 for item in collapsed))

    def test_ground_truth_mapping_preserves_annotations(self):
        source = {
            "images": [{"id": 1, "file_name": "one.jpg"}],
            "categories": [{"id": index, "name": str(index)} for index in range(1, 9)],
            "annotations": [
                {
                    "id": index,
                    "image_id": 1,
                    "category_id": index,
                    "bbox": [index, 0, 2, 3],
                    "area": 6,
                    "iscrowd": 0,
                }
                for index in range(1, 9)
            ],
        }
        collapsed = collapse_ground_truth(source, "dairv2x_vehicle8_to_vehicle5")
        self.assertEqual(len(collapsed["annotations"]), 8)
        self.assertEqual(len(collapsed["categories"]), 5)
        self.assertEqual(
            [item["category_id"] for item in collapsed["annotations"]],
            [1, 1, 1, 1, 2, 3, 4, 5],
        )
        for original, mapped in zip(source["annotations"], collapsed["annotations"]):
            self.assertEqual(mapped["bbox"], original["bbox"])
            self.assertEqual(mapped["area"], original["area"])

    def test_installed_derived_annotations_match_label_collapse(self):
        pairs = [
            (
                Path("/root/autodl-fs/datasets/DAIR-V2X"),
                Path("/root/autodl-fs/datasets/DAIR-V2X-Vehicle5"),
                "dairv2x_vehicle8_to_vehicle5",
            ),
            (
                Path("/root/autodl-fs/datasets/UA-DETRAC_COCO"),
                Path("/root/autodl-fs/datasets/UA-DETRAC-Vehicle1"),
                "uadetrac_vehicle4_to_vehicle1",
            ),
        ]
        for source_root, derived_root, mode in pairs:
            if not source_root.is_dir() or not derived_root.is_dir():
                self.skipTest("installed evaluation datasets are unavailable")
            for split in ("val", "test"):
                source = json.loads(
                    (source_root / "annotations" / f"instances_{split}.json").read_text()
                )
                derived = json.loads(
                    (derived_root / "annotations" / f"instances_{split}.json").read_text()
                )
                collapsed = collapse_ground_truth(source, mode)
                self.assertEqual(collapsed["images"], derived["images"])
                self.assertEqual(
                    [
                        (a["id"], a["image_id"], a["category_id"], a["bbox"], a["area"])
                        for a in collapsed["annotations"]
                    ],
                    [
                        (a["id"], a["image_id"], a["category_id"], a["bbox"], a["area"])
                        for a in derived["annotations"]
                    ],
                )

    def test_unknown_source_category_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside the declared taxonomy"):
            collapse_predictions(
                [{"image_id": 1, "category_id": 9, "bbox": [0, 0, 1, 1], "score": 1}],
                "dairv2x_vehicle8_to_vehicle5",
            )


if __name__ == "__main__":
    unittest.main()
