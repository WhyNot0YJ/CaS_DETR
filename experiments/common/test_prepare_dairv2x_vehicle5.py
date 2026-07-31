from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from prepare_dairv2x_vehicle5 import (
    CLASS_NAMES,
    prepare_vehicle5,
    remap_coco_document,
    remap_yolo_line,
)


class Vehicle5RemappingTest(unittest.TestCase):
    def test_coco_remap_retains_boxes_and_merges_four_vehicle_classes(self):
        categories = [
            {"id": i, "name": name}
            for i, name in enumerate(
                [
                    "Car",
                    "Truck",
                    "Van",
                    "Bus",
                    "Pedestrian",
                    "Cyclist",
                    "Motorcyclist",
                    "Trafficcone",
                ],
                start=1,
            )
        ]
        source = {
            "images": [{"id": 1, "file_name": "image/a.jpg"}],
            "categories": categories,
            "annotations": [
                {"id": i, "image_id": 1, "category_id": i, "bbox": [i, 2, 3, 4]}
                for i in range(1, 9)
            ],
        }

        remapped = remap_coco_document(source)

        self.assertEqual([c["name"] for c in remapped["categories"]], list(CLASS_NAMES))
        self.assertEqual(
            [a["category_id"] for a in remapped["annotations"]],
            [1, 1, 1, 1, 2, 3, 4, 5],
        )
        self.assertEqual(
            [a["bbox"] for a in remapped["annotations"]],
            [a["bbox"] for a in source["annotations"]],
        )

    def test_yolo_remap_retains_coordinates(self):
        self.assertEqual(remap_yolo_line("3 0.1 0.2 0.3 0.4"), "0 0.1 0.2 0.3 0.4")
        self.assertEqual(remap_yolo_line("7 0.1 0.2 0.3 0.4"), "4 0.1 0.2 0.3 0.4")

    def test_end_to_end_temporary_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_coco = root / "source_coco"
            source_yolo = root / "source_yolo"
            output_coco = root / "output_coco"
            output_yolo = root / "output_yolo"
            (source_coco / "image").mkdir(parents=True)
            (source_yolo / "images").mkdir(parents=True)

            categories = [
                {"id": i, "name": name}
                for i, name in enumerate(
                    [
                        "Car",
                        "Truck",
                        "Van",
                        "Bus",
                        "Pedestrian",
                        "Cyclist",
                        "Motorcyclist",
                        "Trafficcone",
                    ],
                    start=1,
                )
            ]
            for split in ("train", "val", "test"):
                image_dir = source_yolo / "images" / split
                image_dir.mkdir(parents=True)
                (image_dir / "a.jpg").write_bytes(b"test image")
                ann_dir = source_coco / "annotations"
                ann_dir.mkdir(parents=True, exist_ok=True)
                document = {
                    "images": [{"id": 1, "file_name": "image/a.jpg"}],
                    "categories": categories,
                    "annotations": [
                        {
                            "id": 1,
                            "image_id": 1,
                            "category_id": 4,
                            "bbox": [1, 2, 3, 4],
                        }
                    ],
                }
                (ann_dir / f"instances_{split}.json").write_text(
                    json.dumps(document), encoding="utf-8"
                )
                label_dir = source_yolo / "labels" / split
                label_dir.mkdir(parents=True)
                (label_dir / "a.txt").write_text(
                    "3 0.1 0.2 0.3 0.4\n", encoding="utf-8"
                )

            prepare_vehicle5(
                source_coco,
                source_yolo,
                output_coco,
                output_yolo,
            )

            self.assertTrue((output_coco / "image").is_symlink())
            self.assertTrue((output_yolo / "images").is_dir())
            self.assertFalse((output_yolo / "images").is_symlink())
            self.assertTrue(
                (output_yolo / "images" / "test" / "a.jpg").samefile(
                    source_yolo / "images" / "test" / "a.jpg"
                )
            )
            data = yaml.safe_load(
                (output_yolo / "dairv2x_vehicle5.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(data["nc"], 5)
            self.assertEqual(data["names"], list(CLASS_NAMES))
            self.assertEqual(
                (output_yolo / "labels" / "test" / "a.txt").read_text(
                    encoding="utf-8"
                ),
                "0 0.1 0.2 0.3 0.4\n",
            )


if __name__ == "__main__":
    unittest.main()
