from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from prepare_uadetrac_vehicle1 import (
    prepare_vehicle1,
    remap_coco_document,
    remap_yolo_line,
)


class UADETRACVehicle1RemappingTest(unittest.TestCase):
    def test_coco_remap_retains_every_box(self):
        source = {
            "images": [{"id": 1, "file_name": "a.jpg"}],
            "categories": [
                {"id": index, "name": name}
                for index, name in enumerate(
                    ("car", "van", "bus", "others"), start=1
                )
            ],
            "annotations": [
                {
                    "id": index,
                    "image_id": 1,
                    "category_id": index,
                    "bbox": [index, 2, 3, 4],
                }
                for index in range(1, 5)
            ],
        }

        remapped = remap_coco_document(source)

        self.assertEqual(remapped["categories"][0]["name"], "vehicle")
        self.assertEqual(
            [annotation["category_id"] for annotation in remapped["annotations"]],
            [1, 1, 1, 1],
        )
        self.assertEqual(
            [annotation["bbox"] for annotation in remapped["annotations"]],
            [annotation["bbox"] for annotation in source["annotations"]],
        )

    def test_yolo_remap_retains_coordinates(self):
        self.assertEqual(remap_yolo_line("3 0.1 0.2 0.3 0.4"), "0 0.1 0.2 0.3 0.4")

    def test_end_to_end_temporary_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_coco = root / "source_coco"
            source_yolo = root / "source_yolo"
            output_coco = root / "output_coco"
            output_yolo = root / "output_yolo"
            categories = [
                {"id": index, "name": name}
                for index, name in enumerate(
                    ("car", "van", "bus", "others"), start=1
                )
            ]
            for split in ("train", "val", "test"):
                (source_coco / split).mkdir(parents=True)
                image_dir = source_yolo / "images" / split
                image_dir.mkdir(parents=True)
                (image_dir / "a.jpg").write_bytes(b"test image")
                ann_dir = source_coco / "annotations"
                ann_dir.mkdir(parents=True, exist_ok=True)
                (ann_dir / f"instances_{split}.json").write_text(
                    json.dumps(
                        {
                            "images": [{"id": 1, "file_name": "a.jpg"}],
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
                    ),
                    encoding="utf-8",
                )
                label_dir = source_yolo / "labels" / split
                label_dir.mkdir(parents=True)
                (label_dir / "a.txt").write_text(
                    "3 0.1 0.2 0.3 0.4\n", encoding="utf-8"
                )
            prepare_vehicle1(
                source_coco, source_yolo, output_coco, output_yolo
            )

            self.assertTrue((output_coco / "train").is_symlink())
            self.assertTrue((output_yolo / "images").is_dir())
            self.assertFalse((output_yolo / "images").is_symlink())
            self.assertTrue(
                (output_yolo / "images" / "test" / "a.jpg").samefile(
                    source_yolo / "images" / "test" / "a.jpg"
                )
            )
            data = yaml.safe_load(
                (output_yolo / "uadetrac_vehicle1.yaml").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(data["nc"], 1)
            self.assertEqual(data["names"], ["vehicle"])
            self.assertEqual(
                (output_yolo / "labels" / "test" / "a.txt").read_text(
                    encoding="utf-8"
                ),
                "0 0.1 0.2 0.3 0.4\n",
            )


if __name__ == "__main__":
    unittest.main()
