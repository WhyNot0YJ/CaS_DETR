#!/usr/bin/env python3
"""Create a non-destructive 5-class DAIR-V2X view for all experiment stacks.

The source 8-class datasets are never modified.  The derived taxonomy is:

    Car + Truck + Van + Bus -> vehicle
    Pedestrian, Cyclist, Motorcyclist, Trafficcone -> unchanged

COCO annotations and YOLO labels keep every original box; only class ids change.
YOLO images are shared through hard links so Ultralytics resolves labels inside
the derived dataset instead of following a source-directory symlink.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


SOURCE_COCO_ROOT = Path("/root/autodl-fs/datasets/DAIR-V2X")
SOURCE_YOLO_ROOT = Path("/root/autodl-fs/datasets/DAIR-V2X_YOLO")
OUTPUT_COCO_ROOT = Path("/root/autodl-fs/datasets/DAIR-V2X-Vehicle5")
OUTPUT_YOLO_ROOT = Path("/root/autodl-fs/datasets/DAIR-V2X_YOLO_Vehicle5")
SPLITS = ("train", "val", "test")

CLASS_NAMES = (
    "vehicle",
    "Pedestrian",
    "Cyclist",
    "Motorcyclist",
    "Trafficcone",
)

SOURCE_NAME_TO_TARGET_ID = {
    "Car": 1,
    "Truck": 1,
    "Van": 1,
    "Bus": 1,
    "Pedestrian": 2,
    "Cyclist": 3,
    "Motorcyclist": 4,
    "Trafficcone": 5,
}

SOURCE_YOLO_TO_TARGET_ID = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 1,
    5: 2,
    6: 3,
    7: 4,
}


def remap_coco_document(source: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow COCO copy with the Vehicle5 category ids."""
    id_map: Dict[int, int] = {}
    source_names = set()
    for category in source.get("categories", []):
        name = str(category["name"])
        if name not in SOURCE_NAME_TO_TARGET_ID:
            raise ValueError(f"Unexpected DAIR-V2X category: {name!r}")
        source_names.add(name)
        id_map[int(category["id"])] = SOURCE_NAME_TO_TARGET_ID[name]

    missing = set(SOURCE_NAME_TO_TARGET_ID) - source_names
    if missing:
        raise ValueError(f"Missing DAIR-V2X categories: {sorted(missing)}")

    annotations = []
    for source_ann in source.get("annotations", []):
        category_id = int(source_ann["category_id"])
        if category_id not in id_map:
            raise ValueError(f"Annotation uses unknown category_id={category_id}")
        ann = dict(source_ann)
        ann["category_id"] = id_map[category_id]
        annotations.append(ann)

    result = dict(source)
    result["images"] = list(source.get("images", []))
    result["annotations"] = annotations
    result["categories"] = [
        {"id": index, "name": name, "supercategory": "object"}
        for index, name in enumerate(CLASS_NAMES, start=1)
    ]
    return result


def remap_yolo_line(line: str) -> str:
    """Map one YOLO label line while preserving all box coordinates."""
    fields = line.split()
    if not fields:
        return ""
    try:
        source_id = int(fields[0])
    except ValueError as exc:
        raise ValueError(f"Invalid YOLO class id in line: {line!r}") from exc
    if source_id not in SOURCE_YOLO_TO_TARGET_ID:
        raise ValueError(f"Unexpected YOLO class id={source_id}")
    fields[0] = str(SOURCE_YOLO_TO_TARGET_ID[source_id])
    return " ".join(fields)


def _ensure_symlink(link: Path, target: Path) -> None:
    target = target.resolve()
    if not target.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {target}")
    if os.path.lexists(link):
        if link.is_symlink() and link.resolve() == target:
            return
        raise FileExistsError(
            f"Refusing to replace existing path {link}; expected symlink to {target}"
        )
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)


def _ensure_hardlinked_images(
    output_root: Path, source_root: Path, splits: Iterable[str]
) -> None:
    source_images = (source_root / "images").resolve()
    output_images = output_root / "images"
    if output_images.is_symlink():
        if output_images.resolve() != source_images:
            raise FileExistsError(
                f"Refusing to replace {output_images}; it points outside {source_images}"
            )
        output_images.unlink()
    elif output_images.exists() and not output_images.is_dir():
        raise FileExistsError(f"Expected an image directory at {output_images}")

    for split in splits:
        source_dir = source_images / split
        if not source_dir.is_dir():
            raise FileNotFoundError(f"YOLO image directory does not exist: {source_dir}")
        output_dir = output_images / split
        output_dir.mkdir(parents=True, exist_ok=True)
        source_files = sorted(path for path in source_dir.iterdir() if path.is_file())
        unexpected = {path.name for path in output_dir.iterdir()} - {
            path.name for path in source_files
        }
        if unexpected:
            raise FileExistsError(
                f"Refusing to remove unexpected generated images in {output_dir}: "
                f"{sorted(unexpected)[:5]}"
            )
        for source_path in source_files:
            output_path = output_dir / source_path.name
            if output_path.exists():
                if not output_path.is_file() or not os.path.samefile(
                    source_path, output_path
                ):
                    raise FileExistsError(
                        f"Expected {output_path} to be a hard link to {source_path}"
                    )
                continue
            os.link(source_path, output_path)


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _prepare_coco(
    source_root: Path,
    output_root: Path,
    splits: Iterable[str],
) -> Dict[str, int]:
    _ensure_symlink(output_root / "image", source_root / "image")
    totals: Dict[str, int] = {}
    for split in splits:
        source_path = source_root / "annotations" / f"instances_{split}.json"
        if not source_path.is_file():
            raise FileNotFoundError(f"COCO annotation does not exist: {source_path}")
        source = json.loads(source_path.read_text(encoding="utf-8"))
        remapped = remap_coco_document(source)
        if len(remapped["annotations"]) != len(source.get("annotations", [])):
            raise AssertionError(f"{split}: annotation count changed during remapping")
        _write_json(
            output_root / "annotations" / f"instances_{split}.json",
            remapped,
        )
        totals[split] = len(remapped["annotations"])
    return totals


def _prepare_yolo(
    source_root: Path,
    output_root: Path,
    output_coco_root: Path,
    splits: Iterable[str],
) -> Dict[str, int]:
    _ensure_hardlinked_images(output_root, source_root, splits)
    totals: Dict[str, int] = {}
    for split in splits:
        source_dir = source_root / "labels" / split
        if not source_dir.is_dir():
            raise FileNotFoundError(f"YOLO label directory does not exist: {source_dir}")
        output_dir = output_root / "labels" / split
        output_dir.mkdir(parents=True, exist_ok=True)
        source_files = sorted(source_dir.glob("*.txt"))
        unexpected = {p.name for p in output_dir.glob("*.txt")} - {
            p.name for p in source_files
        }
        if unexpected:
            raise FileExistsError(
                f"Refusing to remove unexpected generated labels in {output_dir}: "
                f"{sorted(unexpected)[:5]}"
            )

        line_count = 0
        for source_path in source_files:
            source_lines = source_path.read_text(encoding="utf-8").splitlines()
            target_lines = [remap_yolo_line(line) for line in source_lines if line.strip()]
            line_count += len(target_lines)
            text = "\n".join(target_lines)
            if target_lines:
                text += "\n"
            _write_text(output_dir / source_path.name, text)
        totals[split] = line_count

    data_yaml = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "val_coco_ann": str(
            (output_coco_root / "annotations" / "instances_val.json").resolve()
        ),
        "test_coco_ann": str(
            (output_coco_root / "annotations" / "instances_test.json").resolve()
        ),
        "nc": len(CLASS_NAMES),
        "names": list(CLASS_NAMES),
    }
    yaml_path = output_root / "dairv2x_vehicle5.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = yaml_path.with_suffix(".yaml.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data_yaml, stream, sort_keys=False, allow_unicode=True)
    temporary.replace(yaml_path)
    return totals


def prepare_vehicle5(
    source_coco_root: Path,
    source_yolo_root: Path,
    output_coco_root: Path,
    output_yolo_root: Path,
    splits: Iterable[str] = SPLITS,
) -> None:
    splits = tuple(splits)
    coco_totals = _prepare_coco(source_coco_root, output_coco_root, splits)
    yolo_totals = _prepare_yolo(
        source_yolo_root,
        output_yolo_root,
        output_coco_root,
        splits,
    )
    for split in splits:
        if coco_totals[split] != yolo_totals[split]:
            raise ValueError(
                f"{split}: COCO has {coco_totals[split]} boxes, "
                f"but YOLO has {yolo_totals[split]}"
            )
        print(
            f"{split}: {coco_totals[split]} boxes retained; "
            f"classes mapped to 0..{len(CLASS_NAMES) - 1}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-coco-root", type=Path, default=SOURCE_COCO_ROOT)
    parser.add_argument("--source-yolo-root", type=Path, default=SOURCE_YOLO_ROOT)
    parser.add_argument("--output-coco-root", type=Path, default=OUTPUT_COCO_ROOT)
    parser.add_argument("--output-yolo-root", type=Path, default=OUTPUT_YOLO_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_vehicle5(
        args.source_coco_root,
        args.source_yolo_root,
        args.output_coco_root,
        args.output_yolo_root,
    )


if __name__ == "__main__":
    main()
