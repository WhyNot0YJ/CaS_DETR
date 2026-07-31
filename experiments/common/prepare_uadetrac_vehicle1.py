#!/usr/bin/env python3
"""Create a non-destructive one-class UA-DETRAC vehicle dataset.

The source ``car``, ``van``, ``bus`` and ``others`` classes are all mapped to
``vehicle``. Every box and coordinate is retained; images are shared by
hard links for YOLO so label lookup stays inside the derived dataset.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


SOURCE_COCO_ROOT = Path("/root/autodl-fs/datasets/UA-DETRAC_COCO")
SOURCE_YOLO_ROOT = Path("/root/autodl-fs/datasets/UA-DETRAC_YOLO")
OUTPUT_COCO_ROOT = Path("/root/autodl-fs/datasets/UA-DETRAC-Vehicle1")
OUTPUT_YOLO_ROOT = Path("/root/autodl-fs/datasets/UA-DETRAC_YOLO_Vehicle1")
SPLITS = ("train", "val", "test")
SOURCE_CLASS_NAMES = {"car", "van", "bus", "others"}
CLASS_NAMES = ("vehicle",)


def remap_coco_document(source: Dict[str, Any]) -> Dict[str, Any]:
    id_map = {
        int(category["id"]): str(category["name"])
        for category in source.get("categories", [])
    }
    if set(id_map.values()) != SOURCE_CLASS_NAMES:
        raise ValueError(
            f"Unexpected UA-DETRAC categories: {sorted(id_map.values())}"
        )

    annotations = []
    for source_ann in source.get("annotations", []):
        if int(source_ann["category_id"]) not in id_map:
            raise ValueError(
                f"Annotation uses unknown category_id={source_ann['category_id']}"
            )
        ann = dict(source_ann)
        ann["category_id"] = 1
        annotations.append(ann)

    result = dict(source)
    result["images"] = list(source.get("images", []))
    result["annotations"] = annotations
    result["categories"] = [
        {"id": 1, "name": "vehicle", "supercategory": "vehicle"}
    ]
    return result


def remap_yolo_line(line: str) -> str:
    fields = line.split()
    if not fields:
        return ""
    try:
        source_id = int(fields[0])
    except ValueError as exc:
        raise ValueError(f"Invalid YOLO class id in line: {line!r}") from exc
    if source_id not in range(4):
        raise ValueError(f"Unexpected UA-DETRAC YOLO class id={source_id}")
    fields[0] = "0"
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


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _prepare_coco(
    source_root: Path, output_root: Path, splits: Iterable[str]
) -> Dict[str, int]:
    totals = {}
    for split in splits:
        _ensure_symlink(output_root / split, source_root / split)
        source_path = source_root / "annotations" / f"instances_{split}.json"
        source = json.loads(source_path.read_text(encoding="utf-8"))
        remapped = remap_coco_document(source)
        if len(remapped["annotations"]) != len(source.get("annotations", [])):
            raise AssertionError(f"{split}: annotation count changed")
        output_path = output_root / "annotations" / f"instances_{split}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(remapped, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(output_path)
        totals[split] = len(remapped["annotations"])
    return totals


def _prepare_yolo(
    source_root: Path,
    output_root: Path,
    output_coco_root: Path,
    splits: Iterable[str],
) -> Dict[str, int]:
    _ensure_hardlinked_images(output_root, source_root, splits)
    totals = {}
    for split in splits:
        source_dir = source_root / "labels" / split
        output_dir = output_root / "labels" / split
        output_dir.mkdir(parents=True, exist_ok=True)
        source_files = sorted(source_dir.glob("*.txt"))
        unexpected = {p.name for p in output_dir.glob("*.txt")} - {
            p.name for p in source_files
        }
        if unexpected:
            raise FileExistsError(
                f"Refusing to remove generated labels: {sorted(unexpected)[:5]}"
            )
        count = 0
        for source_path in source_files:
            target_lines = [
                remap_yolo_line(line)
                for line in source_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            count += len(target_lines)
            _write_text(
                output_dir / source_path.name,
                "\n".join(target_lines) + ("\n" if target_lines else ""),
            )
        totals[split] = count

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
        "nc": 1,
        "names": list(CLASS_NAMES),
    }
    yaml_path = output_root / "uadetrac_vehicle1.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = yaml_path.with_suffix(".yaml.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data_yaml, stream, sort_keys=False, allow_unicode=True)
    temporary.replace(yaml_path)
    return totals


def prepare_vehicle1(
    source_coco_root: Path,
    source_yolo_root: Path,
    output_coco_root: Path,
    output_yolo_root: Path,
    splits: Iterable[str] = SPLITS,
) -> None:
    splits = tuple(splits)
    coco_totals = _prepare_coco(source_coco_root, output_coco_root, splits)
    yolo_totals = _prepare_yolo(
        source_yolo_root, output_yolo_root, output_coco_root, splits
    )
    for split in splits:
        if coco_totals[split] != yolo_totals[split]:
            raise ValueError(
                f"{split}: COCO has {coco_totals[split]} boxes, "
                f"but YOLO has {yolo_totals[split]}"
            )
        print(f"{split}: {coco_totals[split]} boxes retained; classes mapped to 0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-coco-root", type=Path, default=SOURCE_COCO_ROOT)
    parser.add_argument("--source-yolo-root", type=Path, default=SOURCE_YOLO_ROOT)
    parser.add_argument("--output-coco-root", type=Path, default=OUTPUT_COCO_ROOT)
    parser.add_argument("--output-yolo-root", type=Path, default=OUTPUT_YOLO_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_vehicle1(
        args.source_coco_root,
        args.source_yolo_root,
        args.output_coco_root,
        args.output_yolo_root,
    )


if __name__ == "__main__":
    main()
