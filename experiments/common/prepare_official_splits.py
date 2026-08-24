#!/usr/bin/env python3
"""Restore the repository's native DAIR-V2X and UA-DETRAC evaluation splits.

The source folders are derived datasets.  Original images are never edited:
new split folders are staged first, then replace the old derived folders.

DAIR-V2X follows the available official ``train``/``val`` lists.  As with
COCO train2017/val2017, the official val list is the experiment's labelled
``eval``/selection set; the unlabelled official test is not materialized.
UA-DETRAC uses its original XML provenance: ``train_xml``
is the official 60-sequence training set and ``test_xml`` the 40-sequence test
set.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from copy import deepcopy
from pathlib import Path


DATASETS = Path("/root/autodl-fs/datasets")
DAIR_ROOT = DATASETS / "DAIR-V2X"
DAIR_YOLO = DATASETS / "DAIR-V2X_YOLO"
UA_ROOT = DATASETS / "UA-DETRAC_COCO"
UA_YOLO = DATASETS / "UA-DETRAC_YOLO"


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _stage(path: Path) -> Path:
    staged = path.with_name(f".{path.name}.official_stage")
    if staged.exists():
        raise RuntimeError(f"staging directory already exists: {staged}")
    staged.mkdir(parents=True)
    return staged


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source.resolve(), target)
    except OSError:
        shutil.copy2(source, target)


def _replace_dirs(root: Path, staged: Path, names: tuple[str, ...], apply: bool) -> None:
    if not apply:
        shutil.rmtree(staged)
        return
    for name in names:
        current = root / name
        if current.exists():
            if current.is_dir():
                shutil.rmtree(current)
            else:
                current.unlink()
        (staged / name).rename(current)
    shutil.rmtree(staged)


def _coco_by_stem(annotation_files: list[Path]) -> tuple[dict, dict[str, tuple[dict, list[dict]]]]:
    categories: dict | None = None
    output: dict[str, tuple[dict, list[dict]]] = {}
    for path in annotation_files:
        coco = _load(path)
        categories = categories or coco.get("categories", [])
        anns: dict[int, list[dict]] = {}
        for ann in coco.get("annotations", []):
            anns.setdefault(int(ann["image_id"]), []).append(ann)
        for image in coco.get("images", []):
            stem = Path(str(image["file_name"])).stem
            if stem in output:
                raise RuntimeError(f"duplicate image stem in COCO sources: {stem}")
            output[stem] = (image, anns.get(int(image["id"]), []))
    return {"categories": categories or []}, output


def _subset_coco(base: dict, records: dict[str, tuple[dict, list[dict]]], stems: list[str]) -> dict:
    images, annotations = [], []
    for image_id, stem in enumerate(sorted(stems, key=lambda item: int(item) if item.isdigit() else item)):
        image, anns = records[stem]
        image = deepcopy(image)
        image["id"] = image_id
        images.append(image)
        for ann in anns:
            ann = deepcopy(ann)
            ann["id"] = len(annotations)
            ann["image_id"] = image_id
            annotations.append(ann)
    return {"images": images, "annotations": annotations, "categories": base["categories"]}


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _migrate_dair_test_to_eval(apply: bool) -> tuple[int, int]:
    """Migrate the first restoration's compatibility ``test`` name to ``eval``."""
    ann_dir = DAIR_ROOT / "annotations"
    source = ann_dir / "instances_test.json"
    if not source.is_file():
        raise RuntimeError(f"cannot migrate DAIR eval split; missing {source}")
    data = _load(source)
    if len(data.get("images", [])) != 2016:
        raise RuntimeError("DAIR compatibility test is not the expected official eval split")
    if not apply:
        return 5042, 2016

    _copy(source, ann_dir / "instances_eval.json")
    for name in ("instances_val.json", "instances_test.json"):
        (ann_dir / name).unlink(missing_ok=True)
    _write_json(
        ann_dir / "official_eval_manifest.json",
        {"source": "official_split_data.json", "train": 5042, "eval": 2016, "eval_source": "official_val"},
    )

    test_dir = DAIR_YOLO / "images" / "test"
    if not test_dir.is_dir():
        raise RuntimeError(f"cannot migrate DAIR YOLO eval split; missing {test_dir}")
    for name in ("images", "labels", "labels_meta"):
        source_dir = DAIR_YOLO / name / "test"
        source_dir.rename(DAIR_YOLO / name / "eval")
    (DAIR_YOLO / "dairv2x.yaml").write_text(
        "train: images/train\nval: images/eval\nval_coco_ann: /root/autodl-fs/datasets/DAIR-V2X/annotations/instances_eval.json\nnc: 8\nnames: [Car, Truck, Van, Bus, Pedestrian, Cyclist, Motorcyclist, Trafficcone]\n",
        encoding="utf-8",
    )
    return 5042, 2016


def prepare_dair(apply: bool) -> tuple[int, int]:
    manifest = DAIR_ROOT / "annotations" / "official_eval_manifest.json"
    if manifest.is_file():
        value = _load(manifest)
        if (value.get("train"), value.get("eval")) == (5042, 2016):
            return 5042, 2016
        if (value.get("train"), value.get("test")) == (5042, 2016):
            return _migrate_dair_test_to_eval(apply)
        raise RuntimeError(f"unexpected existing DAIR official manifest: {manifest}")
    official = _load(DAIR_YOLO / "official_split_data.json")
    train, eval_stems = official["train"], official["val"]
    if (len(train), len(eval_stems)) != (5042, 2016):
        raise RuntimeError("unexpected DAIR official train/val list sizes")

    base, records = _coco_by_stem(sorted((DAIR_ROOT / "annotations").glob("instances_*.json")))
    expected = set(train) | set(eval_stems)
    if set(records) != expected:
        raise RuntimeError(f"DAIR sources do not cover the official train+val lists: {len(records)} vs {len(expected)}")
    train_coco = _subset_coco(base, records, train)
    eval_coco = _subset_coco(base, records, eval_stems)

    annotation_stage = _stage(DAIR_ROOT / "annotations")
    _write_json(annotation_stage / "annotations" / "instances_train.json", train_coco)
    _write_json(annotation_stage / "annotations" / "instances_eval.json", eval_coco)
    _write_json(
        annotation_stage / "annotations" / "official_eval_manifest.json",
        {"source": "official_split_data.json", "train": len(train), "eval": len(eval_stems), "eval_source": "official_val"},
    )
    _replace_dirs(DAIR_ROOT, annotation_stage, ("annotations",), apply)

    source = {}
    for split in ("train", "val", "test"):
        for meta in (DAIR_YOLO / "labels_meta" / split).glob("*.json"):
            source[meta.stem] = (DAIR_YOLO / "labels" / split / f"{meta.stem}.txt", meta)
    if set(source) != expected:
        raise RuntimeError("DAIR YOLO metadata does not cover the official train+val lists")
    yolo_stage = _stage(DAIR_YOLO)
    for split, stems in (("train", train), ("eval", eval_stems)):
        for stem in stems:
            label, meta = source[stem]
            _link_or_copy(DAIR_ROOT / "image" / f"{stem}.jpg", yolo_stage / "images" / split / f"{stem}.jpg")
            _copy(label, yolo_stage / "labels" / split / label.name)
            _copy(meta, yolo_stage / "labels_meta" / split / meta.name)
    (yolo_stage / "dairv2x.yaml").write_text(
        "train: images/train\nval: images/eval\nval_coco_ann: /root/autodl-fs/datasets/DAIR-V2X/annotations/instances_eval.json\nnc: 8\nnames: [Car, Truck, Van, Bus, Pedestrian, Cyclist, Motorcyclist, Trafficcone]\n",
        encoding="utf-8",
    )
    _replace_dirs(DAIR_YOLO, yolo_stage, ("images", "labels", "labels_meta", "dairv2x.yaml"), apply)
    return len(train), len(eval_stems)


def prepare_ua(apply: bool) -> tuple[int, int]:
    manifest = UA_ROOT / "annotations" / "official_eval_manifest.json"
    if manifest.is_file():
        value = _load(manifest)
        if (value.get("train_sequences"), value.get("test_sequences")) == (60, 40):
            return int(value["train_images"]), int(value["test_images"])
        raise RuntimeError(f"unexpected existing UA official manifest: {manifest}")
    source_files = sorted((UA_ROOT / "annotations").glob("instances_*.json"))
    base, records = _coco_by_stem(source_files)
    groups: dict[str, list[str]] = {"train_xml": [], "test_xml": []}
    image_sources: dict[str, Path] = {}
    for path in source_files:
        for image in _load(path).get("images", []):
            stem = Path(str(image["file_name"])).stem
            provenance = image.get("weather_source")
            if provenance not in groups:
                raise RuntimeError(f"UA image {stem} has no official XML provenance")
            groups[provenance].append(stem)
            image_sources[stem] = path.parent.parent / path.stem.removeprefix("instances_") / image["file_name"]
    if (len(groups["train_xml"]), len(groups["test_xml"])) != (4134, 2824):
        raise RuntimeError("unexpected UA sampled official split sizes")
    train_coco = _subset_coco(base, records, groups["train_xml"])
    test_coco = _subset_coco(base, records, groups["test_xml"])

    coco_stage = _stage(UA_ROOT)
    for split, stems in (
        ("train", groups["train_xml"]),
        ("val", groups["test_xml"]),
        ("test", groups["test_xml"]),
    ):
        for stem in stems:
            _link_or_copy(image_sources[stem], coco_stage / split / image_sources[stem].name)
    _write_json(coco_stage / "annotations" / "instances_train.json", train_coco)
    _write_json(coco_stage / "annotations" / "instances_val.json", test_coco)
    _write_json(coco_stage / "annotations" / "instances_test.json", test_coco)
    _write_json(coco_stage / "annotations" / "official_eval_manifest.json", {"train_sequences": 60, "test_sequences": 40, "train_images": len(groups["train_xml"]), "test_images": len(groups["test_xml"])})
    _replace_dirs(UA_ROOT, coco_stage, ("train", "val", "test", "annotations"), apply)

    yolo_records: dict[str, tuple[Path, Path, Path, str]] = {}
    for split in ("train", "val", "test"):
        for meta in (UA_YOLO / "labels_meta" / split).glob("*.json"):
            info = _load(meta)
            yolo_records[meta.stem] = (UA_YOLO / "images" / split / f"{meta.stem}.jpg", UA_YOLO / "labels" / split / f"{meta.stem}.txt", meta, info["weather_source"])
    yolo_stage = _stage(UA_YOLO)
    for target, source_name in (("train", "train_xml"), ("test", "test_xml")):
        for stem, (image, label, meta, provenance) in yolo_records.items():
            if provenance != source_name:
                continue
            _link_or_copy(image, yolo_stage / "images" / target / image.name)
            _copy(label, yolo_stage / "labels" / target / label.name)
            _copy(meta, yolo_stage / "labels_meta" / target / meta.name)
    (yolo_stage / "data.yaml").write_text(
        "train: images/train\nval: images/test\ntest: images/test\nnc: 4\nnames: [car, van, bus, others]\n",
        encoding="utf-8",
    )
    _replace_dirs(UA_YOLO, yolo_stage, ("images", "labels", "labels_meta", "data.yaml"), apply)
    return len(groups["train_xml"]), len(groups["test_xml"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="replace derived split folders after staging")
    args = parser.parse_args()
    dair = prepare_dair(args.apply)
    ua = prepare_ua(args.apply)
    print(f"DAIR-V2X train/eval={dair[0]}/{dair[1]}; UA-DETRAC train/test={ua[0]}/{ua[1]}; apply={args.apply}")


if __name__ == "__main__":
    main()
