#!/usr/bin/env python3
"""Run validation/benchmarks for an existing YOLO checkpoint without training."""

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
YOLO_DIR = Path(__file__).resolve().parent
if str(YOLO_DIR) not in sys.path:
    sys.path.insert(0, str(YOLO_DIR))

from train import build_trainer, normalize_version  # noqa: E402
from common.dataset_registry import (  # noqa: E402
    find_dataset_profile_by_data_yaml,
    load_dataset_registry,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="5, 8, or 12")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--no-tensorrt", action="store_true")
    args = parser.parse_args()

    version = normalize_version(args.version)
    config_path = args.config.resolve()
    weights = args.weights.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not weights.is_file():
        raise FileNotFoundError(weights)

    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    profile = find_dataset_profile_by_data_yaml(
        load_dataset_registry(YOLO_DIR / "configs" / "datasets.yaml"),
        config.get("data", {}).get("data_yaml", ""),
    )
    class_names = (profile or {}).get("class_names", [])
    trainer = build_trainer(
        version, config, str(config_path), class_names=class_names,
        resume_checkpoint=str(weights),
    )
    from ultralytics import YOLO

    model = YOLO(str(weights))
    bench = trainer._optional_post_train_benchmark(model)
    trainer._evaluate_coco_scale_after_training(model, bench_dict=bench)
    if not args.no_tensorrt:
        trainer.run_tensorrt_benchmark()
    print(f"validated run_id={trainer.log_dir.name}")


if __name__ == "__main__":
    main()
