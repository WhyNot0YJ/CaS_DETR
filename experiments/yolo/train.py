#!/usr/bin/env python3
"""统一YOLO训练入口（v5/v8/v12）"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

import yaml

_experiments_root = Path(__file__).resolve().parent.parent
if str(_experiments_root) not in sys.path:
    sys.path.insert(0, str(_experiments_root))

_yolo_dir = Path(__file__).resolve().parent
_yolox_repo = _yolo_dir / "external" / "YOLOX"
if _yolox_repo.is_dir() and str(_yolox_repo) not in sys.path:
    sys.path.insert(0, str(_yolox_repo))

from common.dataset_registry import (
    load_dataset_registry,
    resolve_dataset_profile,
    find_dataset_profile_by_data_yaml,
    apply_yolo_dataset_profile as apply_dataset_profile,
)
from common.dataset_protocol import set_report_protocol
from common.train_notifications import notify_training_entry

DEFAULT_CLASS_NAMES: List[str] = []


def normalize_version(version: str) -> str:
    value = version.lower().strip()
    if value.startswith("v"):
        value = value[1:]
    if value not in {"5", "8", "12"}:
        raise ValueError(f"不支持的YOLO版本: {version}，可选: v5/v8/v12")
    return value


def default_config_for_version(version: str) -> Path:
    return Path(f"configs/yolov{version}n_dairv2x.yaml")


def build_trainer(version: str, config: dict, config_path: Optional[str] = None,
                  class_names: Optional[List[str]] = None,
                  resume_checkpoint: Optional[str] = None):
    from base_yolo_trainer import BaseYOLOTrainer

    class UnifiedYOLOTrainer(BaseYOLOTrainer):
        VERSION = "base"

        def __init__(self, trainer_version: str, trainer_config: dict, trainer_config_path: Optional[str] = None):
            self.VERSION = normalize_version(trainer_version)
            super().__init__(
                trainer_config, trainer_config_path, class_names or DEFAULT_CLASS_NAMES,
                resume_checkpoint=resume_checkpoint,
            )

        def create_model(self):
            from ultralytics import YOLO

            model_name = self._resolve_model_path()
            self.logger.info(f"✓ 创建YOLO{self.VERSION}模型: {model_name}")
            return YOLO(model_name)

    return UnifiedYOLOTrainer(version, config, config_path)


def find_latest_checkpoint(log_base: str, config: dict) -> Optional[str]:
    data_yaml = str(config.get("data", {}).get("data_yaml", "")).lower()
    if "dair" in data_yaml:
        dataset_dir = "dairv2x"
    elif "uadetrac" in data_yaml or "ua-detrac" in data_yaml:
        dataset_dir = "uadetrac"
    else:
        dataset_dir = Path(data_yaml).stem
    log_dir = _yolo_dir / log_base / dataset_dir
    if not log_dir.exists():
        return None
    checkpoints = list(log_dir.glob("**/weights/best.pt"))
    if not checkpoints:
        return None
    return str(max(checkpoints, key=lambda path: path.stat().st_mtime))


@notify_training_entry("YOLO")
def main():
    parser = argparse.ArgumentParser(description="统一YOLO训练入口")
    parser.add_argument("--version", type=str, required=True, help="YOLO版本: v5/v8/v12")
    parser.add_argument("--config", type=str, default=None, help="YAML配置文件路径")
    parser.add_argument("--dataset", type=str, default=None, help="数据集键名或别名（在 configs/datasets.yaml 中定义）")
    protocol = parser.add_mutually_exclusive_group()
    protocol.add_argument("--dairv2x", action="store_true")
    protocol.add_argument("--uadetrac", action="store_true")
    parser.add_argument("--dataset_registry", type=str, default="configs/datasets.yaml", help="数据集注册表路径")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="从指定检查点恢复")
    parser.add_argument("--resume", action="store_true", help="自动从最新检查点恢复")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖配置中的epochs")
    parser.add_argument("--seed", type=int, default=None, help="覆盖 training.seed，传给 Ultralytics train")
    args = parser.parse_args()

    version = normalize_version(args.version)
    config_path = Path(args.config) if args.config else default_config_for_version(version)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    selected_class_names = DEFAULT_CLASS_NAMES
    datasets = load_dataset_registry(Path(args.dataset_registry))
    if args.dairv2x:
        args.dataset = "dairv2x"
    elif args.uadetrac:
        args.dataset = "uadetrac"

    if args.dataset:
        profile = resolve_dataset_profile(datasets, args.dataset)
        config = apply_dataset_profile(config, profile)

        profile_classes = profile.get("class_names", [])
        if isinstance(profile_classes, list) and profile_classes:
            selected_class_names = [str(name) for name in profile_classes]

        print(f"🗂️  使用数据集: {args.dataset} -> {config.get('data', {}).get('data_yaml')}")
    else:
        profile = find_dataset_profile_by_data_yaml(datasets, config.get("data", {}).get("data_yaml", ""))
        if profile:
            config = apply_dataset_profile(config, profile)
            profile_classes = profile.get("class_names", [])
            if isinstance(profile_classes, list) and profile_classes:
                selected_class_names = [str(name) for name in profile_classes]
    if profile:
        set_report_protocol(str(profile.get("report_protocol", "dairv2x")))

    if args.resume and not args.resume_from_checkpoint:
        log_base = config.get("checkpoint", {}).get("log_dir", "logs")
        latest_checkpoint = find_latest_checkpoint(log_base, config)
        if latest_checkpoint:
            args.resume_from_checkpoint = latest_checkpoint
            print(f"📦 找到最新检查点: {latest_checkpoint}")

    if args.seed is not None:
        config.setdefault("training", {})["seed"] = args.seed
        print(f"🎲 覆盖 seed: {args.seed}")

    from ultralytics import settings

    settings.update({"tensorboard": True})
    trainer = build_trainer(
        version=version,
        config=config,
        config_path=str(config_path),
        class_names=selected_class_names,
        resume_checkpoint=args.resume_from_checkpoint,
    )
    trainer.start_training(
        resume_checkpoint=args.resume_from_checkpoint,
        epochs_override=args.epochs,
    )
    # TensorRT benchmark 仅用于测速；失败不应让整个实验失败（训练+评测已成功）。
    try:
        trainer.run_tensorrt_benchmark()
    except Exception as exc:
        print(f"[train-notify] TensorRT benchmark 失败: {exc}")
    return {"output_dir": str(trainer.log_dir)}


if __name__ == "__main__":
    main()
