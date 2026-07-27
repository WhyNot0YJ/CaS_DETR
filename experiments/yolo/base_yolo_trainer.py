#!/usr/bin/env python3
"""
YOLO 训练器基类：减少重复，支持 YOLO v8、v10、v11、v12
"""

import sys
import os
import gc
import json
import subprocess
import yaml
import torch
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
from abc import ABC, abstractmethod
from collections import Counter
import shutil

_experiments_root = Path(__file__).resolve().parent.parent
if str(_experiments_root) not in sys.path:
    sys.path.insert(0, str(_experiments_root))

# 本地 ``external/ultralytics`` 与 ``external/YOLOX``（非 site-packages）
_yolo_dir = Path(__file__).resolve().parent
_external = _yolo_dir / "external"
if _external.is_dir() and str(_external) not in sys.path:
    sys.path.insert(0, str(_external))
_yolox_repo = _external / "YOLOX"
if _yolox_repo.is_dir() and str(_yolox_repo) not in sys.path:
    sys.path.insert(0, str(_yolox_repo))

from common.vram_batch import (
    compute_vram_batch_adjustment,
    format_vram_batch_log,
    resolve_cuda_device_index,
)
from common.model_benchmark import (
    BENCHMARK_EVAL_METRIC_KEYS,
    END_TO_END_EVAL_METRIC_KEYS,
    benchmark_to_dict,
    format_benchmark_eval_line,
    format_eval_csv_cell,
    log_benchmark,
    merge_benchmark_dict_into_metrics,
)
from common.result_paths import result_csv
from common.det_eval_metrics import (
    PYCOCOTOOLS_AVAILABLE,
    coco_ap_at_iou50_all,
    coco_area_ap_at_iou50,
    coco_area_bucket_counts_from_xywh_annotations,
    canonical_category_metric_name,
    extract_per_category_ap_from_coco_eval,
    format_area_bucket_counts,
    run_coco_bbox_eval,
)
from common.det_eval_metrics import write_eval_csv

from yolo_validator_utils import MetricsLogger

# Ultralytics ``cfg/default.yaml``：未指定 batch 时为 16
DEFAULT_TRAIN_BATCH = 16


class BaseYOLOTrainer(ABC):
    """所有YOLO训练器的基类"""
    
    # 子类需要实现的版本名称
    VERSION = "base"
    
    def __init__(
        self,
        config: Dict,
        config_path: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        resume_checkpoint: Optional[str] = None,
    ):
        """
        初始化基础训练器
        
        Args:
            config: 配置字典
            config_path: 配置文件路径
            class_names: 类别名称列表
            resume_checkpoint: 若从 ``weights/*.pt`` 续训，传入路径可在首次 ``setup_logging`` 时锚定实验根目录
        """
        self.config = config
        self.config_path = config_path
        self.class_names = class_names or []
        self.num_classes = len(self.class_names)
        self._resume_checkpoint_path = resume_checkpoint

        # 提取配置段
        self.model_config = config.get('model', {})
        self.training_config = config.get('training', {})
        self.data_config = config.get('data', {})
        self.checkpoint_config = config.get('checkpoint', {})
        self.misc_config = config.get('misc', {})
        
        # 日志和指标记录
        self.logger = None
        self.log_dir = None
        self.metrics_logger = None
        
        # 设置日志
        self.setup_logging()
        
        self._apply_vram_batch_size_rule()
        
        # 验证配置
        self._validate_config()
        
        self._log_initialization_info()
    
    def setup_logging(self):
        """设置日志系统"""
        resume_checkpoint = getattr(self, '_resume_checkpoint_path', None)

        def _experiment_root_from_ckpt(ckpt: Path) -> Path:
            ckpt = ckpt.resolve()
            parent = ckpt.parent
            if parent.name == "weights":
                exp = parent.parent
                if (exp / "config.yaml").is_file():
                    return exp
            return parent
        
        if resume_checkpoint and Path(resume_checkpoint).exists():
            self.log_dir = _experiment_root_from_ckpt(Path(resume_checkpoint))
            self.experiment_name = self.log_dir.name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = self.model_config.get('model_name', f'yolo{self.VERSION}n')
            if model_name.endswith('.pt'):
                model_name = model_name[:-3]
            
            self.experiment_name = f"yolo_{model_name.replace(f'yolo{self.VERSION}', f'v{self.VERSION}')}"
            log_base = self.checkpoint_config.get('log_dir', 'logs')
            data_yaml = self.data_config.get('data_yaml', '')
            ds_stem = Path(data_yaml).stem if data_yaml else 'unknown'
            if 'dairv2x' in ds_stem.lower() or 'dair' in ds_stem.lower():
                ds_dir = 'dairv2x'
            elif 'uadetrac' in ds_stem.lower() or 'ua' in ds_stem.lower() or ds_stem == 'data':
                ds_dir = 'uadetrac'
            else:
                ds_dir = ds_stem
            # 锚定到本文件所在目录（experiments/yolo），避免 cwd 与 YOLOX/Ultralytics 不一致时路径错位
            self.log_dir = (_yolo_dir / log_base / ds_dir / f"{self.experiment_name}_{timestamp}").resolve()
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置日志处理器
        handlers = [
            logging.FileHandler(self.log_dir / 'training.log', mode='a'),
            logging.StreamHandler()
        ]
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=handlers,
            force=True
        )
        
        self.logger = logging.getLogger(__name__)
        
        # 保存配置文件（仅新训练时）
        if not resume_checkpoint:
            config_save_path = self.log_dir / 'config.yaml'
            with open(config_save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            self.logger.info(f"✓ 配置已保存到: {config_save_path}")
        
        # 初始化指标记录器
        self.metrics_logger = MetricsLogger(self.log_dir)
    
    def _validate_config(self):
        """验证配置文件"""
        required_keys = {
            'model': ['model_name'],
            'training': ['epochs'],
            'data': ['data_yaml']
        }
        
        missing_keys = []
        for section, keys in required_keys.items():
            if section not in self.config:
                missing_keys.append(f"缺少配置节: {section}")
                continue
            for key in keys:
                if key not in self.config[section]:
                    missing_keys.append(f"{section}.{key}")
        
        if missing_keys:
            error_msg = f"配置文件缺少必需的配置项:\n"
            error_msg += "\n".join(f"  - {key}" for key in missing_keys)
            raise ValueError(error_msg)
    
    def _log_initialization_info(self):
        """记录初始化信息"""
        if self.logger:
            self.logger.info(f"✓ 初始化YOLO{self.VERSION}训练器")
            self.logger.info(f"  类别数量: {self.num_classes}")
            if self.class_names:
                self.logger.info(f"  类别: {', '.join(self.class_names)}")
    
    @abstractmethod
    def create_model(self):
        """创建YOLO模型（由子类实现）"""
        pass
    
    def _resolve_model_path(self, model_name: Optional[str] = None) -> str:
        """
        解析预训练权重路径
        
        Args:
            model_name: 模型名称或路径
            
        Returns:
            解析后的模型路径
        """
        model_name = model_name or self.model_config.get('model_name', f'yolov8n.pt')
        pretrained_weights = self.model_config.get('pretrained_weights', None)
        
        if not pretrained_weights:
            return model_name
        
        pretrained_path = Path(pretrained_weights)
        if not pretrained_path.is_absolute():
            candidates: List[Path] = []
            if self.config_path:
                candidates.append(
                    Path(self.config_path).resolve().parent / pretrained_weights
                )
            candidates.append(Path(__file__).resolve().parent / pretrained_weights)
            found: Optional[Path] = None
            for c in candidates:
                if c.is_file():
                    found = c
                    break
            pretrained_path = found if found is not None else candidates[-1]
        
        if pretrained_path.exists():
            self.logger.info(f"✓ 加载预训练权重: {pretrained_path}")
            return str(pretrained_path)
        else:
            self.logger.warning(f"⚠️  预训练权重文件不存在: {pretrained_path}")
            self.logger.info(f"   将使用模型名称自动加载: {model_name}")
            return model_name
    
    def _resolve_data_yaml(self) -> str:
        """解析 Ultralytics 用的 data.yaml 路径（与 DETR 共用路径解析）。"""
        from common.detr_data_root import resolve_autodl_fs_path
        from common.dataset_registry import (
            load_dataset_registry,
            find_dataset_profile_by_coco_root,
        )

        data_yaml = self.data_config.get("data_yaml")
        if data_yaml and str(data_yaml).strip():
            return resolve_autodl_fs_path(data_yaml)

        # CaS_DETR 等仅配置 data_root（COCO 根），无 data_yaml：从 datasets.yaml 按 coco_data_root 反查
        root_raw = self.data_config.get("data_root") or self.data_config.get("coco_data_root")
        if root_raw and str(root_raw).strip():
            resolved_root = resolve_autodl_fs_path(str(root_raw).strip())
            registry_path = Path(__file__).resolve().parent / "configs" / "datasets.yaml"
            if registry_path.is_file():
                try:
                    datasets = load_dataset_registry(registry_path)
                    profile = find_dataset_profile_by_coco_root(datasets, resolved_root)
                    dy = (profile or {}).get("data_yaml")
                    if dy and str(dy).strip():
                        return resolve_autodl_fs_path(str(dy).strip())
                except Exception:
                    pass
            raise FileNotFoundError(
                f"配置中无 data_yaml，且无法根据 data_root={resolved_root!r} "
                f"在 {registry_path} 中匹配到 coco_data_root；请显式设置 data.data_yaml"
            )

        raise ValueError("路径为空：data.data_yaml 未设置且 data.data_root 为空")
    
    def _apply_vram_batch_size_rule(self):
        """使用配置中的 batch / workers；CUDA 下记录显存信息（与 cas_detr 共用 common.vram_batch）。"""
        device_str = self.misc_config.get('device', 'cuda')
        if 'cuda' not in str(device_str) or not torch.cuda.is_available():
            return

        idx = resolve_cuda_device_index(str(device_str))
        orig_bs = int(self.training_config.get('batch_size', DEFAULT_TRAIN_BATCH))
        orig_nw = int(self.misc_config.get('num_workers', 2))
        orig_pf = int(self.misc_config.get('prefetch_factor', 1))

        r = compute_vram_batch_adjustment(
            orig_bs, orig_nw, orig_pf, device_index=idx
        )
        if r is None:
            return

        self.training_config['batch_size'] = r.batch_size
        self.misc_config['num_workers'] = r.num_workers
        self.misc_config['prefetch_factor'] = r.prefetch_factor

        if self.logger:
            self.logger.info(format_vram_batch_log(r))

    def _build_train_kwargs(self) -> Dict:
        """
        构建传给 ``model.train()`` 的参数：仅包含 YAML 中已写的项，以及输出目录所需的
        ``data`` / ``project`` / ``name``；其余由 Ultralytics 默认配置补齐。
        """
        train_kwargs: Dict = {
            'data': self._resolve_data_yaml(),
            'project': str(self.log_dir.parent),
            'name': self.log_dir.name,
        }
        for k, v in self.training_config.items():
            if k == 'batch_size':
                train_kwargs['batch'] = v
            else:
                train_kwargs[k] = v
        if 'device' in self.misc_config:
            train_kwargs['device'] = self.misc_config['device']
        if 'num_workers' in self.misc_config:
            train_kwargs['workers'] = self.misc_config['num_workers']
        # ``setup_logging`` creates the intended run directory before Ultralytics starts.
        # Reuse it so weights and post-training TensorRT lookup stay in the same path.
        train_kwargs['exist_ok'] = self.checkpoint_config.get('exist_ok', True)
        return train_kwargs

    def _log_training_config(self, train_kwargs: Dict):
        """记录训练配置信息"""
        self.logger.info("=" * 80)
        self.logger.info(f"🚀 开始YOLO{self.VERSION}训练")
        self.logger.info("=" * 80)
        self.logger.info("📝 训练配置:")
        self.logger.info(f"  数据集路径: {train_kwargs['data']}")
        if 'epochs' in train_kwargs:
            self.logger.info(f"  训练轮数: {train_kwargs['epochs']}")
        if 'batch' in train_kwargs:
            self.logger.info(f"  批次大小: {train_kwargs['batch']}")
        for k in ('optimizer', 'lr0', 'weight_decay', 'imgsz'):
            if k in self.training_config:
                self.logger.info(f"  {k}: {self.training_config[k]}")
        self.logger.info(f"  输出目录: {self.log_dir}")
        if self.model_config.get('pretrained_weights'):
            self.logger.info(f"  预训练权重: {self.model_config['pretrained_weights']}")
        if 'seed' in train_kwargs:
            self.logger.info(f"  随机种子 seed: {train_kwargs['seed']}")
        if 'deterministic' in train_kwargs:
            self.logger.info(f"  deterministic: {train_kwargs['deterministic']}")
        self.logger.info("=" * 80)
    
    def start_training(self, resume_checkpoint: Optional[str] = None, 
                      epochs_override: Optional[int] = None):
        """
        开始训练
        
        Args:
            resume_checkpoint: 恢复训练的检查点路径
            epochs_override: 覆盖epochs（用于测试）
        """
        self._resume_checkpoint_path = resume_checkpoint
        self.setup_logging()  # 重新初始化日志

        resume_model_path: Optional[Path] = None
        if resume_checkpoint:
            checkpoint_path = Path(resume_checkpoint)
            if checkpoint_path.is_file():
                resume_model_path = checkpoint_path
            elif checkpoint_path.is_dir():
                for candidate in (
                    checkpoint_path / "weights" / "last.pt",
                    checkpoint_path / "weights" / "best.pt",
                    checkpoint_path / "last.pt",
                    checkpoint_path / "best.pt",
                ):
                    if candidate.is_file():
                        resume_model_path = candidate
                        break
            if resume_model_path is None:
                raise FileNotFoundError(f"未找到可用于恢复的检查点: {resume_checkpoint}")

        # 创建模型；恢复训练时直接从 checkpoint 重新构建，避免 Ultralytics
        # 把 base 权重当成新的起点，从而误判“无可恢复状态”。
        if resume_model_path is not None:
            from ultralytics import YOLO as _YOLO

            self.logger.info(f"📦 从检查点加载模型对象: {resume_model_path}")
            model = _YOLO(str(resume_model_path))
        else:
            model = self.create_model()
        
        # 构建训练参数
        train_kwargs = self._build_train_kwargs()
        
        # 覆盖epochs如果提供了
        if epochs_override is not None:
            config_epochs = train_kwargs['epochs']
            train_kwargs['epochs'] = epochs_override
            self.logger.info(f"⚠️  测试模式：使用命令行参数覆盖epochs ({config_epochs} → {epochs_override})")
        
        # 恢复训练
        if resume_model_path is not None:
            self.logger.info(f"📦 从检查点恢复训练: {resume_model_path}")
            train_kwargs['resume'] = True
        
        # 记录配置
        self._log_training_config(train_kwargs)
        
        # 执行训练
        try:
            results = model.train(**train_kwargs)
            self._post_training_processing(model)
            return results
        except Exception as e:
            self.logger.error(f"训练失败: {e}")
            raise
    
    # ------------------------------------------------------------------
    # Post-training KITTI / multi-scale evaluation
    # ------------------------------------------------------------------

    def _labels_meta_split_dir(self, root: Path, split: str) -> Path:
        """``root/labels_meta/{split}``；若为空且配置了 ``data.coco_data_root``，则尝试该根下的同名路径。"""
        primary = root / 'labels_meta' / split
        if primary.is_dir() and any(primary.glob('*.json')):
            return primary
        cr = self.data_config.get('coco_data_root')
        if not cr:
            return primary
        alt = Path(str(cr)).expanduser().resolve() / 'labels_meta' / split
        if alt.resolve() == primary.resolve():
            return primary
        if alt.is_dir() and any(alt.glob('*.json')):
            self.logger.info(
                "KITTI/scale 使用 data.coco_data_root 下的 labels_meta/%s: %s",
                split,
                alt,
            )
            return alt
        return primary

    def _resolve_labels_meta_dir(self, data_cfg: dict, root: Path, split: str) -> Path:
        override_key = f'{split}_labels_meta'
        override_rel = str(data_cfg.get(override_key, '')).strip()
        if override_rel:
            path = Path(override_rel)
            return path if path.is_absolute() else root / path
        return self._labels_meta_split_dir(root, split)

    def _resolve_eval_image_dirs(self, value: Any, root: Path) -> List[Path]:
        if isinstance(value, (list, tuple)):
            raw_values = value
        else:
            raw_values = [value]
        out = []
        for raw in raw_values:
            rel = str(raw).strip()
            if not rel:
                continue
            path = Path(rel)
            out.append(path if path.is_absolute() else root / path)
        return out

    def _resolve_coco_ann_file(self, data_cfg: dict, root: Path, split: str) -> Optional[Path]:
        ann_rel = str(data_cfg.get(f'{split}_coco_ann', '')).strip()
        if not ann_rel:
            return None
        path = Path(ann_rel)
        return path if path.is_absolute() else root / path

    def _load_coco_meta_by_stem(self, coco_ann_file: Path) -> Dict[str, Dict[str, Any]]:
        raw_coco = json.loads(coco_ann_file.read_text(encoding='utf-8'))
        images_by_id = {
            int(image['id']): image for image in raw_coco.get('images', [])
        }
        annotations_by_image: Dict[int, List[Dict[str, Any]]] = {}
        for ann in raw_coco.get('annotations', []):
            annotations_by_image.setdefault(int(ann['image_id']), []).append(ann)

        meta_by_stem: Dict[str, Dict[str, Any]] = {}
        for image_id, image in images_by_id.items():
            width = float(image['width'])
            height = float(image['height'])
            objects = []
            for ann in annotations_by_image.get(image_id, []):
                x, y, w, h = map(float, ann['bbox'])
                objects.append(
                    {
                        'class_id': int(ann['category_id']) - 1,
                        'bbox_yolo': {
                            'cx': (x + w / 2.0) / width,
                            'cy': (y + h / 2.0) / height,
                            'w': w / width,
                            'h': h / height,
                        },
                        'bbox_xyxy': [x, y, x + w, y + h],
                    }
                )
            meta_by_stem[Path(str(image['file_name'])).stem] = {
                'objects': objects,
                'weather': image.get('weather', ''),
            }
        return meta_by_stem

    def _resolve_kitti_eval_splits(
        self, data_cfg: dict, root: Path
    ) -> List[Tuple[str, List[Path], Optional[Path], Optional[Path]]]:
        """
        返回 [(split, image_dirs, labels_meta_dir, coco_ann_file), ...]，顺序 **val → test**。
        对应目录须存在，且 labels_meta 下有 JSON 或配置了 ``{split}_coco_ann``。
        """
        eval_test = self.data_config.get('eval_test_after_training', True)
        out: List[Tuple[str, List[Path], Optional[Path], Optional[Path]]] = []

        val_img_dirs = self._resolve_eval_image_dirs(
            data_cfg.get('val', 'images/val'), root
        )
        lm_val = self._resolve_labels_meta_dir(data_cfg, root, 'val')
        val_ann = self._resolve_coco_ann_file(data_cfg, root, 'val')
        if (lm_val.is_dir() and any(lm_val.glob('*.json'))) or (
            val_ann is not None and val_ann.is_file()
        ):
            out.append(('val', val_img_dirs, lm_val, val_ann))

        test_rel = str(data_cfg.get('test', '')).strip()
        if eval_test and test_rel:
            test_dirs = self._resolve_eval_image_dirs(data_cfg.get('test', ''), root)
            lm_test = self._resolve_labels_meta_dir(data_cfg, root, 'test')
            test_ann = self._resolve_coco_ann_file(data_cfg, root, 'test')
            has_images = any(path.is_dir() for path in test_dirs)
            has_meta = lm_test.is_dir() and any(lm_test.glob('*.json'))
            has_ann = test_ann is not None and test_ann.is_file()
            if has_images and (has_meta or has_ann):
                out.append(('test', test_dirs, lm_test, test_ann))

        return out

    def _get_kitti_eval_predictor(self, model):
        """
        Return (predictor, num_classes) for KITTI/scale eval.
        Default: Ultralytics ``YOLO`` loaded from ``weights/best.pt``.
        """
        best_pt = self.log_dir / 'weights' / 'best.pt'
        from ultralytics import YOLO as _YOLO
        eval_model = _YOLO(str(best_pt)) if best_pt.exists() else model
        nc = (
            len(eval_model.names)
            if eval_model is not None
            and hasattr(eval_model, 'names')
            and eval_model.names
            else max(len(self.class_names), 1)
        )
        return eval_model, nc

    def _predict_batch_kitti_eval(self, predictor, batch_paths, imgsz, device):
        """Run batch inference for KITTI eval (Ultralytics API)."""
        return predictor.predict(
            source=[str(p) for p in batch_paths],
            conf=0.01,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )

    def _benchmark_eval_predictor(self, eval_predictor) -> Optional[dict]:
        """GFLOPs/FPS on the same weights used for KITTI eval (e.g. ``best.pt``)."""
        return self._run_model_benchmark(eval_predictor)

    def _optional_post_train_benchmark(self, model) -> Optional[dict]:
        """After training: GFLOPs/FPS. Override for non-Ultralytics backends."""
        if model is None:
            return None
        return self._run_model_benchmark(model)

    def _can_run_kitti_eval_without_ultralytics_model(self) -> bool:
        """If True, run KITTI/scale eval even when ``model`` is None (e.g. YOLOX)."""
        return False

    def _kitti_eval_batch_size(self) -> int:
        """Return the initial post-training evaluation batch size.

        Evaluation is independent from the training batch size.  The
        environment override is useful for one-off low-VRAM reruns without
        changing a saved training config; ``data.eval_batch_size`` provides a
        persistent per-dataset override.
        """
        candidates = (
            os.environ.get("CAS_EVAL_BATCH_SIZE"),
            self.data_config.get("eval_batch_size"),
        )
        for raw in candidates:
            if raw is None or not str(raw).strip():
                continue
            try:
                value = int(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"eval_batch_size must be an integer, got {raw!r}"
                ) from exc
            if value < 1:
                raise ValueError(f"eval_batch_size must be positive, got {value}")
            return value
        return 32

    @staticmethod
    def _is_cuda_oom(exc: BaseException) -> bool:
        """Recognize CUDA allocation failures that are safe to retry smaller."""
        message = str(exc).lower()
        return isinstance(exc, RuntimeError) and (
            "out of memory" in message or "cudaerrormemoryallocation" in message
        )

    def _canonical_dataset_name(self) -> str:
        """
        Return a stable human-readable dataset label.

        We prefer matching against the full configured ``data_yaml`` path first,
        because some datasets use generic filenames such as ``data.yaml`` whose
        stem alone is not descriptive enough.
        """
        data_yaml = str(self.data_config.get('data_yaml', '') or '')
        data_lower = data_yaml.lower()
        if 'dairv2x' in data_lower or 'dair-v2x' in data_lower or 'dair' in data_lower:
            return 'DAIR-V2X'
        if 'uadetrac' in data_lower or 'ua-detrac' in data_lower:
            return 'UA-DETRAC'

        stem = Path(data_yaml).stem or 'unknown'
        stem_lower = stem.lower()
        if 'dairv2x' in stem_lower or 'dair' in stem_lower:
            return 'DAIR-V2X'
        if 'uadetrac' in stem_lower or 'ua-detrac' in stem_lower:
            return 'UA-DETRAC'
        return stem

    def _evaluate_kitti_scale_after_training(self, model, bench_dict=None) -> dict:
        """
        训练结束后的 KITTI / multi-scale mAP：

        - **val**：写 ``eval_metrics.csv`` 一行；
        - **test**：当 ``data.eval_test_after_training`` 为真且存在 ``test`` 与 ``labels_meta/test`` 时再评一行。
        - Benchmark（GFLOPs/FPS 等）只算一次，两行共用。

        返回值：若跑了 test 则返回 test 的 metrics，否则返回 val。
        """
        # ── 1. Resolve dataset root ───────────────────────────────────────
        try:
            data_yaml_path = Path(self._resolve_data_yaml())
        except FileNotFoundError as exc:
            self.logger.warning(f"无法定位 data.yaml，跳过 KITTI/scale 评估: {exc}")
            return {}

        with data_yaml_path.open(encoding='utf-8') as fh:
            data_cfg = yaml.safe_load(fh) or {}

        # Resolve dataset root from the optional 'path' field in data.yaml
        root = data_yaml_path.parent.resolve()
        path_field = str(data_cfg.get('path', '')).strip()
        if path_field:
            pc = Path(path_field)
            if pc.is_absolute():
                if pc.is_dir():
                    root = pc
            else:
                proj = Path(__file__).resolve().parent.parent.parent
                for cand in [
                    (data_yaml_path.parent / pc).resolve(),
                    (proj / path_field).resolve(),
                    (proj.parent / path_field).resolve(),
                    (proj.parent / 'datasets' / path_field).resolve(),
                ]:
                    if cand.is_dir():
                        root = cand
                        break

        splits = self._resolve_kitti_eval_splits(data_cfg, root)
        if not splits:
            cr = self.data_config.get('coco_data_root')
            cr_res = Path(str(cr)).expanduser().resolve() if cr else None
            self.logger.warning(
                '未找到可用的 KITTI/scale 评估划分：在 YAML path 对应 root=%s 与 data.coco_data_root=%s '
                '下均未找到含 JSON 的 labels_meta/val 或 labels_meta/test',
                root,
                cr_res,
            )
            return {}

        # ── 2. Load best weights & benchmark（各 split 共用）──────────────
        eval_predictor, nc = self._get_kitti_eval_predictor(model)
        if eval_predictor is None:
            best_pt = (self.log_dir / "weights" / "best.pt").resolve()
            last_pt = (self.log_dir / "weights" / "last.pt").resolve()
            self.logger.warning(
                "无可用评估权重/预测器，跳过 KITTI/scale 评估：未找到 %s "
                "（eval_best_model 未传入内存中的 model，必须依赖该文件）",
                best_pt,
            )
            if last_pt.is_file():
                self.logger.warning(
                    "  发现 last.pt，可执行: cp %s %s 后再评估",
                    last_pt,
                    best_pt,
                )
            return {}

        device = self.misc_config.get('device', 'cuda')
        imgsz = self.training_config.get('imgsz', 640)

        if bench_dict is None and eval_predictor is not None:
            bench_dict = self._benchmark_eval_predictor(eval_predictor)

        model_name = self.model_config.get('model_name', f'yolov{self.VERSION}n')
        if model_name.endswith('.pt'):
            model_name = model_name[:-3]
        dataset_name = self._canonical_dataset_name()

        class_names = self.class_names if self.class_names else [f'cls_{i}' for i in range(nc)]

        def _weather_metric_key(value: str) -> str:
            key = ''.join(ch.lower() if ch.isalnum() else '_' for ch in str(value))
            key = '_'.join(part for part in key.split('_') if part)
            return key or 'unknown'

        weather_names: List[str] = []
        weather_seen = set()
        for _, _, labels_meta_dir, coco_ann_file in splits:
            if labels_meta_dir and labels_meta_dir.is_dir() and any(labels_meta_dir.glob('*.json')):
                for meta_path in labels_meta_dir.glob('*.json'):
                    try:
                        raw_meta = json.loads(meta_path.read_text(encoding='utf-8'))
                    except Exception:
                        continue
                    if not isinstance(raw_meta, dict):
                        continue
                    weather = str(raw_meta.get('weather', '')).strip()
                    if weather and weather not in weather_seen:
                        weather_seen.add(weather)
                        weather_names.append(weather)
            elif coco_ann_file and coco_ann_file.is_file():
                try:
                    raw_coco = json.loads(coco_ann_file.read_text(encoding='utf-8'))
                except Exception:
                    raw_coco = {}
                for image in raw_coco.get('images', []):
                    weather = str(image.get('weather', '')).strip()
                    if weather and weather not in weather_seen:
                        weather_seen.add(weather)
                        weather_names.append(weather)
        weather_names = sorted(weather_names)
        weather_buckets = [_weather_metric_key(weather) for weather in weather_names]
        summary_csv = result_csv('eval_metrics')
        last_metrics: Dict[str, Any] = {}

        if True:
            for eval_split, eval_img_dirs, labels_meta_dir, coco_ann_file in splits:
                # ── 3. Per-split: images + meta ───────────────────────────
                if labels_meta_dir and labels_meta_dir.is_dir() and any(labels_meta_dir.glob('*.json')):
                    meta_by_stem = {p.stem: p for p in labels_meta_dir.glob('*.json')}
                elif coco_ann_file and coco_ann_file.is_file():
                    meta_by_stem = self._load_coco_meta_by_stem(coco_ann_file)
                else:
                    meta_by_stem = {}
                if not meta_by_stem:
                    self.logger.warning(
                        f"{eval_split} 未找到 labels_meta 或 COCO 标注，跳过"
                    )
                    continue

                eval_images = sorted(
                    p
                    for eval_img_dir in eval_img_dirs
                    for ext in ('.jpg', '.jpeg', '.png')
                    for p in eval_img_dir.glob(f'*{ext}')
                    if p.stem in meta_by_stem
                )
                if not eval_images:
                    self.logger.warning(
                        f"{eval_split} 无与 meta 匹配的图像: {eval_img_dirs}"
                    )
                    continue

                self.logger.info(f"📊 评估 [{eval_split}, {len(eval_images)} 张]")

                # ── 4. Collect GT and raw predictions ───────────────────────
                debug_gt_annotations: List[Dict[str, Any]] = []
                debug_pred_annotations: List[Dict[str, Any]] = []
                debug_image_ids: set[int] = set()
                # COCOeval：全类 mAP / S/M/L / 每类 AP
                img_sizes: Dict[int, Tuple[int, int]] = {}
                coco_annotations: List[Dict[str, Any]] = []
                coco_predictions: List[Dict[str, Any]] = []
                weather_by_image_id: Dict[int, str] = {}
                ann_id = 0

                eval_batch_size = self._kitti_eval_batch_size()
                batch_start = 0
                while batch_start < len(eval_images):
                    batch_paths = eval_images[
                        batch_start: batch_start + eval_batch_size
                    ]
                    try:
                        batch_results = self._predict_batch_kitti_eval(
                            eval_predictor, batch_paths, imgsz, device
                        )
                    except RuntimeError as exc:
                        if not self._is_cuda_oom(exc) or len(batch_paths) <= 1:
                            raise
                        next_batch_size = max(1, len(batch_paths) // 2)
                        self.logger.warning(
                            "[%s] 评估 batch=%d OOM，释放显存后降为 batch=%d 重试",
                            eval_split,
                            len(batch_paths),
                            next_batch_size,
                        )
                        eval_batch_size = next_batch_size
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    for i_in_batch, (result, img_path) in enumerate(
                        zip(batch_results, batch_paths)
                    ):
                        img_idx = batch_start + i_in_batch
                        img_h, img_w = result.orig_shape
                        img_sizes[img_idx] = (int(img_w), int(img_h))
                        debug_image_ids.add(img_idx)
                        raw_meta = meta_by_stem[img_path.stem]
                        raw = (
                            json.loads(raw_meta.read_text(encoding='utf-8'))
                            if isinstance(raw_meta, Path)
                            else raw_meta
                        )
                        if isinstance(raw, list):
                            entries = raw
                        elif isinstance(raw, dict) and 'objects' in raw:
                            entries = raw['objects']
                            weather = str(raw.get('weather', '')).strip()
                            if weather:
                                weather_by_image_id[img_idx] = weather
                        else:
                            entries = []

                        for entry in entries:
                            cls = int(entry['class_id'])
                            if not (0 <= cls < nc):
                                continue
                            if 'bbox_yolo' in entry:
                                byo = entry['bbox_yolo']
                                cx, cy, bw, bh = byo['cx'], byo['cy'], byo['w'], byo['h']
                                px1 = (cx - bw / 2) * img_w
                                py1 = (cy - bh / 2) * img_h
                                px2 = (cx + bw / 2) * img_w
                                py2 = (cy + bh / 2) * img_h
                            elif 'bbox_xyxy' in entry:
                                px1, py1, px2, py2 = map(float, entry['bbox_xyxy'][:4])
                            else:
                                continue
                            h_px = py2 - py1
                            w_px = (px2 - px1)
                            area_px = w_px * h_px
                            if w_px <= 0 or h_px <= 0:
                                continue

                            ann_id += 1
                            coco_annotations.append(
                                {
                                    "id": ann_id,
                                    "image_id": img_idx,
                                    "category_id": cls + 1,
                                    "bbox": [float(px1), float(py1), float(w_px), float(h_px)],
                                    "area": float(w_px * h_px),
                                    "iscrowd": 0,
                                }
                            )
                            debug_gt_annotations.append(
                                {
                                    "image_id": img_idx,
                                    "category_id": cls + 1,
                                    "bbox": [float(px1), float(py1), float(w_px), float(h_px)],
                                }
                            )

                        if result.boxes is not None:
                            for box, conf, cls_t in zip(
                                result.boxes.xyxy.cpu().numpy(),
                                result.boxes.conf.cpu().numpy(),
                                result.boxes.cls.cpu().numpy().astype(int),
                            ):
                                c = int(cls_t)
                                if 0 <= c < nc:
                                    box_list = box.tolist()
                                    x1, y1, x2, y2 = map(float, box_list)
                                    w_px = x2 - x1
                                    h_px = y2 - y1
                                    if w_px > 0 and h_px > 0:
                                        coco_predictions.append(
                                            {
                                                "image_id": img_idx,
                                                "category_id": c + 1,
                                                "bbox": [x1, y1, w_px, h_px],
                                                "score": float(conf),
                                            }
                                        )
                                        debug_pred_annotations.append(
                                            {
                                                "image_id": img_idx,
                                                "category_id": c + 1,
                                                "bbox": [x1, y1, w_px, h_px],
                                            }
                                        )

                    batch_start += len(batch_paths)

                if os.getenv("CAS_DEBUG_AREA_METRICS", "0") == "1":
                    gt_counts = coco_area_bucket_counts_from_xywh_annotations(debug_gt_annotations)
                    pred_counts = coco_area_bucket_counts_from_xywh_annotations(debug_pred_annotations)
                    self.logger.info(
                        "[DEBUG][YOLO][AREA][%s] images=%d  %s  %s",
                        eval_split,
                        len(debug_image_ids),
                        format_area_bucket_counts("gt", gt_counts),
                        format_area_bucket_counts("pred", pred_counts),
                    )

                self.logger.info(
                    "[%s] COCO 评估输入: pycocotools=%s, GT=%d 条, pred=%d 条",
                    eval_split,
                    PYCOCOTOOLS_AVAILABLE,
                    len(coco_annotations),
                    len(coco_predictions),
                )

                # ── 5. AP：全类 mAP / S/M/L / 每类 AP（一次 COCOeval，全 GT iscrowd=0）
                metrics: Dict[str, Any] = {}

                categories_coco = [
                    {'id': i + 1, 'name': class_names[i]} for i in range(nc)
                ]
                coco_gt = {
                    'images': [
                        {
                            'id': i,
                            'width': img_sizes[i][0],
                            'height': img_sizes[i][1],
                        }
                        for i in range(len(eval_images))
                    ],
                    'categories': categories_coco,
                    'annotations': coco_annotations,
                }

                coco_eval = run_coco_bbox_eval(coco_gt, coco_predictions)
                per_cls_50: List[float] = []
                per_cls_5095: List[float] = []
                if coco_eval is None:
                    if not PYCOCOTOOLS_AVAILABLE:
                        self.logger.warning(
                            f"[{eval_split}] COCOeval 跳过：未安装 pycocotools，"
                            "请执行: pip install pycocotools  （COCO 口径指标已置 0）"
                        )
                    elif not coco_annotations:
                        self.logger.warning(
                            f"[{eval_split}] COCOeval 跳过：未解析到任何 GT（labels_meta 与图像是否匹配、"
                            "bbox_yolo/bbox_xyxy 是否存在），COCO 口径指标置 0"
                        )
                    else:
                        self.logger.warning(
                            f"[{eval_split}] COCOeval 失败（pycocotools 运行异常），"
                            "COCO 口径指标置 0"
                        )
                    metrics['mAP_50'] = 0.0
                    metrics['mAP_5095'] = 0.0
                    metrics['AP_small_50'] = 0.0
                    metrics['AP_medium_50'] = 0.0
                    metrics['AP_large_50'] = 0.0
                    metrics['AP_small_5095'] = 0.0
                    metrics['AP_medium_5095'] = 0.0
                    metrics['AP_large_5095'] = 0.0
                    per_cls_50 = [0.0] * nc
                    per_cls_5095 = [0.0] * nc
                    for i in range(nc):
                        nm = class_names[i]
                        suffix = canonical_category_metric_name(nm)
                        metrics[f'AP50_{suffix}'] = 0.0
                        metrics[f'AP5095_{suffix}'] = 0.0
                else:
                    metrics['mAP_50'] = coco_ap_at_iou50_all(coco_eval)
                    metrics['mAP_5095'] = (
                        max(0.0, float(coco_eval.stats[0]))
                        if len(coco_eval.stats) > 0
                        else 0.0
                    )
                    s50, m50, l50 = coco_area_ap_at_iou50(coco_eval)
                    metrics['AP_small_50'] = s50
                    metrics['AP_medium_50'] = m50
                    metrics['AP_large_50'] = l50
                    if len(coco_eval.stats) >= 6:
                        metrics['AP_small_5095'] = max(0.0, float(coco_eval.stats[3]))
                        metrics['AP_medium_5095'] = max(0.0, float(coco_eval.stats[4]))
                        metrics['AP_large_5095'] = max(0.0, float(coco_eval.stats[5]))
                    else:
                        metrics['AP_small_5095'] = 0.0
                        metrics['AP_medium_5095'] = 0.0
                        metrics['AP_large_5095'] = 0.0

                    per_cat_50, per_cat_5095 = extract_per_category_ap_from_coco_eval(
                        coco_eval, categories_coco
                    )
                    per_cls_50 = [
                        per_cat_50.get(canonical_category_metric_name(class_names[i]), 0.0)
                        for i in range(nc)
                    ]
                    per_cls_5095 = [
                        per_cat_5095.get(canonical_category_metric_name(class_names[i]), 0.0)
                        for i in range(nc)
                    ]
                    for i in range(nc):
                        nm = class_names[i]
                        suffix = canonical_category_metric_name(nm)
                        metrics[f'AP50_{suffix}'] = per_cat_50.get(suffix, 0.0)
                        metrics[f'AP5095_{suffix}'] = per_cat_5095.get(suffix, 0.0)

                weather_log_parts_50 = []
                weather_log_parts_5095 = []
                for weather in weather_names:
                    image_ids = {
                        image_id
                        for image_id, image_weather in weather_by_image_id.items()
                        if image_weather == weather
                    }
                    if not image_ids:
                        continue
                    sub_annotations = [
                        ann for ann in coco_annotations if ann['image_id'] in image_ids
                    ]
                    sub_predictions = [
                        pred for pred in coco_predictions if pred['image_id'] in image_ids
                    ]
                    sub_gt = {
                        'images': [
                            {
                                'id': image_id,
                                'width': img_sizes[image_id][0],
                                'height': img_sizes[image_id][1],
                            }
                            for image_id in sorted(image_ids)
                            if image_id in img_sizes
                        ],
                        'categories': categories_coco,
                        'annotations': sub_annotations,
                    }
                    sub_eval = run_coco_bbox_eval(sub_gt, sub_predictions)
                    if sub_eval is None:
                        ap50 = 0.0
                        ap5095 = 0.0
                    else:
                        ap50 = coco_ap_at_iou50_all(sub_eval)
                        ap5095 = (
                            max(0.0, float(sub_eval.stats[0]))
                            if len(sub_eval.stats) > 0
                            else 0.0
                        )
                    weather_key = _weather_metric_key(weather)
                    metrics[f'weather_{weather_key}_mAP_50'] = ap50
                    metrics[f'weather_{weather_key}_mAP_5095'] = ap5095
                    weather_log_parts_50.append(f'{weather}={ap50:.4f}')
                    weather_log_parts_5095.append(f'{weather}={ap5095:.4f}')

                metrics['eval_split'] = eval_split
                merge_benchmark_dict_into_metrics(metrics, bench_dict)

                self.logger.info(
                    f"📐 [{eval_split}] S/M/L  "
                    f"@0.5: {metrics['AP_small_50']:.4f} / {metrics['AP_medium_50']:.4f} / "
                    f"{metrics['AP_large_50']:.4f}  |  "
                    f"@0.5:0.95: {metrics['AP_small_5095']:.4f} / "
                    f"{metrics['AP_medium_5095']:.4f} / {metrics['AP_large_5095']:.4f}"
                )
                cls_50_str = ' | '.join(
                    f'{class_names[i]}={per_cls_50[i]:.4f}' for i in range(nc)
                )
                cls_5095_str = ' | '.join(
                    f'{class_names[i]}={per_cls_5095[i]:.4f}' for i in range(nc)
                )
                self.logger.info(f"📋 [{eval_split}] Per-class AP@0.5:  {cls_50_str}")
                self.logger.info(f"📋 [{eval_split}] Per-class AP@0.5:0.95:  {cls_5095_str}")
                if weather_log_parts_50:
                    self.logger.info(
                        f"[{eval_split}] Weather AP@0.5:  "
                        + ' | '.join(weather_log_parts_50)
                    )
                    self.logger.info(
                        f"[{eval_split}] Weather AP@0.5:0.95:  "
                        + ' | '.join(weather_log_parts_5095)
                    )
                if (bm_line := format_benchmark_eval_line(metrics)):
                    self.logger.info(bm_line)

                write_eval_csv(
                    summary_csv,
                    model=model_name,
                    dataset=dataset_name,
                    eval_split=eval_split,
                    metrics=metrics,
                    class_names=class_names,
                    weather_buckets=weather_buckets,
                    benchmark={
                        k: v
                        for k, v in (bench_dict or {}).items()
                        if k in BENCHMARK_EVAL_METRIC_KEYS or k in END_TO_END_EVAL_METRIC_KEYS
                    },
                    metadata={
                        'run_id': self.log_dir.name,
                        'framework': getattr(self, 'REPORT_FRAMEWORK', 'yolo'),
                        'experiment': self.experiment_name,
                        'seed': self.config.get('seed', ''),
                    },
                )
                last_metrics = metrics

        self.logger.info(f"✓ 指标已追加: {summary_csv}")
        self.logger.info(f"{'='*80}")

        return last_metrics

    def _run_model_benchmark(self, model):
        """运行 GFLOPs / FPS benchmark 并记录日志（仅一次）。"""
        from yolo_benchmark import benchmark_yolo
        bench_dict = None
        try:
            model_name = self.model_config.get('model_name', f'yolov{self.VERSION}n')
            if model_name.endswith('.pt'):
                model_name = model_name[:-3]
            bench_result = benchmark_yolo(
                model,
                images=self._benchmark_image_dir(str(self.data_config.get("data_yaml", ""))),
                imgsz=self.training_config.get('imgsz', 640),
                device=self.misc_config.get('device', 'cuda'),
                model_name=model_name,
            )
            log_benchmark(self.logger.info, bench_result, header=model_name)
            bench_dict = benchmark_to_dict(bench_result)
        except Exception as exc:
            self.logger.warning(f"Model benchmark 失败（不影响评估结果）: {exc}")
        return bench_dict

    def _tensorrt_weights_path(self) -> Optional[Path]:
        for path in (
            self.log_dir / "weights" / "best.pt",
            self.log_dir / "weights" / "last.pt",
        ):
            if path.is_file():
                return path
        return None

    def _tensorrt_extra_args(self) -> List[str]:
        return []

    def _benchmark_image_dir(self, data_yaml: str) -> Path:
        yaml_path = Path(data_yaml).expanduser().resolve()
        with yaml_path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        root = Path(data.get("path", yaml_path.parent)).expanduser()
        if not root.is_absolute():
            root = yaml_path.parent / root
        images = Path(data["val"]).expanduser()
        if not images.is_absolute():
            images = root / images
        if not images.is_dir():
            raise FileNotFoundError(f"TensorRT benchmark 图像目录不存在: {images}")
        return images.resolve()

    def run_tensorrt_benchmark(self) -> None:
        """Export the completed run and append its TensorRT result to shared CSVs."""
        enabled = os.environ.get(
            "YOLO_TRT_BENCHMARK", os.environ.get("TRT_BENCHMARK", "1")
        )
        if enabled == "0":
            self.logger.info("TensorRT benchmark 已通过环境变量关闭")
            return

        weights = self._tensorrt_weights_path()
        if weights is None:
            raise FileNotFoundError(f"无可用于 TensorRT benchmark 的权重: {self.log_dir}")

        model_name = str(self.model_config.get("model_name", f"yolov{self.VERSION}n"))
        if model_name.endswith((".pt", ".pth")):
            model_name = model_name.rsplit(".", 1)[0]
        data_yaml = str(self.data_config.get("data_yaml", ""))
        data_lower = data_yaml.lower()
        if "dair" in data_lower:
            dataset = "DAIR-V2X"
        elif "uadetrac" in data_lower or "ua-detrac" in data_lower:
            dataset = "UA-DETRAC"
        else:
            dataset = Path(data_yaml).stem or "unknown"

        command = [
            sys.executable,
            str(_yolo_dir / "tools" / "benchmark_trt.py"),
            "--weights", str(weights.resolve()),
            "--output-dir", str(self.log_dir.resolve()),
            "--images", str(self._benchmark_image_dir(data_yaml)),
            "--run-id", self.log_dir.name,
            "--model", model_name,
            "--dataset", dataset,
            "--seed", str(self.training_config.get("seed", 0)),
            "--imgsz", str(self.training_config.get("imgsz", 640)),
            "--builder", os.environ.get("YOLO_TRT_BUILDER", "auto"),
            "--trtexec", os.environ.get("TRTEXEC", "trtexec"),
            "--warmup", os.environ.get("TRT_WARMUP", "100"),
            "--iterations", os.environ.get("TRT_ITERATIONS", "1000"),
            *self._tensorrt_extra_args(),
        ]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("🚀 运行 TensorRT benchmark: %s", model_name)
        subprocess.run(command, cwd=_yolo_dir, check=True)
        self.logger.info("✓ TensorRT benchmark 已写入统一总表")

    def _post_training_processing(self, model=None):
        """训练后处理"""
        self.logger.info("=" * 80)
        self.logger.info("✅ 训练完成！")
        self.logger.info("=" * 80)

        bench_dict = None
        try:
            self._plot_training_curves()
            self._align_file_naming()

            best_model_path = self.log_dir / "best_model.pth"
            if best_model_path.exists():
                self.logger.info(f"✓ 最佳模型: {best_model_path}")

            try:
                bench_dict = self._optional_post_train_benchmark(model)
            except Exception as exc:
                self.logger.exception(
                    "训练后 benchmark/预测器准备失败（不影响已保存权重；eval 阶段会重试加载）: %s",
                    exc,
                )

            run_eval = model is not None or self._can_run_kitti_eval_without_ultralytics_model()
            if not run_eval:
                self.logger.warning(
                    "跳过 KITTI/scale 与 eval_metrics：无 Ultralytics 模型且未在 log_dir 检测到可评估权重。"
                    " log_dir=%s （请确认 best_ckpt.pth / weights/best.pt 是否在此目录下）",
                    self.log_dir.resolve(),
                )
            else:
                try:
                    self._evaluate_kitti_scale_after_training(model, bench_dict=bench_dict)
                except Exception as exc:
                    self.logger.warning(f"KITTI/scale 评估出错（训练结果不受影响）: {exc}")
        finally:
            self.logger.info(f"✓ 所有输出已保存到: {self.log_dir.resolve()}")
            self.logger.info("=" * 80)
    
    def _parse_and_print_training_results(self):
        """解析results.csv并输出"""
        try:
            results_csv = result_csv("results")
            if not results_csv.exists():
                self.logger.warning(f"未找到results.csv文件: {results_csv}")
                return
            
            df = pd.read_csv(results_csv)
            if 'run_id' in df.columns:
                df = df[df['run_id'].astype(str) == self.log_dir.name]
            # Shared results.csv can contain mixed producers; keep plotting
            # numeric-only rows so a string run_id never reaches matplotlib.
            if 'epoch' in df.columns:
                df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
                df = df.dropna(subset=['epoch'])
            self.logger.info("=" * 80)
            self.logger.info("训练过程摘要:")
            self.logger.info("=" * 80)
            
            # 提取关键指标列
            train_loss_cols = [c for c in df.columns if 'train/box_loss' in c.lower() or 
                             'train/cls_loss' in c.lower() or 'train/dfl_loss' in c.lower()]
            val_loss_cols = [c for c in df.columns if 'val/box_loss' in c.lower() or 
                            'val/cls_loss' in c.lower() or 'val/dfl_loss' in c.lower()]
            
            train_loss = df[train_loss_cols].sum(axis=1) if train_loss_cols else pd.Series(0, index=df.index)
            val_loss = df[val_loss_cols].sum(axis=1) if val_loss_cols else pd.Series(0, index=df.index)
            
            # 查找mAP列
            map50_col = next((c for c in df.columns if 'map50(b)' in c.lower()), None)
            map50_95_col = next((c for c in df.columns if 'map50-95(b)' in c.lower()), None)
            
            # 打印最后5个epoch的结果
            display_rows = min(5, len(df))
            for idx in range(len(df) - display_rows, len(df)):
                row = df.iloc[idx]
                epoch = int(row.get('epoch', idx + 1))
                line = f"Epoch {epoch}: Loss={train_loss.iloc[idx]:.2f}|{val_loss.iloc[idx]:.2f}"
                if map50_col and not pd.isna(row.get(map50_col)):
                    line += f" | mAP@0.5={row[map50_col]:.4f}"
                if map50_95_col and not pd.isna(row.get(map50_95_col)):
                    line += f" | mAP@0.5:0.95={row[map50_95_col]:.4f}"
                self.logger.info(line)
            
            self.logger.info("=" * 80)
        
        except Exception as e:
            self.logger.warning(f"解析训练结果失败: {e}")
    
    def _plot_training_curves(self):
        """生成训练曲线"""
        try:
            results_csv = result_csv("results")
            if not results_csv.exists():
                return
            
            df = pd.read_csv(results_csv)
            if 'run_id' in df.columns:
                df = df[df['run_id'].astype(str) == self.log_dir.name]
            epochs = df.get('epoch', pd.Series(range(1, len(df) + 1), index=df.index))
            epochs = pd.to_numeric(epochs, errors='coerce')
            valid = epochs.notna()
            df = df.loc[valid]
            epochs = epochs.loc[valid].to_numpy()
            
            # 提取损失
            train_loss_cols = [c for c in df.columns if 'train/box_loss' in c.lower() or 
                             'train/cls_loss' in c.lower() or 'train/dfl_loss' in c.lower()]
            val_loss_cols = [c for c in df.columns if 'val/box_loss' in c.lower() or 
                            'val/cls_loss' in c.lower() or 'val/dfl_loss' in c.lower()]
            
            train_loss = (
                df[train_loss_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1).to_numpy()
                if train_loss_cols else None
            )
            val_loss = (
                df[val_loss_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1).to_numpy()
                if val_loss_cols else None
            )
            
            # 提取mAP
            map50_col = next((c for c in df.columns if 'map50(b)' in c.lower()), None)
            map50_95_col = next((c for c in df.columns if 'map50-95(b)' in c.lower()), None)
            
            map50 = pd.to_numeric(df[map50_col], errors='coerce').to_numpy() if map50_col else None
            map50_95 = pd.to_numeric(df[map50_95_col], errors='coerce').to_numpy() if map50_95_col else None
            
            # 提取学习率
            lr_col = next((c for c in df.columns if 'lr' in c.lower()), None)
            lr = pd.to_numeric(df[lr_col], errors='coerce').to_numpy() if lr_col else None
            
            # 绘制
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f'YOLO{self.VERSION} Training Curves', fontsize=16, fontweight='bold')
            
            # 损失曲线
            if train_loss is not None:
                axes[0].plot(epochs, train_loss, 'b-o', label='Train Loss', linewidth=2, markersize=4)
            if val_loss is not None:
                axes[0].plot(epochs, val_loss, 'r-s', label='Val Loss', linewidth=2, markersize=4)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Loss Curves')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # mAP曲线
            if map50 is not None:
                axes[1].plot(epochs, map50, 'g-^', label='mAP@0.5', linewidth=2, markersize=4)
            if map50_95 is not None:
                axes[1].plot(epochs, map50_95, 'm-d', label='mAP@[0.5:0.95]', linewidth=2, markersize=4)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('mAP')
            axes[1].set_title('mAP Metrics')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # 学习率曲线
            if lr is not None:
                axes[2].plot(epochs, lr, 'orange', linewidth=2)
                axes[2].set_yscale('log')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Learning Rate')
            axes[2].set_title('Learning Rate Schedule')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_path = self.log_dir / 'training_curves.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✓ 训练曲线已保存到: {save_path}")
        
        except Exception as e:
            self.logger.warning(f"绘制训练曲线失败: {e}")
    
    def _align_file_naming(self):
        """统一文件命名"""
        try:
            weights_dir = self.log_dir / "weights"
            if not weights_dir.exists():
                return
            
            copies = [
                (weights_dir / "best.pt", self.log_dir / "best_model.pth"),
                (weights_dir / "last.pt", self.log_dir / "latest_checkpoint.pth"),
            ]
            
            for src, dst in copies:
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
                    self.logger.info(f"✓ 已创建: {dst.name}")
        
        except Exception as e:
            self.logger.warning(f"统一文件命名失败: {e}")
