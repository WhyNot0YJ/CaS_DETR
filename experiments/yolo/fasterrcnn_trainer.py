#!/usr/bin/env python3
"""
Faster R-CNN (torchvision) 训练器 — 继承 BaseYOLOTrainer，
复用 COCO / multi-scale 评估管线与 eval_metrics.csv 输出。
"""

import sys
import time
import math
import json
import logging
import gc
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm

_experiments_root = Path(__file__).resolve().parent.parent
if str(_experiments_root) not in sys.path:
    sys.path.insert(0, str(_experiments_root))

_yolo_dir = Path(__file__).resolve().parent
if str(_yolo_dir) not in sys.path:
    sys.path.insert(0, str(_yolo_dir))

from base_yolo_trainer import DEFAULT_TRAIN_BATCH, BaseYOLOTrainer
from fasterrcnn_dataset import (
    DetectionCompose,
    LetterboxDetection,
    YOLOFormatDetectionDataset,
    RandomHorizontalFlipDetection,
    detection_collate_fn,
    letterbox_image,
    resolve_split_dirs,
    unletterbox_boxes,
)
from common.model_benchmark import benchmark_model, benchmark_to_dict, log_benchmark

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result wrapper — mimic Ultralytics result for base-class eval
# ---------------------------------------------------------------------------

class _FasterRCNNResult:
    """Thin wrapper that presents torchvision output in Ultralytics-like API."""

    def __init__(self, orig_shape: Tuple[int, int], output: Dict[str, torch.Tensor]):
        self.orig_shape = orig_shape  # (H, W)
        boxes = output["boxes"]
        scores = output["scores"]
        labels = output["labels"] - 1  # torchvision 1-indexed → YOLO 0-indexed
        self.boxes = SimpleNamespace(
            xyxy=boxes.cpu(),
            conf=scores.cpu(),
            cls=labels.float().cpu(),
        )


# ---------------------------------------------------------------------------
# Benchmark wrapper — convert (1,3,H,W) batch tensor to list for torchvision
# ---------------------------------------------------------------------------

class _BenchmarkInputAdapter(nn.Module):
    """Wraps a torchvision detection model so ``benchmark_model`` can feed
    it a standard ``(1, 3, H, W)`` tensor (converted to ``[tensor(3,H,W)]``)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        return self.model([x[0]])


# ---------------------------------------------------------------------------
# FasterRCNNTrainer
# ---------------------------------------------------------------------------

class FasterRCNNTrainer(BaseYOLOTrainer):
    VERSION = "fasterrcnn"
    REPORT_FRAMEWORK = "fasterrcnn"

    def __init__(
        self,
        config: Dict,
        config_path: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        resume_checkpoint: Optional[str] = None,
    ):
        super().__init__(config, config_path, class_names, resume_checkpoint=resume_checkpoint)

    def _input_size(self) -> int:
        imgsz = int(self.training_config.get("imgsz", 640))
        if imgsz <= 0:
            raise ValueError(f"training.imgsz must be positive, got {imgsz}")
        return imgsz

    # ── model creation ────────────────────────────────────────────────

    def create_model(self) -> nn.Module:
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        self.logger.info("✓ 创建 Faster R-CNN (ResNet-50 FPN, COCO V1 预训练)")
        imgsz = self._input_size()
        model = fasterrcnn_resnet50_fpn(
            weights="DEFAULT", min_size=imgsz, max_size=imgsz,
        )

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        nc_with_bg = self.num_classes + 1
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, nc_with_bg)
        self.logger.info(
            f"  替换分类头: in_features={in_features}, "
            f"num_classes={nc_with_bg} (含背景)"
        )
        return model

    def _create_fresh_model(self) -> nn.Module:
        """Build an un-trained model skeleton for weight loading."""
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        imgsz = self._input_size()
        model = fasterrcnn_resnet50_fpn(
            weights=None, weights_backbone=None, min_size=imgsz, max_size=imgsz,
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, self.num_classes + 1
        )
        return model

    # ── training-time mAP evaluation ─────────────────────────────────

    def _resolve_val_map_paths(self, data_yaml: str, val_img_dir: Path):
        """Resolve the validation GT sources and collect validation image paths.

        Returns:
            (labels_meta_dir, coco_ann_file, val_img_paths) — either GT source may
            be missing (mAP eval gracefully skips only when **both** are missing).
        """
        data_yaml_path = Path(data_yaml)
        with data_yaml_path.open(encoding='utf-8') as fh:
            data_cfg = yaml.safe_load(fh) or {}

        root = data_yaml_path.parent.resolve()
        path_field = str(data_cfg.get('path', '')).strip()
        if path_field:
            pc = Path(path_field)
            if pc.is_absolute() and pc.is_dir():
                root = pc

        labels_meta_dir = self._resolve_labels_meta_dir(data_cfg, root, 'val')
        coco_ann_file = self._resolve_coco_ann_file(data_cfg, root, 'val')

        val_img_paths = sorted(
            p
            for ext in ('.jpg', '.jpeg', '.png')
            for p in val_img_dir.rglob(f'*{ext}')
        )
        return labels_meta_dir, coco_ann_file, val_img_paths

    def _load_val_map_gt(
        self,
        labels_meta_dir: Optional[Path],
        coco_ann_file: Optional[Path],
    ) -> Dict[str, Any]:
        """Load validation GT keyed by image stem for training-time mAP.

        Priority: ``labels_meta/val/*.json`` → ``val_coco_ann`` COCO JSON
        (same two sources the post-training COCO evaluator accepts).
        Parsed once and cached — the COCO JSON can be tens of MB.
        """
        cached = getattr(self, '_val_map_gt_cache', None)
        if cached is not None:
            return cached

        meta_by_stem: Dict[str, Any] = {}
        source = ''
        if labels_meta_dir is not None and labels_meta_dir.is_dir():
            for meta_path in sorted(labels_meta_dir.glob('*.json')):
                try:
                    meta_by_stem[meta_path.stem] = json.loads(
                        meta_path.read_text(encoding='utf-8')
                    )
                except Exception:
                    continue
            if meta_by_stem:
                source = f'labels_meta={labels_meta_dir}'
        if not meta_by_stem and coco_ann_file is not None and coco_ann_file.is_file():
            try:
                meta_by_stem = self._load_coco_meta_by_stem(coco_ann_file)
                source = f'coco_ann={coco_ann_file}'
            except Exception as exc:
                self.logger.warning('解析 val COCO 标注失败（训练期 mAP 不可用）: %s', exc)
                meta_by_stem = {}
        if meta_by_stem:
            self.logger.info(
                '✓ 训练期 mAP GT 源: %s (%d 张标注)', source, len(meta_by_stem),
            )
        self._val_map_gt_cache = meta_by_stem
        return meta_by_stem

    @torch.no_grad()
    def _compute_val_map(
        self,
        model: nn.Module,
        device: torch.device,
        val_img_paths: List[Path],
        labels_meta_dir: Optional[Path] = None,
        coco_ann_file: Optional[Path] = None,
        *,
        max_eval_images: int = 0,
        batch_size: int = 8,
    ) -> Optional[Tuple[float, float]]:
        """Compute COCO mAP on the validation set during training.

        Returns ``(mAP@0.5, mAP@0.5:0.95)``, or ``None`` when no GT source
        is usable, no images match, or pycocotools is unavailable.

        Parameters:
            max_eval_images: If > 0, evaluate only the first N images for speed.
        """
        from common.det_eval_metrics import run_coco_bbox_eval, coco_ap_at_iou50_all, coco_ap_at_iou50_95_all

        meta_by_stem = self._load_val_map_gt(labels_meta_dir, coco_ann_file)
        if not meta_by_stem:
            return None

        eval_images = [p for p in val_img_paths if p.stem in meta_by_stem]
        if not eval_images:
            return None

        # ── optional subset ──
        if max_eval_images > 0 and len(eval_images) > max_eval_images:
            eval_images = eval_images[:max_eval_images]

        model.eval()
        imgsz = self._input_size()

        coco_annotations: List[Dict[str, Any]] = []
        coco_predictions: List[Dict[str, Any]] = []
        img_sizes: Dict[int, Tuple[int, int]] = {}
        ann_id = 0

        for batch_start in range(0, len(eval_images), batch_size):
            batch_paths = eval_images[batch_start: batch_start + batch_size]

            # ── preprocess ──
            images: List[torch.Tensor] = []
            letterbox_params: List[Tuple[float, int, int]] = []
            orig_sizes: List[Tuple[int, int]] = []
            for p in batch_paths:
                img = Image.open(p).convert('RGB')
                w, h = img.size
                orig_sizes.append((h, w))
                image, scale, pad_left, pad_top = letterbox_image(
                    TF.to_tensor(img), imgsz,
                )
                images.append(image.to(device))
                letterbox_params.append((scale, pad_left, pad_top))

            outputs = model(images)

            for i_in_batch, (p, (scale, pad_left, pad_top), out) in enumerate(
                zip(batch_paths, letterbox_params, outputs)
            ):
                img_idx = batch_start + i_in_batch
                img_h, img_w = orig_sizes[i_in_batch]  # (H, W)
                img_id = img_idx
                img_sizes[img_id] = (int(img_w), int(img_h))

                # ── GT from labels_meta / COCO annotations ──
                raw = meta_by_stem[p.stem]
                if isinstance(raw, dict) and 'objects' in raw:
                    entries = raw['objects']
                elif isinstance(raw, list):
                    entries = raw
                else:
                    entries = []

                for entry in entries:
                    cls_id = int(entry['class_id'])
                    if not (0 <= cls_id < self.num_classes):
                        continue
                    if 'bbox_xyxy' in entry:
                        x1, y1, x2, y2 = map(float, entry['bbox_xyxy'][:4])
                    elif 'bbox_yolo' in entry:
                        byo = entry['bbox_yolo']
                        cx, cy, bw, bh = byo['cx'], byo['cy'], byo['w'], byo['h']
                        x1 = (cx - bw / 2.0) * img_w
                        y1 = (cy - bh / 2.0) * img_h
                        x2 = (cx + bw / 2.0) * img_w
                        y2 = (cy + bh / 2.0) * img_h
                    else:
                        continue
                    w_px, h_px = x2 - x1, y2 - y1
                    if w_px <= 0 or h_px <= 0:
                        continue
                    ann_id += 1
                    coco_annotations.append({
                        'id': ann_id,
                        'image_id': img_id,
                        'category_id': cls_id + 1,
                        'bbox': [x1, y1, w_px, h_px],
                        'area': w_px * h_px,
                        'iscrowd': 0,
                    })

                # ── predictions (unletterbox) ──
                out = dict(out)
                out['boxes'] = unletterbox_boxes(
                    out['boxes'], (img_h, img_w), scale, pad_left, pad_top,
                )
                boxes = out['boxes']
                scores = out['scores']
                labels = out['labels'] - 1  # 1-indexed → 0-indexed

                for box, score, label in zip(
                    boxes.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy()
                ):
                    cls = int(label)
                    if not (0 <= cls < self.num_classes) or float(score) < 0.005:
                        continue
                    x1, y1, x2, y2 = map(float, box)
                    w_px, h_px = x2 - x1, y2 - y1
                    if w_px <= 0 or h_px <= 0:
                        continue
                    coco_predictions.append({
                        'image_id': img_id,
                        'category_id': cls + 1,
                        'bbox': [x1, y1, w_px, h_px],
                        'score': float(score),
                    })

        if not coco_annotations or not coco_predictions:
            return None

        # ── build COCO GT & run eval ──
        categories = [
            {'id': i + 1, 'name': self.class_names[i]}
            for i in range(self.num_classes)
        ]
        coco_gt = {
            'images': [
                {'id': i, 'width': img_sizes[i][0], 'height': img_sizes[i][1]}
                for i in sorted(img_sizes)
            ],
            'categories': categories,
            'annotations': coco_annotations,
        }

        coco_eval = run_coco_bbox_eval(coco_gt, coco_predictions)
        if coco_eval is None:
            return None
        return (coco_ap_at_iou50_all(coco_eval), coco_ap_at_iou50_95_all(coco_eval))

    # ── training loop ─────────────────────────────────────────────────

    def start_training(
        self,
        resume_checkpoint: Optional[str] = None,
        epochs_override: Optional[int] = None,
    ):
        self._resume_checkpoint_path = resume_checkpoint
        self.setup_logging()

        device = torch.device(self.misc_config.get("device", "cuda"))
        model = self.create_model()
        model.to(device)

        epochs = epochs_override or self.training_config.get("epochs", 100)
        batch_size = int(
            self.training_config["batch_size"]
            if "batch_size" in self.training_config
            else DEFAULT_TRAIN_BATCH
        )
        num_workers = self.misc_config.get("num_workers", 2)
        data_yaml = self._resolve_data_yaml()
        imgsz = self._input_size()

        # datasets
        train_img_dir, train_lbl_dir = resolve_split_dirs(data_yaml, "train")
        val_img_dir, val_lbl_dir = resolve_split_dirs(data_yaml, "val")

        train_ds = YOLOFormatDetectionDataset(
            str(train_img_dir), str(train_lbl_dir),
            transform=DetectionCompose([
                LetterboxDetection(imgsz),
                RandomHorizontalFlipDetection(0.5),
            ]),
        )
        val_ds = YOLOFormatDetectionDataset(
            str(val_img_dir), str(val_lbl_dir),
            transform=LetterboxDetection(imgsz),
        )

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=detection_collate_fn,
            pin_memory=True, drop_last=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=detection_collate_fn,
            pin_memory=True,
        )

        # ── resolve validation GT for training-time mAP evaluation ──
        _val_labels_meta_dir, _val_coco_ann, _val_img_paths = self._resolve_val_map_paths(
            data_yaml, val_img_dir,
        )
        _val_map_available = bool(
            self._load_val_map_gt(_val_labels_meta_dir, _val_coco_ann)
        ) and bool(_val_img_paths)
        if not _val_map_available:
            self.logger.warning(
                "⚠️  未找到 val GT（labels_meta/val 或 val_coco_ann），"
                "训练时将退化为按 val_loss 选 best 模型。"
                " 检查: labels_meta=%s, coco_ann=%s",
                _val_labels_meta_dir, _val_coco_ann,
            )
        else:
            self.logger.info(
                "✓ 训练期 mAP 评估可用, %d 张 val 图像 → best.pt 按 mAP@0.5:0.95 选择",
                len(_val_img_paths),
            )
        val_map_freq = int(self.training_config.get('val_map_freq', 1))
        val_map_max_images = int(self.training_config.get('val_map_max_images', 0))
        self._val_labels_meta_dir = _val_labels_meta_dir
        self._val_coco_ann_file = _val_coco_ann
        self._val_img_paths = _val_img_paths
        self._val_map_available = _val_map_available

        # resume（checkpoint 内 epoch 为「已成功结束的上一个 epoch 的 1-based 编号」，
        # 与保存时 epoch_loop+1 一致；下次训练从该值作为 0-based 下标开始）
        start_epoch = 0
        resume_ckpt: Optional[Dict[str, Any]] = None
        if resume_checkpoint and Path(resume_checkpoint).exists():
            resume_ckpt = torch.load(resume_checkpoint, map_location=device)
            if isinstance(resume_ckpt, dict) and "model_state_dict" in resume_ckpt:
                model.load_state_dict(resume_ckpt["model_state_dict"])
                start_epoch = int(resume_ckpt.get("epoch", 0))
                self.logger.info(
                    f"📦 从检查点恢复: 下一轮将从 Epoch {start_epoch + 1}/{epochs} 继续 "
                    f"(checkpoint['epoch']={start_epoch})"
                )
            else:
                model.load_state_dict(resume_ckpt)
                self.logger.info(f"📦 从检查点恢复权重: {resume_checkpoint}")

        # optimizer & scheduler
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params, lr=0.005, momentum=0.9, weight_decay=0.0005,
        )
        warmup_epochs = min(3, epochs)
        main_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(epochs * 0.6), int(epochs * 0.8)],
            gamma=0.1,
        )

        best_val_loss = float("inf")
        best_map_50: float = -1.0    # mAP@0.5
        best_map_50_95: float = -1.0  # mAP@0.5:0.95 (primary criterion for best.pt)
        best_map_epoch: int = 0
        results_rows: List[Dict[str, Any]] = []
        csv_path = self.log_dir / "results.csv"
        if resume_ckpt is not None and csv_path.exists():
            try:
                import pandas as pd
                prev = pd.read_csv(csv_path)
                if "run_id" in prev.columns:
                    prev = prev[prev["run_id"].astype(str) == self.log_dir.name]
                results_rows = prev.to_dict("records")
                if results_rows:
                    best_val_loss = float(
                        min(float(r["val/total_loss"]) for r in results_rows)
                    )
                if "val/mAP_50" in prev.columns:
                    map_vals = pd.to_numeric(
                        prev["val/mAP_50"], errors="coerce"
                    ).dropna()
                    if not map_vals.empty:
                        best_map_50 = float(map_vals.max())
                if "val/mAP_50_95" in prev.columns:
                    map95_vals = pd.to_numeric(
                        prev["val/mAP_50_95"], errors="coerce"
                    ).dropna()
                    if not map95_vals.empty:
                        best_map_50_95 = float(map95_vals.max())
                        best_map_epoch = int(prev.loc[map95_vals.idxmax(), "epoch"])
                elif best_map_50 >= 0:
                    best_map_50_95 = best_map_50  # legacy CSV fallback
                self.logger.info(
                    f"📈 已载入本地 results.csv 共 {len(results_rows)} 行, "
                    f"历史 best val_loss≈{best_val_loss:.4f}"
                    f"{', best mAP@0.5:0.95≈%.4f' % best_map_50_95 if best_map_50_95 >= 0 else ''}"
                )
            except Exception as exc:
                self.logger.warning(f"读取已有 results.csv 失败（将从头记曲线）: {exc}")

        if resume_ckpt is not None and isinstance(resume_ckpt, dict):
            if "optimizer_state_dict" in resume_ckpt:
                try:
                    optimizer.load_state_dict(
                        resume_ckpt["optimizer_state_dict"]
                    )
                except Exception as exc:
                    self.logger.warning(
                        "优化器状态加载失败（将用全新优化器）: %s", exc
                    )
            if "scheduler_state_dict" in resume_ckpt:
                try:
                    main_scheduler.load_state_dict(
                        resume_ckpt["scheduler_state_dict"]
                    )
                except Exception as exc:
                    self.logger.warning(
                        "调度器状态加载失败，尝试按 epoch 快进: %s", exc
                    )
                    n_ff = max(0, start_epoch - warmup_epochs)
                    for _ in range(n_ff):
                        main_scheduler.step()
            elif start_epoch > 0:
                self.logger.warning(
                    "检查点无 scheduler 状态：仅按 epoch 快进 MultiStepLR（旧版 .pt）"
                )
                n_ff = max(0, start_epoch - warmup_epochs)
                for _ in range(n_ff):
                    main_scheduler.step()
            if "best_val_loss" in resume_ckpt:
                try:
                    best_val_loss = float(resume_ckpt["best_val_loss"])
                except (TypeError, ValueError):
                    pass
            if "best_map" in resume_ckpt:
                try:
                    ck_map = float(resume_ckpt["best_map"])
                    if ck_map > best_map_50:
                        best_map_50 = ck_map
                except (TypeError, ValueError):
                    pass
            if "best_map_50_95" in resume_ckpt:
                try:
                    ck_map95 = float(resume_ckpt["best_map_50_95"])
                    if ck_map95 > best_map_50_95:
                        best_map_50_95 = ck_map95
                        best_map_epoch = int(resume_ckpt.get("best_map_epoch", 0))
                except (TypeError, ValueError):
                    pass
            elif best_map_50 >= 0 and best_map_50_95 < 0:
                best_map_50_95 = best_map_50  # legacy checkpoint fallback

        weights_dir = self.log_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)

        if start_epoch > 0:
            self._log_fasterrcnn_resume(epochs, batch_size, data_yaml, device, train_ds, val_ds, start_epoch)
        else:
            self._log_fasterrcnn_config(
                epochs, batch_size, data_yaml, device, train_ds, val_ds,
            )

        total_batches = math.ceil(len(train_ds) / batch_size)

        for epoch in range(start_epoch, epochs):
            # ── warmup LR (linear) ──
            if epoch < warmup_epochs:
                warmup_factor = min(1.0, (epoch + 1) / warmup_epochs)
                for pg in optimizer.param_groups:
                    pg["lr"] = 0.005 * warmup_factor

            # ── train ──
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            t0 = time.time()

            pbar = tqdm(
                train_loader, total=total_batches,
                desc=f"Epoch {epoch + 1}/{epochs}",
                ncols=120, leave=True,
            )
            for images, targets in pbar:
                images = [img.to(device) for img in images]
                targets = [
                    {k: v.to(device) for k, v in t.items()} for t in targets
                ]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
                optimizer.step()

                batch_loss = losses.item()
                epoch_loss += batch_loss
                n_batches += 1
                pbar.set_postfix(loss=f"{batch_loss:.3f}", lr=f"{optimizer.param_groups[0]['lr']:.5f}")

            if epoch >= warmup_epochs:
                main_scheduler.step()

            train_loss = epoch_loss / max(n_batches, 1)
            epoch_time = time.time() - t0

            # ── validation loss ──
            self.logger.info(f"  计算验证集 loss ...")
            val_loss = self._validate_loss(model, val_loader, device)

            # ── training-time mAP evaluation (every val_map_freq epochs) ──
            val_map: Optional[Tuple[float, float]] = None
            map_50 = float('nan')
            map_50_95 = float('nan')
            if (
                _val_map_available
                and val_map_freq > 0
                and (epoch + 1) % val_map_freq == 0
            ):
                self.logger.info(f"  计算验证集 mAP@0.5 / mAP@0.5:0.95 ...")
                val_map = self._compute_val_map(
                    model, device, _val_img_paths,
                    _val_labels_meta_dir, _val_coco_ann,
                    max_eval_images=val_map_max_images,
                )
                if val_map is not None:
                    map_50, map_50_95 = val_map
            else:
                val_map = None

            lr_now = optimizer.param_groups[0]["lr"]
            log_parts = [
                f"Epoch {epoch + 1}/{epochs}",
                f"train_loss={train_loss:.4f}",
                f"val_loss={val_loss:.4f}",
            ]
            if val_map is not None:
                log_parts.append(f"mAP@0.5={map_50:.4f}")
                log_parts.append(f"mAP@0.5:0.95={map_50_95:.4f}")
            log_parts.append(f"lr={lr_now:.6f}")
            log_parts.append(f"time={epoch_time:.1f}s")
            self.logger.info("  ".join(log_parts))

            results_rows.append({
                "epoch": epoch + 1,
                "train/total_loss": train_loss,
                "val/total_loss": val_loss,
                "val/mAP_50": map_50,
                "val/mAP_50_95": map_50_95,
                "lr/pg0": lr_now,
            })

            # ── checkpointing ──
            map50_95_improved = val_map is not None and map_50_95 > best_map_50_95
            if map50_95_improved:
                best_map_50_95 = map_50_95
                best_map_50 = map_50
                best_map_epoch = epoch + 1
            if val_map is not None and map_50 > best_map_50:
                best_map_50 = map_50  # track mAP@0.5 independently

            loss_improved = val_loss < best_val_loss
            if loss_improved:
                best_val_loss = val_loss

            # best.pt 判据：有 mAP 时**只**看 mAP@0.5:0.95；mAP 不可用时退回 val_loss
            if val_map is not None:
                best_is_new = map50_95_improved
                best_reason = f"mAP@0.5:0.95={map_50_95:.4f}"
            else:
                best_is_new = loss_improved and not self._val_map_available
                best_reason = f"val_loss={val_loss:.4f}, mAP 不可用"

            ckpt_payload = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": main_scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "best_map": best_map_50,
                "best_map_50_95": best_map_50_95,
                "best_map_epoch": best_map_epoch,
            }
            torch.save(ckpt_payload, weights_dir / "last.pt")

            if best_is_new:
                torch.save(ckpt_payload, weights_dir / "best.pt")
                self.logger.info(f"  ✓ Best model saved ({best_reason})")

            # periodic checkpoint removed — only last.pt + best.pt are kept

        # ── write results.csv ──
        self._write_results_csv(results_rows)

        # ── best.pt selection summary (verifiable audit trail) ──
        best_pt = weights_dir / "best.pt"
        if best_pt.exists():
            ck = torch.load(best_pt, map_location="cpu")
            sel_epoch = int(ck.get("epoch", 0)) if isinstance(ck, dict) else 0
            del ck
            loss_epochs = [
                (float(r["val/total_loss"]), int(r["epoch"]))
                for r in results_rows
                if r.get("val/total_loss") is not None
            ]
            loss_best_epoch = min(loss_epochs)[1] if loss_epochs else 0
            criterion = "mAP@0.5:0.95" if best_map_50_95 >= 0 else "val_loss"
            self.logger.info(
                "🏁 best.pt 选择依据=%s → epoch %d (best mAP@0.5:0.95=%.4f, "
                "mAP@0.5=%.4f @epoch %s; "
                "val_loss 最优 epoch=%d，仅供参考)",
                criterion, sel_epoch, best_map_50_95, best_map_50,
                best_map_epoch or "-", loss_best_epoch,
            )
            (self.log_dir / "best_selection.json").write_text(
                json.dumps({
                    "criterion": criterion,
                    "best_pt_epoch": sel_epoch,
                    "best_map_50": best_map_50 if best_map_50 >= 0 else None,
                    "best_map_50_95": best_map_50_95 if best_map_50_95 >= 0 else None,
                    "best_map_epoch": best_map_epoch,
                    "best_val_loss": best_val_loss,
                    "val_loss_best_epoch": loss_best_epoch,
                }, indent=2),
                encoding="utf-8",
            )
            if best_map_50_95 >= 0 and best_map_epoch and sel_epoch != best_map_epoch:
                self.logger.warning(
                    "⚠️best.pt epoch(%d) 与 best mAP epoch(%d) 不一致，请检查",
                    sel_epoch, best_map_epoch,
                )

        # ── post-training (benchmark + COCO/scale eval) ──
        if best_pt.exists():
            ckpt = torch.load(best_pt, map_location=device)
            state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state)
        model.eval()
        self._post_training_processing(model)

    # ── helpers for training loop ─────────────────────────────────────

    @torch.no_grad()
    def _validate_loss(self, model, val_loader, device) -> float:
        model.train()  # torchvision returns losses only in train mode
        total_loss = 0.0
        n = 0
        for images, targets in tqdm(val_loader, desc="  Val", ncols=100, leave=False):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            total_loss += sum(l.item() for l in loss_dict.values())
            n += 1
        return total_loss / max(n, 1)

    def _write_results_csv(self, rows: List[Dict[str, Any]]):
        import pandas as pd
        if not rows:
            return
        df = pd.DataFrame(rows)
        df.to_csv(self.log_dir / "results.csv", index=False)
        self.logger.info(f"✓ 本地训练曲线数据: {self.log_dir / 'results.csv'}")

    def _log_fasterrcnn_resume(
        self,
        epochs,
        batch_size,
        data_yaml,
        device,
        train_ds,
        val_ds,
        start_epoch: int,
    ):
        self.logger.info("=" * 80)
        self.logger.info("▶ 恢复 Faster R-CNN (ResNet-50 FPN) 训练")
        self.logger.info("=" * 80)
        self.logger.info(f"  数据集路径: {data_yaml}")
        self.logger.info(f"  训练集: {len(train_ds)} 张")
        self.logger.info(f"  验证集: {len(val_ds)} 张")
        self.logger.info(
            f"  进度: 从 Epoch {start_epoch + 1}/{epochs} 继续 → 共 {epochs} epoch 配置"
        )
        self.logger.info(f"  批次大小: {batch_size}")
        self.logger.info(f"  设备: {device}")
        self.logger.info(f"  输出目录: {self.log_dir}")
        self.logger.info("=" * 80)

    def _log_fasterrcnn_config(self, epochs, batch_size, data_yaml, device, train_ds, val_ds):
        self.logger.info("=" * 80)
        self.logger.info("🚀 开始 Faster R-CNN (ResNet-50 FPN) 训练")
        self.logger.info("=" * 80)
        self.logger.info(f"  数据集路径: {data_yaml}")
        self.logger.info(f"  训练集: {len(train_ds)} 张")
        self.logger.info(f"  验证集: {len(val_ds)} 张")
        self.logger.info(f"  训练轮数: {epochs}")
        self.logger.info(f"  批次大小: {batch_size}")
        self.logger.info(f"  优化器: SGD (lr=0.005, momentum=0.9, wd=0.0005)")
        self.logger.info(f"  设备: {device}")
        self.logger.info(f"  类别数: {self.num_classes}  类别: {self.class_names}")
        self.logger.info(f"  输出目录: {self.log_dir}")
        self.logger.info("=" * 80)

    # ── plot override (reads our results.csv) ─────────────────────────

    def _plot_training_curves(self):
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        csv_path = self.log_dir / "results.csv"
        if not csv_path.exists():
            return
        try:
            df = pd.read_csv(csv_path)
            epochs = df["epoch"].values

            has_map = (
                "val/mAP_50" in df.columns and df["val/mAP_50_95"].notna().any()
                if "val/mAP_50_95" in df.columns
                else "val/mAP_50" in df.columns and df["val/mAP_50"].notna().any()
            )
            has_map95 = "val/mAP_50_95" in df.columns and df["val/mAP_50_95"].notna().any()
            if has_map95:
                fig, axes = plt.subplots(1, 4, figsize=(27, 5))
            elif has_map:
                fig, axes = plt.subplots(1, 3, figsize=(21, 5))
            else:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                axes = [axes[0], axes[1], None, None]
            if not has_map95 and has_map:
                axes = [axes[0], axes[1], None, axes[2]]
            if not has_map and not has_map95:
                axes = [axes[0], None, None, axes[1]]
            fig.suptitle("Faster R-CNN Training Curves", fontsize=16, fontweight="bold")

            if "train/total_loss" in df.columns:
                axes[0].plot(epochs, df["train/total_loss"], "b-o", label="Train Loss", linewidth=2, markersize=3)
            if "val/total_loss" in df.columns:
                axes[0].plot(epochs, df["val/total_loss"], "r-s", label="Val Loss", linewidth=2, markersize=3)
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].set_title("Loss Curves")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # mAP@0.5 curve (axes[1])
            map50_ax = axes[1] if has_map else None
            if map50_ax is not None:
                map_vals = pd.to_numeric(df["val/mAP_50"], errors='coerce')
                valid = map_vals.notna()
                if valid.any():
                    map50_ax.plot(
                        epochs[valid], map_vals[valid].values,
                        "g-^", label="mAP@0.5", linewidth=2, markersize=4,
                    )
                    best_idx = map_vals[valid].idxmax()
                    best_ep = int(df.loc[best_idx, "epoch"])
                    best_val = map_vals.loc[best_idx]
                    map50_ax.annotate(
                        f'E{best_ep}\n{best_val:.3f}',
                        xy=(best_ep, best_val), fontsize=8, fontweight='bold',
                        color='darkgreen',
                    )
                map50_ax.set_xlabel("Epoch")
                map50_ax.set_ylabel("mAP@0.5")
                map50_ax.set_title("Validation mAP@0.5")
                map50_ax.legend()
                map50_ax.grid(True, alpha=0.3)

            # mAP@0.5:0.95 curve (axes[2]) — primary criterion
            map95_ax = axes[2] if has_map95 else None
            if map95_ax is not None:
                map95_vals = pd.to_numeric(df["val/mAP_50_95"], errors='coerce')
                valid95 = map95_vals.notna()
                if valid95.any():
                    map95_ax.plot(
                        epochs[valid95], map95_vals[valid95].values,
                        "m-D", label="mAP@0.5:0.95", linewidth=2, markersize=4,
                    )
                    best95_idx = map95_vals[valid95].idxmax()
                    best95_ep = int(df.loc[best95_idx, "epoch"])
                    best95_val = map95_vals.loc[best95_idx]
                    map95_ax.annotate(
                        f'E{best95_ep}\n{best95_val:.3f}',
                        xy=(best95_ep, best95_val), fontsize=8, fontweight='bold',
                        color='purple',
                    )
                map95_ax.set_xlabel("Epoch")
                map95_ax.set_ylabel("mAP@0.5:0.95")
                map95_ax.set_title("Validation mAP@0.5:0.95")
                map95_ax.legend()
                map95_ax.grid(True, alpha=0.3)

            # LR curve (always in the last subplot)
            lr_ax = axes[3] if len(axes) > 3 else axes[-1]
            if lr_ax is not None:
                if "lr/pg0" in df.columns:
                    lr_ax.plot(epochs, df["lr/pg0"], color="orange", linewidth=2)
                    lr_ax.set_yscale("log")
                lr_ax.set_xlabel("Epoch")
                lr_ax.set_ylabel("Learning Rate")
                lr_ax.set_title("Learning Rate Schedule")
                lr_ax.grid(True, alpha=0.3)

            plt.tight_layout()
            save_path = self.log_dir / "training_curves.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            self.logger.info(f"✓ 训练曲线已保存: {save_path}")
        except Exception as exc:
            self.logger.warning(f"绘制训练曲线失败: {exc}")

    # ── COCO / scale eval overrides ──────────────────────────────────

    def _get_coco_eval_predictor(self, model):
        device = torch.device(self.misc_config.get("device", "cuda"))
        best_pt = self.log_dir / "weights" / "best.pt"

        if best_pt.exists():
            eval_model = self._create_fresh_model()
            ckpt = torch.load(best_pt, map_location=device)
            state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            eval_model.load_state_dict(state, strict=True)
            eval_model.to(device).eval()
        elif model is not None:
            eval_model = model
            eval_model.eval()
        else:
            return None, max(len(self.class_names), 1)

        nc = max(len(self.class_names), 1)
        return eval_model, nc

    def _predict_batch_coco_eval(self, predictor, batch_paths, imgsz, device):
        device = torch.device(device) if isinstance(device, str) else device
        predictor.eval()

        images = []
        orig_shapes = []
        letterbox_params = []
        input_size = self._input_size()
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            w, h = img.size
            orig_shapes.append((h, w))
            image, scale, pad_left, pad_top = letterbox_image(
                TF.to_tensor(img), input_size,
            )
            images.append(image.to(device))
            letterbox_params.append((scale, pad_left, pad_top))

        with torch.no_grad():
            outputs = predictor(images)

        results = []
        for orig_shape, params, out in zip(orig_shapes, letterbox_params, outputs):
            scale, pad_left, pad_top = params
            out = dict(out)
            out["boxes"] = unletterbox_boxes(
                out["boxes"], orig_shape, scale, pad_left, pad_top,
            )
            results.append(_FasterRCNNResult(orig_shape, out))
        return results

    def _can_run_coco_eval_without_ultralytics_model(self) -> bool:
        return True

    # ── benchmark overrides ───────────────────────────────────────────

    @staticmethod
    def _copy_benchmark_output_to_host(value):
        """Force detection outputs back to host memory before stopping the timer."""
        if isinstance(value, torch.Tensor):
            return value.cpu()
        if isinstance(value, dict):
            return {key: FasterRCNNTrainer._copy_benchmark_output_to_host(item)
                    for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(
                FasterRCNNTrainer._copy_benchmark_output_to_host(item)
                for item in value
            )
        return value

    def _measure_end_to_end(
        self,
        model: nn.Module,
        images,
        device,
        warmup_iters: int = 50,
        measure_iters: int = 200,
    ) -> Tuple[float, float, List[float]]:
        """Measure real-image preprocessing + Faster R-CNN inference pipeline."""
        image_paths = sorted(
            path
            for path in Path(images).rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not image_paths:
            raise ValueError(f"未找到 benchmark 图像: {images}")

        device = torch.device(device)
        input_size = self._input_size()
        model.to(device).eval()

        def pipeline(path: Path):
            with Image.open(path) as pil_image:
                image = TF.to_tensor(pil_image.convert("RGB"))
            image, _, _, _ = letterbox_image(image, input_size)
            outputs = model([image.to(device)])
            self._copy_benchmark_output_to_host(outputs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        with torch.no_grad():
            for index in range(warmup_iters):
                pipeline(image_paths[index % len(image_paths)])

            latencies: List[float] = []
            for index in range(measure_iters):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                pipeline(image_paths[index % len(image_paths)])
                latencies.append((time.perf_counter() - start) * 1000.0)

        latency_ms = float(np.mean(latencies))
        return 1000.0 / latency_ms, latency_ms, latencies

    def _run_model_benchmark(self, model_or_predictor) -> Optional[dict]:
        try:
            if isinstance(model_or_predictor, nn.Module):
                raw_model = model_or_predictor
            else:
                return None

            raw_model.eval()
            wrapper = _BenchmarkInputAdapter(raw_model)
            model_name = self.model_config.get("model_name", "fasterrcnn_resnet50_fpn")
            result = benchmark_model(
                wrapper,
                imgsz=self.training_config.get("imgsz", 640),
                device=self.misc_config.get("device", "cuda"),
                model_name=model_name,
                includes_nms=True,
            )
            try:
                e2e_fps, e2e_latency, e2e_latencies = self._measure_end_to_end(
                    raw_model,
                    self._benchmark_image_dir(str(self.data_config.get("data_yaml", ""))),
                    self.misc_config.get("device", "cuda"),
                )
                result.end_to_end_fps = e2e_fps
                result.end_to_end_latency_ms = e2e_latency
                result.end_to_end_latencies_ms = e2e_latencies
            except Exception as exc:
                self.logger.warning(f"Faster R-CNN 端到端 benchmark 失败（保留模型推理指标）: {exc}")
            log_benchmark(self.logger.info, result, header=model_name)
            return benchmark_to_dict(result)
        except Exception as exc:
            self.logger.warning(f"Faster R-CNN benchmark 失败: {exc}")
            return None

    def _benchmark_eval_predictor(self, eval_predictor) -> Optional[dict]:
        return self._run_model_benchmark(eval_predictor)

    def _optional_post_train_benchmark(self, model) -> Optional[dict]:
        if model is None:
            return None
        return self._run_model_benchmark(model)

    def run_tensorrt_benchmark(self) -> None:
        """Export the completed Faster R-CNN run and record TensorRT results."""
        enabled = os.environ.get(
            "YOLO_TRT_BENCHMARK", os.environ.get("TRT_BENCHMARK", "1")
        )
        if enabled == "0":
            self.logger.info("TensorRT benchmark 已通过环境变量关闭")
            return

        weights = self._tensorrt_weights_path()
        if weights is None:
            raise FileNotFoundError(f"无可用于 TensorRT benchmark 的权重: {self.log_dir}")

        data_yaml = str(self.data_config.get("data_yaml", ""))
        data_lower = data_yaml.lower()
        if "dair" in data_lower:
            dataset = "DAIR-V2X"
        elif "uadetrac" in data_lower or "ua-detrac" in data_lower:
            dataset = "UA-DETRAC"
        else:
            dataset = Path(data_yaml).stem or "unknown"

        model_name = str(self.model_config.get("model_name", "fasterrcnn_resnet50_fpn"))
        command = [
            sys.executable,
            str(_yolo_dir / "tools" / "benchmark_fasterrcnn_trt.py"),
            "--weights", str(weights.resolve()),
            "--config", str((self.log_dir / "config.yaml").resolve()),
            "--output-dir", str(self.log_dir.resolve()),
            "--images", str(self._benchmark_image_dir(data_yaml)),
            "--num-classes", str(self.num_classes),
            "--run-id", self.log_dir.name,
            "--model", model_name,
            "--dataset", dataset,
            "--seed", str(self.training_config.get("seed", 0)),
            "--imgsz", str(self._input_size()),
            "--warmup", os.environ.get("TRT_WARMUP", "100"),
            "--iterations", os.environ.get("TRT_ITERATIONS", "1000"),
        ]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("🚀 运行 Faster R-CNN TensorRT benchmark: %s", model_name)
        subprocess.run(command, cwd=_yolo_dir, check=True)
        self.logger.info("✓ TensorRT benchmark 已写入统一总表")

    # ── file-naming override (weights are state_dict, not Ultralytics) ─

    def _align_file_naming(self):
        """Copy weights to standard names if they exist."""
        import shutil
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
