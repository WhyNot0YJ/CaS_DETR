"""UA-DETRAC ignore-region integration for Ultralytics detection training."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors

_experiments_root = Path(__file__).resolve().parent.parent
if str(_experiments_root) not in sys.path:
    sys.path.insert(0, str(_experiments_root))

from common.uadetrac_ignore import IGNORE_IOA_THRESHOLD, ignore_regions_xyxy, prediction_ioa_xyxy


def ignored_prediction_mask(pred_boxes: torch.Tensor, ignore_boxes: torch.Tensor) -> torch.Tensor:
    """Return boxes whose fixed IoA with an ignore region is at least 0.5."""
    if pred_boxes.numel() == 0 or ignore_boxes.numel() == 0:
        return torch.zeros(pred_boxes.shape[0], dtype=torch.bool, device=pred_boxes.device)
    return prediction_ioa_xyxy(pred_boxes, ignore_boxes) >= IGNORE_IOA_THRESHOLD


def _coco_ignore_by_stem(annotation_files: str | Path | list[str]) -> dict[str, tuple[float, float, list[list[float]]]]:
    result = {}
    for annotation_file in annotation_files if isinstance(annotation_files, list) else [annotation_files]:
        data = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
        for image in data.get("images", []):
            ignores = ignore_regions_xyxy(image)
            if ignores:
                result[Path(str(image["file_name"])).stem] = (
                    float(image["width"]), float(image["height"]), ignores,
                )
    return result


class UADETRACYOLODatasetMixin:
    """Append ignore rectangles as class -1 so stock geometric augments preserve them."""

    def __init__(self, *args, data: dict | None = None, **kwargs):
        annotation_files = (data or {}).get("ignore_coco_anns")
        self._ignore_by_stem = _coco_ignore_by_stem(annotation_files) if annotation_files else {}
        super().__init__(*args, data=data, **kwargs)

    def get_labels(self):
        labels = super().get_labels()
        for label in labels:
            record = self._ignore_by_stem.get(Path(label["im_file"]).stem)
            if not record:
                continue
            width, height, ignores = record
            boxes = np.asarray(
                [[(x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height]
                 for x1, y1, x2, y2 in ignores],
                dtype=np.float32,
            )
            label["cls"] = np.concatenate((label["cls"], np.full((len(boxes), 1), -1, dtype=np.float32)))
            label["bboxes"] = np.concatenate((label["bboxes"], boxes))
        return labels


def _split_targets(batch: dict[str, torch.Tensor], batch_size: int, scale: torch.Tensor, preprocess) -> tuple[torch.Tensor, torch.Tensor]:
    flat = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
    positive = flat[flat[:, 1] >= 0]
    ignored = flat[flat[:, 1] < 0]
    targets = preprocess(positive, batch_size, scale)
    ignore_boxes = preprocess(ignored, batch_size, scale)[..., 1:5]
    return targets, ignore_boxes


def _ignore_anchor_mask(pred_boxes: torch.Tensor, ignore_boxes: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros(pred_boxes.shape[:2], dtype=torch.bool, device=pred_boxes.device)
    for index, boxes in enumerate(ignore_boxes):
        boxes = boxes[boxes.sum(-1) > 0]
        if len(boxes):
            mask[index] = ignored_prediction_mask(pred_boxes[index], boxes)
    return mask


class UADETRACDetectionLoss(v8DetectionLoss):
    """必须定义在模块级：BaseModel.loss() 会把 criterion 挂到 model 上进 checkpoint，
    函数内 local 类无法被 torch.save 序列化。"""

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]):
        loss = torch.zeros(3, device=self.device)
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [item.view(feats[0].shape[0], self.no, -1) for item in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)
        pred_scores, pred_distri = pred_scores.permute(0, 2, 1).contiguous(), pred_distri.permute(0, 2, 1).contiguous()
        dtype, batch_size = pred_scores.dtype, pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        targets, ignore_boxes = _split_targets(batch, batch_size, imgsz[[1, 0, 1, 0]], self.preprocess)
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)
        cls_loss = self.bce(pred_scores, target_scores.to(dtype))
        ignored = _ignore_anchor_mask(pred_bboxes.detach() * stride_tensor, ignore_boxes)
        cls_loss[ignored & ~fg_mask] = 0
        loss[1] = cls_loss.sum() / target_scores_sum
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return loss * batch_size, loss.detach()


class _IgnoreCriterionFactory:
    """torch.save 无法 pickle 函数内 lambda；用模块级可 pickle 工厂替代 init_criterion 赋值，
    保证 checkpoint 序列化（save_model / EMA）不炸。"""

    def __init__(self, model):
        self.model = model

    def __call__(self):
        return UADETRACDetectionLoss(self.model)


def build_ignore_detection_trainer():
    """Create version-pinned subclasses after importing the installed Ultralytics package."""
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.models.yolo.detect.val import DetectionValidator

    class UADETRACYOLODataset(UADETRACYOLODatasetMixin, YOLODataset):
        pass

    class UADETRACDetectionValidator(DetectionValidator):
        def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
            gs = max(int(self.stride), 32)
            return UADETRACYOLODataset(
                img_path=img_path, imgsz=self.args.imgsz, batch_size=batch, augment=False, hyp=self.args,
                rect=True, cache=self.args.cache or None, single_cls=self.args.single_cls, stride=gs,
                pad=0.5, prefix="val: ", task=self.args.task, classes=self.args.classes, data=self.data,
                fraction=self.args.fraction,
            )

        def update_metrics(self, preds, batch):
            for si, pred in enumerate(preds):
                self.seen += 1
                pbatch = self._prepare_batch(si, batch)
                ignored_gt = pbatch["cls"] < 0
                ignore_boxes = pbatch["bboxes"][ignored_gt]
                pbatch["cls"], pbatch["bboxes"] = pbatch["cls"][~ignored_gt], pbatch["bboxes"][~ignored_gt]
                keep_gt = ~ignored_prediction_mask(pbatch["bboxes"], ignore_boxes)
                pbatch["cls"], pbatch["bboxes"] = pbatch["cls"][keep_gt], pbatch["bboxes"][keep_gt]
                predn = self._prepare_pred(pred)
                keep = ~ignored_prediction_mask(predn["bboxes"], ignore_boxes)
                predn = {key: value[keep] if isinstance(value, torch.Tensor) and value.shape[0] == keep.shape[0] else value for key, value in predn.items()}
                cls = pbatch["cls"].cpu().numpy()
                no_pred = len(predn["cls"]) == 0
                self.metrics.update_stats({
                    **self._process_batch(predn, pbatch), "target_cls": cls, "target_img": np.unique(cls),
                    "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                    "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                })
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, pbatch, conf=self.args.conf)
                    if self.args.visualize:
                        self.confusion_matrix.plot_matches(batch["img"][si], pbatch["im_file"], self.save_dir)
                if no_pred:
                    continue
                if self.args.save_json or self.args.save_txt:
                    predn_scaled = self.scale_preds(predn, pbatch)
                if self.args.save_json:
                    self.pred_to_json(predn_scaled, pbatch)
                if self.args.save_txt:
                    self.save_one_txt(predn_scaled, self.args.save_conf, pbatch["ori_shape"], self.save_dir / "labels" / f"{Path(pbatch['im_file']).stem}.txt")

    class UADETRACDetectionTrainer(DetectionTrainer):
        def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
            gs = max(int(self.model.stride.max() if self.model else 0), 32)
            return UADETRACYOLODataset(
                img_path=img_path, imgsz=self.args.imgsz, batch_size=batch, augment=mode == "train", hyp=self.args,
                rect=mode == "val", cache=self.args.cache or None, single_cls=self.args.single_cls, stride=gs,
                pad=0.0 if mode == "train" else 0.5, prefix=f"{mode}: ", task=self.args.task,
                classes=self.args.classes, data=self.data, fraction=self.args.fraction,
            )

        def get_validator(self):
            self.loss_names = "box_loss", "cls_loss", "dfl_loss"
            return UADETRACDetectionValidator(self.test_loader, save_dir=self.save_dir, args=self.args, _callbacks=self.callbacks)

        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg, weights, verbose)
            model.init_criterion = _IgnoreCriterionFactory(model)
            return model

        def setup_model(self):
            checkpoint = super().setup_model()
            self.model.init_criterion = _IgnoreCriterionFactory(self.model)
            return checkpoint

        def save_model(self):
            # Checkpoint 是交付物：剥离引用本仓库模块的 criterion 对象，
            # 保证任意进程（benchmark/eval/外部工具）torch.load 无需 import
            # uadetrac_ultralytics。保存后恢复，训练不受影响。
            restore = []
            holders = [self.model] + ([self.ema.ema] if self.ema else [])
            for holder in holders:
                for attr in ("criterion", "init_criterion"):
                    if attr in vars(holder):
                        restore.append((holder, attr, vars(holder).pop(attr)))
            try:
                return super().save_model()
            finally:
                for holder, attr, value in restore:
                    setattr(holder, attr, value)

    return UADETRACDetectionTrainer
