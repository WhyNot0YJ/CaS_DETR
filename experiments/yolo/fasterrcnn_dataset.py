#!/usr/bin/env python3
"""
YOLO 格式数据集 → torchvision Faster R-CNN 目标检测 Dataset。

读取 Ultralytics data_yaml 定义的 images/{split} 与 labels/{split}，
将 YOLO 归一化标注 (class_id cx cy w h) 转换为 torchvision 目标格式：
  boxes: FloatTensor[N, 4]  (x1, y1, x2, y2) 绝对像素坐标
  labels: Int64Tensor[N]    class_id + 1（torchvision 保留 0 给背景）
"""

import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


def letterbox_image(
    image: torch.Tensor, size: int
) -> Tuple[torch.Tensor, float, int, int]:
    """Resize an image into a square canvas while preserving its aspect ratio."""
    height, width = image.shape[-2:]
    scale = min(size / height, size / width)
    resized_height = round(height * scale)
    resized_width = round(width * scale)
    image = F.resize(image, [resized_height, resized_width])

    pad_left = (size - resized_width) // 2
    pad_top = (size - resized_height) // 2
    image = F.pad(
        image,
        [pad_left, pad_top, size - resized_width - pad_left, size - resized_height - pad_top],
        fill=114.0 / 255.0,
    )
    return image, scale, pad_left, pad_top


def unletterbox_boxes(
    boxes: torch.Tensor,
    original_size: Tuple[int, int],
    scale: float,
    pad_left: int,
    pad_top: int,
) -> torch.Tensor:
    """Map boxes predicted on a letterboxed image back to original coordinates."""
    height, width = original_size
    boxes = boxes.clone()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / scale
    boxes[:, [0, 2]].clamp_(0, width)
    boxes[:, [1, 3]].clamp_(0, height)
    return boxes


class YOLOFormatDetectionDataset(Dataset):
    """从 YOLO 格式目录加载图像和标注，输出 torchvision detection 所需格式。"""

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        transform: Optional[Callable] = None,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform

        self.image_files: List[Path] = sorted(
            p
            for ext in (".jpg", ".jpeg", ".png")
            for p in self.images_dir.glob(f"*{ext}")
        )
        if not self.image_files:
            raise FileNotFoundError(
                f"No images found in {self.images_dir}"
            )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        label_path = self.labels_dir / f"{img_path.stem}.txt"
        boxes: List[List[float]] = []
        labels: List[int] = []

        if label_path.exists():
            with label_path.open() as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    x1 = (cx - bw / 2) * w
                    y1 = (cy - bh / 2) * h
                    x2 = (cx + bw / 2) * w
                    y2 = (cy + bh / 2) * h
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(cls_id + 1)

        if boxes:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)

        target: Dict[str, Any] = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx]),
        }

        img_tensor = F.to_tensor(img)

        if self.transform is not None:
            img_tensor, target = self.transform(img_tensor, target)

        return img_tensor, target


class RandomHorizontalFlipDetection:
    """对图像和目标框同时做水平翻转。"""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self, image: torch.Tensor, target: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if random.random() < self.p:
            image = F.hflip(image)
            boxes = target["boxes"]
            if boxes.numel() > 0:
                w = image.shape[-1]
                boxes = boxes.clone()
                x1 = boxes[:, 0].clone()
                boxes[:, 0] = w - boxes[:, 2]
                boxes[:, 2] = w - x1
                target["boxes"] = boxes
        return image, target


class LetterboxDetection:
    """Apply fixed square letterbox preprocessing to an image and its boxes."""

    def __init__(self, size: int):
        self.size = size

    def __call__(
        self, image: torch.Tensor, target: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        image, scale, pad_left, pad_top = letterbox_image(image, self.size)
        boxes = target["boxes"]
        if boxes.numel() > 0:
            boxes = boxes.clone()
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_left
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_top
            target["boxes"] = boxes
        return image, target


class DetectionCompose:
    """Compose detection transforms that update both image and target."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(
        self, image: torch.Tensor, target: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


def detection_collate_fn(batch):
    """DataLoader collate：图像和目标长度不同，不能 stack。"""
    return tuple(zip(*batch))


def resolve_split_dirs(
    data_yaml_path: str, split: str
) -> Tuple[Path, Path]:
    """从 data_yaml 解析 images 和 labels 目录。"""
    data_yaml_path = Path(data_yaml_path)
    with data_yaml_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    root = data_yaml_path.parent.resolve()
    path_field = str(cfg.get("path", "")).strip()
    if path_field:
        pc = Path(path_field)
        if pc.is_absolute() and pc.is_dir():
            root = pc
        elif not pc.is_absolute():
            candidate = (data_yaml_path.parent / pc).resolve()
            if candidate.is_dir():
                root = candidate

    images_rel = str(cfg.get(split, f"images/{split}")).strip()
    images_dir = root / images_rel if not Path(images_rel).is_absolute() else Path(images_rel)

    labels_dir = Path(str(images_dir).replace("images", "labels", 1))
    return images_dir.resolve(), labels_dir.resolve()
