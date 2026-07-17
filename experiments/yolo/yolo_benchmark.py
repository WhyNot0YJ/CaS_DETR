"""YOLO GFLOPs, parameters, model speed, and end-to-end batch-1 speed."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

_experiments_root = Path(__file__).resolve().parent.parent
if str(_experiments_root) not in sys.path:
    sys.path.insert(0, str(_experiments_root))

_bm_yolo_dir = Path(__file__).resolve().parent
_bm_ext = _bm_yolo_dir / "external"
if _bm_ext.is_dir() and str(_bm_ext) not in sys.path:
    sys.path.insert(0, str(_bm_ext))
_bm_yolox = _bm_ext / "YOLOX"
if _bm_yolox.is_dir() and str(_bm_yolox) not in sys.path:
    sys.path.insert(0, str(_bm_yolox))

from common.model_benchmark import (
    BenchmarkResult,
    benchmark_to_dict,
    compute_active_params,
    compute_gflops,
    compute_params,
    format_benchmark_report,
    measure_fps,
)

logger = logging.getLogger(__name__)


def _build_nms_postprocess(conf_thres=0.25, iou_thres=0.7, max_det=300):
    try:
        from ultralytics.utils.nms import non_max_suppression
    except ImportError:
        from ultralytics.utils.ops import non_max_suppression

    def postprocess(raw_output):
        predictions = raw_output[0] if isinstance(raw_output, (tuple, list)) else raw_output
        return non_max_suppression(
            predictions,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
        )

    return postprocess


def _extract_yolo_nn_module(model_or_path):
    if isinstance(model_or_path, (str, Path)):
        pt_path = Path(model_or_path)
        if not pt_path.exists():
            raise FileNotFoundError(f"权重文件不存在: {pt_path}")
        from ultralytics import YOLO
        return _unwrap_ultralytics(YOLO(str(pt_path))), pt_path.stem

    if hasattr(model_or_path, "model") and isinstance(model_or_path.model, nn.Module):
        name = getattr(model_or_path, "model_name", "yolo")
        if hasattr(model_or_path, "ckpt_path"):
            name = Path(str(model_or_path.ckpt_path)).stem
        return _unwrap_ultralytics(model_or_path), str(name)
    if isinstance(model_or_path, nn.Module):
        return model_or_path, "yolo"
    raise TypeError(f"不支持的模型类型: {type(model_or_path)}")


def _unwrap_ultralytics(yolo_obj):
    inner = getattr(yolo_obj, "model", yolo_obj)
    if hasattr(inner, "module"):
        inner = inner.module
    if hasattr(inner, "fuse"):
        try:
            inner.fuse()
        except Exception:
            pass
    return inner


def _image_paths(root):
    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = sorted(path for path in Path(root).rglob("*") if path.suffix.lower() in suffixes)
    if not paths:
        raise ValueError(f"未找到 benchmark 图像: {root}")
    return paths


def _preprocess(path, height, width):
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图像: {path}")
    source_height, source_width = image.shape[:2]
    scale = min(height / source_height, width / source_width)
    resized_width = round(source_width * scale)
    resized_height = round(source_height * scale)
    image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    image = cv2.copyMakeBorder(
        image,
        top,
        height - resized_height - top,
        left,
        width - resized_width - left,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    array = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)
    return torch.from_numpy(array).div_(255.0).unsqueeze(0)


def _copy_to_host(value):
    if isinstance(value, torch.Tensor):
        return value.cpu()
    if isinstance(value, (tuple, list)):
        return [_copy_to_host(item) for item in value]
    return value


def _measure_end_to_end(
    model, postprocess, images, imgsz, device, warmup_iters, measure_iters, use_fp16
):
    paths = _image_paths(images)
    model = model.to(device).eval()
    if use_fp16 and device.type == "cuda":
        model = model.half()

    def pipeline(path):
        image = _preprocess(path, *imgsz).to(device)
        if use_fp16 and device.type == "cuda":
            image = image.half()
        detections = postprocess(model(image))
        _copy_to_host(detections)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.no_grad():
        for index in range(warmup_iters):
            pipeline(paths[index % len(paths)])
        latencies = []
        for index in range(measure_iters):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            pipeline(paths[index % len(paths)])
            latencies.append((time.perf_counter() - start) * 1000.0)

    latency_ms = float(np.mean(latencies))
    return 1000.0 / latency_ms, latency_ms, latencies


def _benchmark(model, postprocess, images, imgsz, device, name, warmup, iterations, fp16):
    size = (int(imgsz), int(imgsz)) if not isinstance(imgsz, (list, tuple)) else tuple(map(int, imgsz))
    total, trainable = compute_params(model)
    active = compute_active_params(model)
    gflops = compute_gflops(model, size, device)
    fps, latency, latencies = measure_fps(
        model, size, device, warmup, iterations, fp16, postprocess
    )
    end_to_end_fps, end_to_end_latency, end_to_end_latencies = _measure_end_to_end(
        model, postprocess, images, size, device, warmup, iterations, fp16
    )
    return BenchmarkResult(
        model_name=name,
        gflops=gflops,
        params_total=total,
        params_active=active,
        params_trainable=trainable,
        fps=fps,
        latency_ms=latency,
        imgsz=size,
        device=str(device),
        warmup_iters=warmup,
        measure_iters=iterations,
        latencies_ms=latencies,
        includes_nms=True,
        end_to_end_fps=end_to_end_fps,
        end_to_end_latency_ms=end_to_end_latency,
        end_to_end_latencies_ms=end_to_end_latencies,
    )


def benchmark_yolo(
    model_or_path,
    images: Union[str, Path],
    imgsz: Union[int, List[int], Tuple[int, int]] = 640,
    device: Optional[Union[str, torch.device]] = None,
    model_name: Optional[str] = None,
    warmup_iters: int = 50,
    measure_iters: int = 200,
    use_fp16: bool = False,
    conf_thres: float = 0.25,
    iou_thres: float = 0.7,
    max_det: int = 300,
) -> BenchmarkResult:
    """Measure the complete YOLO deployment path on real images."""
    model, auto_name = _extract_yolo_nn_module(model_or_path)
    if device is None:
        device = next(model.parameters()).device
    postprocess = _build_nms_postprocess(conf_thres, iou_thres, max_det)
    return _benchmark(
        model, postprocess, images, imgsz, torch.device(device), model_name or auto_name,
        warmup_iters, measure_iters, use_fp16,
    )


def benchmark_yolox(
    model: nn.Module,
    exp,
    images: Union[str, Path],
    imgsz: Union[int, List[int], Tuple[int, int]] = 640,
    device: Optional[Union[str, torch.device]] = None,
    model_name: Optional[str] = None,
    warmup_iters: int = 50,
    measure_iters: int = 200,
) -> BenchmarkResult:
    """Measure the complete YOLOX deployment path on real images."""
    from yolox_predict import benchmark_yolox_forward_nms

    if device is None:
        device = next(model.parameters()).device
    return _benchmark(
        model, benchmark_yolox_forward_nms(model, exp), images, imgsz,
        torch.device(device), model_name or "yolox", warmup_iters, measure_iters, False,
    )
