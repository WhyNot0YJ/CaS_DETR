#!/usr/bin/env python3
"""Model and end-to-end TensorRT benchmark for the batch-1 deployment protocol."""

import argparse
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
import sys

EXPERIMENTS_DIR = Path(__file__).resolve().parents[3]
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))
from common.result_paths import update_csv_rows, upsert_csv_rows
from benchmark_runtime import command_output, ensure_gpu_idle, summarize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--eval-csv",
        type=Path,
        help="Optional eval_metrics.csv whose matching run_id receives TensorRT speed columns.",
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument("--framework", default="")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--seed", default="")
    parser.add_argument("--checkpoint-sha256", default="")
    parser.add_argument("--config-sha256", default="")
    parser.add_argument("--training-taxonomy", default="")
    parser.add_argument("--evaluation-taxonomy", default="")
    parser.add_argument("--postprocess", default="")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--preprocess", choices=("resize", "letterbox"), default="resize")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    return parser.parse_args()


def torch_dtype(dtype):
    return torch.from_numpy(np.empty((), dtype=trt.nptype(dtype))).dtype


class EngineRunner:
    def __init__(self, engine_path, imgsz):
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        self.runtime = trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.imgsz = imgsz
        self.tensors = {}

        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                if name == "images":
                    shape = (1, 3, imgsz, imgsz)
                elif name == "orig_target_sizes":
                    shape = (1, 2)
                else:
                    raise ValueError(f"dynamic output shape is unresolved: {name} {shape}")
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.context.set_input_shape(name, shape)

        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            shape = tuple(self.context.get_tensor_shape(name))
            tensor = torch.empty(shape, dtype=torch_dtype(self.engine.get_tensor_dtype(name)), device="cuda")
            self.tensors[name] = tensor
            self.context.set_tensor_address(name, tensor.data_ptr())

    def enqueue(self):
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")

    def run(self):
        self.enqueue()

    @property
    def outputs(self):
        return [
            tensor for name, tensor in self.tensors.items()
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]


def benchmark_model(runner, warmup, iterations):
    with torch.cuda.stream(runner.stream):
        runner.tensors["images"].fill_(0.5)
        if "orig_target_sizes" in runner.tensors:
            runner.tensors["orig_target_sizes"].copy_(torch.tensor(
                [[runner.imgsz, runner.imgsz]],
                device="cuda",
                dtype=runner.tensors["orig_target_sizes"].dtype,
            ))
    for _ in range(warmup):
        runner.run()
    runner.stream.synchronize()

    times = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(runner.stream)
        runner.run()
        end.record(runner.stream)
        end.synchronize()
        times.append(start.elapsed_time(end))
    return times


def image_paths(root):
    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in suffixes)


def preprocess(path, imgsz, mode):
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to read image: {path}")
    height, width = image.shape[:2]
    if mode == "letterbox":
        scale = min(imgsz / height, imgsz / width)
        resized_width = round(width * scale)
        resized_height = round(height * scale)
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        left = (imgsz - resized_width) // 2
        top = (imgsz - resized_height) // 2
        image = cv2.copyMakeBorder(
            image,
            top,
            imgsz - resized_height - top,
            left,
            imgsz - resized_width - left,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
    else:
        image = cv2.resize(image, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    array = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)
    return torch.from_numpy(array).div_(255.0).unsqueeze(0), (height, width)


def run_pipeline(runner, path, imgsz, preprocess_mode):
    image, original_size = preprocess(path, imgsz, preprocess_mode)
    with torch.cuda.stream(runner.stream):
        runner.tensors["images"].copy_(image, non_blocking=False)
        if "orig_target_sizes" in runner.tensors:
            runner.tensors["orig_target_sizes"].copy_(torch.tensor(
                [original_size], dtype=runner.tensors["orig_target_sizes"].dtype
            ))
        runner.run()
    runner.stream.synchronize()
    return [output.cpu() for output in runner.outputs]


def benchmark_end_to_end(runner, root, imgsz, preprocess_mode, warmup, iterations):
    paths = image_paths(root)
    if not paths:
        raise ValueError(f"no images found under {root}")
    for index in range(warmup):
        run_pipeline(runner, paths[index % len(paths)], imgsz, preprocess_mode)
    torch.cuda.synchronize()

    times = []
    for index in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        run_pipeline(runner, paths[index % len(paths)], imgsz, preprocess_mode)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    return times


def main():
    args = parse_args()
    ensure_gpu_idle()
    torch.cuda.reset_peak_memory_stats()
    runner = EngineRunner(args.engine, args.imgsz)
    common = {
        "run_id": args.run_id,
        "framework": args.framework,
        "dataset": args.dataset,
        "seed": args.seed,
        "checkpoint_sha256": args.checkpoint_sha256,
        "config_sha256": args.config_sha256,
        "training_taxonomy": args.training_taxonomy,
        "evaluation_taxonomy": args.evaluation_taxonomy,
        "postprocess": args.postprocess,
        "result_type": "benchmark",
        "model": args.model,
        "engine": str(args.engine.resolve()),
        "batch_size": 1,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "execution": "",
        "aux_streams": getattr(runner.engine, "num_aux_streams", ""),
        "peak_memory_mib": "",
        "gpu": torch.cuda.get_device_name(0),
        "driver": command_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "tensorrt": trt.__version__,
        "cuda": torch.version.cuda,
    }
    rows = [
        {
            **common,
            "mode": "model",
            "execution": "cuda_event",
            **summarize(benchmark_model(runner, args.warmup, args.iterations)),
        },
        {
            **common,
            "mode": "end-to-end",
            "execution": "synchronous_end_to_end",
            **summarize(benchmark_end_to_end(
                runner, args.images, args.imgsz, args.preprocess, args.warmup, args.iterations
            )),
        },
    ]
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    for row in rows:
        row["peak_memory_mib"] = peak_memory

    upsert_csv_rows(
        args.output_csv,
        rows,
        key_fields=("run_id", "mode"),
    )
    if args.eval_csv:
        model_row, end_to_end_row = rows
        updated = update_csv_rows(
            args.eval_csv,
            match={"run_id": args.run_id},
            updates={
                "Inference_FPS": f"{float(model_row['fps']):.2f}",
                "Inference_Latency_ms": f"{float(model_row['mean_latency_ms']):.2f}",
                "EndToEnd_FPS": f"{float(end_to_end_row['fps']):.2f}",
                "EndToEnd_Latency_ms": f"{float(end_to_end_row['mean_latency_ms']):.2f}",
            },
        )
        if not updated:
            raise RuntimeError(
                f"TensorRT rows were written but no eval_metrics rows matched run_id={args.run_id}"
            )


if __name__ == "__main__":
    main()
