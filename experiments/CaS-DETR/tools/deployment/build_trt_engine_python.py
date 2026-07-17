#!/usr/bin/env python3
"""Build a TensorRT FP16 engine through the Python API."""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import tensorrt as trt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--workspace-mib", type=int, default=4096)
    parser.add_argument("--max-aux-streams", type=int)
    parser.add_argument("--builder-optimization-level", type=int, choices=range(0, 6), default=5)
    return parser.parse_args()


def command_output(command):
    try:
        return subprocess.run(command, check=False, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def main():
    args = parse_args()
    onnx_path = args.onnx.resolve()
    engine_path = args.engine.resolve()
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    explicit_batch = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
    network_flags = 0 if explicit_batch is None else 1 << int(explicit_batch)
    fp16_flag = getattr(trt.BuilderFlag, "FP16", None)
    if fp16_flag is None:
        network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    onnx_bytes = onnx_path.read_bytes()
    if fp16_flag is None:
        import onnx
        from onnxconverter_common import float16

        model = onnx.load_from_string(onnx_bytes)
        model = float16.convert_float_to_float16(model, keep_io_types=False)
        for node in model.graph.node:
            if node.op_type != "Cast":
                continue
            for attribute in node.attribute:
                if attribute.name == "to" and attribute.i == onnx.TensorProto.FLOAT:
                    attribute.i = onnx.TensorProto.FLOAT16
        onnx_bytes = model.SerializeToString()
    if not parser.parse(onnx_bytes):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mib * 1024 * 1024)
    config.builder_optimization_level = args.builder_optimization_level
    if args.max_aux_streams is not None:
        config.max_aux_streams = args.max_aux_streams
    if fp16_flag is not None:
        config.set_flag(fp16_flag)
    started = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT returned an empty serialized network")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)
    metadata = {
        "onnx": str(onnx_path),
        "engine": str(engine_path),
        "precision": "fp16",
        "workspace_mib": args.workspace_mib,
        "max_aux_streams": args.max_aux_streams,
        "builder_optimization_level": args.builder_optimization_level,
        "build_seconds": time.time() - started,
        "tensorrt": trt.__version__,
        "platform": platform.platform(),
        "gpu": command_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]),
        "driver": command_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
    }
    engine_path.with_suffix(".build.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
