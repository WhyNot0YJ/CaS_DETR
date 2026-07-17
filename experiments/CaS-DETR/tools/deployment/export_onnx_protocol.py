#!/usr/bin/env python3
"""Export a static batch-1 deploy graph for the benchmark protocol."""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


FRAMEWORK_DIRS = {
    "casdeim": "CaS-DETR",
    "deim": "DEIM",
    "dfine": "D-FINE",
    "rtdetr": "RT-DETR/rtdetrv2_pytorch",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=FRAMEWORK_DIRS, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def build(framework, config_path, checkpoint_path):
    experiments_dir = Path(__file__).resolve().parents[3]
    framework_dir = experiments_dir / FRAMEWORK_DIRS[framework]
    sys.path.insert(0, str(framework_dir))
    os.chdir(framework_dir)
    if framework in ("dfine", "rtdetr"):
        from src.core import YAMLConfig
    else:
        from engine.core import YAMLConfig

    cfg = YAMLConfig(str(config_path))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("ema", {}).get("module") or checkpoint["model"]
    cfg.model.load_state_dict(state, strict=True)
    if hasattr(getattr(cfg.model, "encoder", None), "set_epoch"):
        cfg.model.encoder.set_epoch(int(checkpoint.get("last_epoch", 0)))
    return cfg.model.deploy(), cfg.postprocessor.deploy()


class DeployModel(nn.Module):
    def __init__(self, model, postprocessor):
        super().__init__()
        self.model = model
        self.postprocessor = postprocessor

    def forward(self, images, orig_target_sizes):
        return self.postprocessor(self.model(images), orig_target_sizes)


def main():
    args = parse_args()
    config = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    model, postprocessor = build(args.framework, config, checkpoint)
    torch.manual_seed(123)
    wrapper = DeployModel(model, postprocessor).eval()
    inputs = (
        torch.rand(1, 3, 640, 640),
        torch.tensor([[640, 640]]),
    )
    input_names = ["images", "orig_target_sizes"]
    output_names = ["labels", "boxes", "scores"]
    with torch.inference_mode():
        wrapper(*inputs)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        inputs,
        str(output),
        input_names=input_names,
        output_names=output_names,
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    import onnx
    exported = onnx.load(str(output))
    onnx.checker.check_model(exported)
    print(f"exported {output} ({output.stat().st_size / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()
