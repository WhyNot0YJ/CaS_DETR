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
    "dqmdeim": "DQM-DETR",
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
    parser.add_argument("--cass-static-keep-eval", action="store_true")
    parser.add_argument("--cass-eval-keep-ratio", type=float)
    parser.add_argument(
        "--dataset-protocol",
        choices=(
            "dairv2x",
            "uadetrac",
        ),
    )
    return parser.parse_args()


def build(
    framework,
    config_path,
    checkpoint_path,
    dataset_protocol=None,
    cass_static_keep_eval=False,
    cass_eval_keep_ratio=None,
):
    experiments_dir = Path(__file__).resolve().parents[3]
    framework_dir = experiments_dir / FRAMEWORK_DIRS[framework]
    sys.path.insert(0, str(experiments_dir))
    sys.path.insert(0, str(framework_dir))
    os.chdir(framework_dir)
    from common.dataset_protocol import apply_detr_protocol_overrides
    if framework in ("dfine", "rtdetr"):
        from src.core import YAMLConfig
    else:
        from engine.core import YAMLConfig

    overrides = {}
    apply_detr_protocol_overrides(
        overrides,
        config_path,
        dataset_protocol,
        rtdetr_layout=framework == "rtdetr",
    )
    cfg = YAMLConfig(str(config_path), **overrides)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("ema", {}).get("module") or checkpoint["model"]
    model = cfg.model
    model.load_state_dict(state, strict=True)
    encoder = getattr(model, "encoder", None)
    if hasattr(encoder, "set_epoch"):
        encoder.set_epoch(int(checkpoint.get("last_epoch", 0)))
    if cass_static_keep_eval:
        if encoder is None or not hasattr(encoder, "cass_static_keep_eval"):
            raise ValueError("model does not support --cass-static-keep-eval")
        encoder.cass_static_keep_eval = True
    if cass_eval_keep_ratio is not None:
        pruner = getattr(encoder, "shared_token_pruner", None)
        if pruner is None or not hasattr(pruner, "keep_ratio"):
            raise ValueError("model does not support --cass-eval-keep-ratio")
        pruner.keep_ratio = float(cass_eval_keep_ratio)
    return model.deploy(), cfg.postprocessor.deploy()


class DeployModel(nn.Module):
    def __init__(self, model, postprocessor):
        super().__init__()
        self.model = model
        self.postprocessor = postprocessor

    def forward(self, images, orig_target_sizes):
        return self.postprocessor(self.model(images), orig_target_sizes)


def main():
    args = parse_args()
    if args.cass_eval_keep_ratio is not None:
        if not args.cass_static_keep_eval:
            raise ValueError("--cass-eval-keep-ratio requires --cass-static-keep-eval")
        if not 0.0 < args.cass_eval_keep_ratio <= 1.0:
            raise ValueError("--cass-eval-keep-ratio must be within (0, 1]")
    config = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    model, postprocessor = build(
        args.framework,
        config,
        checkpoint,
        args.dataset_protocol,
        args.cass_static_keep_eval,
        args.cass_eval_keep_ratio,
    )
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
