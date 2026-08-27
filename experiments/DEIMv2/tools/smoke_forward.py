"""Build DEIMv2-S and run one inference forward pass."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from engine.core import YAMLConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-r', '--resume')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    config = Path(args.config).resolve()
    resume = Path(args.resume).resolve() if args.resume else None
    os.chdir(ROOT)

    cfg = YAMLConfig(str(config), resume=None, tuning=None)
    model = cfg.model.eval().to(args.device)
    if resume:
        checkpoint = torch.load(resume, map_location='cpu')
        state = checkpoint.get('ema', {}).get('module', checkpoint.get('model', checkpoint))
        current = model.state_dict()
        matched = {key: value for key, value in state.items()
                   if key in current and value.shape == current[key].shape}
        model.load_state_dict(matched, strict=False)
        print(f'checkpoint tensors: {len(matched)}/{len(current)} matched')

    height, width = cfg.yaml_cfg['eval_spatial_size']
    images = torch.rand(1, 3, height, width, device=args.device)
    with torch.inference_mode():
        outputs = model(images)
    print('pred_logits:', tuple(outputs['pred_logits'].shape))
    print('pred_boxes:', tuple(outputs['pred_boxes'].shape))


if __name__ == '__main__':
    main()
