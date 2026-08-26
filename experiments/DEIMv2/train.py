"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
from pathlib import Path

from engine.misc import dist_utils
from engine.core import YAMLConfig, yaml_utils
from engine.solver import TASKS
from common.dataset_protocol import apply_detr_protocol_overrides

debug=False

if debug:
    import torch
    def custom_repr(self):
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr


def _resolve_tuning_checkpoint(cfg):
    tuning = getattr(cfg, 'tuning', None)
    if not tuning:
        return
    path = Path(tuning)
    if not path.is_absolute():
        path = (Path(__file__).resolve().parent / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'tuning checkpoint not found: {path}')
    cfg.tuning = str(path)
    cfg.yaml_cfg['tuning'] = str(path)

def main(args, ) -> None:
    """main
    """
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() \
        if k not in ['update', 'dataset_protocol'] and v is not None})
    apply_detr_protocol_overrides(
        update_dict, args.config, args.dataset_protocol
    )

    cfg = YAMLConfig(args.config, **update_dict)
    _resolve_tuning_checkpoint(cfg)

    if args.resume and cfg.tuning:
        raise RuntimeError('Use either resume or tuning, not both.')

    if args.resume or args.tuning:
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    print('cfg: ', cfg.__dict__)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # priority 0
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, help='resume from checkpoint')
    parser.add_argument('-t', '--tuning', type=str, help='tuning from checkpoint')
    parser.add_argument('-d', '--device', type=str, help='device',)
    parser.add_argument('--seed', type=int, default=0, help='exp reproducibility')
    parser.add_argument('--use-amp', action='store_true', help='auto mixed precision training')
    parser.add_argument('--output-dir', type=str, help='output directoy')
    parser.add_argument('--summary-dir', type=str, help='tensorboard summry')
    parser.add_argument('--test-only', action='store_true', default=False,)
    protocol = parser.add_mutually_exclusive_group()
    protocol.add_argument('--dairv2x', dest='dataset_protocol', action='store_const', const='dairv2x')
    protocol.add_argument('--uadetrac', dest='dataset_protocol', action='store_const', const='uadetrac')

    # priority 1
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')

    # env
    parser.add_argument('--print-method', type=str, default='builtin', help='print method')
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')

    parser.add_argument('--local-rank', type=int, help='local rank id')
    args = parser.parse_args()

    main(args)
