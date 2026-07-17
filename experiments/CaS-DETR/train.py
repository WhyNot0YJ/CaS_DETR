"""
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
import copy
import json
import platform
import shutil
import subprocess
from pathlib import Path

import torch
import yaml

from engine.misc import dist_utils
from engine.core import YAMLConfig, yaml_utils
from engine.solver import TASKS

debug=False

if debug:
    import torch
    def custom_repr(self):
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr


def _resolve_tuning_checkpoint(cfg):
    """``tuning`` 可为 yaml 或 ``-t``；相对路径相对 DEIM 目录解析。缺文件则放弃整网微调。"""
    t = getattr(cfg, 'tuning', None)
    if not t:
        return
    root = Path(__file__).resolve().parent
    p = Path(t)
    if not p.is_absolute():
        p = (root / t).resolve()
    p = str(p)
    if os.path.isfile(p):
        cfg.tuning = p
        if getattr(cfg, 'yaml_cfg', None) is not None:
            cfg.yaml_cfg['tuning'] = p
    else:
        print(f'[WARN] tuning checkpoint not found: {p}, use HGNet stage1 only.')
        cfg.tuning = None
        if getattr(cfg, 'yaml_cfg', None) is not None:
            cfg.yaml_cfg.pop('tuning', None)


def _save_run_metadata(cfg, config_path):
    """Save the resolved protocol and runtime identity beside checkpoints."""
    if not dist_utils.is_main_process():
        return

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = copy.deepcopy(cfg.yaml_cfg)
    resolved_config.pop('__include__', None)
    with (output_dir / 'resolved_config.yml').open('w', encoding='utf-8') as f:
        yaml.safe_dump(resolved_config, f, sort_keys=False, allow_unicode=True)
    shutil.copy2(config_path, output_dir / 'source_config.yml')

    def command_output(command):
        try:
            return subprocess.run(
                command, check=False, capture_output=True, text=True
            ).stdout.strip()
        except OSError:
            return ''

    environment = {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'driver': command_output([
            'nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'
        ]),
        'git_commit': command_output(['git', 'rev-parse', 'HEAD']),
        'git_status': command_output(['git', 'status', '--short']),
        'pip_freeze': command_output([sys.executable, '-m', 'pip', 'freeze']).splitlines(),
    }
    with (output_dir / 'environment.json').open('w', encoding='utf-8') as f:
        json.dump(environment, f, indent=2, ensure_ascii=False)


def main(args, ) -> None:
    """main
    """
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() \
        if k not in ['update', ] and v is not None})

    cfg = YAMLConfig(args.config, **update_dict)
    _resolve_tuning_checkpoint(cfg)
    _save_run_metadata(cfg, args.config)

    if args.resume and getattr(cfg, 'tuning', None):
        raise RuntimeError('Use either resume or tuning, not both.')

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    if args.resume or getattr(cfg, 'tuning', None):
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
    parser.add_argument('--seed', type=int, help='exp reproducibility')
    parser.add_argument('--output-dir', type=str, help='output directoy')
    parser.add_argument('--summary-dir', type=str, help='tensorboard summry')
    parser.add_argument('--test-only', action='store_true', default=False,)

    # priority 1
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')

    # env
    parser.add_argument('--print-method', type=str, default='builtin', help='print method')
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')

    parser.add_argument('--local-rank', type=int, help='local rank id')
    args = parser.parse_args()

    main(args)
