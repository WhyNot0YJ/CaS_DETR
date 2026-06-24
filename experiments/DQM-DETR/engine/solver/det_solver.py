"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 D-FINE authors. All Rights Reserved.
"""

import time
import json
import datetime

import torch

from ..misc import dist_utils, stats

from ._solver import BaseSolver


def _load_train_end_vis():
    import importlib.util
    from pathlib import Path

    p = Path(__file__).resolve().parents[3] / "common" / "train_end_inference_vis.py"
    if not p.is_file():
        return None, None
    try:
        spec = importlib.util.spec_from_file_location("train_end_inference_vis", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.cfg_dict_for_vis, mod.maybe_run_train_end_vis
    except Exception:
        return None, None


_cfg_dict_for_vis, _maybe_run_train_end_vis = _load_train_end_vis()
from .det_engine import train_one_epoch, evaluate
from ..optim.lr_scheduler import FlatCosineLRScheduler


# COCO bbox stats vector layout (faster_coco_eval): [0]=mAP@[.5:.95], [1]=mAP@.5, [2]=mAP@.75, ...
_COCO_MAP_5095_IDX = 0
_COCO_MAP_50_IDX = 1


def _format_weather_summary(test_stats: dict, prefix: str = "Weather") -> str | None:
    """One-line summary of per-weather mAP50/mAP95, or None if no weather subsets present."""
    keys = sorted(
        k for k in test_stats
        if k.startswith('weather_') and k.endswith('_coco_eval_bbox')
    )
    if not keys:
        return None
    parts = []
    for k in keys:
        name = k[len('weather_'):-len('_coco_eval_bbox')]
        v = test_stats[k]
        parts.append(f"{name}: mAP50={v[_COCO_MAP_50_IDX]:.4f} mAP95={v[_COCO_MAP_5095_IDX]:.4f}")
    return f"[{prefix}] " + " | ".join(parts)


class DetSolver(BaseSolver):

    def fit(self, ):
        self.train()
        args = self.cfg

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-"*42 + "Start training" + "-"*43)

        self.self_lr_scheduler = False
        if args.lrsheduler is not None:
            iter_per_epoch = len(self.train_dataloader)
            print("     ## Using Self-defined Scheduler-{} ## ".format(args.lrsheduler))
            self.lr_scheduler = FlatCosineLRScheduler(self.optimizer, args.lr_gamma, iter_per_epoch, total_epochs=args.epoches, 
                                                warmup_iter=args.warmup_iter, flat_epochs=args.flat_epoch, no_aug_epochs=args.no_aug_epoch)
            self.self_lr_scheduler = True
        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        top1 = 0
        best_stat = {'epoch': -1, }
        # evaluate again before resume training
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )
            for k in test_stats:
                if k.startswith('weather_'):
                    continue
                best_stat['epoch'] = self.last_epoch
                best_stat[k] = test_stats[k][0]
                top1 = test_stats[k][0]
                print(f'best_stat: {best_stat}')

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            train_stats = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model, 
                self.criterion, 
                self.train_dataloader, 
                self.optimizer, 
                self.device, 
                epoch, 
                max_norm=args.clip_max_norm, 
                print_freq=args.print_freq, 
                ema=self.ema, 
                scaler=self.scaler, 
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer
            )

            if not self.self_lr_scheduler:  # update by epoch 
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir and epoch < self.train_dataloader.collate_fn.stop_epoch:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            ws_cfg = self.cfg.yaml_cfg.get('weather_subset_eval') if hasattr(self.cfg, 'yaml_cfg') else None
            ws_every = int(ws_cfg.get('every_n_epochs', 1)) if ws_cfg else 1
            compute_ws = ws_every <= 1 or (epoch + 1) % ws_every == 0
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                compute_weather_subsets=compute_ws,
            )

            weather_line = _format_weather_summary(test_stats)
            if weather_line and dist_utils.is_main_process():
                print(weather_line)

            # TODO
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)

                if k.startswith('weather_'):
                    continue

                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat[k] > top1:
                    best_stat_print['epoch'] = epoch
                    top1 = best_stat[k]
                    if self.output_dir:
                        if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
                        else:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')

                best_stat_print[k] = max(best_stat[k], top1)
                print(f'best_stat: {best_stat_print}')  # global best

                if best_stat['epoch'] == epoch and self.output_dir:
                    if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        if test_stats[k][0] > top1:
                            top1 = test_stats[k][0]
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
                    else:
                        top1 = max(test_stats[k][0], top1)
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')

                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    best_stat = {'epoch': -1, }
                    self.ema.decay -= 0.0001
                    self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                    print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')


            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            cd_cfg = self.cfg.yaml_cfg.get('cross_domain_eval') if hasattr(self.cfg, 'yaml_cfg') else None
            cd_every = int(cd_cfg.get('every_n_epochs', 0)) if cd_cfg and cd_cfg.get('enable', True) else 0
            if cd_every > 0 and (epoch + 1) % cd_every == 0 and dist_utils.is_main_process():
                module = self.ema.module if self.ema else self.model
                cd_stats = self._run_cross_domain_eval(module, epoch=epoch, dump_json=False)
                if cd_stats:
                    log_stats.update({f'cd_{k}': v for k, v in cd_stats.items()})

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        self._maybe_run_cross_domain_eval()

        if _maybe_run_train_end_vis is not None:
            _maybe_run_train_end_vis(
                dist_utils.is_main_process(),
                self.ema.module if self.ema else self.model,
                self.postprocessor,
                self.val_dataloader,
                self.device,
                self.output_dir,
                _cfg_dict_for_vis(self.cfg),
            )

    def _maybe_run_cross_domain_eval(self):
        """If `cross_domain_eval: {config: ..., enable: true}` is set in cfg, run evaluate
        once on a different val_dataloader/evaluator and dump stats. No-op otherwise.

        Used at the end of training. The per-epoch hook calls _run_cross_domain_eval directly.
        """
        cd_cfg = self.cfg.yaml_cfg.get('cross_domain_eval') if hasattr(self.cfg, 'yaml_cfg') else None
        if not cd_cfg or not cd_cfg.get('enable', True):
            return
        if not dist_utils.is_main_process():
            return

        # Final pass: prefer best_stg2 weights if present; fall back to current EMA.
        module = self._final_eval_module()
        self._run_cross_domain_eval(module, epoch=None, dump_json=True)

    def _final_eval_module(self):
        """Return a model module loaded from best_stg2.pth if it exists, else current EMA/model."""
        best_path = self.output_dir / 'best_stg2.pth' if self.output_dir else None
        live_module = self.ema.module if self.ema else self.model
        if not best_path or not best_path.is_file():
            print('[CrossDomain] best_stg2.pth not found, using live EMA for final eval')
            return live_module

        try:
            state = torch.load(str(best_path), map_location='cpu')
        except Exception as e:
            print(f'[CrossDomain] failed to load {best_path}: {e}, using live EMA')
            return live_module

        if self.ema is not None and 'ema' in state:
            from copy import deepcopy
            ema_clone = deepcopy(dist_utils.de_parallel(self.ema))
            ema_clone.load_state_dict(state['ema'])
            print(f'[CrossDomain] loaded EMA weights from {best_path.name}')
            return ema_clone.module
        if 'model' in state:
            from copy import deepcopy
            model_clone = deepcopy(dist_utils.de_parallel(self.model))
            model_clone.load_state_dict(state['model'])
            model_clone.eval()
            print(f'[CrossDomain] loaded model weights from {best_path.name}')
            return model_clone
        print(f'[CrossDomain] no ema/model key in {best_path.name}, using live EMA')
        return live_module

    def _run_cross_domain_eval(self, module, epoch=None, dump_json=False):
        """Core cross-domain eval. Caller is responsible for cfg gating and main-process check."""
        cd_cfg = self.cfg.yaml_cfg.get('cross_domain_eval') if hasattr(self.cfg, 'yaml_cfg') else None
        if not cd_cfg:
            return None

        cd_yaml = cd_cfg.get('config')
        if not cd_yaml:
            print('[CrossDomain] cross_domain_eval.config not set, skipping')
            return None

        from pathlib import Path
        cd_path = Path(cd_yaml)
        if not cd_path.is_absolute():
            cd_path = (Path(__file__).resolve().parents[2] / cd_path).resolve()
        if not cd_path.is_file():
            print(f'[CrossDomain] config not found: {cd_path}, skipping')
            return None

        from ..core import YAMLConfig
        cd = YAMLConfig(str(cd_path))
        cd_loader = dist_utils.warp_loader(cd.val_dataloader, shuffle=cd.val_dataloader.shuffle)
        cd_evaluator = cd.evaluator

        tag = f'epoch={epoch}' if epoch is not None else 'final'
        print(f'[CrossDomain][{tag}] running eval with config={cd_path.name}, '
              f'val_anns={cd.yaml_cfg["val_dataloader"]["dataset"]["ann_file"]}')
        cd_stats, cd_coco_evaluator = evaluate(
            module, self.criterion, self.postprocessor,
            cd_loader, cd_evaluator, self.device,
        )

        weather_line = _format_weather_summary(cd_stats, prefix=f'Weather/CrossDomain[{tag}]')
        if weather_line:
            print(weather_line)
        if 'coco_eval_bbox' in cd_stats:
            v = cd_stats['coco_eval_bbox']
            print(f'[CrossDomain][{tag}] Overall mAP50={v[_COCO_MAP_50_IDX]:.4f} '
                  f'mAP95={v[_COCO_MAP_5095_IDX]:.4f}')

        if dump_json and self.output_dir:
            dump_path = self.output_dir / 'cross_domain_eval.json'
            with dump_path.open('w') as f:
                json.dump(cd_stats, f, indent=2)
            print(f'[CrossDomain] saved {dump_path}')

        return cd_stats

    def val(self, ):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)

        weather_line = _format_weather_summary(test_stats)
        if weather_line and dist_utils.is_main_process():
            print(weather_line)

        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return
