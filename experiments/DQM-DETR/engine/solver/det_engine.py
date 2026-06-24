"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""


import sys
import math
from collections import defaultdict
from typing import Iterable

import torch
import torch.amp
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils


def _build_weather_evaluators(coco_evaluator: CocoEvaluator):
    images = coco_evaluator.coco_gt.dataset.get("images", [])
    weather_to_img_ids = defaultdict(set)
    for image in images:
        weather = image.get("weather")
        if weather:
            weather_to_img_ids[str(weather)].add(int(image["id"]))

    evaluators = {}
    for weather, image_ids in sorted(weather_to_img_ids.items()):
        evaluator = coco_evaluator.__class__(coco_evaluator.coco_gt, coco_evaluator.iou_types)
        evaluators[weather] = (image_ids, evaluator)
    return evaluators


def train_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler :Warmup = kwargs.get('lr_warmup_scheduler', None)

    cur_iters = epoch * len(data_loader)

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if isinstance(batch, dict):
            samples = {
                'clean': batch['clean'].to(device),
                'degraded': batch['degraded'].to(device),
            }
            targets = [{k: v.to(device) for k, v in t.items()} for t in batch['targets']]
            paired_batch = True
        else:
            samples, targets = batch
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            paired_batch = False
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                if paired_batch:
                    outputs = {
                        'clean': model(samples['clean'], targets=targets, dqm_branch='clean'),
                        'degraded': model(samples['degraded'], targets=targets, dqm_branch='degraded'),
                    }
                else:
                    outputs = model(samples, targets=targets)

            pred_boxes = outputs['degraded']['pred_boxes'] if paired_batch else outputs['pred_boxes']
            if torch.isnan(pred_boxes).any() or torch.isinf(pred_boxes).any():
                print(pred_boxes)
                state = model.state_dict()
                new_state = {}
                for key, value in model.state_dict().items():
                    # Replace 'module' with 'model' in each key
                    new_key = key.replace('module.', '')
                    # Add the updated key-value pair to the state dictionary
                    state[new_key] = value
                new_state['model'] = state
                dist_utils.save_on_master(new_state, "./NaN.pth")

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            if paired_batch:
                outputs = {
                    'clean': model(samples['clean'], targets=targets, dqm_branch='clean'),
                    'degraded': model(samples['degraded'], targets=targets, dqm_branch='degraded'),
                }
            else:
                outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)

            loss : torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        # ema
        if ema is not None:
            ema.update(model)

        if self_lr_scheduler:
            optimizer = lr_scheduler.step(cur_iters + i, optimizer)
        else:
            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device, compute_weather_subsets: bool = True):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()
    weather_evaluators = (
        _build_weather_evaluators(coco_evaluator)
        if coco_evaluator is not None and compute_weather_subsets
        else {}
    )

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessor(outputs, orig_target_sizes)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)
            for image_ids, weather_evaluator in weather_evaluators.values():
                subset_res = {image_id: output for image_id, output in res.items() if int(image_id) in image_ids}
                if subset_res:
                    weather_evaluator.update(subset_res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        for _, weather_evaluator in weather_evaluators.values():
            weather_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        for weather, (_, weather_evaluator) in weather_evaluators.items():
            print(f"Weather subset: {weather}")
            weather_evaluator.accumulate()
            weather_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
        for weather, (_, weather_evaluator) in weather_evaluators.items():
            key = weather.lower().replace(" ", "_")
            if 'bbox' in iou_types:
                stats[f'weather_{key}_coco_eval_bbox'] = weather_evaluator.coco_eval['bbox'].stats.tolist()
            if 'segm' in iou_types:
                stats[f'weather_{key}_coco_eval_masks'] = weather_evaluator.coco_eval['segm'].stats.tolist()

    return stats, coco_evaluator
