"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import sys
import math
from contextlib import nullcontext
from typing import Iterable

import torch
import torch.amp
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils

def _compute_encoder_transformer_grad_percentage(model: torch.nn.Module) -> float:
    """Compute percentage of gradients attributed to encoder transformer only.
    This avoids collecting/printing any other stats for speed.
    """
    total_l1 = 0.0
    enc_l1 = 0.0
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        val = grad.detach().abs().sum().item()
        total_l1 += val
        # Support both DDP ('module.') and non-DDP naming
        if name.startswith('module.encoder.encoder'):
            enc_l1 += val
    if total_l1 <= 0.0 or not math.isfinite(total_l1):
        return 0.0
    return 100.0 * enc_l1 / total_l1


def train_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    print_freq = kwargs.get('print_freq', 10)
    writer: SummaryWriter = kwargs.get('writer', None)
    ema: ModelEMA = kwargs.get('ema', None)
    scaler: GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler: Warmup = kwargs.get('lr_warmup_scheduler', None)
    teacher_model = kwargs.get('teacher_model', None)
    batch_augments = kwargs.get('batch_augments', ())
    accumulate_steps = int(kwargs.get('accumulate_steps', 1))
    if accumulate_steps < 1:
        raise ValueError('accumulate_steps must be at least 1')

    encoder_grad_percentages = []
    updates_per_epoch = math.ceil(len(data_loader) / accumulate_steps)
    cur_iters = epoch * updates_per_epoch
    optimizer.zero_grad()

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))
        for augment in batch_augments:
            samples, targets = augment(samples, targets, global_step)

        window_start = (i // accumulate_steps) * accumulate_steps
        window_size = min(accumulate_steps, len(data_loader) - window_start)
        should_step = (i + 1) % accumulate_steps == 0 or i + 1 == len(data_loader)
        sync_context = model.no_sync() if not should_step and hasattr(model, 'no_sync') else nullcontext()

        with sync_context:
            teacher_output = None
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_output = teacher_model(samples).detach()

            if scaler is not None:
                with torch.autocast(device_type=str(device), cache_enabled=True):
                    outputs = model(samples, targets=targets,
                                    teacher_encoder_output=teacher_output)
                if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
                    state = {key.replace('module.', ''): value
                             for key, value in model.state_dict().items()}
                    dist_utils.save_on_master({'model': state}, "./NaN.pth")
                with torch.autocast(device_type=str(device), enabled=False):
                    loss_dict = criterion(outputs, targets, **metas)
                loss = sum(loss_dict.values())
                scaler.scale(loss / window_size).backward()
            else:
                outputs = model(samples, targets=targets,
                                teacher_encoder_output=teacher_output)
                loss_dict = criterion(outputs, targets, **metas)
                loss = sum(loss_dict.values())
                (loss / window_size).backward()

        if should_step:
            if scaler is not None and max_norm > 0:
                scaler.unscale_(optimizer)
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            if dist_utils.is_main_process() and hasattr(criterion, 'distill_adaptive_params') and \
               criterion.distill_adaptive_params and \
               criterion.distill_adaptive_params.get('enabled', False):
                encoder_grad_percentages.append(_compute_encoder_transformer_grad_percentage(model))

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

            if ema is not None:
                ema.update(model)
            if self_lr_scheduler:
                update_index = cur_iters + i // accumulate_steps
                optimizer = lr_scheduler.step(update_index, optimizer)
            elif lr_warmup_scheduler is not None:
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
            for j, group in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', group['lr'], global_step)
            for key, value in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{key}', value.item(), global_step)

    metric_logger.synchronize_between_processes()
    return {key: meter.global_avg for key, meter in metric_logger.meters.items()}, encoder_grad_percentages


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

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

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
        if hasattr(coco_evaluator, 'vehicle_metrics'):
            stats.update(coco_evaluator.vehicle_metrics)

    return stats, coco_evaluator
