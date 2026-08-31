"""Vehicle pose losses adapted from MMPose for the native RT-DETRv4 registry."""
import runpy
from typing import Optional

import torch
import torch.nn as nn

from ..core import register


@register()
class OKSLoss(nn.Module):
    """Object Keypoint Similarity loss compatible with MMPose's OKSLoss API."""
    def __init__(self, metainfo: Optional[str] = None, reduction='mean',
                 mode='linear', eps=1e-8, norm_target_weight=False,
                 loss_weight=1.0):
        super().__init__()
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f'Unsupported reduction: {reduction}')
        if mode not in ('linear', 'square', 'log'):
            raise ValueError(f'Unsupported OKS mode: {mode}')
        self.reduction, self.mode, self.eps = reduction, mode, eps
        self.norm_target_weight, self.loss_weight = norm_target_weight, loss_weight
        if metainfo:
            info = runpy.run_path(metainfo)
            info = info.get('dataset_info', info.get('metainfo', info))
            sigmas = info.get('sigmas')
            joint_weights = info.get('dataset_keypoint_weights')
            if sigmas is not None:
                self.register_buffer('sigmas', torch.as_tensor(sigmas, dtype=torch.float))
            if joint_weights is not None:
                self.register_buffer('joint_weights', torch.as_tensor(joint_weights, dtype=torch.float))

    def forward(self, output, target, target_weight=None, areas=None):
        output, target = output.float(), target.float()
        if target_weight is None:
            target_weight = output.new_ones(output.shape[:-1])
        visibility = (target_weight > 0).float()
        weight = visibility
        if hasattr(self, 'joint_weights'):
            weight = weight * self.joint_weights.to(weight).reshape(1, -1)
        has_keypoints = weight.sum(-1) > 0
        distance = torch.linalg.vector_norm(output - target, dim=-1)
        if areas is not None:
            distance = distance / areas.float().sqrt().clamp(min=self.eps).unsqueeze(-1)
        if hasattr(self, 'sigmas'):
            distance = distance / (self.sigmas.to(distance).reshape(1, -1) * 2)
        oks = torch.exp(-distance.square() / 2)
        denominator = weight.sum(-1, keepdim=True).clamp(min=self.eps)
        if self.norm_target_weight:
            weight = weight / denominator
        else:
            weight = weight / denominator
        oks = (oks * weight).sum(-1)
        if self.mode == 'linear':
            loss = 1 - oks
        elif self.mode == 'square':
            loss = 1 - oks.square()
        else:
            loss = -oks.clamp(min=self.eps).log()
        loss = loss[has_keypoints]
        if loss.numel() == 0:
            return output.sum() * 0
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss * self.loss_weight
