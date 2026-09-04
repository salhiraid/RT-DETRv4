"""Batch-level augmentations applied on the training device."""

import random
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import register


@register()
class BatchSyncRandomResize(nn.Module):
    """Randomly resize a whole batch, using the same shape on every rank.

    Exactly one of ``random_size_range`` and ``random_sizes`` must be given.
    Sizes are image heights in pixels; width is derived from the input aspect
    ratio and rounded to ``size_divisor``. A newly sampled shape is retained
    for ``interval`` data-loader iterations.
    """

    def __init__(
        self,
        random_size_range: Optional[Tuple[int, int]] = None,
        interval: int = 10,
        size_divisor: int = 32,
        interpolations: Union[str, Sequence[str]] = 'bilinear',
        random_sizes: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        if (random_size_range is None) == (random_sizes is None):
            raise ValueError('Specify exactly one of random_size_range or random_sizes')
        if interval <= 0 or size_divisor <= 0:
            raise ValueError('interval and size_divisor must be positive')

        if random_size_range is not None:
            lower, upper = random_size_range
            if lower > upper:
                raise ValueError('random_size_range must be ordered')
            sizes = range(
                (lower + size_divisor - 1) // size_divisor,
                upper // size_divisor + 1,
            )
            self._random_sizes = list(sizes)
        else:
            if not random_sizes:
                raise ValueError('random_sizes cannot be empty')
            if any(size % size_divisor for size in random_sizes):
                raise ValueError('Every random size must be divisible by size_divisor')
            self._random_sizes = [size // size_divisor for size in random_sizes]
        if not self._random_sizes:
            raise ValueError('The requested range contains no divisor-aligned size')

        if isinstance(interpolations, str):
            interpolations = [interpolations]
        supported = {'nearest', 'bilinear', 'bicubic', 'area'}
        if not interpolations or any(mode not in supported for mode in interpolations):
            raise ValueError(f'interpolations must be selected from {sorted(supported)}')

        self.interval = interval
        self.size_divisor = size_divisor
        self.interpolations = list(interpolations)
        self._input_size = None
        self._interp = self.interpolations[0]

    def forward(self, inputs, targets, iteration=0):
        height, width = inputs.shape[-2:]
        if self._input_size is None or iteration % self.interval == 0:
            self._input_size, self._interp = self._sample(
                width / height, inputs.device)

        if tuple(inputs.shape[-2:]) != self._input_size:
            kwargs = {}
            if self._interp in {'bilinear', 'bicubic'}:
                kwargs['align_corners'] = False
            inputs = F.interpolate(
                inputs, size=self._input_size, mode=self._interp, **kwargs)
            for target in targets:
                if 'masks' in target:
                    target['masks'] = F.interpolate(
                        target['masks'][:, None].float(),
                        size=self._input_size,
                        mode='nearest',
                    )[:, 0].to(target['masks'].dtype)
                if 'size' in target:
                    target['size'] = target['size'].new_tensor(self._input_size)
        return inputs, targets

    def _sample(self, aspect_ratio, device):
        choice = torch.zeros(3, dtype=torch.long, device=device)
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if distributed else 0
        if rank == 0:
            height_units = random.choice(self._random_sizes)
            width_units = max(1, round(aspect_ratio * height_units))
            choice.copy_(choice.new_tensor([
                height_units * self.size_divisor,
                width_units * self.size_divisor,
                random.randrange(len(self.interpolations)),
            ]))
        if distributed:
            torch.distributed.broadcast(choice, src=0)
        return (choice[0].item(), choice[1].item()), self.interpolations[choice[2].item()]
