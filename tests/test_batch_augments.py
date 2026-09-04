import random

import pytest
import torch

from engine.data.batch_augments import BatchSyncRandomResize


def test_batch_sync_random_resize_reuses_shape_during_interval():
    random.seed(3)
    augment = BatchSyncRandomResize(
        random_sizes=[480, 640], interval=2, size_divisor=32)
    images = torch.rand(2, 3, 320, 640)
    targets = [{'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
                'size': torch.tensor([320, 640])} for _ in range(2)]

    resized, resized_targets = augment(images, targets, iteration=0)
    first_shape = resized.shape[-2:]
    resized_again, _ = augment(images, targets, iteration=1)

    assert resized_again.shape[-2:] == first_shape
    assert first_shape[1] == first_shape[0] * 2
    assert tuple(resized_targets[0]['size'].tolist()) == first_shape
    # Coordinates are normalized by the input pipeline and must not be scaled.
    assert torch.equal(resized_targets[0]['boxes'], targets[0]['boxes'])


def test_batch_sync_random_resize_validates_configuration():
    with pytest.raises(ValueError, match='exactly one'):
        BatchSyncRandomResize()
    with pytest.raises(ValueError, match='divisible'):
        BatchSyncRandomResize(random_sizes=[500], size_divisor=32)
