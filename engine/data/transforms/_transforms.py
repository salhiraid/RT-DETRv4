"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

import PIL
import PIL.Image

from typing import Any, Dict, List, Optional

from .._misc import convert_to_tv_tensor, _boxes_keys
from .._misc import Image, Video, Mask, BoundingBoxes
from .._misc import SanitizeBoundingBoxes as TVSanitizeBoundingBoxes

from ...core import register
torchvision.disable_beta_transforms_warning()


def _spatial_size(value):
    getter = F.get_size if hasattr(F, 'get_size') else F.get_spatial_size
    return getter(value)


RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
# ToImageTensor = register()(T.ToImageTensor)
# ConvertDtype = register()(T.ConvertDtype)
# PILToTensor = register()(T.PILToTensor)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)


@register(name='SanitizeBoundingBoxes')
class SanitizeBoundingBoxes(TVSanitizeBoundingBoxes):
    def forward(self, image, target=None, *extra):
        if target is None or 'boxes' not in target:
            return super().forward(image, target, *extra)
        boxes = target['boxes']
        xyxy = torchvision.ops.box_convert(boxes.as_subclass(torch.Tensor),
                                           in_fmt=boxes.format.value.lower(), out_fmt='xyxy')
        keep = ((xyxy[:, 2] - xyxy[:, 0] >= self.min_size) &
                (xyxy[:, 3] - xyxy[:, 1] >= self.min_size))
        pose = {key: target.pop(key) for key in
                ('keypoints', 'keypoints_visible', 'ignore_keypoints') if key in target}
        output = super().forward(image, target)
        image, target = output
        for key, value in pose.items():
            target[key] = value[keep]
        return (image, target, *extra) if extra else (image, target)


@register()
class FillKeypoints(T.Transform):
    """Enforce aligned 31-point placeholders without removing bbox-only GT."""
    def __init__(self, num_keypoints=31):
        super().__init__()
        self.num_keypoints = num_keypoints

    def forward(self, image, target, *extra):
        n = len(target.get('boxes', []))
        if 'keypoints' not in target or target['keypoints'].ndim != 3:
            target['keypoints'] = torch.zeros(n, self.num_keypoints, 2)
            target['keypoints_visible'] = torch.zeros(n, self.num_keypoints)
            target['ignore_keypoints'] = torch.ones(n, dtype=torch.bool)
        if not (n == len(target['labels']) == len(target['keypoints']) ==
                len(target['keypoints_visible']) == len(target['ignore_keypoints'])):
            raise RuntimeError('bbox/keypoint instance fields are not aligned')
        return (image, target, *extra) if extra else (image, target)


@register()
class Resize(T.Resize):
    def forward(self, image, target=None, *extra):
        if target is None:
            return super().forward(image)
        old_h, old_w = _spatial_size(image)
        keypoints = target.pop('keypoints', None)
        output = super().forward(image, target)
        image, target = output
        new_h, new_w = _spatial_size(image)
        if keypoints is not None:
            keypoints = keypoints.clone()
            valid = target['keypoints_visible'] > 0
            keypoints[..., 0][valid] *= new_w / old_w
            keypoints[..., 1][valid] *= new_h / old_h
            target['keypoints'] = keypoints
        return (image, target, *extra) if extra else (image, target)


@register()
class RandomHorizontalFlip(T.RandomHorizontalFlip):
    """Flip coordinates only; no semantic permutation is invented."""
    def forward(self, image, target=None, *extra):
        if target is None or torch.rand(1) >= self.p:
            return (image, target, *extra) if target is not None else image
        width = _spatial_size(image)[1]
        keypoints = target.pop('keypoints', None)
        image, target = F.horizontal_flip(image), F.horizontal_flip(target)
        if keypoints is not None:
            keypoints = keypoints.clone()
            valid = target['keypoints_visible'] > 0
            keypoints[..., 0][valid] = width - keypoints[..., 0][valid]
            target['keypoints'] = keypoints
        return (image, target, *extra) if extra else (image, target)


@register()
class EmptyTransform(T.Transform):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register()
class PadToSize(T.Pad):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sp = F.get_spatial_size(flat_inputs[0])
        h, w = self.size[1] - sp[0], self.size[0] - sp[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, size, fill=0, padding_mode='constant') -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)
    
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = self._fill[type(inpt)]
        padding = params['padding']
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)
        return outputs


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        image, target, *extra = inputs
        params = self.make_params([image, target['boxes']])
        if not params:
            return inputs if len(inputs) > 1 else inputs[0]
        crop = dict(top=params['top'], left=params['left'],
                    height=params['height'], width=params['width'])
        image = F.crop(image, **crop)
        target['boxes'] = F.crop(target['boxes'], **crop)
        target['boxes'][~params['is_within_crop_area']] = 0
        if 'masks' in target:
            target['masks'] = F.crop(target['masks'], **crop)
        if 'keypoints' in target:
            keypoints = target['keypoints'].clone()
            visible = target['keypoints_visible'].clone()
            valid = visible > 0
            keypoints[..., 0][valid] -= params['left']
            keypoints[..., 1][valid] -= params['top']
            inside = ((keypoints[..., 0] >= 0) & (keypoints[..., 0] <= params['width']) &
                      (keypoints[..., 1] >= 0) & (keypoints[..., 1] <= params['height']))
            visible[~inside] = 0
            keypoints[~inside] = 0
            target['keypoints'], target['keypoints_visible'] = keypoints, visible
        return (image, target, *extra)


@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (
        BoundingBoxes,
    )
    def __init__(self, fmt='', normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)
    
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(inpt, key='boxes', box_format=self.fmt.upper(), spatial_size=spatial_size)

        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt

    def forward(self, image, target=None, *extra):
        output = super().forward(image, target) if target is not None else super().forward(image)
        if target is not None and self.normalize and 'keypoints' in output[1]:
            image, target = output
            h, w = _spatial_size(image)
            target['keypoints'] = target['keypoints'] / target['keypoints'].new_tensor([w, h])
            target['size'] = torch.as_tensor([w, h], device=target['keypoints'].device)
            output = (image, target)
        return (*output, *extra) if extra else output


@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (
        PIL.Image.Image,
    )
    def __init__(self, dtype='float32', scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)
    
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == 'float32':
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.

        inpt = Image(inpt)

        return inpt
