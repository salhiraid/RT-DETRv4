"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.utils.data

import torchvision

from PIL import Image
import faster_coco_eval
import faster_coco_eval.core.mask as coco_mask
from ._dataset import DetDataset
from .._misc import convert_to_tv_tensor
from ...core import register

torchvision.disable_beta_transforms_warning()
faster_coco_eval.init_as_pycocotools()
Image.MAX_IMAGE_PIXELS = None

__all__ = ['CocoDetection', 'MultiCocoDetection', 'WeightedMultiDataset']
NUM_KEYPOINTS = 31


@register()
class CocoDetection(torchvision.datasets.CocoDetection, DetDataset):
    __inject__ = ['transforms', ]
    __share__ = ['remap_mscoco_category', 'class_names', 'num_keypoints']

    def __init__(self, img_folder, ann_file, transforms, return_masks=False,
                 remap_mscoco_category=False, class_names=None,
                 filter_unknown_categories=True, num_keypoints=NUM_KEYPOINTS):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, num_keypoints)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category
        self.class_names = list(class_names) if class_names is not None else None
        self.filter_unknown_categories = filter_unknown_categories
        if self.class_names is not None:
            if len(set(self.class_names)) != len(self.class_names):
                raise ValueError('class_names must not contain duplicates')
            unknown = set(self.category2name.values()) - set(self.class_names)
            if unknown and not self.filter_unknown_categories:
                raise ValueError(
                    f'{ann_file} contains categories absent from class_names: '
                    f'{sorted(unknown)}')

    def __getitem__(self, idx):
        img, target = self.load_item(idx)
        if self._transforms is not None:
            img, target, _ = self._transforms(img, target, self)
        return img, target

    def load_item(self, idx):
        image, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}

        if self.class_names is not None:
            name_to_label = {name: index for index, name in enumerate(self.class_names)}
            category2label = {
                category_id: name_to_label[name]
                for category_id, name in self.category2name.items()
                if name in name_to_label
            }
            image, target = self.prepare(image, target, category2label=category2label)
        elif self.remap_mscoco_category:
            image, target = self.prepare(image, target, category2label=mscoco_category2label)
        else:
            image, target = self.prepare(image, target)

        target['idx'] = torch.tensor([idx])

        if 'boxes' in target:
            target['boxes'] = convert_to_tv_tensor(target['boxes'], key='boxes', spatial_size=image.size[::-1])

        if 'masks' in target:
            target['masks'] = convert_to_tv_tensor(target['masks'], key='masks')

        return image, target

    def extra_repr(self) -> str:
        s = f' img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n'
        s += f' return_masks: {self.return_masks}\n'
        if hasattr(self, '_transforms') and self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'
        if hasattr(self, '_preset') and self._preset is not None:
            s += f' preset:\n   {repr(self._preset)}'
        return s

    @property
    def categories(self, ):
        return self.coco.dataset['categories']

    @property
    def category2name(self, ):
        return {cat['id']: cat['name'] for cat in self.categories}

    @property
    def category2label(self, ):
        return {cat['id']: i for i, cat in enumerate(self.categories)}

    @property
    def label2category(self, ):
        if self.class_names is not None:
            name_to_category = {cat['name']: cat['id'] for cat in self.categories}
            return {index: name_to_category[name] for index, name in enumerate(self.class_names)
                    if name in name_to_category}
        return {i: cat['id'] for i, cat in enumerate(self.categories)}


@register()
class MultiCocoDetection(DetDataset):
    """Concatenate multiple COCO datasets while retaining Mosaic ``load_item``.

    Dataset sizes determine their natural sampling ratio. ``repeat_factors`` can
    be used to oversample smaller datasets by an integer factor.
    """
    __inject__ = ['transforms']
    __share__ = ['remap_mscoco_category', 'class_names']

    def __init__(self, img_folders, ann_files, transforms,
                 repeat_factors=None, return_masks=False,
                 remap_mscoco_category=False, class_names=None,
                 filter_unknown_categories=True,
                 img_folder=None, ann_file=None):
        if len(img_folders) != len(ann_files):
            raise ValueError('img_folders and ann_files must have the same length')
        if not img_folders:
            raise ValueError('MultiCocoDetection requires at least one dataset')
        factors = repeat_factors or [1] * len(img_folders)
        if len(factors) != len(img_folders) or any(int(x) != x or x < 1 for x in factors):
            raise ValueError('repeat_factors must contain one positive integer per dataset')
        self._transforms = transforms
        self.datasets = [
            CocoDetection(folder, annotation, transforms=None,
                          return_masks=return_masks,
                          remap_mscoco_category=remap_mscoco_category,
                          class_names=class_names,
                          filter_unknown_categories=filter_unknown_categories)
            for folder, annotation in zip(img_folders, ann_files)
        ]
        self.class_names = list(class_names) if class_names is not None else None
        if self.class_names is None:
            reference_categories = self.datasets[0].categories
            for dataset in self.datasets[1:]:
                if dataset.categories != reference_categories:
                    raise ValueError(
                        'Datasets use different COCO categories. Set class_names '
                        'to the canonical class-name list to remap category ids by name.')
        self.index_map = [
            (dataset_index, sample_index)
            for dataset_index, (dataset, factor) in enumerate(zip(self.datasets, factors))
            for _ in range(int(factor)) for sample_index in range(len(dataset))
        ]
        self.remap_mscoco_category = remap_mscoco_category

    def __len__(self):
        return len(self.index_map)

    def load_item(self, index):
        dataset_index, sample_index = self.index_map[index]
        return self.datasets[dataset_index].load_item(sample_index)

    def __getitem__(self, index):
        image, target = self.load_item(index)
        if self._transforms is not None:
            image, target, _ = self._transforms(image, target, self)
        return image, target

    @property
    def categories(self):
        if self.class_names is not None:
            return [{'id': index, 'name': name}
                    for index, name in enumerate(self.class_names)]
        return self.datasets[0].categories


@register()
class WeightedMultiDataset(DetDataset):
    """Sample child datasets with fixed probabilities for a virtual epoch.

    The mapping is rebuilt deterministically by :meth:`set_epoch`. Child
    datasets deliberately receive their own configuration (normally
    ``transforms: null``); the shared augmentation pipeline belongs here so
    Mosaic and similar transforms can sample through this wrapper.
    """
    __inject__ = ['datasets', 'transforms']

    def __init__(self, datasets, weights, samples_per_epoch, seed=0,
                 transforms=None):
        if not datasets:
            raise ValueError('WeightedMultiDataset requires at least one dataset')
        if len(weights) != len(datasets):
            raise ValueError('weights must contain one value per dataset')
        probabilities = torch.as_tensor(weights, dtype=torch.float64)
        if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
            raise ValueError('weights must be finite and non-negative')
        if probabilities.sum() <= 0:
            raise ValueError('weights must have a positive sum')
        if int(samples_per_epoch) != samples_per_epoch or samples_per_epoch <= 0:
            raise ValueError('samples_per_epoch must be a positive integer')
        if any(len(dataset) == 0 for dataset in datasets):
            raise ValueError('WeightedMultiDataset does not support empty datasets')

        self.datasets = list(datasets)
        self.weights = probabilities / probabilities.sum()
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self._transforms = transforms
        self.set_epoch(0)

    def __len__(self):
        return self.samples_per_epoch

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        generator = torch.Generator().manual_seed(self.seed + int(epoch))
        choices = torch.multinomial(
            self.weights, self.samples_per_epoch, replacement=True,
            generator=generator)
        self.index_map = []
        for dataset_index in choices.tolist():
            sample_index = torch.randint(
                len(self.datasets[dataset_index]), (1,), generator=generator).item()
            self.index_map.append((dataset_index, sample_index))
        for dataset in self.datasets:
            if hasattr(dataset, 'set_epoch'):
                dataset.set_epoch(epoch)

    def load_item(self, index):
        dataset_index, sample_index = self.index_map[index]
        dataset = self.datasets[dataset_index]
        if hasattr(dataset, 'load_item'):
            return dataset.load_item(sample_index)
        return dataset[sample_index]

    def __getitem__(self, index):
        image, target = self.load_item(index)
        if self._transforms is not None:
            image, target, _ = self._transforms(image, target, self)
        return image, target

    @property
    def categories(self):
        return self.datasets[0].categories


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, num_keypoints=NUM_KEYPOINTS):
        self.return_masks = return_masks
        self.num_keypoints = num_keypoints

    def __call__(self, image: Image.Image, target, **kwargs):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        category2label = kwargs.get('category2label', None)
        if category2label is not None:
            # Match MMDetection semantics: annotations whose category is not in
            # the configured class subset (for example `person` in a vehicle
            # task) are ignored rather than treated as a configuration error.
            anno = [obj for obj in anno if obj['category_id'] in category2label]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        if category2label is not None:
            labels = [category2label[obj["category_id"]] for obj in anno]
        else:
            labels = [obj["category_id"] for obj in anno]

        labels = torch.tensor(labels, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        # Every bbox has pose fields.  Bbox-only objects are not dropped: their
        # placeholders are ignored exclusively by pose losses.
        keypoints, keypoints_visible, ignore_keypoints = [], [], []
        for obj in anno:
            raw = obj.get('keypoints')
            if raw:
                kp = torch.as_tensor(raw, dtype=torch.float32).reshape(-1, 3)
                if kp.shape[0] != self.num_keypoints:
                    raise ValueError(
                        f'Expected {self.num_keypoints} keypoints, got {kp.shape[0]}')
                keypoints.append(kp[:, :2])
                keypoints_visible.append(kp[:, 2])
                ignore_keypoints.append(False)
            else:
                keypoints.append(torch.zeros(self.num_keypoints, 2))
                keypoints_visible.append(torch.zeros(self.num_keypoints))
                ignore_keypoints.append(True)
        keypoints = (torch.stack(keypoints) if keypoints else
                     torch.zeros(0, self.num_keypoints, 2))
        keypoints_visible = (torch.stack(keypoints_visible) if keypoints_visible else
                             torch.zeros(0, self.num_keypoints))
        ignore_keypoints = torch.tensor(ignore_keypoints, dtype=torch.bool)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]
        if self.return_masks:
            masks = masks[keep]
        keypoints = keypoints[keep]
        keypoints_visible = keypoints_visible[keep]
        ignore_keypoints = ignore_keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        target["keypoints"] = keypoints
        target["keypoints_visible"] = keypoints_visible
        target["ignore_keypoints"] = ignore_keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(w), int(h)])
        # target["size"] = torch.as_tensor([int(w), int(h)])

        return image, target


mscoco_category2name = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}

mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}
