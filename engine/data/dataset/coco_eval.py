"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
COCO evaluator that works in distributed mode.
Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import os
import contextlib
import copy
import numpy as np
import torch

from faster_coco_eval import COCO, COCOeval_faster
import faster_coco_eval.core.mask as mask_util
from ...core import register
from ...misc import dist_utils
__all__ = ['CocoEvaluator', 'VehicleCocoEvaluator']


@register()
class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt : COCO = coco_gt
        self.iou_types = iou_types

        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval_faster(coco_gt, iouType=iou_type, print_function=print, separate_eval=True)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def cleanup(self):
        self.coco_eval = {}
        for iou_type in self.iou_types:
            self.coco_eval[iou_type] = COCOeval_faster(self.coco_gt, iouType=iou_type, print_function=print, separate_eval=True)
        self.img_ids = []
        self.eval_imgs = {k: [] for k in self.iou_types}


    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            coco_eval = self.coco_eval[iou_type]

            # suppress pycocotools prints
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = self.coco_gt.loadRes(results) if results else COCO()
                    coco_eval.cocoDt = coco_dt
                    coco_eval.params.imgIds = list(img_ids)
                    coco_eval.evaluate()

            self.eval_imgs[iou_type].append(np.array(coco_eval._evalImgs_cpp).reshape(len(coco_eval.params.catIds), len(coco_eval.params.areaRng), len(coco_eval.params.imgIds)))

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            img_ids, eval_imgs = merge(self.img_ids, self.eval_imgs[iou_type])

            coco_eval = self.coco_eval[iou_type]
            coco_eval.params.imgIds = img_ids
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
            coco_eval._evalImgs_cpp = eval_imgs

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        'keypoints': keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def merge(img_ids, eval_imgs):
    all_img_ids = dist_utils.all_gather(img_ids)
    all_eval_imgs = dist_utils.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.extend(p)


    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, axis=2).ravel()
    # merged_eval_imgs = np.array(merged_eval_imgs).T.ravel()

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)

    return merged_img_ids.tolist(), merged_eval_imgs.tolist()


@register()
class VehicleCocoEvaluator(CocoEvaluator):
    """Run the existing COCO bbox evaluator and vehicle pose metric together."""
    def __init__(self, coco_gt, iou_types=('bbox',), num_keypoints=31,
                 thresholds=(5.0, 10.0), iou_thr=.5, vis_thr=.5,
                 score_thr=.05, margin=.05, crop_size=512,
                 min_bbox_size=64., class_names=None):
        super().__init__(coco_gt, iou_types)
        self.class_names = list(class_names) if class_names is not None else None
        self.label2category = None
        if self.class_names is not None:
            name_to_category = {
                category['name']: category['id']
                for category in self.coco_gt.dataset['categories']
            }
            missing = set(self.class_names) - set(name_to_category)
            if missing:
                raise ValueError(
                    f'Validation JSON is missing configured classes: {sorted(missing)}')
            self.label2category = {
                index: name_to_category[name]
                for index, name in enumerate(self.class_names)
            }
        from .vehicle_keypoint_metric import VehicleKeypointMetric
        self.vehicle_metric = VehicleKeypointMetric(
            num_keypoints, thresholds, iou_thr, vis_thr, score_thr,
            margin, crop_size, min_bbox_size)
        self.vehicle_metrics = {}

    def cleanup(self):
        super().cleanup()
        self.vehicle_metric.results.clear()
        self.vehicle_metrics = {}

    def update(self, predictions):
        if self.label2category is not None:
            predictions = {
                image_id: {
                    **prediction,
                    'labels': prediction['labels'].new_tensor([
                        self.label2category[int(label)]
                        for label in prediction['labels']
                    ])
                }
                for image_id, prediction in predictions.items()
            }
        super().update(predictions)
        samples = []
        for image_id, pred in predictions.items():
            annotations = [ann for ann in self.coco_gt.imgToAnns.get(image_id, [])
                           if not ann.get('iscrowd', 0)]
            boxes, labels, keypoints, visibility = [], [], [], []
            for ann in annotations:
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h])
                labels.append(ann['category_id'])
                raw = ann.get('keypoints')
                if raw:
                    kp = torch.as_tensor(raw, dtype=torch.float).reshape(-1, 3)
                    keypoints.append(kp[:, :2]); visibility.append(kp[:, 2])
                else:
                    keypoints.append(torch.zeros(self.vehicle_metric.num_keypoints, 2))
                    visibility.append(torch.zeros(self.vehicle_metric.num_keypoints))
            gt = {
                'boxes': torch.as_tensor(boxes, dtype=torch.float).reshape(-1, 4),
                'labels': torch.as_tensor(labels, dtype=torch.long),
                'keypoints': (torch.stack(keypoints) if keypoints else
                              torch.zeros(0, self.vehicle_metric.num_keypoints, 2)),
                'keypoints_visible': (torch.stack(visibility) if visibility else
                                      torch.zeros(0, self.vehicle_metric.num_keypoints)),
            }
            samples.append({'pred_instances': pred, 'gt_instances': gt})
        self.vehicle_metric.process(None, samples)

    def synchronize_between_processes(self):
        super().synchronize_between_processes()
        gathered = dist_utils.all_gather(self.vehicle_metric.results)
        self.vehicle_metric.results = [item for rank_results in gathered for item in rank_results]

    def summarize(self):
        super().summarize()
        self.vehicle_metrics = self.vehicle_metric.compute_metrics()
        print('Vehicle keypoint metrics:')
        for name, value in self.vehicle_metrics.items():
            print(f'  {name}: {value:.6f}' if isinstance(value, float) else f'  {name}: {value}')
