"""Run checkpoint inference over the validation dataset from an RT-DETRv4 config."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from engine.core import YAMLConfig
from engine.misc import dist_utils
from engine.solver import TASKS
from train import configure_evaluation


def _cpu_result(result):
    return {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in result.items()
    }


def serialize_prediction(image_id, image_index, result, score_threshold):
    """Convert one postprocessed prediction to JSON-compatible values."""
    result = _cpu_result(result)
    keep = result['scores'] >= score_threshold
    labels = result['labels'][keep]
    boxes = result['boxes'][keep]
    scores = result['scores'][keep]
    keypoints = result.get('keypoints_xy')
    keypoint_scores = result.get('keypoint_scores')
    if keypoints is not None:
        keypoints = keypoints[keep]
        keypoint_scores = keypoint_scores[keep]

    detections = []
    for detection_index in range(len(scores)):
        box = boxes[detection_index].tolist()
        detection = {
            'label': int(labels[detection_index]),
            'score': float(scores[detection_index]),
            'bbox_xyxy': box,
            'bbox_xywh': [box[0], box[1], box[2] - box[0], box[3] - box[1]],
        }
        if keypoints is not None:
            detection['keypoints_xy'] = keypoints[detection_index].tolist()
            detection['keypoint_scores'] = keypoint_scores[detection_index].tolist()
        detections.append(detection)
    return {
        'index': image_index,
        'image_id': int(image_id),
        'detections': detections,
    }


def draw_prediction(image, result, score_threshold, keypoint_threshold,
                    class_names=None, ground_truth=None):
    """Draw predictions and optional COCO ground truth on an original image.

    Predictions use red boxes and green keypoints; ground truth uses blue boxes
    and cyan keypoints.  Keypoint labels are one-based and only visible points
    are drawn (score threshold for predictions and COCO visibility > 0 for GT).
    """
    result = _cpu_result(result)
    canvas = image.copy()
    painter = ImageDraw.Draw(canvas)
    for label, box, score, keypoints, keypoint_scores in zip(
            result['labels'], result['boxes'], result['scores'],
            result.get('keypoints_xy', [None] * len(result['scores'])),
            result.get('keypoint_scores', [None] * len(result['scores']))):
        if float(score) < score_threshold:
            continue
        box = box.tolist()
        painter.rectangle(box, outline='red', width=3)
        label_index = int(label)
        label_text = (class_names[label_index]
                      if class_names and label_index < len(class_names)
                      else str(label_index))
        painter.text((box[0], box[1]), f'{label_text} {float(score):.3f}',
                     fill='red')
        if keypoints is not None:
            for keypoint_index, (point, point_score) in enumerate(
                    zip(keypoints, keypoint_scores), start=1):
                if float(point_score) >= keypoint_threshold:
                    x, y = point.tolist()
                    radius = 3
                    painter.ellipse(
                        (x - radius, y - radius, x + radius, y + radius),
                        fill='lime', outline='green')
                    painter.text((x + radius + 1, y - radius),
                                 str(keypoint_index), fill='lime')

    for annotation in ground_truth or []:
        x, y, width, height = annotation['bbox']
        painter.rectangle((x, y, x + width, y + height),
                          outline='blue', width=3)
        label = annotation.get('category_name', annotation.get('category_id', 'GT'))
        painter.text((x, y + 12), f'GT {label}', fill='blue')
        raw_keypoints = annotation.get('keypoints') or []
        for keypoint_index in range(len(raw_keypoints) // 3):
            point_x, point_y, visibility = raw_keypoints[keypoint_index * 3:keypoint_index * 3 + 3]
            if visibility <= 0:
                continue
            radius = 3
            painter.ellipse(
                (point_x - radius, point_y - radius,
                 point_x + radius, point_y + radius),
                fill='cyan', outline='blue')
            painter.text((point_x + radius + 1, point_y - radius),
                         str(keypoint_index + 1), fill='cyan')
    return canvas


def load_ground_truth(dataset, image_id):
    """Return original-coordinate COCO annotations from a wrapped dataset."""
    current = dataset
    for _ in range(10):
        if hasattr(current, 'coco'):
            annotation_ids = current.coco.getAnnIds(imgIds=[int(image_id)])
            annotations = current.coco.loadAnns(annotation_ids)
            categories = getattr(current.coco, 'cats', {})
            for annotation in annotations:
                category = categories.get(annotation.get('category_id'), {})
                annotation['category_name'] = category.get(
                    'name', annotation.get('category_id', 'GT'))
            return annotations
        if hasattr(current, 'dataset'):
            current = current.dataset
        else:
            break
    return []


def load_original_image(dataset, image_id, fallback_tensor, original_size):
    """Load the source image from a COCO dataset, with a tensor fallback."""
    current = dataset
    for _ in range(10):
        if hasattr(current, 'coco') and hasattr(current, 'root'):
            image_info = current.coco.loadImgs([int(image_id)])[0]
            return Image.open(Path(current.root) / image_info['file_name']).convert('RGB')
        if hasattr(current, 'dataset'):
            current = current.dataset
        else:
            break

    tensor = fallback_tensor.detach().cpu().clamp(0, 1)
    array = (tensor.permute(1, 2, 0).numpy() * 255).astype('uint8')
    width, height = map(int, original_size)
    return Image.fromarray(array).resize((width, height), Image.Resampling.BILINEAR)


def collect_metrics(evaluator):
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    metrics = {}
    detection_metric_names = (
        'AP', 'AP50', 'AP75', 'AP_S', 'AP_M', 'AP_L',
        'AR_1', 'AR_10', 'AR_100', 'AR_S', 'AR_M', 'AR_L')
    keypoint_metric_names = (
        'AP', 'AP50', 'AP75', 'AP_M', 'AP_L',
        'AR', 'AR50', 'AR75', 'AR_M', 'AR_L')
    for iou_type, coco_eval in evaluator.coco_eval.items():
        metric_names = (keypoint_metric_names if iou_type == 'keypoints'
                        else detection_metric_names)
        for index, value in enumerate(coco_eval.stats.tolist()):
            metric_name = (metric_names[index]
                           if index < len(metric_names) else str(index))
            metrics[f'{iou_type}_{metric_name}'] = float(value)
    if hasattr(evaluator, 'vehicle_metrics'):
        vehicle_metric_names = {
            'precision_5px': 'keypoints_Precision@5px',
            'recall_5px': 'keypoints_Recall@5px',
            'f1_5px': 'keypoints_F1@5px',
            'precision_10px': 'keypoints_Precision@10px',
            'recall_10px': 'keypoints_Recall@10px',
            'f1_10px': 'keypoints_F1@10px',
            'vis_precision': 'keypoints_VisibilityPrecision',
            'vis_recall': 'keypoints_VisibilityRecall',
            'vis_f1': 'keypoints_VisibilityF1',
            'vis_accuracy': 'keypoints_VisibilityAccuracy',
            'num_matched': 'keypoints_MatchedDetections',
            'num_gt': 'keypoints_GroundTruthDetections',
            'matched_ratio': 'keypoints_MatchedRatio',
        }
        for name, value in evaluator.vehicle_metrics.items():
            metrics[vehicle_metric_names.get(name, f'keypoints_{name}')] = float(value)
    return metrics


@torch.inference_mode()
def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir = output_dir / 'visualizations'
    if args.save_visualizations:
        visualization_dir.mkdir(parents=True, exist_ok=True)

    cfg = YAMLConfig(
        args.config, resume=args.checkpoint, device=args.device,
        output_dir=str(output_dir))
    configure_evaluation(cfg, args.batch_size)
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    solver.eval()
    model = solver.ema.module if solver.ema else solver.model
    model.eval()
    postprocessor = solver.postprocessor
    postprocessor.eval()
    data_loader = solver.val_dataloader
    evaluator = solver.evaluator
    evaluator.cleanup()

    predictions_json = []
    image_index = 0
    class_names = cfg.yaml_cfg.get('class_names')
    for samples, targets in data_loader:
        samples_device = samples.to(solver.device)
        outputs = model(samples_device)
        original_sizes = torch.stack(
            [target['orig_size'] for target in targets]).to(solver.device)
        results = postprocessor(outputs, original_sizes)
        evaluator.update({
            int(target['image_id'].item()): result
            for target, result in zip(targets, results)
        })

        for sample, target, result in zip(samples, targets, results):
            image_index += 1
            image_id = int(target['image_id'].item())
            record = serialize_prediction(
                image_id, image_index, result, args.score_threshold)
            width, height = map(int, target['orig_size'].tolist())
            record.update(width=width, height=height)
            predictions_json.append(record)

            if args.save_visualizations:
                original = load_original_image(
                    data_loader.dataset, image_id, sample, (width, height))
                visualization = draw_prediction(
                    original, result, args.score_threshold,
                    args.keypoint_threshold, class_names,
                    load_ground_truth(data_loader.dataset, image_id))
                visualization.save(visualization_dir / f'{image_index}.jpg')

    (output_dir / 'predictions.json').write_text(
        json.dumps(predictions_json, indent=2), encoding='utf-8')
    metrics = collect_metrics(evaluator)
    (output_dir / 'metrics.txt').write_text(
        ''.join(f'{name}: {value}\n' for name, value in metrics.items()),
        encoding='utf-8')
    print(f'Saved predictions to {output_dir / "predictions.json"}')
    print(f'Saved metrics to {output_dir / "metrics.txt"}')
    if args.save_visualizations:
        print(f'Saved visualizations to {visualization_dir}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Infer on the validation dataset configured in YAML.')
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-r', '--checkpoint', required=True)
    parser.add_argument('-o', '--output-dir', required=True)
    parser.add_argument('-d', '--device', default='cuda:0')
    parser.add_argument('-b', '--batch-size', type=int, default=1)
    parser.add_argument('--score-threshold', type=float, default=0.4)
    parser.add_argument('--keypoint-threshold', type=float, default=0.5)
    parser.add_argument('--save-visualizations', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
