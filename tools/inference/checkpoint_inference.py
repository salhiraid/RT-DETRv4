"""Run a native PyTorch checkpoint on images and export JSON/visualizations."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torchvision.transforms.v2.functional as F
from PIL import Image, ImageDraw, ImageFile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine.core import YAMLConfig


ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_SUFFIXES = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}


def discover_images(source, recursive=False):
    """Return image paths from a file or directory in deterministic order."""
    source = Path(source)
    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f'Unsupported image extension: {source.suffix}')
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f'Input does not exist: {source}')
    iterator = source.rglob('*') if recursive else source.glob('*')
    images = sorted(path for path in iterator
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f'No supported images found in: {source}')
    return images


def checkpoint_state(checkpoint):
    """Select EMA when available, while accepting common checkpoint layouts."""
    if not isinstance(checkpoint, dict):
        raise TypeError('Checkpoint must contain a state-dict mapping')
    if 'ema' in checkpoint:
        ema = checkpoint['ema']
        return ema.get('module', ema) if isinstance(ema, dict) else ema
    if 'model' in checkpoint:
        return checkpoint['model']
    if 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    return checkpoint


def prediction_to_json(path, width, height, result, score_threshold,
                       keypoint_threshold):
    detections = []
    for label, box, score, index in zip(
            result['labels'], result['boxes'], result['scores'],
            range(len(result['scores']))):
        score = float(score)
        if score < score_threshold:
            continue
        xyxy = [float(value) for value in box]
        item = {
            'label': int(label),
            'score': score,
            'bbox_xyxy': xyxy,
            'bbox_xywh': [xyxy[0], xyxy[1], xyxy[2] - xyxy[0],
                          xyxy[3] - xyxy[1]],
        }
        if 'keypoints' in result:
            points = result['keypoints'][index]
            point_scores = result.get('keypoint_scores')
            point_scores = (point_scores[index] if point_scores is not None
                            else torch.ones(len(points), device=points.device))
            item['keypoints'] = [
                [float(point[0]), float(point[1]), float(kp_score)]
                for point, kp_score in zip(points, point_scores)
            ]
            item['visible_keypoints'] = [score >= keypoint_threshold
                                         for _, _, score in item['keypoints']]
        detections.append(item)
    return {'file_name': str(path), 'width': width, 'height': height,
            'detections': detections}


def visualize(image, prediction, keypoint_threshold):
    """Draw coordinates that are already expressed in original-image pixels."""
    output = image.copy()
    draw = ImageDraw.Draw(output)
    radius = max(2, round(min(image.size) / 300))
    line_width = max(2, round(min(image.size) / 400))
    for detection in prediction['detections']:
        box = detection['bbox_xyxy']
        draw.rectangle(box, outline='red', width=line_width)
        draw.text((box[0], max(0, box[1] - 12)),
                  f"{detection['label']} {detection['score']:.2f}", fill='red')
        for x, y, score in detection.get('keypoints', []):
            if score >= keypoint_threshold:
                draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                             fill='lime', outline='black')
    return output


def load_model(config_path, checkpoint_path, device):
    cfg = YAMLConfig(str(config_path))
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    model = cfg.model
    state = checkpoint_state(torch.load(checkpoint_path, map_location='cpu'))
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    model.load_state_dict(state)
    return model.to(device).eval(), cfg.postprocessor.to(device).eval(), cfg


@torch.inference_mode()
def run(args):
    device = torch.device(args.device)
    model, postprocessor, cfg = load_model(args.config, args.checkpoint, device)
    input_size = args.input_size or cfg.eval_spatial_size or [640, 640]
    if len(input_size) != 2 or min(input_size) <= 0:
        raise ValueError('--input-size must be two positive values: HEIGHT WIDTH')

    paths = discover_images(args.input, args.recursive)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    source_root = Path(args.input) if Path(args.input).is_dir() else None

    for path in paths:
        with Image.open(path) as opened:
            image = opened.convert('RGB')
        width, height = image.size
        tensor = F.to_image(image)
        tensor = F.resize(tensor, input_size, antialias=True)
        tensor = F.to_dtype(tensor, torch.float32, scale=True).unsqueeze(0).to(device)
        outputs = model(tensor)
        # PostProcessor expects [width, height], and therefore directly maps
        # normalized boxes/keypoints back to the original (not resized) image.
        original_size = torch.tensor([[width, height]], device=device)
        result = postprocessor(outputs, original_size)[0]
        record = prediction_to_json(path, width, height, result,
                                    args.score_threshold,
                                    args.keypoint_threshold)
        records.append(record)
        if args.visualize:
            relative = path.relative_to(source_root) if source_root else Path(path.name)
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            visualize(image, record, args.keypoint_threshold).save(destination)

    json_path = Path(args.json_output) if args.json_output else output_dir / 'predictions.json'
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(f'Saved {sum(len(x["detections"]) for x in records)} detections '
          f'for {len(records)} images to {json_path}')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-r', '--checkpoint', required=True)
    parser.add_argument('-i', '--input', required=True,
                        help='An image or directory of images')
    parser.add_argument('-o', '--output-dir', default='inference_results')
    parser.add_argument('--json-output')
    parser.add_argument('-d', '--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--input-size', nargs=2, type=int, metavar=('HEIGHT', 'WIDTH'))
    parser.add_argument('--score-threshold', type=float, default=0.4)
    parser.add_argument('--keypoint-threshold', type=float, default=0.5)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--recursive', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
