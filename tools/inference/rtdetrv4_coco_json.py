"""Write RT-DETRv4 predictions as ``predictions_class_aware.json``.

The output is a COCO results list, one entry per detection, with the same
fields as the YOLOX+RTMPose path of the external evaluation toolkit::

    {
        'image_id':        int,              # GT image id
        'category_id':     int,              # GT category id (matched by class name)
        'bbox':            [x, y, w, h],     # original-image pixels
        'score':           float,
        'keypoints':       [x1, y1, v1, ...],  # v = sigmoid(visibility)
        'keypoint_scores': [v1, v2, ...],
        'img_path':        str,
    }

Model labels are mapped to GT category ids by class name, so the result is
independent of the evaluation's MODEL_CLASSES order. Save it as
``<OUT_DIR>/<model_name>/<dataset>@<W>x<H>/predictions_class_aware.json``
and ``main.py`` reuses it without running inference.

Either run inference (``--config/--checkpoint``, same flags as
``rtdetrv4_bridge.py``) or convert an existing bridge pkl (``--pkl``).
"""

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rtdetrv4_bridge import build_parser, predict


def records_to_coco(records, label_names, cat_name_to_id, score_threshold=0.0):
    """Flatten bridge pkl records into a class-aware COCO results list."""
    coco, unmapped = [], {}
    for record in records:
        instances = record['pred_instances']
        keypoints = instances.get('keypoints')
        visible = instances.get('keypoints_visible')
        keypoint_scores = instances.get('keypoint_scores')
        for index, score in enumerate(instances['scores']):
            score = float(score)
            if score < score_threshold:
                continue
            name = label_names[int(instances['labels'][index])]
            if name not in cat_name_to_id:
                unmapped[name] = unmapped.get(name, 0) + 1
                continue
            x1, y1, x2, y2 = (float(v) for v in instances['bboxes'][index])
            prediction = {
                'image_id': record['img_id'],
                'category_id': cat_name_to_id[name],
                'bbox': [x1, y1, x2 - x1, y2 - y1],
                'score': score,
                'img_path': record['img_path'],
            }
            if keypoints is not None:
                prediction['keypoints'] = [
                    float(value)
                    for (x, y), v in zip(keypoints[index], visible[index])
                    for value in (x, y, v)
                ]
                prediction['keypoint_scores'] = [float(v) for v in keypoint_scores[index]]
            coco.append(prediction)
    if unmapped:
        print(f'Warning: dropped detections of classes absent from the GT file: {unmapped}')
    return coco


def label_names_from_config(config_path):
    from engine.core import YAMLConfig
    names = YAMLConfig(str(config_path)).yaml_cfg.get('class_names')
    if not names:
        raise ValueError(f'{config_path} has no class_names; pass --class-names')
    return list(names)


def run(args):
    if not args.gt_file:
        raise ValueError('--gt-file is required (category ids come from it)')
    with open(args.gt_file) as f:
        gt = json.load(f)
    cat_name_to_id = {c['name']: c['id'] for c in gt['categories']}

    if args.pkl:
        with open(args.pkl, 'rb') as f:
            records = pickle.load(f)
        label_names = args.class_names or (
            label_names_from_config(args.config) if args.config else None)
        if not label_names:
            raise ValueError('--pkl needs --class-names (label order of the pkl) or --config')
        gt_img_ids = {img['id'] for img in gt['images'][:args.max_samples]}
        records = [r for r in records if r['img_id'] in gt_img_ids]
    else:
        if not args.config or not args.checkpoint:
            raise ValueError('Pass --config and --checkpoint, or --pkl')
        records, label_names = predict(args)

    print(f'Label names : {label_names}')
    print(f'GT classes  : {cat_name_to_id}')
    coco = records_to_coco(records, label_names, cat_name_to_id, args.score_threshold)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        json.dump(coco, f, indent=2)
    print(f'Saved {len(coco)} class-aware predictions for {len(records)} images -> {output}')
    return str(output)


def parse_args(argv=None):
    parser = build_parser(description=__doc__, require_model=False,
                          output_help='Output json, e.g. .../predictions_class_aware.json')
    parser.add_argument('--pkl', help='Convert an existing rtdetrv4_bridge.py pkl instead of '
                                      'running inference')
    parser.add_argument('--class-names', nargs='+',
                        help='With --pkl: class name of each pkl label index '
                             '(default: YAML class_names from --config)')
    return parser.parse_args(argv)


if __name__ == '__main__':
    run(parse_args())
