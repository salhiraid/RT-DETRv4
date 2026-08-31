import torch
import json
from PIL import Image

from engine.data.dataset.coco_dataset import ConvertCocoPolysToMask
from engine.data.dataset.coco_dataset import MultiCocoDetection
from engine.data.dataloader import MultiDataSampler
from engine.data.dataset.vehicle_keypoint_metric import VehicleKeypointMetric
from engine.data.dataset.coco_eval import VehicleCocoEvaluator
from engine.rtv4.dfine_decoder import MLP, TransformerDecoder
from engine.rtv4.postprocessor import PostProcessor
from engine.rtv4.rtv4_criterion import RTv4Criterion


def annotation(with_pose, x=1):
    ann = dict(bbox=[x, 2, 20, 10], category_id=0, area=200, iscrowd=0)
    if with_pose:
        ann['keypoints'] = sum(([x + i / 10, 3, 2] for i in range(31)), [])
    return ann


def test_parser_mixed_preserves_bbox_only():
    _, target = ConvertCocoPolysToMask()(Image.new('RGB', (64, 32)), {
        'image_id': 1, 'annotations': [annotation(True), annotation(False, 30)]})
    assert target['boxes'].shape == (2, 4)
    assert target['keypoints'].shape == (2, 31, 2)
    assert target['keypoints_visible'].shape == (2, 31)
    assert target['ignore_keypoints'].tolist() == [False, True]


def test_bbox_local_decode_shapes_geometry_and_backward():
    query = torch.randn(2, 5, 16, requires_grad=True)
    branch_xy, branch_vis = MLP(16, 16, 62, 5), MLP(16, 16, 31, 5)
    boxes = torch.tensor([.5, .5, .4, .2]).expand(2, 5, 4).clone().requires_grad_()
    points = TransformerDecoder._decode_keypoints_inside_bbox(branch_xy(query), boxes)
    visibility = branch_vis(query)
    assert points.shape == (2, 5, 62) and visibility.shape == (2, 5, 31)
    p = points.reshape(2, 5, 31, 2)
    assert ((p[..., 0] >= .3) & (p[..., 0] <= .7)).all()
    assert ((p[..., 1] >= .4) & (p[..., 1] <= .6)).all()
    (points.sum() + visibility.sum()).backward()
    assert boxes.grad is None
    assert branch_xy.layers[-1].weight.grad.abs().sum() > 0
    assert branch_vis.layers[-1].weight.grad.abs().sum() > 0


class Matcher:
    def __call__(self, outputs, targets):
        return {'indices': [(torch.arange(len(t['labels'])), torch.arange(len(t['labels']))) for t in targets]}


def criterion():
    return RTv4Criterion(Matcher(), {'loss_keypoints_xy': 1, 'loss_keypoints_vis': 1},
                         ['keypoints'], which_keypoints=list(range(31)))


def test_bbox_only_and_mixed_pose_loss():
    outputs = dict(pred_boxes=torch.rand(1, 2, 4), pred_logits=torch.rand(1, 2, 1),
                   pred_keypoints=torch.rand(1, 2, 62, requires_grad=True),
                   pred_keypoints_vis=torch.rand(1, 2, 31, requires_grad=True),
                   aux_outputs=[], enc_aux_outputs=[])
    bbox_only = dict(labels=torch.zeros(1, dtype=torch.long), boxes=torch.rand(1, 4),
                     keypoints=torch.zeros(1, 31, 2), keypoints_visible=torch.zeros(1, 31),
                     ignore_keypoints=torch.ones(1, dtype=torch.bool))
    losses = criterion().loss_keypoints(outputs, [bbox_only], [(torch.tensor([0]), torch.tensor([0]))], 1)
    assert losses['loss_keypoints_xy'].item() == losses['loss_keypoints_vis'].item() == 0
    assert losses['loss_keypoints_xy'].requires_grad
    pose = {k: v.clone() for k, v in bbox_only.items()}
    pose['keypoints'] = torch.rand(1, 31, 2); pose['keypoints_visible'].fill_(2)
    pose['ignore_keypoints'].fill_(False)
    mixed = {k: torch.cat((pose[k], bbox_only[k])) for k in pose}
    losses = criterion().loss_keypoints(outputs, [mixed], [(torch.arange(2), torch.arange(2))], 2)
    sum(losses.values()).backward()
    assert losses['loss_keypoints_xy'] > 0 and outputs['pred_keypoints'].grad.abs().sum() > 0


def test_inference_alignment_and_metric_contract():
    outputs = dict(pred_logits=torch.tensor([[[8.], [7.], [6.]]]),
                   pred_boxes=torch.rand(1, 3, 4), pred_keypoints=torch.rand(1, 3, 62),
                   pred_keypoints_vis=torch.rand(1, 3, 31))
    result = PostProcessor(num_classes=1, num_top_queries=2)(outputs, torch.tensor([[100, 50]]))[0]
    assert len(result['boxes']) == len(result['keypoints_xy']) == len(result['keypoints_vis']) == 2
    metric = VehicleKeypointMetric()
    values = metric.compute_metrics([])
    assert {'precision_5px', 'f1_10px', 'vis_accuracy', 'matched_ratio'} <= values.keys()


def test_oks_loss_pixels_area_mask_and_empty_graph():
    from engine.rtv4.keypoint_loss import OKSLoss
    loss_fn = OKSLoss(loss_weight=25.)
    output = torch.zeros(2, 31, 2, requires_grad=True)
    target = torch.zeros_like(output)
    visible = torch.zeros(2, 31)
    empty = loss_fn(output, target, visible, torch.ones(2))
    assert empty.item() == 0 and empty.requires_grad
    visible[0] = 2
    target[0, :, 0] = 1
    loss = loss_fn(output, target, visible, torch.tensor([100., 100.]))
    assert loss > 0
    loss.backward()
    assert output.grad is not None and torch.isfinite(output.grad).all()


def test_criterion_exposes_oks_for_pose_and_empty_batches():
    from engine.rtv4.keypoint_loss import OKSLoss
    loss_fn = OKSLoss(loss_weight=25.)
    crit = RTv4Criterion(
        Matcher(), {'loss_keypoints_xy': 1, 'loss_keypoints_vis': 1,
                    'loss_keypoints_oks': 1}, ['keypoints'],
        which_keypoints=list(range(31)), loss_keypoints_oks=loss_fn)
    outputs = dict(pred_boxes=torch.tensor([[[.5, .5, .5, .5]]]),
                   pred_logits=torch.zeros(1, 1, 1),
                   pred_keypoints=torch.full((1, 1, 62), .5, requires_grad=True),
                   pred_keypoints_vis=torch.zeros(1, 1, 31, requires_grad=True))
    bbox_only = dict(labels=torch.zeros(1, dtype=torch.long),
                     boxes=torch.tensor([[.5, .5, .5, .5]]),
                     keypoints=torch.zeros(1, 31, 2),
                     keypoints_visible=torch.zeros(1, 31),
                     ignore_keypoints=torch.ones(1, dtype=torch.bool),
                     size=torch.tensor([1184, 666]))
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    empty = crit.loss_keypoints(outputs, [bbox_only], indices, 1)
    assert empty['loss_keypoints_oks'].item() == 0
    pose = {key: value.clone() for key, value in bbox_only.items()}
    pose['ignore_keypoints'].fill_(False)
    pose['keypoints_visible'].fill_(2)
    pose['keypoints'][..., 0] = .6
    losses = crit.loss_keypoints(outputs, [pose], indices, 1)
    assert losses['loss_keypoints_oks'] > 0
    sum(losses.values()).backward()
    assert outputs['pred_keypoints'].grad.abs().sum() > 0


def test_multiple_coco_datasets_concat_and_repeat(tmp_path):
    image_folders, annotation_files = [], []
    for dataset_index in range(2):
        folder = tmp_path / f'images_{dataset_index}'
        folder.mkdir()
        Image.new('RGB', (32, 16)).save(folder / 'sample.jpg')
        annotation = {
            'images': [{'id': dataset_index + 1, 'file_name': 'sample.jpg',
                        'width': 32, 'height': 16}],
            'annotations': [
                {'id': dataset_index + 1, 'image_id': dataset_index + 1,
                 'category_id': 10 + dataset_index, 'bbox': [1, 1, 10, 8],
                 'area': 80, 'iscrowd': 0},
                {'id': 100 + dataset_index, 'image_id': dataset_index + 1,
                 'category_id': 0, 'bbox': [15, 1, 8, 8],
                 'area': 64, 'iscrowd': 0},
            ],
            'categories': [
                {'id': 0, 'name': 'person'},
                {'id': 10 + dataset_index, 'name': 'vehicle'},
            ],
        }
        annotation_file = tmp_path / f'dataset_{dataset_index}.json'
        annotation_file.write_text(json.dumps(annotation))
        image_folders.append(str(folder)); annotation_files.append(str(annotation_file))
    dataset = MultiCocoDetection(
        image_folders, annotation_files, transforms=None, repeat_factors=[1, 2],
        class_names=['vehicle'])
    assert len(dataset) == 3
    _, target = dataset.load_item(2)
    assert target['boxes'].shape == (1, 4)
    assert target['keypoints'].shape == (1, 31, 2)
    assert target['ignore_keypoints'].item()
    assert target['labels'].item() == 0
    evaluator = VehicleCocoEvaluator(
        dataset.datasets[0].coco, class_names=['vehicle'])
    assert evaluator.coco_eval['bbox'].params.catIds == [10]

    sampler = MultiDataSampler(dataset, dataset_ratio=[25, 75],
                               max_samples=2000, seed=7)
    sampled_dataset_ids = [dataset.index_map[index][0] for index in sampler]
    fraction_second = sum(x == 1 for x in sampled_dataset_ids) / len(sampled_dataset_ids)
    assert 0.70 < fraction_second < 0.80
    first_epoch = list(iter(sampler))
    sampler.set_epoch(1)
    assert first_epoch != list(iter(sampler))


def test_dinov3_teacher_validates_paths_and_supports_rectangular_grid(tmp_path, monkeypatch):
    from engine.rtv4.dinov3_teacher import DINOv3TeacherModel
    missing = tmp_path / 'missing_repo'
    try:
        DINOv3TeacherModel(str(missing), str(tmp_path / 'missing.pth'))
    except FileNotFoundError as error:
        assert 'hubconf.py' in str(error)
    else:
        raise AssertionError('missing DINOv3 repository must fail before torch.hub.load')

    repo = tmp_path / 'dinov3'; repo.mkdir(); (repo / 'hubconf.py').touch()
    weights = tmp_path / 'teacher.pth'; weights.touch()

    class FakeDINO(torch.nn.Module):
        embed_dim = 8

        def forward(self, images, is_training=True, masks=None):
            h, w = images.shape[-2] // 16, images.shape[-1] // 16
            return {'x_norm_patchtokens': images.new_zeros(images.shape[0], h * w, 8)}

    monkeypatch.setattr(torch.hub, 'load', lambda *args, **kwargs: FakeDINO())
    teacher = DINOv3TeacherModel(str(repo), str(weights), patch_size=16)
    features = teacher(torch.zeros(1, 3, 672, 1184))
    assert features.shape == (1, 8, 21, 37)


def test_profiler_uses_rectangular_eval_spatial_size(monkeypatch):
    import engine.misc.profiler_utils as profiler

    captured = {}

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def deploy(self):
            return self

    def fake_flops(model, input_shape, **kwargs):
        captured['input_shape'] = input_shape
        return '1', '1', None

    class Config:
        eval_spatial_size = [672, 1184]
        model = DummyModel()

    monkeypatch.setattr(profiler, 'calculate_flops', fake_flops)
    profiler.stats(Config())
    assert captured['input_shape'] == (1, 3, 672, 1184)


def test_yaml_config_exposes_eval_spatial_size(tmp_path):
    from engine.core.yaml_config import YAMLConfig

    config_path = tmp_path / 'geometry.yml'
    config_path.write_text('eval_spatial_size: [672, 1184]\n')
    config = YAMLConfig(str(config_path))
    assert config.eval_spatial_size == [672, 1184]


def test_checkpoint_inference_json_preserves_original_coordinates(tmp_path):
    from tools.inference.checkpoint_inference import (
        checkpoint_state, discover_images, prediction_to_json, visualize)

    image_path = tmp_path / 'frame.jpg'
    Image.new('RGB', (1200, 700)).save(image_path)
    assert discover_images(tmp_path) == [image_path]
    weights = {'weight': torch.ones(1)}
    assert checkpoint_state({'ema': {'module': weights}}) is weights

    result = {
        'labels': torch.tensor([3]),
        'scores': torch.tensor([.9]),
        'boxes': torch.tensor([[100., 50., 1100., 650.]]),
        'keypoints': torch.tensor([[[600., 350.], [900., 500.]]]),
        'keypoint_scores': torch.tensor([[.8, .2]]),
    }
    record = prediction_to_json(image_path, 1200, 700, result, .4, .5)
    detection = record['detections'][0]
    assert detection['bbox_xyxy'] == [100., 50., 1100., 650.]
    assert detection['bbox_xywh'] == [100., 50., 1000., 600.]
    assert detection['visible_keypoints'] == [True, False]
    rendered = visualize(Image.open(image_path), record, .5)
    assert rendered.size == (1200, 700)
