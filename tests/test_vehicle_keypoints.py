import torch
import json
import pytest
from PIL import Image

from engine.data.dataset.coco_dataset import ConvertCocoPolysToMask
from engine.data.dataset.coco_dataset import MultiCocoDetection
from engine.data.dataset.coco_dataset import WeightedMultiDataset
from engine.data.dataloader import MultiDataSampler
from engine.data.dataset.vehicle_keypoint_metric import VehicleKeypointMetric
from engine.data.dataset.coco_eval import VehicleCocoEvaluator
from engine.rtv4.dfine_decoder import DFINETransformer, MLP, TransformerDecoder
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


def test_keypoint_heads_are_shared_deep_branches():
    model = DFINETransformer(
        num_classes=1, hidden_dim=16, num_queries=10,
        feat_channels=[16, 16, 16], num_layers=3,
        dim_feedforward=32, nhead=4, which_keypoints=list(range(31)),
        num_reg_fcs=2)

    assert len(model.kps_xy_branches) == len(model.kps_vis_branches) == 1
    xy_layers = list(model.kps_xy_branches[0])
    vis_layers = list(model.kps_vis_branches[0])
    assert len([layer for layer in xy_layers if isinstance(layer, torch.nn.Linear)]) == 5
    assert len([layer for layer in xy_layers if isinstance(layer, torch.nn.ReLU)]) == 4
    assert xy_layers[-1].out_features == 62
    assert vis_layers[-1].out_features == 31


def test_keypoint_parameter_count_is_independent_of_evaluation_resolution():
    kwargs = dict(
        num_classes=1, hidden_dim=16, num_queries=10,
        feat_channels=[16, 16, 16], num_layers=3,
        dim_feedforward=32, nhead=4, which_keypoints=list(range(31)))
    small = DFINETransformer(eval_spatial_size=[400, 400], **kwargs)
    large = DFINETransformer(eval_spatial_size=[672, 1184], **kwargs)

    count = lambda model: sum(parameter.numel() for parameter in model.parameters())
    assert count(small) == count(large)


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


def test_pose_loss_accepts_cpu_match_indices_for_cuda_predictions():
    if not torch.cuda.is_available():
        pytest.skip('CUDA is required for the cross-device regression test')

    device = torch.device('cuda')
    outputs = dict(
        pred_boxes=torch.rand(1, 2, 4, device=device),
        pred_logits=torch.rand(1, 2, 1, device=device),
        pred_keypoints=torch.rand(
            1, 2, 62, device=device, requires_grad=True),
        pred_keypoints_vis=torch.rand(
            1, 2, 31, device=device, requires_grad=True))
    target = dict(
        labels=torch.zeros(2, dtype=torch.long, device=device),
        boxes=torch.rand(2, 4, device=device),
        keypoints=torch.rand(2, 31, 2, device=device),
        keypoints_visible=torch.full((2, 31), 2, device=device),
        ignore_keypoints=torch.zeros(2, dtype=torch.bool, device=device))
    cpu_indices = [(torch.arange(2), torch.arange(2))]

    losses = criterion().loss_keypoints(outputs, [target], cpu_indices, 2)
    sum(losses.values()).backward()

    assert outputs['pred_keypoints'].grad.abs().sum() > 0
    assert outputs['pred_keypoints_vis'].grad.abs().sum() > 0


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


def test_configured_keypoint_losses_apply_requested_weights():
    from engine.rtv4.keypoint_loss import CrossEntropyLoss, L1Loss, OKSLoss

    crit = RTv4Criterion(
        Matcher(), {
            'loss_keypoints_xy': 1, 'loss_keypoints_vis': 1,
            'loss_keypoints_oks': 1}, ['keypoints'],
        which_keypoints=list(range(31)),
        loss_keypoints_xy=L1Loss(reduction='mean', loss_weight=10),
        loss_keypoints_vis=CrossEntropyLoss(
            use_sigmoid=True, reduction='mean', loss_weight=3),
        loss_keypoints_oks=OKSLoss(reduction='mean', loss_weight=35))
    outputs = dict(
        pred_boxes=torch.tensor([[[.5, .5, .5, .5]]]),
        pred_logits=torch.zeros(1, 1, 1),
        pred_keypoints=torch.zeros(1, 1, 62, requires_grad=True),
        pred_keypoints_vis=torch.zeros(1, 1, 31, requires_grad=True))
    target = dict(
        labels=torch.zeros(1, dtype=torch.long),
        boxes=torch.tensor([[.5, .5, .5, .5]]),
        keypoints=torch.ones(1, 31, 2),
        keypoints_visible=torch.full((1, 31), 2),
        ignore_keypoints=torch.zeros(1, dtype=torch.bool),
        size=torch.tensor([100, 100]))

    losses = crit.loss_keypoints(
        outputs, [target], [(torch.tensor([0]), torch.tensor([0]))], 1)

    assert torch.isclose(losses['loss_keypoints_xy'], torch.tensor(10.0))
    expected_vis = torch.log(torch.tensor(2.0)) * 3
    assert torch.isclose(losses['loss_keypoints_vis'], expected_vis)
    expected_oks = (1 - torch.exp(torch.tensor(-4.0))) * 35
    assert torch.isclose(losses['loss_keypoints_oks'], expected_oks)


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


def test_weighted_multi_dataset_uses_nested_coco_configs(tmp_path):
    from engine.core.workspace import create
    from engine.core.yaml_utils import merge_config

    dataset_configs = []
    for dataset_index in range(2):
        folder = tmp_path / f'weighted_images_{dataset_index}'
        folder.mkdir()
        Image.new('RGB', (32, 16)).save(folder / 'sample.jpg')
        annotation_file = tmp_path / f'weighted_{dataset_index}.json'
        annotation_file.write_text(json.dumps({
            'images': [{'id': 1, 'file_name': 'sample.jpg',
                        'width': 32, 'height': 16}],
            'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1,
                             'bbox': [1, 1, 10, 8], 'area': 80,
                             'iscrowd': 0}],
            'categories': [{'id': 1, 'name': 'vehicle'}],
        }))
        dataset_configs.append({
            'type': 'CocoDetection', 'img_folder': str(folder),
            'ann_file': str(annotation_file), 'transforms': None,
            'return_masks': False, 'remap_mscoco_category': False,
            'num_keypoints': 31,
        })

    config = merge_config({
        'weighted_test_dataset': {
            'type': 'WeightedMultiDataset', 'datasets': dataset_configs,
            'weights': [0.7, 0.3], 'samples_per_epoch': 1000, 'seed': 42,
            'transforms': None,
        }
    })
    dataset = create('weighted_test_dataset', config)

    assert isinstance(dataset, WeightedMultiDataset)
    assert all(dataset._transforms is None for dataset in dataset.datasets)
    assert len(dataset) == 1000
    selected = [dataset_index for dataset_index, _ in dataset.index_map]
    assert 0.65 < selected.count(0) / len(selected) < 0.75
    first_epoch = list(dataset.index_map)
    dataset.set_epoch(1)
    assert first_epoch != dataset.index_map
    image, target = dataset[0]
    assert image.size == (32, 16)
    assert target['keypoints'].shape == (1, 31, 2)


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


def test_eval_mode_forces_validation_batch_size_one():
    from train import configure_evaluation

    class Config:
        yaml_cfg = {
            'val_dataloader': {'total_batch_size': 64, 'shuffle': False}}

    config = Config()
    configure_evaluation(config, 1)

    assert config.yaml_cfg['val_dataloader']['batch_size'] == 1
    assert 'total_batch_size' not in config.yaml_cfg['val_dataloader']


def test_validation_inference_json_and_visualization_keep_original_geometry():
    from tools.inference.validation_inference import (
        draw_prediction, serialize_prediction)

    result = {
        'labels': torch.tensor([2]),
        'scores': torch.tensor([.9]),
        'boxes': torch.tensor([[10., 20., 100., 80.]]),
        'keypoints_xy': torch.tensor([[[30., 40.], [50., 60.]]]),
        'keypoint_scores': torch.tensor([[.8, .2]]),
    }
    record = serialize_prediction(7, 1, result, .4)
    visualization = draw_prediction(
        Image.new('RGB', (320, 180)), result, .4, .5,
        class_names=['a', 'b', 'vehicle'])

    assert record['image_id'] == 7 and record['index'] == 1
    assert record['detections'][0]['bbox_xywh'] == [10., 20., 90., 60.]
    assert record['detections'][0]['keypoints_xy'] == [[30., 40.], [50., 60.]]
    assert visualization.size == (320, 180)
