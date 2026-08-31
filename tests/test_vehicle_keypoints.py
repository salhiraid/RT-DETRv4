import torch
import json
from PIL import Image

from engine.data.dataset.coco_dataset import ConvertCocoPolysToMask
from engine.data.dataset.coco_dataset import MultiCocoDetection
from engine.data.dataloader import MultiDataSampler
from engine.data.dataset.vehicle_keypoint_metric import VehicleKeypointMetric
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
            'annotations': [{'id': dataset_index + 1, 'image_id': dataset_index + 1,
                             'category_id': 0, 'bbox': [1, 1, 10, 8],
                             'area': 80, 'iscrowd': 0}],
            'categories': [{'id': 0, 'name': 'vehicle'}],
        }
        annotation_file = tmp_path / f'dataset_{dataset_index}.json'
        annotation_file.write_text(json.dumps(annotation))
        image_folders.append(str(folder)); annotation_files.append(str(annotation_file))
    dataset = MultiCocoDetection(
        image_folders, annotation_files, transforms=None, repeat_factors=[1, 2])
    assert len(dataset) == 3
    _, target = dataset.load_item(2)
    assert target['boxes'].shape == (1, 4)
    assert target['keypoints'].shape == (1, 31, 2)
    assert target['ignore_keypoints'].item()

    sampler = MultiDataSampler(dataset, dataset_ratio=[25, 75],
                               max_samples=2000, seed=7)
    sampled_dataset_ids = [dataset.index_map[index][0] for index in sampler]
    fraction_second = sum(x == 1 for x in sampled_dataset_ids) / len(sampled_dataset_ids)
    assert 0.70 < fraction_second < 0.80
    first_epoch = list(iter(sampler))
    sampler.set_epoch(1)
    assert first_epoch != list(iter(sampler))
