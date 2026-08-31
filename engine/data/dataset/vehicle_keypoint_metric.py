"""Vehicle keypoint metric with detection-first, one-to-one matching."""
from collections import OrderedDict
import torch

from ...core import register


@register()
class VehicleKeypointMetric:
    def __init__(self, num_keypoints=31, thresholds=(5.0, 10.0), iou_thr=.5,
                 vis_thr=.5, score_thr=.05, margin=.05, crop_size=512,
                 min_bbox_size=64., collect_device='cpu', prefix=None):
        self.num_keypoints = num_keypoints
        self.thresholds = thresholds
        self.iou_thr, self.vis_thr, self.score_thr = iou_thr, vis_thr, score_thr
        self.margin, self.crop_size, self.min_bbox_size = margin, crop_size, min_bbox_size
        self.collect_device, self.prefix = collect_device, prefix
        self.results = []

    @staticmethod
    def _iou(a, b):
        lt, rb = torch.maximum(a[:2], b[:2]), torch.minimum(a[2:], b[2:])
        inter = (rb - lt).clamp(min=0).prod()
        return inter / ((a[2:] - a[:2]).clamp(min=0).prod() +
                        (b[2:] - b[:2]).clamp(min=0).prod() - inter).clamp(min=1e-7)

    def process(self, data_batch, data_samples):
        for sample in data_samples:
            pred = sample.get('pred_instances', sample)
            gt = sample.get('gt_instances', sample.get('target', {}))
            pred = {key: value.detach().cpu() if torch.is_tensor(value) else value
                    for key, value in pred.items()}
            gt = {key: value.detach().cpu() if torch.is_tensor(value) else value
                  for key, value in gt.items()}
            self.results.append((pred, gt))

    def compute_metrics(self, results=None):
        counts = {t: [0, 0, 0] for t in self.thresholds}  # tp, pred-visible, gt-annotated
        vis_tp = vis_fp = vis_fn = vis_tn = matched = num_gt = 0
        for pred, gt in results or self.results:
            pb, ps, pl = pred['boxes'], pred['scores'], pred['labels']
            gb, gl = gt['boxes'], gt['labels']
            gt_size = gb[:, 2:] - gb[:, :2]
            valid_gt = ((gt_size[:, 0] >= self.min_bbox_size) &
                        (gt_size[:, 1] >= self.min_bbox_size))
            num_gt += int(valid_gt.sum())
            pk = pred.get('keypoints', pred.get('keypoints_xy'))
            pv = pred.get('keypoint_scores', pred.get('keypoints_vis'))
            if pv is not None and pv.numel() and (pv.min() < 0 or pv.max() > 1):
                pv = pv.sigmoid()
            gk, gv = gt.get('keypoints'), gt.get('keypoints_visible')
            used = set()
            for pi in torch.argsort(ps, descending=True).tolist():
                if ps[pi] < self.score_thr:
                    continue
                candidates = [(self._iou(pb[pi], gb[gi]), gi) for gi in range(len(gb))
                              if valid_gt[gi] and gi not in used and pl[pi] == gl[gi]]
                if not candidates:
                    continue
                iou, gi = max(candidates, key=lambda x: float(x[0]))
                if iou < self.iou_thr:
                    continue
                used.add(gi); matched += 1
                pred_visible, annotated = pv[pi] >= self.vis_thr, gv[gi] > 0
                truly_visible = gv[gi] == 2 if gv[gi].max() > 1 else annotated
                vis_tp += int((pred_visible & truly_visible).sum())
                vis_fp += int((pred_visible & ~truly_visible).sum())
                vis_fn += int((~pred_visible & truly_visible).sum())
                vis_tn += int((~pred_visible & ~truly_visible).sum())
                distance = torch.linalg.vector_norm(pk[pi].reshape(-1, 2) - gk[gi], dim=-1)
                for threshold in self.thresholds:
                    counts[threshold][0] += int(((distance <= threshold) & annotated & pred_visible).sum())
                    counts[threshold][1] += int(pred_visible.sum())
                    counts[threshold][2] += int(annotated.sum())
        metrics = OrderedDict()
        for threshold, (tp, npred, ngt) in counts.items():
            p, r = tp / max(npred, 1), tp / max(ngt, 1)
            suffix = f'{threshold:g}px'
            metrics[f'precision_{suffix}'] = p; metrics[f'recall_{suffix}'] = r
            metrics[f'f1_{suffix}'] = 2 * p * r / max(p + r, 1e-12)
        vp, vr = vis_tp / max(vis_tp + vis_fp, 1), vis_tp / max(vis_tp + vis_fn, 1)
        metrics.update(vis_precision=vp, vis_recall=vr,
                       vis_f1=2 * vp * vr / max(vp + vr, 1e-12),
                       vis_accuracy=(vis_tp + vis_tn) / max(vis_tp + vis_fp + vis_fn + vis_tn, 1),
                       num_matched=matched, num_gt=num_gt,
                       matched_ratio=matched / max(num_gt, 1))
        return metrics
