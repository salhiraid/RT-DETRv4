"""
evaluation.py — COCO bbox evaluation, error analysis and keypoint metrics.

Port of the external evaluation toolkit's ``evaluation.py`` with identical
logic. The only change: ``from config import MAX_DETS, NUM_KPT`` (and the
``KPT_NAMES`` fallback) are module constants here, set by rtdetrv4_bridge.py
from its command line (``--max-dets``, ``--kpt-names``).
"""

from typing import List

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


MAX_DETS = 100
NUM_KPT = 31
KPT_NAMES = None

BBOX_SIZE_RANGES = (
    ('small', 0, 32),
    ('medium', 32, 96),
    ('large', 96, 256),
    ('xlarge', 256, 512),
    ('xxlarge', 512, 1e5),
)

ERROR_IOU_THRS = tuple(np.arange(0.50, 0.96, 0.05).round(2))


# ── Bounding-box evaluation ───────────────────────────────────────────────────

def evaluate_bbox(
    preds: List[dict],
    cat_ids: List[int],
    coco_gt: COCO,
    extended_sizes: bool = True,
) -> dict:
    """
    Run COCO bbox evaluation with:
      - standard COCO AP/AR
      - per-category AP/AR
      - PR curves
      - custom S/M/L/XL/XXL metrics

    Custom size bins are based on ORIGINAL GT bbox coordinates:
        sqrt(width * height)
    """
    coco_dt = coco_gt.loadRes(preds)

    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.catIds = cat_ids
    coco_eval.params.maxDets = [1, 10, MAX_DETS]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    _debug_category_ids(preds, cat_ids, coco_gt)

    eval_cat_ids = list(cat_ids)

    per_cat = _per_category_stats(coco_gt, coco_dt, eval_cat_ids)
    pr_curves = _pr_curves(coco_gt, coco_dt, eval_cat_ids)

    results = dict(
        stats=coco_eval.stats,
        per_cat=per_cat,
        pr_curves=pr_curves,
    )

    if extended_sizes:
        results['size_stats'] = evaluate_bbox_by_size(preds, cat_ids, coco_gt)
        results['per_cat_size_stats'] = {
            cat_id: evaluate_bbox_by_size(preds, [cat_id], coco_gt, verbose=False)
            for cat_id in eval_cat_ids
        }

    return results


def evaluate_bbox_by_size(
    preds: List[dict],
    cat_ids: List[int],
    coco_gt: COCO,
    verbose: bool = True,
) -> dict:
    """
    Evaluate custom S/M/L/XL/XXL metrics (AP, AP50, AP75, AR1, AR10, ARMAX).

    Size is calculated from ORIGINAL GT bbox: sqrt(bbox_width * bbox_height)
    """
    size_coco_gt = _coco_with_bbox_area(coco_gt)
    coco_dt = size_coco_gt.loadRes(preds)

    results = {}

    for name, low, high in BBOX_SIZE_RANGES:
        ce = COCOeval(size_coco_gt, coco_dt, 'bbox')
        ce.params.catIds = cat_ids
        ce.params.maxDets = [1, 10, MAX_DETS]
        ce.params.areaRng = [[low ** 2, high ** 2]]
        ce.params.areaRngLbl = [name]
        ce.evaluate()
        ce.accumulate()

        metrics = _bbox_metrics_from_eval(ce)

        metrics['num_gt'] = sum(
            1
            for ann in size_coco_gt.dataset['annotations']
            if (
                ann.get('category_id') in cat_ids
                and not ann.get('iscrowd', 0)
                and low ** 2 <= ann.get('area', 0) < high ** 2
            )
        )

        results[name] = metrics

        if verbose:
            high_label = 'inf' if high >= 1e5 else str(int(high))
            print(
                f'  {name:8s} '
                f'[{int(low)},{high_label})  '
                f'GT={metrics["num_gt"]}  '
                f'AP={metrics["ap"]:.3f}  '
                f'AP50={metrics["ap50"]:.3f}  '
                f'AP75={metrics["ap75"]:.3f}  '
                f'AR={metrics["ar"]:.3f}'
            )

    return results


# ── TP / FP / FN analysis ─────────────────────────────────────────────────────

def evaluate_bbox_errors(
    preds: List[dict],
    cat_ids: List[int],
    coco_gt: COCO,
    iou_thrs=ERROR_IOU_THRS,
) -> dict:
    """
    COCO-aligned TP / FP / FN counts for IoU 0.50 → 0.95, for all and each
    size bin, per-category counts, and FP diagnosis (background,
    localization, duplicate, class_confusion).
    """
    size_coco_gt = _coco_with_bbox_area(coco_gt)
    coco_dt = size_coco_gt.loadRes(preds)

    area_defs = [('all', 0, 1e5)] + list(BBOX_SIZE_RANGES)

    ce = COCOeval(size_coco_gt, coco_dt, 'bbox')
    ce.params.catIds = cat_ids
    ce.params.maxDets = [MAX_DETS]
    ce.params.iouThrs = np.asarray(iou_thrs, dtype=float)
    ce.params.areaRng = [[low ** 2, high ** 2] for _, low, high in area_defs]
    ce.params.areaRngLbl = [name for name, _, _ in area_defs]
    ce.evaluate()

    range_to_name = {
        tuple(float(v) for v in rng): name
        for name, rng in zip(ce.params.areaRngLbl, ce.params.areaRng)
    }

    counts = {
        _iou_key(t): {name: _empty_error_count() for name, _, _ in area_defs}
        for t in ce.params.iouThrs
    }

    per_cat = {
        _iou_key(t): {cat_id: _empty_error_count() for cat_id in cat_ids}
        for t in ce.params.iouThrs
    }

    for e in ce.evalImgs:
        if e is None:
            continue

        area_name = range_to_name.get(tuple(float(v) for v in e['aRng']))
        if area_name is None:
            continue

        gt_ignore = np.asarray(e['gtIgnore'], dtype=bool)

        for t_idx, thr in enumerate(ce.params.iouThrs):
            key = _iou_key(thr)

            dt_match = np.asarray(e['dtMatches'][t_idx]) > 0
            gt_match = np.asarray(e['gtMatches'][t_idx]) > 0
            dt_ignore = np.asarray(e['dtIgnore'][t_idx], dtype=bool)

            tp = int(np.sum(dt_match & ~dt_ignore))
            fp = int(np.sum(~dt_match & ~dt_ignore))
            fn = int(np.sum(~gt_match & ~gt_ignore))

            _add_error_count(counts[key][area_name], tp, fp, fn)

            if area_name == 'all' and e['category_id'] in per_cat[key]:
                _add_error_count(per_cat[key][e['category_id']], tp, fp, fn)

    for key in counts:
        for d in counts[key].values():
            _finalize_error_count(d)
        for d in per_cat[key].values():
            _finalize_error_count(d)

    fp_types = {}

    for thr in (0.50, 0.75):
        if any(np.isclose(ce.params.iouThrs, thr)):
            fp_types[_iou_key(thr)] = _diagnose_fp_types(
                preds, size_coco_gt, cat_ids, thr)

    return dict(
        iou=counts,
        per_category=per_cat,
        fp_types=fp_types,
    )


def _empty_error_count():
    return dict(
        tp=0,
        fp=0,
        fn=0,
        precision=0.0,
        recall=0.0,
        f1=0.0,
        fp_rate=0.0,
        fn_rate=0.0,
    )


def _add_error_count(d, tp, fp, fn):
    d['tp'] += int(tp)
    d['fp'] += int(fp)
    d['fn'] += int(fn)


def _finalize_error_count(d):
    tp = d['tp']
    fp = d['fp']
    fn = d['fn']

    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0

    d['precision'] = float(p)
    d['recall'] = float(r)
    d['f1'] = float(2 * p * r / (p + r)) if p + r else 0.0
    d['fp_rate'] = float(fp / (tp + fp)) if tp + fp else 0.0
    d['fn_rate'] = float(fn / (tp + fn)) if tp + fn else 0.0


# ── FP diagnosis ──────────────────────────────────────────────────────────────

def _diagnose_fp_types(preds, coco_gt, cat_ids, iou_thr):
    """
    Diagnostic FP categories:

    duplicate:       overlaps same-class GT >= requested IoU, but that GT was
                     already matched.
    class_confusion: overlaps another-class GT >= requested IoU.
    localization:    overlaps a GT at IoU >= 0.10 but not enough.
    background:      no meaningful GT overlap.
    """
    types = ['background', 'localization', 'duplicate', 'class_confusion']

    totals = {k: 0 for k in types}

    by_size = {name: {k: 0 for k in types} for name, _, _ in BBOX_SIZE_RANGES}

    preds_by_img = {}

    for p in preds:
        if p.get('category_id') in cat_ids:
            preds_by_img.setdefault(p['image_id'], []).append(p)

    for img_id in coco_gt.getImgIds():
        gts = [
            a
            for a in coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id], catIds=cat_ids))
            if not a.get('iscrowd', 0)
        ]

        dts = preds_by_img.get(img_id, [])

        if not dts:
            continue

        capped = []

        for cat_id in cat_ids:
            cat_dts = sorted(
                [d for d in dts if d.get('category_id') == cat_id],
                key=lambda d: d.get('score', 0),
                reverse=True,
            )[:MAX_DETS]
            capped.extend(cat_dts)

        capped.sort(key=lambda d: d.get('score', 0), reverse=True)

        matched_gt = set()

        for dt in capped:
            ious = np.array([_bbox_iou(dt['bbox'], gt['bbox']) for gt in gts], dtype=float)

            same_idx = [
                i for i, gt in enumerate(gts)
                if gt['category_id'] == dt['category_id']
            ]

            valid = [(i, ious[i]) for i in same_idx if i not in matched_gt]

            if valid:
                best_i, best_iou = max(valid, key=lambda x: x[1])

                if best_iou >= iou_thr:
                    matched_gt.add(best_i)
                    continue

            best_same = max([ious[i] for i in same_idx], default=0.0)

            best_any_i = int(np.argmax(ious)) if len(ious) else -1

            best_any = float(ious[best_any_i]) if best_any_i >= 0 else 0.0

            if best_same >= iou_thr:
                reason = 'duplicate'

            elif (
                best_any >= iou_thr
                and best_any_i >= 0
                and gts[best_any_i]['category_id'] != dt['category_id']
            ):
                reason = 'class_confusion'

            elif max(best_same, best_any) >= 0.10:
                reason = 'localization'

            else:
                reason = 'background'

            totals[reason] += 1
            by_size[_size_name_from_bbox(dt['bbox'])][reason] += 1

    return dict(
        total=totals,
        by_size=by_size,
    )


def _bbox_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter

    return inter / union if union > 0 else 0.0


def _size_name_from_bbox(bbox):
    size = float(np.sqrt(max(0.0, bbox[2] * bbox[3])))

    for name, low, high in BBOX_SIZE_RANGES:
        if low <= size < high:
            return name

    return 'xxlarge'


def _iou_key(iou):
    return f'{float(iou):.2f}'


def _coco_with_bbox_area(coco_gt):
    """
    Make a COCO copy whose annotation area is explicitly
    bbox_width * bbox_height (ORIGINAL annotation coordinates).
    """
    import copy

    gt_dict = copy.deepcopy(coco_gt.dataset)

    for ann in gt_dict['annotations']:
        if 'bbox' in ann:
            ann['area'] = float(ann['bbox'][2] * ann['bbox'][3])

    out = COCO()
    out.dataset = gt_dict
    out.createIndex()

    return out


def _bbox_metrics_from_eval(ce: COCOeval) -> dict:
    precision = ce.eval['precision']
    recall = ce.eval['recall']

    max_dets = list(ce.params.maxDets)
    max_idx = max_dets.index(MAX_DETS)

    def mean_valid(values):
        values = np.asarray(values)
        values = values[values > -1]
        return float(values.mean()) if values.size else -1.0

    def precision_at(iou=None):
        values = precision[:, :, :, 0, max_idx]

        if iou is not None:
            idx = np.where(np.isclose(ce.params.iouThrs, iou))[0]
            values = values[idx] if len(idx) else np.array([])

        return mean_valid(values)

    def recall_at(max_det):
        idx = max_dets.index(max_det)
        return mean_valid(recall[:, :, 0, idx])

    return dict(
        ap=precision_at(),
        ap50=precision_at(0.50),
        ap75=precision_at(0.75),
        ar1=recall_at(1),
        ar10=recall_at(10),
        ar=recall_at(MAX_DETS),
    )


# ── Class agnostic ────────────────────────────────────────────────────────────

def make_class_agnostic(
    preds: List[dict],
    coco_gt: COCO,
    group_ids: List[int] = None,
    agnostic_cat_id: int = 9999,
) -> tuple:
    """
    Deep-copy predictions + GT and merge requested categories into one
    `vehicle` category. Original predictions are not modified.
    """
    import copy

    def remap(cat_id):
        if group_ids is None:
            return agnostic_cat_id
        return agnostic_cat_id if cat_id in group_ids else cat_id

    agnostic_preds = copy.deepcopy(preds)

    for p in agnostic_preds:
        p['category_id'] = remap(p['category_id'])

    gt_dict = copy.deepcopy(coco_gt.dataset)

    for ann in gt_dict['annotations']:
        ann['category_id'] = remap(ann['category_id'])

    if group_ids is None:
        gt_dict['categories'] = [
            {'id': agnostic_cat_id, 'name': 'vehicle', 'supercategory': 'vehicle'}
        ]

    else:
        kept = [c for c in gt_dict['categories'] if c['id'] not in group_ids]
        kept.append({'id': agnostic_cat_id, 'name': 'vehicle', 'supercategory': 'vehicle'})
        gt_dict['categories'] = kept

    agnostic_coco_gt = COCO()
    agnostic_coco_gt.dataset = gt_dict
    agnostic_coco_gt.createIndex()

    return agnostic_preds, agnostic_coco_gt


# ── Occlusion ─────────────────────────────────────────────────────────────────

def make_occlusion_subset_gt(coco_gt: COCO, occlusion_level: str) -> COCO:
    import copy

    gt_dict = copy.deepcopy(coco_gt.dataset)

    for ann in gt_dict['annotations']:
        ann_occ = ann.get('occlusion', 'no')

        if ann_occ != occlusion_level:
            ann['iscrowd'] = 1
        else:
            ann['iscrowd'] = ann.get('iscrowd', 0)

    occ_coco_gt = COCO()
    occ_coco_gt.dataset = gt_dict
    occ_coco_gt.createIndex()

    return occ_coco_gt


def evaluate_bbox_by_occlusion(
    preds: List[dict],
    cat_ids: List[int],
    coco_gt: COCO,
    occlusion_levels: List[str] = ('no', 'light', 'medium', 'heavy'),
) -> dict:

    results = {}

    for occ in occlusion_levels:
        occ_coco_gt = make_occlusion_subset_gt(coco_gt, occ)
        coco_dt = occ_coco_gt.loadRes(preds)

        ce = COCOeval(occ_coco_gt, coco_dt, 'bbox')
        ce.params.catIds = cat_ids
        ce.params.maxDets = [1, 10, MAX_DETS]
        ce.evaluate()
        ce.accumulate()

        print(f'\n── Occlusion: {occ} ──')

        ce.summarize()

        results[occ] = (
            ce.stats
            if (ce.stats is not None and len(ce.stats))
            else np.zeros(12)
        )

    return results


# ── Standard category metrics ─────────────────────────────────────────────────

def _per_category_stats(coco_gt, coco_dt, eval_cat_ids: list) -> dict:
    per_cat = {}

    for cat_id in eval_cat_ids:
        ce = COCOeval(coco_gt, coco_dt, 'bbox')
        ce.params.catIds = [cat_id]
        ce.params.maxDets = [1, 10, MAX_DETS]
        ce.evaluate()
        ce.accumulate()
        ce.summarize()

        per_cat[cat_id] = (
            ce.stats
            if (ce.stats is not None and len(ce.stats))
            else np.zeros(12)
        )

    return per_cat


def _pr_curves(coco_gt, coco_dt, eval_cat_ids: list) -> dict:
    pr_curves = {}

    for cat_id in eval_cat_ids:
        ce = COCOeval(coco_gt, coco_dt, 'bbox')
        ce.params.catIds = [cat_id]
        ce.params.iouThrs = np.array([0.5])
        ce.params.maxDets = [MAX_DETS]
        ce.evaluate()
        ce.accumulate()

        try:
            prec = ce.eval['precision']
            precision = prec[0, :, 0, 0, 0]
            precision = np.where(precision == -1, 0, precision)

        except (KeyError, IndexError) as e:
            print(f'  cat_id={cat_id} precision error: {e}')
            precision = np.zeros(101)

        pr_curves[cat_id] = {
            'precision': precision,
            'recall': ce.params.recThrs,
        }

    return pr_curves


def _debug_category_ids(preds, cat_ids, coco_gt):
    pred_cat_ids = list(set(p['category_id'] for p in preds))
    gt_cat_ids = coco_gt.getCatIds()
    eval_cat_ids = [c for c in cat_ids if c in pred_cat_ids]

    print(f'cat_ids passed:  {cat_ids}')
    print(f'pred cat ids:    {pred_cat_ids}')
    print(f'GT cat ids:      {gt_cat_ids}')
    print(f'eval_cat_ids:    {eval_cat_ids}')


# ── Keypoint evaluation ───────────────────────────────────────────────────────

def evaluate_keypoints(
    preds: List[dict],
    coco_gt: COCO,
    cat_ids: List[int],
    thresholds: List[float] = (5.0, 10.0),
    vis_thr: float = 0.5,
    conf_thrs: List[float] = (0.0,),
    model_type: str = 'yoloxpose',
    margin: float = 0.05,
    crop_size: int = 512,
    kpt_names: List[str] = None,
) -> dict:
    """
    Pixel-level keypoint evaluation using bbox matching @ IoU=0.5.
    """
    kp_preds = [p for p in preds if 'keypoints' in p]

    if not kp_preds:
        print('No keypoint predictions found.')
        return {}

    coco_dt = coco_gt.loadRes(preds)

    coco_eval = _run_bbox_eval(coco_gt, coco_dt, cat_ids)

    if kpt_names is not None:
        num_kpt = len(kpt_names)

    else:
        sample_anns = coco_gt.loadAnns(coco_gt.getAnnIds())

        gt_num_kpt = next(
            (
                len(a['keypoints']) // 3
                for a in sample_anns
                if a.get('num_keypoints', 0) > 0
            ),
            None,
        )

        if gt_num_kpt is None:
            print('No GT keypoint annotations found — skipping keypoint evaluation')
            return {}

        num_kpt = gt_num_kpt

        kpt_names = KPT_NAMES

    pred_num_kpt = len(kp_preds[0]['keypoints']) // 3

    assert pred_num_kpt >= num_kpt, (
        f'predictions have {pred_num_kpt} keypoints '
        f'but evaluation expects {num_kpt}'
    )

    if model_type == 'yoloxpose':
        conf_thrs = [0.0]

    coord_tp = {conf: {thr: np.zeros(num_kpt) for thr in thresholds} for conf in conf_thrs}
    coord_fp = {conf: {thr: np.zeros(num_kpt) for thr in thresholds} for conf in conf_thrs}
    coord_fn = {conf: {thr: np.zeros(num_kpt) for thr in thresholds} for conf in conf_thrs}

    vis_tp = {conf: np.zeros(num_kpt) for conf in conf_thrs}
    vis_fp = {conf: np.zeros(num_kpt) for conf in conf_thrs}
    vis_fn = {conf: np.zeros(num_kpt) for conf in conf_thrs}
    vis_tn = {conf: np.zeros(num_kpt) for conf in conf_thrs}

    num_matched = {conf: 0 for conf in conf_thrs}

    num_gt_total = 0

    matched_pairs = []

    for eval_img in coco_eval.evalImgs:
        if eval_img is None:
            continue

        if eval_img['aRng'] != coco_eval.params.areaRng[0]:
            continue

        img_id = eval_img['image_id']
        cat_id = eval_img['category_id']
        gt_ids = eval_img['gtIds']
        dt_ids = eval_img['dtIds']
        dt_matches = eval_img['dtMatches'][0]
        dt_ignore = eval_img['dtIgnore'][0]
        gt_ignore = eval_img['gtIgnore']

        gt_anns = {
            ann['id']: ann
            for ann in coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id, catIds=[cat_id]))
        }

        dt_list = [coco_dt.anns[did] for did in dt_ids if did in coco_dt.anns]

        num_gt_total += sum(
            1
            for i, gid in enumerate(gt_ids)
            if (
                not gt_ignore[i]
                and gt_anns.get(gid, {}).get('num_keypoints', 0) > 0
            )
        )

        for dt_idx, (dt, gt_matched_id) in enumerate(zip(dt_list, dt_matches)):
            if (
                'keypoints' not in dt
                or gt_matched_id == 0
                or dt_ignore[dt_idx]
            ):
                continue

            gt_ann = gt_anns.get(int(gt_matched_id))

            if gt_ann is None or gt_ann.get('num_keypoints', 0) == 0:
                continue

            # Keep your original KP size filter.
            gx, gy, gw, gh = gt_ann['bbox']

            if gw < 64 or gh < 64:
                continue

            pair = _build_matched_pair(gt_ann, dt, vis_thr, margin, crop_size)

            matched_pairs.append(pair)

            for conf in conf_thrs:
                pred_visible_k_coord = (
                    pair['pred_visible_k_vis']
                    & (pair['pred_dcc'] > conf)
                )

                pred_visible_k_vis = pair['pred_visible_k_vis']

                distances = pair['distances']

                gt_visible = pair['gt_visible']

                for thr in thresholds:
                    dist_ok = distances <= thr

                    tp = pred_visible_k_coord & gt_visible & dist_ok
                    fp = pred_visible_k_coord & gt_visible & ~dist_ok
                    fn = (gt_visible & ~pred_visible_k_coord) | fp

                    coord_tp[conf][thr] += tp.astype(float)
                    coord_fp[conf][thr] += fp.astype(float)
                    coord_fn[conf][thr] += fn.astype(float)

                vis_tp[conf] += (pred_visible_k_vis & gt_visible).astype(float)
                vis_fp[conf] += (pred_visible_k_vis & ~gt_visible).astype(float)
                vis_fn[conf] += (~pred_visible_k_vis & gt_visible).astype(float)
                vis_tn[conf] += (~pred_visible_k_vis & ~gt_visible).astype(float)

                num_matched[conf] += 1

    results = {}

    for conf in conf_thrs:
        results[conf] = _aggregate_coord_metrics(
            coord_tp[conf],
            coord_fp[conf],
            coord_fn[conf],
            thresholds,
            num_matched[conf],
            num_gt_total,
        )

        results[conf]['vis'] = _aggregate_vis_metrics(
            vis_tp[conf],
            vis_fp[conf],
            vis_fn[conf],
            vis_tn[conf],
        )

        results[conf].update(
            dict(
                num_matched=num_matched[conf],
                num_gt=num_gt_total,
            )
        )

    results['matched_pairs'] = matched_pairs
    results['coco_eval'] = coco_eval
    results['coco_dt'] = coco_dt

    return results


# ── KP private helpers ─────────────────────────────────────────────────────────

def _run_bbox_eval(coco_gt, coco_dt, cat_ids) -> COCOeval:
    ce = COCOeval(coco_gt, coco_dt, 'bbox')
    ce.params.catIds = cat_ids
    ce.params.maxDets = [1, 10, 100]
    ce.params.iouThrs = np.array([0.5])
    ce.evaluate()

    return ce


def _build_matched_pair(
    gt_ann: dict,
    dt: dict,
    vis_thr: float,
    margin: float,
    crop_size: int,
) -> dict:

    gt_kpts = np.array(gt_ann['keypoints']).reshape(-1, 3)
    gt_xy = gt_kpts[:, :2]
    gt_vis = gt_kpts[:, 2]
    gt_visible = gt_vis == 2

    pred_kpts = np.array(dt['keypoints']).reshape(-1, 3)
    pred_kpts = pred_kpts[:len(gt_kpts)]
    pred_xy = pred_kpts[:, :2]
    pred_vis_raw = pred_kpts[:, 2]
    pred_visible_k = pred_vis_raw > vis_thr

    if 'keypoint_scores' in dt and dt['keypoint_scores'] is not None:
        pred_dcc = np.array(dt['keypoint_scores'])[:len(gt_kpts)]
    else:
        pred_dcc = np.ones(len(gt_kpts))

    gx, gy, gw, gh = gt_ann['bbox']

    mx = gw * margin
    my = gh * margin

    x1 = float(int(max(0, gx - mx)))
    y1 = float(int(max(0, gy - my)))
    x2 = float(int(gx + gw + mx))
    y2 = float(int(gy + gh + my))

    cw = x2 - x1
    ch = y2 - y1

    scale = min(crop_size / cw, crop_size / ch)

    new_w = int(cw * scale)
    new_h = int(ch * scale)

    pad_x = (crop_size - new_w) // 2
    pad_y = (crop_size - new_h) // 2

    offset = np.array([x1, y1])
    padding = np.array([pad_x, pad_y])

    gt_crop = (gt_xy - offset) * scale + padding
    pred_crop = (pred_xy - offset) * scale + padding

    distances = np.sqrt(((gt_crop - pred_crop) ** 2).sum(axis=1))

    return dict(
        img_id=gt_ann['image_id'] if 'image_id' in gt_ann else None,
        gt_ann=gt_ann,
        pred_ann=dt,
        gt_xy=gt_xy,
        pred_xy=pred_xy,
        gt_vis=gt_vis,
        gt_visible=gt_visible,
        pred_visible_k_vis=pred_visible_k,
        pred_dcc=pred_dcc,
        pred_vis_raw=pred_vis_raw,
        gt_crop=gt_crop,
        pred_crop=pred_crop,
        distances=distances,
        crop_coords=(x1, y1, x2, y2, new_w, new_h, pad_x, pad_y, scale, scale),
    )


def _aggregate_coord_metrics(
    coord_tp,
    coord_fp,
    coord_fn,
    thresholds,
    num_matched,
    num_gt_total,
) -> dict:
    results = {}

    for thr in thresholds:
        tp = coord_tp[thr]
        fp = coord_fp[thr]
        fn = coord_fn[thr]

        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)

        results[thr] = dict(
            coord_precision_per_kpt=prec.tolist(),
            coord_recall_per_kpt=rec.tolist(),
            coord_f1_per_kpt=f1.tolist(),
            mean_coord_precision=float(prec.mean()),
            mean_coord_recall=float(rec.mean()),
            mean_coord_f1=float(f1.mean()),
            tp=tp.tolist(),
            fp=fp.tolist(),
            fn=fn.tolist(),
        )

        print(f'\nCoord eval @{thr}px (crop 512×512):')
        print(f'  matched:   {num_matched} / {num_gt_total} GT with kpts')
        print(f'  precision: {prec.mean():.3f}')
        print(f'  recall:    {rec.mean():.3f}')
        print(f'  F1:        {f1.mean():.3f}')

    return results


def _aggregate_vis_metrics(vis_tp, vis_fp, vis_fn, vis_tn) -> dict:

    total = vis_tp + vis_fp + vis_fn + vis_tn

    prec = np.where(vis_tp + vis_fp > 0, vis_tp / (vis_tp + vis_fp), 0.0)
    rec = np.where(vis_tp + vis_fn > 0, vis_tp / (vis_tp + vis_fn), 0.0)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    acc = np.where(total > 0, (vis_tp + vis_tn) / total, 0.0)

    print('\nVisibility eval:')
    print(f'  precision: {prec.mean():.3f}')
    print(f'  recall:    {rec.mean():.3f}')
    print(f'  F1:        {f1.mean():.3f}')
    print(f'  accuracy:  {acc.mean():.3f}')

    return dict(
        vis_precision_per_kpt=prec.tolist(),
        vis_recall_per_kpt=rec.tolist(),
        vis_f1_per_kpt=f1.tolist(),
        vis_accuracy_per_kpt=acc.tolist(),
        mean_vis_precision=float(prec.mean()),
        mean_vis_recall=float(rec.mean()),
        mean_vis_f1=float(f1.mean()),
        mean_vis_accuracy=float(acc.mean()),
        vis_tp=vis_tp.tolist(),
        vis_fp=vis_fp.tolist(),
        vis_fn=vis_fn.tolist(),
        vis_tn=vis_tn.tolist(),
    )
