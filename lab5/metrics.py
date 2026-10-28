"""Own metric implementations (tasks 5, 6): IoU, NMS, TP/FP, Precision/Recall, AP, mAP.

Boxes are [x1, y1, x2, y2] in pixels (COCO convention, no "+1").
"""
import numpy as np

from .config import IOU_MATCH


# IoU = intersection area / union area
def iou(box_a, box_b):
    """IoU of two boxes."""
    x1, y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x2, y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def iou_matrix(a, b):
    """Vectorized IoU, returns a (len(a), len(b)) matrix. Works with empty arrays."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


# NMS: 1) sort by confidence; 2) keep the best box; 3) drop all boxes with IoU > iou_thr;
#      4) repeat. Same rule as torchvision: a box is dropped if IoU is strictly above the threshold.
def nms(boxes, scores, iou_thr):
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    scores = np.asarray(scores, float)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        best, rest = order[0], order[1:]
        keep.append(int(best))
        if rest.size == 0:
            break
        ious = iou_matrix(boxes[best:best + 1], boxes[rest])[0]
        order = rest[ious <= iou_thr]
    return keep


def nms_per_class(boxes, scores, classes, iou_thr):
    """Class-wise NMS, as in YOLO: boxes of different classes are shifted apart so they
    never overlap (shift = max coordinate + 1)."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    if len(boxes) == 0:
        return []
    offset = np.asarray(classes, float)[:, None] * (boxes.max() + 1)
    return nms(boxes + offset, scores, iou_thr)


# Matching detections to ground truth (TP / FP)
def match_predictions(pred_boxes, pred_scores, pred_cls, gt_boxes, gt_cls, thr=IOU_MATCH):
    """Greedy matching from the most confident detection: TP if there is an unused
    ground-truth box of the same class with IoU >= thr. Each ground-truth box is matched
    at most once, so duplicates become FP."""
    tp = np.zeros(len(pred_scores), bool)
    if len(pred_scores) == 0 or len(gt_cls) == 0:
        return tp
    ious = iou_matrix(pred_boxes, gt_boxes)
    used = np.zeros(len(gt_cls), bool)
    for i in np.argsort(-np.asarray(pred_scores), kind="stable"):
        cand = np.where((gt_cls == pred_cls[i]) & ~used, ious[i], -1.0)
        j = int(cand.argmax())
        if cand[j] >= thr:
            tp[i], used[j] = True, True
    return tp


def precision_recall_at(conf, tp, n_gt_total, c):
    """Micro-averaged Precision / Recall / F1 for detections with confidence >= c."""
    sel = conf >= c
    n_det, n_tp = int(sel.sum()), int(tp[sel].sum())
    p = n_tp / n_det if n_det else float("nan")
    r = n_tp / n_gt_total if n_gt_total else float("nan")
    f1 = 2 * p * r / (p + r) if n_tp else 0.0
    return dict(c=c, P=p, R=r, F1=f1, TP=n_tp, FP=n_det - n_tp, FN=int(n_gt_total) - n_tp)


def pr_curve(conf, tp, n_gt):
    """Precision-Recall curve: one point after each detection, in descending confidence order."""
    order = np.argsort(-conf, kind="stable")
    tpc = np.cumsum(tp[order])
    precision = tpc / np.arange(1, len(order) + 1)
    recall = tpc / max(n_gt, 1)
    return recall, precision, conf[order]


# AP@0.5: area under the envelope of the P(R) curve (continuous interpolation, PASCAL VOC 2010+).
# Ultralytics uses 101-point interpolation (COCO style), so a difference of about 0.01 is expected.
def average_precision(conf, tp, n_gt):
    if n_gt == 0:
        return float("nan")
    if len(conf) == 0:
        return 0.0
    recall, precision, _ = pr_curve(conf, tp, n_gt)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))  # envelope: P(R) is non-increasing
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def map50(conf, tp, cls, n_gt):
    """mAP@0.5 = mean AP over classes present in the ground truth. Returns (mAP, {class: AP})."""
    per_class = {int(k): average_precision(conf[cls == k], tp[cls == k], n_gt[k])
                 for k in np.where(n_gt > 0)[0]}
    return float(np.mean(list(per_class.values()))), per_class


def spearman(x, y):
    """Spearman rank correlation (no tie correction)."""
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])
