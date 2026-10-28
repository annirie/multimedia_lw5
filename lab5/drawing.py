"""Task 4. Drawing bounding boxes with class names and confidence."""
import cv2
import numpy as np


def class_color(cls_id):
    rng = np.random.default_rng(int(cls_id) * 7 + 3)
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def draw_detections(img, boxes, scores, classes, names):
    """Draws rectangles and 'class conf' labels on top of the image (BGR)."""
    out = img.copy()
    thickness = max(2, round(sum(img.shape[:2]) / 600))
    font_scale = thickness / 3
    for (x1, y1, x2, y2), s, c in zip(boxes, scores, classes):
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        color = class_color(c)
        cv2.rectangle(out, p1, p2, color, thickness)
        label = f"{names[int(c)]} {s:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thickness - 1))
        ty = p1[1] - 4 if p1[1] - th - 6 > 0 else p1[1] + th + 4
        cv2.rectangle(out, (p1[0], ty - th - 4), (p1[0] + tw + 4, ty + 4), color, -1)
        cv2.putText(out, label, (p1[0] + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), max(1, thickness - 1), cv2.LINE_AA)
    return out


def unpack(result):
    """Extracts box (N,4), confidence (N,) and class (N,) arrays from an ultralytics result."""
    b = result.boxes
    return (b.xyxy.cpu().numpy().reshape(-1, 4), b.conf.cpu().numpy(),
            b.cls.cpu().numpy().astype(int))
