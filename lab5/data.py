"""Data: static images, video, ground-truth annotations for metrics."""
from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np
from ultralytics.data.utils import check_det_dataset, img2label_paths
from ultralytics.utils import ASSETS

from .config import (COCO_SUBSET_DIR, COCO_SUBSET_YAML, DATA_DIR, FALLBACK_DATA, IMAGES_DIR,
                     VIDEO_SOURCE, VIDEOS_DIR)

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


GTItem = namedtuple("GTItem", "path w h boxes cls rel_area px_area")


def list_files(folder, exts):
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)


# ----------------------------------------------------------------------------------------
# Static images (task 2)
# ----------------------------------------------------------------------------------------
def static_images():
    """Ultralytics sample images plus all photos from data/images."""
    return [ASSETS / "bus.jpg", ASSETS / "zidane.jpg"] + list_files(IMAGES_DIR, IMG_EXT)


# ----------------------------------------------------------------------------------------
# Annotated dataset (for Precision / Recall / mAP)
# ----------------------------------------------------------------------------------------
def resolve_dataset():
    """Returns (yaml, short name, whether it is part of the model's training data)."""
    if COCO_SUBSET_YAML.exists():
        return str(COCO_SUBSET_YAML), COCO_SUBSET_DIR.name, False
    print("\nCOCO val2017 subset not found, using coco128 instead.")
    print("coco128 images come from COCO train2017, so the metrics on it are optimistic.")
    return FALLBACK_DATA, "coco128", True


def load_ground_truth(data_yaml):
    """Reads validation images and their YOLO labels. Returns (items, names)."""
    data = check_det_dataset(data_yaml)  # coco128 is downloaded automatically if missing
    val = data["val"]
    img_dir = Path(val[0] if isinstance(val, (list, tuple)) else val)
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Expected an image folder, got: {img_dir}")

    paths = list_files(img_dir, IMG_EXT)
    items = []
    for p, lbl in zip(paths, img2label_paths([str(p) for p in paths])):
        img = cv2.imread(str(p))
        if img is None:
            print(f"  Skipping unreadable image {p.name}")
            continue
        h, w = img.shape[:2]
        lbl = Path(lbl)
        arr = np.loadtxt(lbl, ndmin=2) if lbl.exists() and lbl.stat().st_size > 0 else np.zeros((0, 5))
        if arr.size == 0:
            arr = np.zeros((0, 5))
        if arr.shape[1] != 5:
            raise ValueError(f"{lbl}: expected 'cls cx cy w h' (5 columns), found {arr.shape[1]}")
        cls = arr[:, 0].astype(int)
        cx, cy, bw, bh = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
        boxes = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h,
                          (cx + bw / 2) * w, (cy + bh / 2) * h], 1)
        items.append(GTItem(p, w, h, boxes, cls, bw * bh, bw * w * bh * h))
    return items, data["names"]


# ----------------------------------------------------------------------------------------
# Video (task 3)
# ----------------------------------------------------------------------------------------
def make_synthetic_video(path, n_frames=60, size=540):
    """Fallback video: the camera pans across bus.jpg."""
    img = cv2.imread(str(ASSETS / "bus.jpg"))
    h, w = img.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (size, size))
    for i in range(n_frames):
        x = int((w - size) * i / (n_frames - 1))
        y = int(0.25 * (h - size) + 0.5 * (h - size) * i / (n_frames - 1))
        writer.write(img[y:y + size, x:x + size])
    writer.release()


def resolve_video_source(cli_value=None):
    """Returns (source for cv2.VideoCapture, description). Priority: --video, VIDEO_SOURCE,
    the first video in data/videos, the synthetic video."""
    src = cli_value if cli_value is not None else VIDEO_SOURCE
    if src is not None:
        if str(src).isdigit():
            return int(src), f"webcam {src}"
        return Path(src), Path(src).name
    videos = list_files(VIDEOS_DIR, VIDEO_EXT)
    if videos:
        return videos[0], videos[0].name
    DATA_DIR.mkdir(exist_ok=True)
    syn = DATA_DIR / "synthetic.mp4"
    make_synthetic_video(syn)
    print("No video found in data/videos, using a synthetic video (pan across bus.jpg).")
    return syn, "synthetic (pan across bus.jpg)"
