"""Data preparation: sample photos for task 2 and a COCO val2017 subset for the metrics.

Both steps skip files that already exist, so they can be re-run after an interruption.
"""
import random
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ultralytics
import yaml

from .config import COCO_SUBSET_DIR, COCO_SUBSET_SIZE, COCO_SUBSET_YAML, DATASETS, IMAGES_DIR, SEED, VIDEOS_DIR
from .data import IMG_EXT, list_files
from .utils import header

# Sample photos for task 2 
PHOTOS = [
    ("traffic_highway", "qOJhhDD2b8c", "Michael Myers", "car, truck, bus, motorcycle"),
    ("osaka_street", "P3vvI9GZogg", "Mario Scheibl", "person, bicycle, traffic light"),
    ("park_cycling", "jZgLIW0UpvA", "Halil Celik", "person, bicycle, dog"),
    ("people_with_dogs", "oa8sbi-DRmg", "Marin Tulard", "person, dog"),
    ("snow_dog_walk", "eMeekHa41FM", "Jim Luo", "person, dog"),
    ("cat_on_couch", "DLKybuvRwyo", "Thomas Haas", "cat, couch"),
    ("horses_field", "L5dBOfICzRE", "Xie Jian", "horse"),
    ("desk_workspace", "TEXJaKQTw2c", "Kate Tepla", "tv, keyboard, mouse, cup"),
]
PHOTO_WIDTH = 1920  # originals can be 5000+ px wide
PHOTO_URL = "https://unsplash.com/photos/{id}/download?force=true&w={w}"
PHOTO_PAGE = "https://unsplash.com/photos/{id}"

# COCO val2017 subset: YOLO-format labels from ultralytics releases + individual images
LABELS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"
COCO_IMG_URL = "http://images.cocodataset.org/val2017/{}.jpg"


def download(url, dst, show_progress=False, require_image=False):
    """Downloads to a temporary .part file and renames it at the end, so an interrupted
    download does not leave a broken file. Existing files are skipped."""
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (lab5-yolo)"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            ctype = r.headers.get("Content-Type", "")
            if require_image and not ctype.startswith("image/"):
                raise ValueError(f"response is not an image ({ctype or 'unknown type'})")
            total, done, next_report = int(r.headers.get("Content-Length") or 0), 0, 0.1
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if show_progress and total and done / total >= next_report:  # every 10%
                    print(f"  {done / 2 ** 20:.0f} / {total / 2 ** 20:.0f} MB")
                    next_report += 0.1
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)
    return dst


def download_sample_photos():
    header("Preparation: photos for task 2")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    existing = list_files(IMAGES_DIR, IMG_EXT)
    if existing:
        print(f"data/images already contains {len(existing)} photos, nothing to download.")
        return

    failed = []
    for name, photo_id, author, expected in PHOTOS:
        dst = IMAGES_DIR / f"{name}.jpg"
        try:
            download(PHOTO_URL.format(id=photo_id, w=PHOTO_WIDTH), dst, require_image=True)
            print(f"  {dst.name} downloaded ({dst.stat().st_size / 2 ** 20:.1f} MB), expected: {expected}")
        except Exception as e:  # keep going with the other photos
            failed.append((name, photo_id))
            print(f"  {dst.name} failed: {e}")

    lines = ["Photos: Unsplash (https://unsplash.com), Unsplash License.", ""]
    lines += [f"{name}.jpg - {author}, {PHOTO_PAGE.format(id=pid)}" for name, pid, author, _ in PHOTOS]
    (IMAGES_DIR / "CREDITS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for name, pid in failed:
        print(f"  Manual download for {name}.jpg: {PHOTO_PAGE.format(id=pid)}")


# Step "dataset": COCO val2017 subset -> datasets/coco_val2017_subset
# val2017 images were not used for training, unlike coco128 (train2017).
def prepare_coco_subset(n=COCO_SUBSET_SIZE, workers=8):
    header(f"Preparation: COCO val2017 subset ({n} images)")
    lbl_dir = COCO_SUBSET_DIR / "labels" / "val2017"
    img_dir = COCO_SUBSET_DIR / "images" / "val2017"
    if (COCO_SUBSET_YAML.exists() and len(list(img_dir.glob("*.jpg"))) == n
            and len(list(lbl_dir.glob("*.txt"))) == n):
        print("Dataset is already prepared.")
        return

    # 1. Labels (downloaded once)
    zip_path = DATASETS / "_downloads" / "coco2017labels.zip"
    print("Downloading COCO labels (YOLO format)" if not zip_path.exists() else "COCO labels already downloaded")
    download(LABELS_URL, zip_path, show_progress=True)
    if not zipfile.is_zipfile(zip_path):
        zip_path.unlink()
        raise RuntimeError("label archive is corrupted and was removed, run the step again")

    lbl_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        members = {Path(m).stem: m for m in z.namelist() if "labels/val2017/" in m and m.endswith(".txt")}
        if not members:
            raise RuntimeError(f"labels/val2017 not found in {zip_path}")
        # 2. Pick images: shuffle with a fixed seed, so the subset is the same on every run
        stems = sorted(members)
        random.Random(SEED).shuffle(stems)
        chosen = sorted(stems[:n])
        keep = set(chosen)
        for d, ext in ((lbl_dir, ".txt"), (img_dir, ".jpg")):  # remove files from an older subset
            for p in d.glob(f"*{ext}"):
                if p.stem not in keep:
                    p.unlink()
        for s in chosen:
            (lbl_dir / f"{s}.txt").write_bytes(z.read(members[s]))
    for p in COCO_SUBSET_DIR.glob("labels/*.cache"):  # stale ultralytics label cache
        p.unlink()

    # 3. Images (only the chosen ones, in parallel)
    missing = [s for s in chosen if not (img_dir / f"{s}.jpg").exists()]
    print(f"Selected {len(chosen)} of {len(members)} val2017 images, to download: {len(missing)}")

    def fetch(stem):
        try:
            download(COCO_IMG_URL.format(stem), img_dir / f"{stem}.jpg")
            return None
        except Exception as e:
            return stem, e

    failed = []
    with ThreadPoolExecutor(workers) as ex:
        for i, res in enumerate(ex.map(fetch, missing), 1):
            if res:
                failed.append(res)
            if i % 100 == 0 or i == len(missing):
                print(f"  {i} / {len(missing)}")

    # 4. data.yaml (class names from coco.yaml shipped with ultralytics)
    coco_yaml = Path(ultralytics.__file__).parent / "cfg" / "datasets" / "coco.yaml"
    names = yaml.safe_load(coco_yaml.read_text(encoding="utf-8"))["names"]
    cfg = {"path": str(COCO_SUBSET_DIR.resolve()), "train": "images/val2017", "val": "images/val2017",
           "names": names}
    COCO_SUBSET_YAML.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    print(f"Done: {len(list(img_dir.glob('*.jpg')))} images")
    if failed:
        print(f"{len(failed)} images failed to download (first: {failed[0][0]}: {failed[0][1]}). "
              f"Running the step again downloads only the missing ones.")
