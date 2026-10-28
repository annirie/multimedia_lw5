"""
Lab work 5. Multimedia data processing.
Object detection with YOLO.

Usage:
  python main.py                                   all steps
  python main.py --steps static,nms                selected steps only
  python main.py --steps video --video path.mp4    a specific video (0 = webcam)
  python main.py --coco-n 1000                     bigger COCO subset for the metrics
  python main.py --fresh                           recompute results (downloads are kept)

Steps, in execution order:
  images   download sample photos into data/images (skipped if the folder has photos)
  dataset  download a COCO val2017 subset with labels (skipped if already prepared)
  static   tasks 1, 2, 4: inference on static images, bounding boxes
  iou      task 6: IoU
  nms      task 5: NMS
  video    task 3: video stream (first video in data/videos, otherwise a synthetic one)
  conf     study 1: Precision = f(c), Recall = f(c), own mAP50
  res      study 2 and extended study: mAP and FPS vs imgsz, nano vs small
  classes  study 3: worst detected classes and the reasons

Downloads and predictions are cached, so after an interruption the same command can be run again.
Results: output/ (plots, CSV tables, log.txt, summary.json).
Pretrained YOLOv8 weights are used, no training is done.
"""
import argparse
import json
import shutil
import sys

from lab5 import config as C
from lab5.utils import Tee, header

STEPS = ["images", "dataset", "static", "iou", "nms", "video", "conf", "res", "classes"]


def parse_args():
    ap = argparse.ArgumentParser(description="Lab 5: object detection with YOLO")
    ap.add_argument("--steps", default="all", help=f"comma-separated steps from {STEPS}, or all")
    ap.add_argument("--video", default=None, help="video file path or webcam index (0)")
    ap.add_argument("--coco-n", type=int, default=C.COCO_SUBSET_SIZE,
                    help="number of COCO val2017 images used for the metrics")
    ap.add_argument("--fresh", action="store_true", help="delete cached results and recompute")
    args = ap.parse_args()
    args.steps = set(STEPS) if args.steps == "all" else {s.strip() for s in args.steps.split(",")}
    unknown = args.steps - set(STEPS)
    if unknown:
        ap.error(f"unknown steps: {sorted(unknown)}")
    return args


def load_models():
    from ultralytics import YOLO
    C.WEIGHTS_DIR.mkdir(exist_ok=True)
    models = {}
    for name, fname in C.MODELS.items():
        local = C.ROOT / fname  # use weights next to main.py if they are there
        models[name] = YOLO(str(local if local.exists() else C.WEIGHTS_DIR / fname))
    return models


def save_summary(new):
    """Merges results into output/summary.json, so partial runs keep earlier results."""
    path = C.OUT / "summary.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    old.update(new)
    path.write_text(json.dumps(old, indent=2,
                               default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                    encoding="utf-8")


def run_preparation(step, *args):
    """A network error during data preparation should not stop the lab;
    the studies then use coco128 and the video task uses a synthetic video."""
    try:
        step(*args)
    except Exception as e:
        print(f"\nData preparation failed: {e}")
        print("Continuing with the data that is available.")


def run(args):
    from lab5 import prepare

    # Data preparation
    if "images" in args.steps:
        run_preparation(prepare.download_sample_photos)
    if "dataset" in args.steps:
        run_preparation(prepare.prepare_coco_subset, args.coco_n)

    lab_steps = args.steps - {"images", "dataset"}
    if not lab_steps:
        return

    from lab5 import data, studies, tasks

    models = load_models()
    main_model = models[C.MAIN_MODEL]
    print(f"\nModels: {', '.join(C.MODELS.values())}. Main model for tasks 1-5: {C.MODELS[C.MAIN_MODEL]}")
    print(f"Steps: {', '.join(s for s in STEPS if s in lab_steps)}")
    summary = {}

    # Tasks 1-6
    if "static" in lab_steps:
        tasks.task_static_images(main_model)
    if "iou" in lab_steps:
        tasks.task_iou_demo()
    if "nms" in lab_steps:
        tasks.task_nms(main_model)
    if "video" in lab_steps:
        summary["video"] = tasks.task_video(main_model, args.video)

    # Studies on the annotated dataset
    if lab_steps & {"conf", "res", "classes"}:
        data_yaml, ds_name, is_train = data.resolve_dataset()
        gt, names = data.load_ground_truth(data_yaml)
        ds_key = studies.dataset_key(ds_name, gt)
        n_obj = sum(len(it.cls) for it in gt)
        header(f"Dataset for metrics: {ds_name}, {len(gt)} images, {n_obj} labeled objects")
        summary["dataset"] = dict(name=ds_name, images=len(gt), objects=n_obj, is_training_data=is_train)
        if "conf" in lab_steps:
            studies.study_confidence(models, gt, ds_key)
        if "res" in lab_steps:
            studies.study_resolution(models, gt, data_yaml, ds_key)
        if "classes" in lab_steps:
            studies.study_classes(models, gt, ds_key, names)
        summary.update(studies.SUMMARY)

    save_summary(summary)
    print(f"\nDone. Results are in {C.OUT}")


def main():
    args = parse_args()
    C.OUT.mkdir(exist_ok=True)
    if args.fresh:  # only computed results are deleted; photos and datasets are kept
        shutil.rmtree(C.CACHE, ignore_errors=True)
        (C.OUT / "results_resolution_models.csv").unlink(missing_ok=True)
    tee = Tee(C.OUT / "log.txt")
    sys.stdout = tee
    try:
        run(args)
    finally:
        sys.stdout = tee.stdout
        tee.close()


if __name__ == "__main__":
    main()
