"""Tasks 1-6: inference, static images, video, bounding boxes, NMS, IoU."""
import time

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")  # save plots to files, no windows
import matplotlib.pyplot as plt
from ultralytics.utils import ASSETS

from .config import MAIN_MODEL, MODELS, NMS_IOU_THRESHOLDS, OUT, PRED, VIDEO_MAX_FRAMES
from .data import resolve_video_source, static_images
from .drawing import draw_detections, unpack
from .metrics import iou, nms, nms_per_class
from .utils import header, write_csv


# ========================================================================================
# Tasks 1, 2, 4. Inference of a pretrained YOLO on static images
# ========================================================================================
def task_static_images(model, n_runs=5):
    header(f"Tasks 1, 2, 4. Inference of {MODELS[MAIN_MODEL]} on static images "
           f"(conf >= {PRED['conf']}, NMS iou = {PRED['iou']})")
    print(f"Number of COCO classes: {len(model.names)}")
    rows = []
    for path in static_images():
        img = cv2.imread(str(path))
        if img is None:
            print(f"\nCould not read {path}, skipped")
            continue
        model.predict(img, **PRED, verbose=False)  # warm-up
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            result = model.predict(img, **PRED, verbose=False)[0]
            times.append((time.perf_counter() - t0) * 1000)
        boxes, scores, classes = unpack(result)
        sp = result.speed
        print(f"\n{path.name}: {img.shape[1]}x{img.shape[0]}, predict time (median of {n_runs}) = "
              f"{np.median(times):.1f} ms [preprocess {sp['preprocess']:.1f} + network "
              f"{sp['inference']:.1f} + NMS {sp['postprocess']:.1f} ms], objects: {len(boxes)}")
        for b, s, c in zip(boxes, scores, classes):
            print(f"  {model.names[c]:<14} conf={s:.2f}  box=[{', '.join(f'{v:.1f}' for v in b)}]")
            rows.append([path.name, model.names[c], float(s), *map(float, b)])

        annotated = draw_detections(img, boxes, scores, classes, model.names)
        cv2.imwrite(str(OUT / f"detect_{path.stem}.jpg"), annotated)

    write_csv(OUT / "detections_static.csv", ["image", "class", "conf", "x1", "y1", "x2", "y2"], rows)
    print("\nSaved: detect_*.jpg, detections_static.csv")


# ========================================================================================
# Task 6 (part 1). IoU
# ========================================================================================
def task_iou_demo():
    header("Task 6. IoU (Intersection over Union)")
    box1, box2, box3 = [50, 50, 150, 150], [90, 90, 190, 190], [300, 300, 400, 400]
    v12 = iou(box1, box2)
    print(f"IoU(box1, box2) = {v12:.3f}   (partial overlap)")
    print(f"IoU(box1, box3) = {iou(box1, box3):.3f}   (no overlap)")
    print(f"IoU(box1, box1) = {iou(box1, box1):.3f}   (identical boxes)")

    inter = [max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])]
    fig, ax = plt.subplots(figsize=(5, 5))
    for b, name, col in [(box1, "box1", "tab:blue"), (box2, "box2", "tab:orange")]:
        ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, lw=2,
                                   edgecolor=col, label=name))
    if inter[2] > inter[0] and inter[3] > inter[1]:
        ax.add_patch(plt.Rectangle((inter[0], inter[1]), inter[2] - inter[0], inter[3] - inter[1],
                                   color="tab:green", alpha=0.3, label="intersection"))
    ax.set_xlim(0, 220); ax.set_ylim(220, 0); ax.set_aspect("equal"); ax.legend()
    ax.set_title(f"IoU(box1, box2) = {v12:.3f}")
    fig.tight_layout(); fig.savefig(OUT / "iou_demo.png", dpi=120); plt.close(fig)
    print("Saved: iou_demo.png")


# Task 5. NMS
#   a) own NMS on a synthetic example;
#   b) own NMS on the raw boxes of the model (YOLO NMS disabled with iou=1.0),
#      compared with the built-in ultralytics NMS at different iou thresholds.
def task_nms(model):
    header("Task 5. NMS (Non-Maximum Suppression)")

    # a) synthetic example: 5 boxes, two clusters of duplicates
    boxes = np.array([[50, 50, 200, 200], [55, 55, 205, 205], [60, 60, 195, 195],
                      [300, 80, 420, 220], [310, 90, 410, 210]], float)
    scores = np.array([0.90, 0.75, 0.60, 0.85, 0.50])
    keep = nms(boxes, scores, iou_thr=0.5)
    print(f"Synthetic example: boxes before NMS = {len(boxes)}, after NMS = {len(keep)}, kept indices: {keep}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, idx, title in [(axes[0], range(len(boxes)), "Before NMS"), (axes[1], keep, "After NMS")]:
        for i in idx:
            b = boxes[i]
            ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="red"))
            ax.text(b[0], b[1] - 3, f"{scores[i]:.2f}", color="red", fontsize=8)
        ax.set_xlim(0, 450); ax.set_ylim(260, 0); ax.set_title(title)
    fig.tight_layout(); fig.savefig(OUT / "nms_synthetic.png", dpi=120); plt.close(fig)

    # b) real image
    img = cv2.imread(str(ASSETS / "bus.jpg"))
    raw = model.predict(img, **{**PRED, "iou": 1.0}, verbose=False)[0]  # iou=1.0: NMS removes nothing
    rb, rs, rc = unpack(raw)
    print(f"\nbus.jpg: raw boxes without NMS (conf >= {PRED['conf']}) = {len(rb)}")
    print(f"{'iou_threshold':>14} | {'own NMS':>8} | {'ultralytics NMS':>15}")
    rows = []
    for thr in NMS_IOU_THRESHOLDS:
        own = len(nms_per_class(rb, rs, rc, thr))
        lib = len(model.predict(img, **{**PRED, "iou": thr}, verbose=False)[0].boxes)
        rows.append([thr, len(rb), own, lib])
        print(f"{thr:>14} | {own:>8} | {lib:>15}")
    # A difference of one box is possible: ultralytics runs NMS before clipping boxes to the
    # image border, own NMS runs after clipping.
    write_csv(OUT / "nms_iou_threshold.csv",
              ["iou_threshold", "boxes_raw", "boxes_own_nms", "boxes_ultralytics_nms"], rows)

    keep = nms_per_class(rb, rs, rc, PRED["iou"])
    before = draw_detections(img, rb, rs, rc, model.names)
    after = draw_detections(img, rb[keep], rs[keep], rc[keep], model.names)
    fig, axes = plt.subplots(1, 2, figsize=(10, 7))
    axes[0].imshow(before[..., ::-1]); axes[0].set_title(f"Before NMS: {len(rb)} boxes")
    axes[1].imshow(after[..., ::-1]); axes[1].set_title(f"After NMS (iou={PRED['iou']}): {len(keep)} boxes")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "nms_real_bus.png", dpi=110); plt.close(fig)
    print("Saved: nms_synthetic.png, nms_real_bus.png, nms_iou_threshold.csv")


# Task 3. Video stream: frame -> detection -> boxes -> output file
def task_video(model, source_arg=None):
    header("Task 3. Video stream processing")
    source, label = resolve_video_source(source_arg)
    print(f"Video source: {label}")
    cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
    if not cap.isOpened():
        print("Could not open the video, task skipped.")
        return {}
    ok, frame = cap.read()
    if not ok:
        print("The video has no frames, task skipped.")
        return {}

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if not fps_in or not np.isfinite(fps_in) or fps_in <= 0 or fps_in > 240:
        fps_in = 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 0 or -1 for a webcam
    n_max = min(total, VIDEO_MAX_FRAMES) if total > 0 else VIDEO_MAX_FRAMES
    sample_at = set(np.linspace(1, n_max, 3).astype(int).tolist())
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(str(OUT / "video_detected.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps_in, (w, h))

    model.predict(frame, **PRED, verbose=False)  # warm-up
    n_frames, n_det, t_pipeline, t_model_ms, samples = 0, 0, 0.0, 0.0, []
    while ok and n_frames < n_max:
        t0 = time.perf_counter()
        result = model.predict(frame, **PRED, verbose=False)[0]
        boxes, scores, classes = unpack(result)
        annotated = draw_detections(frame, boxes, scores, classes, model.names)
        t_pipeline += time.perf_counter() - t0
        t_model_ms += sum(result.speed.values())
        n_frames += 1
        n_det += len(boxes)
        cv2.putText(annotated, f"FPS {n_frames / t_pipeline:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2, cv2.LINE_AA)
        writer.write(annotated)
        if n_frames in sample_at:
            samples.append((n_frames, annotated))
        ok, frame = cap.read()
    cap.release(); writer.release()

    fps_pipe = n_frames / t_pipeline
    fps_model = 1000 * n_frames / t_model_ms
    print(f"Processed {n_frames} frames ({w}x{h}, input video {fps_in:.0f} FPS)")
    print(f"  Pipeline FPS (predict + drawing boxes): {fps_pipe:.1f}")
    print(f"  Model FPS (preprocess + network + NMS): {fps_model:.1f}")
    print(f"  Detections: {n_det} ({n_det / n_frames:.1f} per frame on average)")
    print(f"  Real time (pipeline FPS >= video FPS): {'yes' if fps_pipe >= fps_in else 'no'}")

    if samples:
        fig, axes = plt.subplots(1, len(samples), figsize=(4 * len(samples), 4), squeeze=False)
        for ax, (i, fr) in zip(axes[0], samples):
            ax.imshow(fr[..., ::-1]); ax.set_title(f"Frame {i}"); ax.axis("off")
        fig.suptitle(f"Video frames with detections ({label})")
        fig.tight_layout(); fig.savefig(OUT / "video_frames.png", dpi=110); plt.close(fig)
    print("Saved: video_detected.mp4, video_frames.png")
    return dict(source=label, frames=n_frames, resolution=f"{w}x{h}", fps_pipeline=fps_pipe,
                fps_model=fps_model, detections_per_frame=n_det / n_frames)
