"""Studies 1-3 and the nano vs small comparison.

Predictions on the annotated dataset are computed once per model and stored in output/cache.
"""
import hashlib
import time

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import (CACHE, CONF_GRID, CONF_THRESHOLDS, EVAL, FPS_N_IMAGES, FPS_REPEATS, IMG_SIZES,
                     IOU_MATCH, MAIN_MODEL, MIN_INSTANCES, OUT, PRED, REALTIME_FPS, VAL)
from .drawing import unpack
from .metrics import iou_matrix, map50, match_predictions, pr_curve, precision_recall_at, spearman
from .utils import header, read_csv, write_csv

SUMMARY = {}  # key numbers for the conclusions, saved to output/summary.json


# Inference on the dataset (cached) and matching with ground truth
def dataset_key(name, gt):
    h = hashlib.md5("".join(it.path.stem for it in gt).encode()).hexdigest()[:8]
    return f"{name}_{len(gt)}_{h}"


def _cache_file(name, ds_key, imgsz):
    return CACHE / f"preds_{name}_{ds_key}_{imgsz}.npz"


def collect_predictions(name, model, gt, ds_key, imgsz=640):
    """All detections with conf >= 0.001 for every image; they are filtered by c later.
    This is equivalent to a separate run with threshold c, because in greedy NMS a box with
    lower confidence can never suppress a box with higher confidence."""
    f = _cache_file(name, ds_key, imgsz)
    if f.exists():
        d = np.load(f)
        print(f"  [{name}] predictions loaded from cache ({f.name})")
        return {k: d[k] for k in d.files}

    CACHE.mkdir(parents=True, exist_ok=True)
    img_idx, boxes, conf, cls = [], [], [], []
    t0 = time.perf_counter()
    for i, it in enumerate(gt):
        r = model.predict(str(it.path), imgsz=imgsz, **EVAL, verbose=False)[0]
        b, s, c = unpack(r)
        img_idx.append(np.full(len(s), i, int)); boxes.append(b); conf.append(s); cls.append(c)
        if (i + 1) % 100 == 0 or i + 1 == len(gt):
            print(f"  [{name}] inference {i + 1}/{len(gt)} images, {time.perf_counter() - t0:.0f} s")
    preds = dict(img=np.concatenate(img_idx), boxes=np.concatenate(boxes).astype(np.float32),
                 conf=np.concatenate(conf), cls=np.concatenate(cls))
    tmp = CACHE / f"{f.stem}.tmp.npz"
    np.savez_compressed(tmp, **preds)
    tmp.replace(f)  # atomic rename, an interrupted write leaves no broken cache file
    return preds


def _image_bounds(preds, n_images):
    """Detections are stored in image order; returns [start, end) bounds for each image."""
    return np.searchsorted(preds["img"], np.arange(n_images + 1))


def match_all(preds, gt):
    st = _image_bounds(preds, len(gt))
    tp = np.zeros(len(preds["conf"]), bool)
    for i, it in enumerate(gt):
        a, b = st[i], st[i + 1]
        if b > a:
            tp[a:b] = match_predictions(preds["boxes"][a:b], preds["conf"][a:b], preds["cls"][a:b],
                                        it.boxes, it.cls)
    return tp


def gt_counts(gt, n_classes):
    return np.bincount(np.concatenate([it.cls for it in gt]).astype(int), minlength=n_classes)


# ========================================================================================
# Study 1. Precision = f(c), Recall = f(c), own mAP50
# ========================================================================================
def study_confidence(models, gt, ds_key):
    header(f"Study 1. Precision = f(c), Recall = f(c)  (IoU >= {IOU_MATCH}, imgsz = 640)")
    n_classes = len(next(iter(models.values())).names)
    n_gt = gt_counts(gt, n_classes)
    n_total = int(n_gt.sum())
    print(f"Ground-truth objects: {n_total}. Precision and Recall are micro-averaged "
          f"(all detections of all classes together).")

    rows, curves, res = [], {}, {}
    for name, model in models.items():
        preds = collect_predictions(name, model, gt, ds_key)
        tp = match_all(preds, gt)
        conf, cls = preds["conf"], preds["cls"]
        m50, _ = map50(conf, tp, cls, n_gt)
        print(f"\nYOLOv8 {name}: own mAP50 = {m50:.3f}")
        print(f"{'c':>5} | {'Precision':>9} | {'Recall':>6} | {'F1':>5} | {'TP':>5} | {'FP':>6} | {'FN':>5}")
        req = []
        for c in CONF_THRESHOLDS:
            d = precision_recall_at(conf, tp, n_total, c)
            req.append(d)
            rows.append([name, c, d["P"], d["R"], d["F1"], d["TP"], d["FP"], d["FN"]])
            print(f"{c:>5} | {d['P']:>9.3f} | {d['R']:>6.3f} | {d['F1']:>5.3f} | "
                  f"{d['TP']:>5} | {d['FP']:>6} | {d['FN']:>5}")
        grid = [precision_recall_at(conf, tp, n_total, c) for c in CONF_GRID]
        best = max(grid, key=lambda d: d["F1"])
        print(f"Optimal threshold by max F1 (grid 0.05..0.95): c = {best['c']:.2f} "
              f"(P = {best['P']:.3f}, R = {best['R']:.3f}, F1 = {best['F1']:.3f})")
        curves[name] = (grid, req, pr_curve(conf, tp, n_total))
        res[name] = dict(own_mAP50=m50, best_conf=best["c"], best_F1=best["F1"],
                         by_conf={str(d["c"]): dict(P=d["P"], R=d["R"], F1=d["F1"]) for d in req})
    SUMMARY["confidence_study"] = res

    write_csv(OUT / "precision_recall_vs_conf.csv",
              ["model", "conf", "precision", "recall", "f1", "TP", "FP", "FN"], rows)

    # Plot 1: P, R, F1 as functions of c (line: fine grid, markers: thresholds from the assignment)
    fig, axes = plt.subplots(1, len(curves), figsize=(6 * len(curves), 4.5), sharey=True, squeeze=False)
    for ax, (name, (grid, req, _)) in zip(axes[0], curves.items()):
        cs = [d["c"] for d in grid]
        for key, mk, col in (("P", "o", "C0"), ("R", "s", "C1"), ("F1", "^", "C2")):
            ax.plot(cs, [d[key] for d in grid], "-", color=col, alpha=0.5)
            ax.plot(CONF_THRESHOLDS, [d[key] for d in req], mk, color=col,
                    label={"P": "Precision", "R": "Recall", "F1": "F1"}[key])
        best = max(grid, key=lambda d: d["F1"])
        ax.axvline(best["c"], color="gray", ls=":", label=f"max F1 at c={best['c']:.2f}")
        ax.set_title(f"YOLOv8 {name}"); ax.set_xlabel("confidence threshold c")
        ax.set_xticks(CONF_THRESHOLDS); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axes[0][0].set_ylabel("metric value")
    fig.suptitle("Precision = f(c), Recall = f(c)")
    fig.tight_layout(); fig.savefig(OUT / "precision_recall_vs_conf.png", dpi=120); plt.close(fig)

    # Plot 2: Precision-Recall curves with the c thresholds marked
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for i, (name, (_, _, (rec, prec, cs))) in enumerate(curves.items()):
        ax.plot(rec, prec, color=f"C{i}", label=f"YOLOv8 {name}")
        for c in CONF_THRESHOLDS:
            k = int((cs >= c).sum())
            if k:
                ax.plot(rec[k - 1], prec[k - 1], "o", color=f"C{i}")
                ax.annotate(f"{c}", (rec[k - 1], prec[k - 1]), textcoords="offset points",
                            xytext=(4, 4), fontsize=7)
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve (labels: c)",
           xlim=(0, 1), ylim=(0, 1.05))
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "pr_curve.png", dpi=120); plt.close(fig)
    print("\nSaved: precision_recall_vs_conf.png, pr_curve.png, precision_recall_vs_conf.csv")


# Study 2 + extended study: mAP and FPS vs resolution, nano vs small
def measure_fps(model, images, imgsz, repeats=FPS_REPEATS):
    """Full predict time (preprocess + network + NMS) on single images, batch = 1."""
    for img in images[:3]:  # warm-up
        model.predict(img, imgsz=imgsz, **PRED, verbose=False)
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for img in images:
            r = model.predict(img, imgsz=imgsz, **PRED, verbose=False)[0]
            r.boxes.xyxy.cpu()  # wait for GPU / MPS to finish
        runs.append(len(images) / (time.perf_counter() - t0))
    return float(np.median(runs)), float(np.std(runs))


def study_resolution(models, gt, data_yaml, ds_key):
    header("Study 2 + extended study. mAP and FPS vs resolution, nano vs small")
    csv_path = OUT / "results_resolution_models.csv"
    cols = ["dataset", "model", "imgsz", "P_macro", "R_macro", "mAP50", "mAP50_95", "FPS", "FPS_std", "ms"]
    num = cols[3:]
    # Rows already computed for the same dataset are read back from the CSV
    done = {}
    for r in read_csv(csv_path):
        if r.get("dataset") == ds_key:
            done[(r["model"], int(r["imgsz"]))] = {**r, "imgsz": int(r["imgsz"]),
                                                   **{k: float(r[k]) for k in num}}

    images = [cv2.imread(str(it.path)) for it in gt[:FPS_N_IMAGES]]
    first = next(iter(models.values()))
    first.predict(images[0], **PRED, verbose=False)
    print(f"Device: {first.device}. FPS is the median of {FPS_REPEATS} runs over {len(images)} images, "
          f"batch = 1. P_macro / R_macro come from ultralytics (mean over classes at the max-F1 threshold).")

    order = [(n, s) for n in models for s in IMG_SIZES]
    for name, s in order:
        if (name, s) in done:
            print(f"YOLOv8 {name:<5} imgsz={s:<4} loaded from the previous run")
            continue
        m = models[name].val(data=data_yaml, imgsz=s, **VAL)
        fps, fps_std = measure_fps(models[name], images, s)
        done[(name, s)] = dict(dataset=ds_key, model=name, imgsz=s, P_macro=float(m.box.mp),
                               R_macro=float(m.box.mr), mAP50=float(m.box.map50), mAP50_95=float(m.box.map),
                               FPS=fps, FPS_std=fps_std, ms=1000 / fps)
        r = done[(name, s)]
        print(f"YOLOv8 {name:<5} imgsz={s:<4} mAP50={r['mAP50']:.3f} mAP50-95={r['mAP50_95']:.3f} "
              f"FPS={fps:.1f} +/- {fps_std:.1f}")
        write_csv(csv_path, cols, [[done[k][c] for c in cols] for k in order if k in done])
    table = [done[k] for k in order]

    print(f"\n{'Model':<7}|{'imgsz':>6} |{'P_macro':>8} |{'R_macro':>8} |{'mAP50':>6} |{'mAP50-95':>9} |"
          f"{'FPS':>6} |{'ms/frame':>9}")
    for r in table:
        print(f"{r['model']:<7}|{r['imgsz']:>6} |{r['P_macro']:>8.3f} |{r['R_macro']:>8.3f} |"
              f"{r['mAP50']:>6.3f} |{r['mAP50_95']:>9.3f} |{r['FPS']:>6.1f} |{r['ms']:>9.1f}")

    # Own mAP50 vs ultralytics mAP50 (if predictions at imgsz = 640 are cached)
    n_gt = gt_counts(gt, len(first.names))
    for name in models:
        if _cache_file(name, ds_key, 640).exists() and (name, 640) in done:
            preds = collect_predictions(name, models[name], gt, ds_key)
            own, _ = map50(preds["conf"], match_all(preds, gt), preds["cls"], n_gt)
            print(f"mAP50 check ({name}, imgsz 640): own = {own:.3f}, ultralytics = "
                  f"{done[(name, 640)]['mAP50']:.3f} (difference comes from interpolation and preprocessing)")

    # Optimal configuration: highest mAP50-95 among those with FPS >= REALTIME_FPS
    rt = [r for r in table if r["FPS"] >= REALTIME_FPS]
    best_rt = max(rt, key=lambda r: r["mAP50_95"]) if rt else None
    if best_rt:
        print(f"\nMost accurate configuration with FPS >= {REALTIME_FPS}: YOLOv8 {best_rt['model']}, "
              f"imgsz={best_rt['imgsz']} (mAP50-95={best_rt['mAP50_95']:.3f}, FPS={best_rt['FPS']:.1f})")
    else:
        print(f"\nNo configuration reached {REALTIME_FPS} FPS on this device.")

    names = list(models)
    if len(names) == 2 and (names[0], 640) in done and (names[1], 640) in done:
        a, b = done[(names[0], 640)], done[(names[1], 640)]
        print(f"imgsz=640: {names[1]} vs {names[0]}: mAP50-95 {b['mAP50_95'] - a['mAP50_95']:+.3f}, "
              f"{a['FPS'] / b['FPS']:.2f}x slower")
    SUMMARY["resolution_study"] = dict(table=table, best_realtime=best_rt, realtime_fps=REALTIME_FPS)

    # Plots: FPS = f(imgsz), mAP = f(imgsz), accuracy vs speed
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for i, name in enumerate(models):
        col = f"C{i}"
        rs = [r for r in table if r["model"] == name]
        x = [r["imgsz"] for r in rs]
        axes[0].errorbar(x, [r["FPS"] for r in rs], yerr=[r["FPS_std"] for r in rs], fmt="o-",
                         color=col, capsize=3, label=f"YOLOv8 {name}")
        axes[1].plot(x, [r["mAP50"] for r in rs], "o-", color=col, label=f"{name}: mAP50")
        axes[1].plot(x, [r["mAP50_95"] for r in rs], "s--", color=col, label=f"{name}: mAP50-95")
        axes[2].plot([r["FPS"] for r in rs], [r["mAP50_95"] for r in rs], "o-", color=col,
                     label=f"YOLOv8 {name}")
        for r in rs:
            axes[2].annotate(str(r["imgsz"]), (r["FPS"], r["mAP50_95"]), textcoords="offset points",
                             xytext=(4, 4), fontsize=8)
    axes[0].axhline(REALTIME_FPS, color="gray", ls=":", label=f"{REALTIME_FPS} FPS")
    axes[2].axvline(REALTIME_FPS, color="gray", ls=":", label=f"{REALTIME_FPS} FPS")
    axes[0].set(title="FPS = f(resolution)", xlabel="imgsz, pixels", ylabel="FPS")
    axes[1].set(title="mAP = f(resolution)", xlabel="imgsz, pixels", ylabel="mAP")
    axes[2].set(title="Accuracy vs speed (labels: imgsz)", xlabel="FPS", ylabel="mAP50-95")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fps_map_vs_resolution.png", dpi=120); plt.close(fig)
    print("\nSaved: fps_map_vs_resolution.png, results_resolution_models.csv")


# Study 3. Which classes are detected worst, and why
CATS = ["found", "poor localization", "wrong class", "missed"]


def error_breakdown(preds, gt, conf_thr):
    """For every ground-truth object, decides what happened to it at the given confidence:
      0 found              - a detection of the same class with IoU >= 0.5
      1 poor localization  - a detection of the same class with 0.1 <= IoU < 0.5
      2 wrong class        - a detection of another class with IoU >= 0.5
      3 missed             - nothing close
    Returns (category per object, the confused class or -1)."""
    st = _image_bounds(preds, len(gt))
    cats, other = [], []
    for i, it in enumerate(gt):
        a, b = st[i], st[i + 1]
        keep = preds["conf"][a:b] >= conf_thr
        pb, pc = preds["boxes"][a:b][keep], preds["cls"][a:b][keep]
        ious = iou_matrix(it.boxes, pb)
        for g, k in enumerate(it.cls):
            same, diff = pc == k, pc != k
            best_same = ious[g, same].max(initial=0.0)
            best_diff = ious[g, diff].max(initial=0.0)
            if best_same >= IOU_MATCH:
                cats.append(0); other.append(-1)
            elif best_diff >= IOU_MATCH:
                cats.append(2); other.append(int(pc[diff][ious[g, diff].argmax()]))
            elif best_same >= 0.1:
                cats.append(1); other.append(-1)
            else:
                cats.append(3); other.append(-1)
    return np.array(cats, int), np.array(other, int)


def study_classes(models, gt, ds_key, names):
    conf_thr = PRED["conf"]
    header("Study 3. Worst and best detected classes (own AP@0.5, imgsz = 640)")
    n_classes = len(names)
    n_gt = gt_counts(gt, n_classes)
    all_cls = np.concatenate([it.cls for it in gt]).astype(int)
    rel = np.concatenate([it.rel_area for it in gt])
    px = np.concatenate([it.px_area for it in gt])

    aps = {}
    for name, model in models.items():
        preds = collect_predictions(name, model, gt, ds_key)
        _, aps[name] = map50(preds["conf"], match_all(preds, gt), preds["cls"], n_gt)
    cats, other = error_breakdown(collect_predictions(MAIN_MODEL, models[MAIN_MODEL], gt, ds_key),
                                  gt, conf_thr)

    info = {}
    for k in np.where(n_gt > 0)[0]:
        sel = all_cls == k
        fr = np.bincount(cats[sel], minlength=4) / sel.sum()
        conf_with = other[sel & (cats == 2)]
        top = names[int(np.bincount(conf_with).argmax())] if conf_with.size else "-"
        info[int(k)] = dict(n=int(sel.sum()), rel_area=float(np.median(rel[sel])),
                            small=float((px[sel] < 32 ** 2).mean()),
                            **{f"AP_{n}": aps[n][k] for n in models},
                            **{CATS[j]: float(fr[j]) for j in range(4)}, confused_with=top)

    cols = (["class", "instances", "reliable", "median_rel_area", "share_small"]
            + [f"AP50_{n}" for n in models] + ["found", "poor_loc", "wrong_class", "missed", "confused_with"])
    key = f"AP_{MAIN_MODEL}"
    write_csv(OUT / "per_class_ap.csv", cols, [
        [names[k], d["n"], int(d["n"] >= MIN_INSTANCES), d["rel_area"], d["small"]]
        + [d[f"AP_{n}"] for n in models] + [d[c] for c in CATS] + [d["confused_with"]]
        for k, d in sorted(info.items(), key=lambda kv: kv[1][key])])

    reliable = {k: d for k, d in info.items() if d["n"] >= MIN_INSTANCES}
    print(f"Classes in the dataset: {len(info)}, ranked (>= {MIN_INSTANCES} objects): {len(reliable)}.")
    print(f"Error breakdown for YOLOv8 {MAIN_MODEL} at c = {conf_thr}.")
    ranked = sorted(reliable.items(), key=lambda kv: kv[1][key])
    head = (f"{'class':<15}|{'objects':>8} |{'med.area':>9} |{'<32^2px':>7} |"
            + "|".join(f"{'AP ' + n:>8}" for n in models)
            + f" |{'found':>7}|{'loc.':>7}|{'wrong':>7}|{'missed':>8}| confused with")
    for title, part in [("Worst 10 classes:", ranked[:10]), ("Best 5 classes:", ranked[-5:][::-1])]:
        print(f"\n{title}\n{head}")
        for k, d in part:
            print(f"{names[k]:<15}|{d['n']:>8} |{d['rel_area'] * 100:>8.2f}% |{d['small'] * 100:>6.0f}% |"
                  + "|".join(f"{d[f'AP_{n}']:>8.3f}" for n in models)
                  + f" |{d[CATS[0]] * 100:>6.0f}%|{d[CATS[1]] * 100:>6.0f}%|{d[CATS[2]] * 100:>6.0f}%|"
                    f"{d[CATS[3]] * 100:>7.0f}%| {d['confused_with']}")

    # Dependence on object size
    ks = list(reliable)
    ap_arr = np.array([reliable[k][key] for k in ks])
    rho_area = spearman([reliable[k]["rel_area"] for k in ks], ap_arr)
    rho_small = spearman([reliable[k]["small"] for k in ks], ap_arr)
    print(f"\nSpearman correlation between class AP and median object area: {rho_area:+.2f}")
    print(f"Spearman correlation between class AP and share of small (< 32^2 px) objects: {rho_small:+.2f}")

    print(f"\nOutcome by object size (COCO size groups), YOLOv8 {MAIN_MODEL}:")
    print(f"{'size':<26}|{'objects':>9} |" + "|".join(f"{c:>20}" for c in CATS))
    size_bins = [("small (< 32^2 px)", px < 32 ** 2), ("medium (32^2-96^2 px)", (px >= 32 ** 2) & (px < 96 ** 2)),
                 ("large (>= 96^2 px)", px >= 96 ** 2)]
    by_size = {}
    for label, m in size_bins:
        fr = np.bincount(cats[m], minlength=4) / max(m.sum(), 1)
        by_size[label] = {CATS[j]: float(fr[j]) for j in range(4)}
        print(f"{label:<26}|{int(m.sum()):>9} |" + "|".join(f"{v * 100:>19.1f}%" for v in fr))

    # Plot: class AP vs median object area
    fig, ax = plt.subplots(figsize=(7, 5))
    areas = np.array([reliable[k]["rel_area"] for k in ks])
    sizes = np.array([reliable[k]["n"] for k in ks])
    ax.scatter(areas, ap_arr, s=20 + 3 * np.sqrt(sizes) * 4, alpha=0.6)
    for k, d in ranked[:6] + ranked[-3:]:
        ax.annotate(names[k], (d["rel_area"], d[key]), fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set(xlabel="median relative object area of the class (log scale)", ylabel=f"AP50 (YOLOv8 {MAIN_MODEL})",
           title=f"Class AP vs object size (Spearman rho = {rho_area:+.2f})")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "ap_vs_object_size.png", dpi=120); plt.close(fig)

    SUMMARY["classes_study"] = dict(
        worst=[dict(cls=names[k], n=d["n"], AP=d[key], confused_with=d["confused_with"]) for k, d in ranked[:5]],
        best=[dict(cls=names[k], n=d["n"], AP=d[key]) for k, d in ranked[-5:][::-1]],
        spearman_ap_vs_area=rho_area, spearman_ap_vs_small_share=rho_small, outcome_by_size=by_size)
    print("\nSaved: per_class_ap.csv, ap_vs_object_size.png")
