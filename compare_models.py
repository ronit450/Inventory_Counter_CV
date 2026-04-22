"""
Model Comparison Script
=======================
Runs two YOLO models (no tracking, no ReID) on the same video frames
and produces side-by-side images, a raw detections CSV, and a summary CSV.

Usage:
    python compare_models.py \
        --model1 path/to/old.pt \
        --model2 path/to/new.pt \
        --videos path/to/video_folder \
        --output comparison_results \
        [--frame-skip 5] \
        [--conf 0.35] \
        [--iou 0.45] \
        [--max-sample-frames 20] \
        [--img-size 640]
"""

import argparse
import csv
import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

import config

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_color(class_id: int) -> tuple:
    np.random.seed(class_id * 42 + 7)
    return tuple(int(c) for c in np.random.randint(50, 230, size=3))


def draw_boxes(frame, detections, model_label: str, conf_thresh: float):
    """Draw bounding boxes on a copy of frame. detections = list of (x1,y1,x2,y2,conf,cls_id)."""
    out = frame.copy()
    h, w = out.shape[:2]

    for (x1, y1, x2, y2, conf, cls_id) in detections:
        color = get_color(cls_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cls_name = config.CLASS_NAMES.get(cls_id, f"cls_{cls_id}")
        if len(cls_name) > 18:
            cls_name = cls_name[:16] + ".."
        label = f"{cls_name} {conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Count overlay
    counts = defaultdict(int)
    for (_, _, _, _, _, cls_id) in detections:
        counts[config.CLASS_NAMES.get(cls_id, f"cls_{cls_id}")] += 1

    panel_h = 30 + len(counts) * 22 + 10
    overlay = out.copy()
    cv2.rectangle(overlay, (5, 5), (280, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)

    cv2.putText(out, model_label, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    y = 48
    for cls_name, cnt in sorted(counts.items()):
        cv2.putText(out, f"  {cls_name}: {cnt}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        y += 22
    cv2.putText(out, f"  TOTAL: {sum(counts.values())}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return out


def run_detect(model: YOLO, frame, conf: float, iou: float, img_size: int):
    """Run raw detection (no tracking). Returns list of (x1,y1,x2,y2,conf,cls_id)."""
    results = model.predict(frame, conf=conf, iou=iou, imgsz=img_size, verbose=False)
    result = results[0]
    dets = []
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        for box, c, cid in zip(boxes, confs, cls_ids):
            x1, y1, x2, y2 = map(int, box)
            dets.append((x1, y1, x2, y2, float(c), int(cid)))
    return dets


def make_side_by_side(frame_left, frame_right):
    """Stack two frames horizontally with a divider."""
    h = max(frame_left.shape[0], frame_right.shape[0])
    # Pad to same height if needed
    def pad_h(img, target_h):
        dh = target_h - img.shape[0]
        if dh > 0:
            img = cv2.copyMakeBorder(img, 0, dh, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        return img
    l = pad_h(frame_left, h)
    r = pad_h(frame_right, h)
    divider = np.full((h, 4, 3), (80, 80, 80), dtype=np.uint8)
    return np.hstack([l, divider, r])


# ── Per-video processing ──────────────────────────────────────────────────────

def process_video(video_path: str, model1: YOLO, model2: YOLO,
                  model1_name: str, model2_name: str,
                  output_dir: str, args) -> dict:
    video_name = Path(video_path).stem
    video_out_dir = os.path.join(output_dir, video_name)
    frames_dir = os.path.join(video_out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: cannot open {video_path}")
        return {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # Which frames to save as images (evenly sampled)
    processed_frame_indices = list(range(0, total_frames, args.frame_skip))
    step = max(1, len(processed_frame_indices) // args.max_sample_frames)
    save_set = set(processed_frame_indices[::step][:args.max_sample_frames])

    print(f"  Frames total      : {total_frames}  |  FPS: {fps:.1f}")
    print(f"  Processed frames  : {len(processed_frame_indices)}  (every {args.frame_skip})")
    print(f"  Saved side-by-side: up to {args.max_sample_frames}")

    raw_rows = []  # for detections.csv

    # Accumulators: per-class raw detection counts (sum of per-frame counts)
    m1_class_counts = defaultdict(int)
    m2_class_counts = defaultdict(int)
    m1_conf_sum, m1_det_total = 0.0, 0
    m2_conf_sum, m2_det_total = 0.0, 0
    m1_time_total, m2_time_total = 0.0, 0.0

    frame_idx = -1
    pbar = tqdm(total=total_frames, desc="  Comparing", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        pbar.update(1)

        if frame_idx % args.frame_skip != 0:
            continue

        # ── Model 1 ──
        t0 = time.perf_counter()
        dets1 = run_detect(model1, frame, args.conf, args.iou, args.img_size)
        m1_time_total += time.perf_counter() - t0

        # ── Model 2 ──
        t0 = time.perf_counter()
        dets2 = run_detect(model2, frame, args.conf, args.iou, args.img_size)
        m2_time_total += time.perf_counter() - t0

        # Accumulate counts
        for (_, _, _, _, c, cid) in dets1:
            m1_class_counts[cid] += 1
            m1_conf_sum += c
            m1_det_total += 1
        for (_, _, _, _, c, cid) in dets2:
            m2_class_counts[cid] += 1
            m2_conf_sum += c
            m2_det_total += 1

        # CSV rows
        for (x1, y1, x2, y2, c, cid) in dets1:
            raw_rows.append([video_name, frame_idx, model1_name,
                             config.CLASS_NAMES.get(cid, f"cls_{cid}"), cid,
                             f"{c:.4f}", x1, y1, x2, y2])
        for (x1, y1, x2, y2, c, cid) in dets2:
            raw_rows.append([video_name, frame_idx, model2_name,
                             config.CLASS_NAMES.get(cid, f"cls_{cid}"), cid,
                             f"{c:.4f}", x1, y1, x2, y2])

        # Save side-by-side image
        if frame_idx in save_set:
            ann1 = draw_boxes(frame, dets1, model1_name, args.conf)
            ann2 = draw_boxes(frame, dets2, model2_name, args.conf)
            combined = make_side_by_side(ann1, ann2)
            out_img = os.path.join(frames_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(out_img, combined, [cv2.IMWRITE_JPEG_QUALITY, 88])

    pbar.close()
    cap.release()

    n_processed = len(processed_frame_indices)
    return {
        "video": video_name,
        "total_frames": total_frames,
        "processed_frames": n_processed,
        "m1_name": model1_name,
        "m2_name": model2_name,
        "m1_class_counts": dict(m1_class_counts),
        "m2_class_counts": dict(m2_class_counts),
        "m1_total_dets": m1_det_total,
        "m2_total_dets": m2_det_total,
        "m1_avg_conf": m1_conf_sum / max(m1_det_total, 1),
        "m2_avg_conf": m2_conf_sum / max(m2_det_total, 1),
        "m1_ms_per_frame": (m1_time_total / max(n_processed, 1)) * 1000,
        "m2_ms_per_frame": (m2_time_total / max(n_processed, 1)) * 1000,
        "raw_rows": raw_rows,
        "frames_dir": frames_dir,
    }


# ── Write CSVs ────────────────────────────────────────────────────────────────

def write_detections_csv(all_results: list, output_dir: str):
    path = os.path.join(output_dir, "detections.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame", "model", "class_name", "class_id",
                    "confidence", "x1", "y1", "x2", "y2"])
        for res in all_results:
            w.writerows(res["raw_rows"])
    print(f"\n  Detections CSV  : {path}")
    return path


def write_summary_csv(all_results: list, output_dir: str):
    # Collect all class ids across both models
    all_class_ids = set()
    for res in all_results:
        all_class_ids.update(res["m1_class_counts"].keys())
        all_class_ids.update(res["m2_class_counts"].keys())
    sorted_classes = sorted(all_class_ids)

    path = os.path.join(output_dir, "summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)

        # Header
        header = ["video", "model", "total_detections", "avg_confidence", "ms_per_frame"]
        for cid in sorted_classes:
            header.append(config.CLASS_NAMES.get(cid, f"cls_{cid}"))
        w.writerow(header)

        for res in all_results:
            for model_key in ("m1", "m2"):
                row = [
                    res["video"],
                    res[f"{model_key}_name"],
                    res[f"{model_key}_total_dets"],
                    f"{res[f'{model_key}_avg_conf']:.4f}",
                    f"{res[f'{model_key}_ms_per_frame']:.1f}",
                ]
                counts = res[f"{model_key}_class_counts"]
                for cid in sorted_classes:
                    row.append(counts.get(cid, 0))
                w.writerow(row)

    print(f"  Summary CSV     : {path}")
    return path


def print_comparison(all_results: list):
    for res in all_results:
        print(f"\n  {'─'*60}")
        print(f"  Video: {res['video']}")
        print(f"  {'─'*60}")
        print(f"  {'CLASS':<35} {res['m1_name']:>12} {res['m2_name']:>12}  DIFF")
        print(f"  {'─'*60}")

        all_cids = sorted(set(list(res["m1_class_counts"].keys()) +
                              list(res["m2_class_counts"].keys())))
        for cid in all_cids:
            name = config.CLASS_NAMES.get(cid, f"cls_{cid}")
            c1 = res["m1_class_counts"].get(cid, 0)
            c2 = res["m2_class_counts"].get(cid, 0)
            diff = c2 - c1
            diff_str = f"+{diff}" if diff > 0 else str(diff)
            print(f"  {name:<35} {c1:>12} {c2:>12}  {diff_str}")

        print(f"  {'─'*60}")
        print(f"  {'TOTAL DETECTIONS':<35} {res['m1_total_dets']:>12} {res['m2_total_dets']:>12}"
              f"  {res['m2_total_dets'] - res['m1_total_dets']:+}")
        print(f"  {'AVG CONFIDENCE':<35} {res['m1_avg_conf']:>12.3f} {res['m2_avg_conf']:>12.3f}")
        print(f"  {'MS / FRAME':<35} {res['m1_ms_per_frame']:>11.1f} {res['m2_ms_per_frame']:>11.1f}")
        print(f"  Saved frames    : {res['frames_dir']}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="YOLO model comparison (no tracking, no ReID)")
    p.add_argument("--model1", required=True, help="Path to first (old) model .pt")
    p.add_argument("--model2", required=True, help="Path to second (new) model .pt")
    p.add_argument("--videos", default=config.INPUT_FOLDER,
                   help="Folder containing input videos")
    p.add_argument("--output", default="comparison_results",
                   help="Output folder")
    p.add_argument("--frame-skip", type=int, default=config.FRAME_SKIP,
                   help="Process every Nth frame")
    p.add_argument("--conf", type=float, default=config.YOLO_CONFIDENCE,
                   help="Detection confidence threshold")
    p.add_argument("--iou", type=float, default=config.YOLO_IOU,
                   help="NMS IoU threshold")
    p.add_argument("--img-size", type=int, default=config.YOLO_IMG_SIZE,
                   help="Inference image size")
    p.add_argument("--max-sample-frames", type=int, default=20,
                   help="Max side-by-side images saved per video")
    p.add_argument("--model1-name", default="Model-v1",
                   help="Display name for model 1")
    p.add_argument("--model2-name", default="Model-v2",
                   help="Display name for model 2")
    return p.parse_args()


def main():
    args = parse_args()

    video_folder = Path(args.videos)
    if not video_folder.is_dir():
        print(f"ERROR: videos folder not found: {args.videos}")
        return

    videos = sorted(p for p in video_folder.iterdir()
                    if p.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        print(f"ERROR: no videos in {args.videos}")
        return

    os.makedirs(args.output, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"  Model 1  : {args.model1_name}  ({args.model1})")
    print(f"  Model 2  : {args.model2_name}  ({args.model2})")
    print(f"  Videos   : {len(videos)}  in  {args.videos}")
    print(f"  Output   : {args.output}")
    print(f"  Conf     : {args.conf}  |  IoU: {args.iou}  |  Skip: {args.frame_skip}")
    print(f"{'='*60}\n")

    print("[*] Loading models...")
    model1 = YOLO(args.model1)
    model2 = YOLO(args.model2)
    print("    Done.\n")

    all_results = []
    for idx, vp in enumerate(videos, 1):
        print(f"{'='*60}")
        print(f"  [{idx}/{len(videos)}] {vp.name}")
        print(f"{'='*60}")
        res = process_video(
            str(vp), model1, model2,
            args.model1_name, args.model2_name,
            args.output, args,
        )
        if res:
            all_results.append(res)

    if not all_results:
        print("No results. Exiting.")
        return

    print_comparison(all_results)
    write_detections_csv(all_results, args.output)
    write_summary_csv(all_results, args.output)

    print(f"\n{'='*60}")
    print(f"  DONE. Output in: {args.output}/")
    print(f"  - comparison_results/<video>/frames/   ← side-by-side JPGs")
    print(f"  - detections.csv                       ← raw per-detection rows")
    print(f"  - summary.csv                          ← per-class totals")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
