"""
Model Comparison WITH ReID
===========================
Runs two YOLO models through the full pipeline (ByteTrack + CLIP ReID)
independently on the same videos and compares deduplicated unique counts.

Each model gets its own fresh tracker + CLIPReIdentifier per video.
The video is read twice (once per model) so results are fully independent.

Usage:
    python compare_models_reid.py \
        --model1 old.pt \
        --model2 new.pt \
        --videos /path/to/videos \
        --output comparison_reid_results \
        [--model1-name "v1"] [--model2-name "v2"] \
        [--frame-skip 2] [--conf 0.35] [--iou 0.45] \
        [--reid-threshold 0.82] [--max-sample-frames 20]
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
from reid_module import CLIPReIdentifier

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


# ── Drawing helpers ───────────────────────────────────────────────────────────

def get_color(class_id: int) -> tuple:
    np.random.seed(class_id * 42 + 7)
    return tuple(int(c) for c in np.random.randint(50, 230, size=3))


def draw_detections(frame, boxes, track_ids, class_ids, confidences, canonical_map):
    out = frame.copy()
    for box, tid, cid, conf in zip(boxes, track_ids, class_ids, confidences):
        x1, y1, x2, y2 = map(int, box)
        color = get_color(int(cid))
        display_id = canonical_map.get(tid, tid)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cls_name = config.CLASS_NAMES.get(int(cid), f"cls_{int(cid)}")
        if len(cls_name) > 18:
            cls_name = cls_name[:16] + ".."
        label = f"ID:{display_id} {cls_name} {conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return out


def draw_counter_overlay(frame, model_label: str, unique_counts: dict,
                          raw_tracks: int, frame_idx: int, total_frames: int):
    h, w = frame.shape[:2]
    counts_list = sorted(unique_counts.items())
    panel_h = 55 + len(counts_list) * 22 + 30
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (300, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, f"{model_label}  [{frame_idx}/{total_frames}]",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Raw tracks: {raw_tracks}",
                (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    y = 68
    total_unique = 0
    for cls_name, cnt in counts_list:
        cv2.putText(frame, f"  {cls_name}: {cnt}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
        y += 22
        total_unique += cnt
    cv2.putText(frame, f"  UNIQUE TOTAL: {total_unique}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2)
    return frame


def make_side_by_side(left, right):
    h = max(left.shape[0], right.shape[0])
    def pad(img):
        dh = h - img.shape[0]
        if dh > 0:
            img = cv2.copyMakeBorder(img, 0, dh, 0, 0,
                                     cv2.BORDER_CONSTANT, value=(30, 30, 30))
        return img
    divider = np.full((h, 4, 3), (80, 80, 80), dtype=np.uint8)
    return np.hstack([pad(left), divider, pad(right)])


# ── Single-model pass ─────────────────────────────────────────────────────────

def run_model_on_video(video_path: str, model: YOLO, model_name: str,
                        args, save_frames: bool, frames_dir: str) -> dict:
    """
    Full pipeline pass: YOLO track + CLIP ReID.
    Returns summary dict with unique counts, raw tracks, timing, etc.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    reid = CLIPReIdentifier(
        similarity_threshold=args.reid_threshold,
        model_name=config.CLIP_MODEL_NAME,
    )

    # Determine which processed frames to save
    processed_indices = list(range(0, total_frames, args.frame_skip))
    step = max(1, len(processed_indices) // args.max_sample_frames)
    save_set = set(processed_indices[::step][:args.max_sample_frames])

    track_class_map   = {}
    track_conf_map    = {}
    canonical_map     = {}
    processed_count   = 0
    det_time_total    = 0.0
    saved_frames      = {}   # frame_idx -> annotated frame (for side-by-side later)

    frame_idx = -1
    pbar = tqdm(total=total_frames, desc=f"    {model_name}", unit="frame", leave=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        pbar.update(1)

        if frame_idx % args.frame_skip != 0:
            continue

        processed_count += 1

        t0 = time.perf_counter()
        results = model.track(
            frame,
            persist=True,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.img_size,
            tracker=f"{config.TRACKER_TYPE}.yaml",
            verbose=False,
        )
        det_time_total += time.perf_counter() - t0

        result = results[0]
        boxes, track_ids, class_ids, confidences = [], [], [], []

        if result.boxes is not None and result.boxes.id is not None:
            boxes       = result.boxes.xyxy.cpu().numpy()
            track_ids   = result.boxes.id.cpu().numpy().astype(int)
            class_ids   = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()

            reid.register_frame_tracks(track_ids, frame, frame_idx)

            for box, tid, cid, conf in zip(boxes, track_ids, class_ids, confidences):
                track_class_map[tid] = cid
                track_conf_map[tid]  = max(track_conf_map.get(tid, 0), conf)

                if processed_count % config.REID_CHECK_INTERVAL == 0:
                    x1c, y1c, x2c, y2c = map(int, box)
                    x1c, y1c = max(0, x1c), max(0, y1c)
                    x2c, y2c = min(width, x2c), min(height, y2c)
                    crop = frame[y1c:y2c, x1c:x2c]
                    if crop.size > 0:
                        reid.update_track(tid, cid, crop,
                                          (x1c, y1c, x2c, y2c),
                                          frame=frame)

        # Save annotated frame for side-by-side output
        if save_frames and frame_idx in save_set:
            ann = draw_detections(frame, boxes, track_ids, class_ids, confidences, canonical_map)
            unique_counts = reid.get_unique_counts()
            ann = draw_counter_overlay(ann, model_name, unique_counts,
                                        len(track_class_map), frame_idx, total_frames)
            saved_frames[frame_idx] = ann

    pbar.close()
    cap.release()

    reid.finalize()

    unique_counts  = reid.get_unique_counts()
    raw_tracks     = len(track_class_map)
    total_unique   = sum(unique_counts.values())
    dupes_removed  = raw_tracks - total_unique
    ms_per_frame   = (det_time_total / max(processed_count, 1)) * 1000

    return {
        "model_name":     model_name,
        "total_frames":   total_frames,
        "processed":      processed_count,
        "unique_counts":  unique_counts,
        "raw_tracks":     raw_tracks,
        "total_unique":   total_unique,
        "dupes_removed":  dupes_removed,
        "ms_per_frame":   ms_per_frame,
        "saved_frames":   saved_frames,   # frame_idx -> annotated BGR image
    }


# ── Per-video comparison ──────────────────────────────────────────────────────

def compare_video(video_path: str, model1: YOLO, model2: YOLO,
                   model1_name: str, model2_name: str,
                   output_dir: str, args) -> dict:
    video_name = Path(video_path).stem
    video_out  = os.path.join(output_dir, video_name)
    frames_dir = os.path.join(video_out, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    print(f"\n  -- {model1_name} pass --")
    res1 = run_model_on_video(video_path, model1, model1_name, args,
                               save_frames=True, frames_dir=frames_dir)

    print(f"\n  -- {model2_name} pass --")
    res2 = run_model_on_video(video_path, model2, model2_name, args,
                               save_frames=True, frames_dir=frames_dir)

    if not res1 or not res2:
        return {}

    # Save side-by-side images for frames present in both
    common_frames = sorted(set(res1["saved_frames"]) & set(res2["saved_frames"]))
    for fidx in common_frames:
        combined = make_side_by_side(res1["saved_frames"][fidx],
                                      res2["saved_frames"][fidx])
        cv2.imwrite(os.path.join(frames_dir, f"frame_{fidx:06d}.jpg"),
                    combined, [cv2.IMWRITE_JPEG_QUALITY, 88])

    print(f"\n  Saved {len(common_frames)} side-by-side frames → {frames_dir}")

    return {
        "video":      video_name,
        "model1":     res1,
        "model2":     res2,
    }


# ── Output ────────────────────────────────────────────────────────────────────

def print_comparison(all_results: list):
    for res in all_results:
        r1, r2 = res["model1"], res["model2"]
        m1, m2 = r1["model_name"], r2["model_name"]

        print(f"\n  {'='*65}")
        print(f"  VIDEO: {res['video']}")
        print(f"  {'='*65}")
        print(f"  {'CLASS':<35} {m1:>12} {m2:>12}  DIFF")
        print(f"  {'-'*65}")

        all_classes = sorted(set(list(r1["unique_counts"].keys()) +
                                  list(r2["unique_counts"].keys())))
        for cls in all_classes:
            c1 = r1["unique_counts"].get(cls, 0)
            c2 = r2["unique_counts"].get(cls, 0)
            diff = c2 - c1
            diff_str = f"+{diff}" if diff > 0 else str(diff)
            print(f"  {cls:<35} {c1:>12} {c2:>12}  {diff_str}")

        print(f"  {'-'*65}")
        print(f"  {'UNIQUE TOTAL (after ReID)':<35} {r1['total_unique']:>12} {r2['total_unique']:>12}"
              f"  {r2['total_unique'] - r1['total_unique']:+}")
        print(f"  {'RAW TRACKS (before ReID)':<35} {r1['raw_tracks']:>12} {r2['raw_tracks']:>12}"
              f"  {r2['raw_tracks'] - r1['raw_tracks']:+}")
        print(f"  {'DUPES REMOVED':<35} {r1['dupes_removed']:>12} {r2['dupes_removed']:>12}"
              f"  {r2['dupes_removed'] - r1['dupes_removed']:+}")
        print(f"  {'MS / FRAME (YOLO only)':<35} {r1['ms_per_frame']:>11.1f} {r2['ms_per_frame']:>11.1f}")


def write_summary_csv(all_results: list, output_dir: str):
    all_classes = set()
    for res in all_results:
        all_classes.update(res["model1"]["unique_counts"].keys())
        all_classes.update(res["model2"]["unique_counts"].keys())
    all_classes = sorted(all_classes)

    path = os.path.join(output_dir, "summary_reid.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["video", "model", "unique_total", "raw_tracks",
                   "dupes_removed", "ms_per_frame"] + all_classes
        w.writerow(header)
        for res in all_results:
            for key in ("model1", "model2"):
                r = res[key]
                row = [res["video"], r["model_name"], r["total_unique"],
                        r["raw_tracks"], r["dupes_removed"],
                        f"{r['ms_per_frame']:.1f}"]
                for cls in all_classes:
                    row.append(r["unique_counts"].get(cls, 0))
                w.writerow(row)
    print(f"\n  Summary CSV : {path}")
    return path


# ── Args + main ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="YOLO model comparison with ByteTrack + CLIP ReID")
    p.add_argument("--model1", required=True)
    p.add_argument("--model2", required=True)
    p.add_argument("--videos", default=config.INPUT_FOLDER)
    p.add_argument("--output", default="comparison_reid_results")
    p.add_argument("--model1-name", default="Model-v1")
    p.add_argument("--model2-name", default="Model-v2")
    p.add_argument("--frame-skip", type=int, default=config.FRAME_SKIP)
    p.add_argument("--conf", type=float, default=config.YOLO_CONFIDENCE)
    p.add_argument("--iou", type=float, default=config.YOLO_IOU)
    p.add_argument("--img-size", type=int, default=config.YOLO_IMG_SIZE)
    p.add_argument("--reid-threshold", type=float,
                   default=config.REID_SIMILARITY_THRESHOLD)
    p.add_argument("--max-sample-frames", type=int, default=20,
                   help="Max side-by-side images saved per video")
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

    print(f"\n{'='*65}")
    print(f"  MODEL COMPARISON  (with ByteTrack + CLIP ReID)")
    print(f"{'='*65}")
    print(f"  Model 1        : {args.model1_name}  ({args.model1})")
    print(f"  Model 2        : {args.model2_name}  ({args.model2})")
    print(f"  Videos         : {len(videos)} in {args.videos}")
    print(f"  Output         : {args.output}")
    print(f"  Conf/IoU/Skip  : {args.conf} / {args.iou} / {args.frame_skip}")
    print(f"  ReID threshold : {args.reid_threshold}")
    print(f"{'='*65}\n")

    print("[*] Loading YOLO models...")
    model1 = YOLO(args.model1)
    model2 = YOLO(args.model2)
    print("    Done.\n")

    all_results = []
    for idx, vp in enumerate(videos, 1):
        print(f"\n{'='*65}")
        print(f"  [{idx}/{len(videos)}] {vp.name}")
        print(f"{'='*65}")
        res = compare_video(str(vp), model1, model2,
                             args.model1_name, args.model2_name,
                             args.output, args)
        if res:
            all_results.append(res)

    if not all_results:
        print("No results.")
        return

    print_comparison(all_results)
    write_summary_csv(all_results, args.output)

    print(f"\n{'='*65}")
    print(f"  DONE. Output in: {args.output}/")
    print(f"  - <video>/frames/   ← side-by-side frames (after ReID overlay)")
    print(f"  - summary_reid.csv  ← unique counts per class per model")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
