"""
Unique Object Counter Pipeline
================================
Scans all videos in the configured input folder, runs YOLO detection +
ByteTrack tracking + CLIP re-identification on each, and writes per-video
results (JSON + annotated video) to the output folder.

The pipeline handles:
  - Camera rotation and revisiting same areas
  - Deduplication of objects seen multiple times
  - Frame skipping for performance
  - Flicker-free output video (only annotated frames, at a configurable FPS)
"""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

import config

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


def get_color_for_class(class_id: int) -> tuple:
    """Generate a consistent color for each class ID."""
    np.random.seed(class_id * 42 + 7)
    return tuple(int(c) for c in np.random.randint(50, 255, size=3))


def draw_detections(frame, boxes, track_ids, class_ids, confidences,
                     canonical_map=None):
    """Draw bounding boxes with track info on frame."""
    annotated = frame.copy()

    for i, (box, track_id, cls_id, conf) in enumerate(
            zip(boxes, track_ids, class_ids, confidences)):

        x1, y1, x2, y2 = map(int, box)
        color = get_color_for_class(int(cls_id))

        # Get canonical ID if re-id is active
        display_id = track_id
        if canonical_map and track_id in canonical_map:
            display_id = canonical_map[track_id]

        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, config.BBOX_THICKNESS)

        # Build label
        parts = []
        if config.SHOW_TRACK_ID:
            parts.append(f"ID:{display_id}")
        if config.SHOW_CLASS_NAME:
            class_name = config.CLASS_NAMES.get(int(cls_id), f"cls_{int(cls_id)}")
            if len(class_name) > 20:
                class_name = class_name[:18] + ".."
            parts.append(class_name)
        if config.SHOW_CONFIDENCE:
            parts.append(f"{conf:.2f}")

        label = " | ".join(parts)

        # Draw label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                      config.FONT_SCALE, 1)
        cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w + 5, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE,
                    (255, 255, 255), 2)

    return annotated


def draw_counter_overlay(frame, unique_counts: dict, frame_num: int, total_frames: int):
    """Draw a live counter overlay on the frame."""
    h, w = frame.shape[:2]

    # Semi-transparent background
    overlay = frame.copy()
    panel_h = 30 + len(unique_counts) * 25 + 10
    panel_w = 320
    cv2.rectangle(overlay, (w - panel_w - 10, 5), (w - 5, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Title
    cv2.putText(frame, f"Unique Objects (Frame {frame_num}/{total_frames})",
                (w - panel_w - 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 255), 1)

    # Counts
    y_offset = 50
    total = 0
    for class_name, count in sorted(unique_counts.items()):
        text = f"{class_name}: {count}"
        cv2.putText(frame, text, (w - panel_w - 5, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y_offset += 25
        total += count

    cv2.putText(frame, f"TOTAL: {total}", (w - panel_w - 5, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    return frame


def process_video(video_path: str, model: YOLO, output_dir: str):
    """Process a single video through the detection pipeline."""

    video_name = Path(video_path).stem

    # ── Validate Input ──────────────────────────────────────────
    if not os.path.exists(video_path):
        print(f"  ERROR: Video not found: {video_path}")
        return None

    os.makedirs(output_dir, exist_ok=True)

    # ── Load Video ──────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Resolution  : {width}x{height}")
    print(f"  FPS         : {fps:.1f}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Frame Skip  : {config.FRAME_SKIP}")
    print(f"  Frames to Process: ~{total_frames // config.FRAME_SKIP}")

    # ── Initialize ReID Module ─────────────────────────────────
    reid = None
    if config.ENABLE_CLIP_REID:
        from reid_module import CLIPReIdentifier
        reid = CLIPReIdentifier(
            similarity_threshold=config.REID_SIMILARITY_THRESHOLD,
            model_name=config.CLIP_MODEL_NAME,
        )
        print("  CLIP Re-ID  : Enabled")
    else:
        print("  CLIP Re-ID  : Disabled (tracker IDs only)")

    # ── Initialize Video Writer ────────────────────────────────
    # Only annotated frames are written → no flickering.
    # Output FPS controls playback speed (lower = slower, easier to inspect).
    out_fps = config.OUTPUT_VIDEO_FPS or max(1, fps / config.FRAME_SKIP)
    output_video_path = os.path.join(output_dir, f"{video_name}_detected.mp4")
    fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)
    writer = cv2.VideoWriter(output_video_path, fourcc, out_fps, (width, height))
    print(f"  Output FPS  : {out_fps}  (only annotated frames written)")

    # ── Run Detection + Tracking ───────────────────────────────
    track_class_map = {}
    track_confidence_map = {}
    track_first_seen = {}
    track_last_seen = {}
    canonical_map = {}

    frame_idx = 0
    processed_count = 0
    start_time = time.time()

    pbar = tqdm(total=total_frames, desc="  Processing", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        pbar.update(1)

        # Skip frames that aren't scheduled for processing
        if frame_idx % config.FRAME_SKIP != 0:
            continue

        processed_count += 1

        # ── Run YOLO with built-in tracking ───────────────────
        results = model.track(
            frame,
            persist=True,
            conf=config.YOLO_CONFIDENCE,
            iou=config.YOLO_IOU,
            imgsz=config.YOLO_IMG_SIZE,
            tracker=f"{config.TRACKER_TYPE}.yaml",
            verbose=False,
        )

        result = results[0]
        boxes = []
        track_ids = []
        class_ids = []
        confidences = []

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()

            # Record which tracks are visible together in this frame
            # so reid knows they are different physical objects.
            if reid:
                reid.register_frame_tracks(track_ids)

            for i, (box, tid, cid, conf) in enumerate(
                    zip(boxes, track_ids, class_ids, confidences)):

                track_class_map[tid] = cid
                track_confidence_map[tid] = max(
                    track_confidence_map.get(tid, 0), conf
                )
                if tid not in track_first_seen:
                    track_first_seen[tid] = frame_idx
                track_last_seen[tid] = frame_idx

                # ── CLIP Re-Identification ────────────────────
                if reid and processed_count % config.REID_CHECK_INTERVAL == 0:
                    x1, y1, x2, y2 = map(int, box)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)

                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        canonical_id = reid.update_track(tid, cid, crop)
                        if canonical_id != tid:
                            canonical_map[tid] = canonical_id

        # ── Draw & Write Frame ────────────────────────────────
        annotated = draw_detections(
            frame, boxes, track_ids, class_ids, confidences,
            canonical_map=canonical_map,
        )

        # Build live counter
        if reid:
            current_counts = reid.get_unique_counts()
        else:
            current_counts = defaultdict(int)
            seen_canonical = set()
            for tid, cid in track_class_map.items():
                c_id = canonical_map.get(tid, tid)
                if c_id not in seen_canonical:
                    seen_canonical.add(c_id)
                    class_name = config.CLASS_NAMES.get(cid, f"class_{cid}")
                    current_counts[class_name] += 1
            current_counts = dict(current_counts)

        annotated = draw_counter_overlay(
            annotated, current_counts, frame_idx, total_frames
        )
        writer.write(annotated)

    pbar.close()
    cap.release()
    writer.release()

    elapsed = time.time() - start_time

    # ── Compute Final Results ──────────────────────────────────
    if reid:
        unique_counts = reid.get_unique_counts()
        unique_objects = reid.get_unique_objects_detail()
    else:
        unique_counts = defaultdict(int)
        unique_objects = []
        seen = set()
        for tid, cid in track_class_map.items():
            c_id = canonical_map.get(tid, tid)
            if c_id not in seen:
                seen.add(c_id)
                class_name = config.CLASS_NAMES.get(cid, f"class_{cid}")
                unique_counts[class_name] += 1
                unique_objects.append({
                    "unique_id": int(c_id),
                    "class_id": int(cid),
                    "class_name": class_name,
                })
        unique_counts = dict(unique_counts)

    total_unique = sum(unique_counts.values())
    total_raw_tracks = len(track_class_map)

    # ── Build Output JSON ──────────────────────────────────────
    output_data = {
        "summary": {
            "total_unique_objects": total_unique,
            "total_raw_tracks": total_raw_tracks,
            "duplicates_removed": total_raw_tracks - total_unique,
            "processing_time_seconds": round(elapsed, 2),
            "frames_processed": processed_count,
            "total_frames": total_frames,
            "reid_enabled": config.ENABLE_CLIP_REID,
        },
        "counts_by_class": dict(sorted(unique_counts.items())),
        "unique_objects": unique_objects,
        "config": {
            "yolo_model": config.YOLO_MODEL_PATH,
            "confidence_threshold": config.YOLO_CONFIDENCE,
            "frame_skip": config.FRAME_SKIP,
            "reid_threshold": config.REID_SIMILARITY_THRESHOLD if config.ENABLE_CLIP_REID else None,
            "tracker": config.TRACKER_TYPE,
        },
    }

    output_json_path = os.path.join(output_dir, f"{video_name}_counts.json")
    with open(output_json_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # ── Print Summary ──────────────────────────────────────────
    print(f"\n  {'─'*50}")
    print(f"  Total Raw Tracks   : {total_raw_tracks}")
    print(f"  Duplicates Removed : {total_raw_tracks - total_unique}")
    print(f"  Total Unique Objects: {total_unique}")
    for class_name, count in sorted(unique_counts.items()):
        print(f"    {class_name:.<40} {count}")
    print(f"  Processing Time: {elapsed:.1f}s ({processed_count / max(elapsed, 0.1):.1f} frames/sec)")
    print(f"  Output JSON : {output_json_path}")
    print(f"  Output Video: {output_video_path}")
    print(f"  {'─'*50}")

    return output_data


def main():
    input_folder = config.INPUT_FOLDER
    output_folder = config.OUTPUT_FOLDER

    if not os.path.isdir(input_folder):
        print(f"ERROR: Input folder not found: {input_folder}")
        sys.exit(1)

    # Collect video files
    video_files = sorted(
        p for p in Path(input_folder).iterdir()
        if p.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not video_files:
        print(f"ERROR: No video files found in {input_folder}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  UNIQUE OBJECT COUNTER PIPELINE")
    print(f"{'='*60}")
    print(f"  Input Folder : {input_folder}")
    print(f"  Output Folder: {output_folder}")
    print(f"  Videos Found : {len(video_files)}")
    print(f"  YOLO Model   : {config.YOLO_MODEL_PATH}")
    print(f"  Confidence   : {config.YOLO_CONFIDENCE}")
    print(f"  Frame Skip   : {config.FRAME_SKIP}")
    print(f"  CLIP Re-ID   : {'Enabled' if config.ENABLE_CLIP_REID else 'Disabled'}")
    print(f"{'='*60}\n")

    # Load YOLO model once (shared across all videos)
    if not os.path.exists(config.YOLO_MODEL_PATH):
        print(f"ERROR: YOLO model not found: {config.YOLO_MODEL_PATH}")
        sys.exit(1)

    print("[*] Loading YOLO model...")
    model = YOLO(config.YOLO_MODEL_PATH)

    os.makedirs(output_folder, exist_ok=True)

    all_results = {}
    for idx, video_path in enumerate(video_files, 1):
        print(f"\n{'='*60}")
        print(f"  [{idx}/{len(video_files)}] {video_path.name}")
        print(f"{'='*60}")

        result = process_video(str(video_path), model, output_folder)
        if result:
            all_results[video_path.name] = result

    # ── Overall Summary ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ALL DONE — {len(all_results)}/{len(video_files)} videos processed")
    print(f"{'='*60}")
    for name, res in all_results.items():
        total = res["summary"]["total_unique_objects"]
        print(f"  {name}: {total} unique objects")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
