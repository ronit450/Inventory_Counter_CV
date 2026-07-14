"""
Unique Object Counter Pipeline
================================
Scans all videos in the configured input folder, runs YOLO detection +
ByteTrack tracking + CLIP re-identification on each, and writes per-video
results (JSON + annotated video) to the output folder.
"""

import json
import math
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

# Reverse lookup: class name → class_id (for consistent coloring in overlay)
_CLASS_NAME_TO_ID = {v: k for k, v in config.CLASS_NAMES.items()}

CROP_PAD        = 20
CONTACT_THUMB_W = 200
CONTACT_THUMB_H = 200
CONTACT_COLS    = 6
LABEL_H         = 28
_BG_COLOR       = (30, 30, 30)
_BORDER_COLOR   = (80, 80, 80)



def get_color_for_class(class_id: int) -> tuple:
    np.random.seed(class_id * 42 + 7)
    return tuple(int(c) for c in np.random.randint(50, 255, size=3))


def draw_detections(frame, boxes, track_ids, class_ids, confidences, canonical_map=None):
    annotated = frame.copy()
    for box, track_id, cls_id, conf in zip(boxes, track_ids, class_ids, confidences):
        x1, y1, x2, y2 = map(int, box)
        color = get_color_for_class(int(cls_id))
        display_id = canonical_map.get(track_id, track_id) if canonical_map else track_id
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, config.BBOX_THICKNESS)

        parts = []
        if config.SHOW_CLASS_NAME:
            cn = config.CLASS_NAMES.get(int(cls_id), f"cls_{int(cls_id)}")
            parts.append(cn[:18] + ".." if len(cn) > 20 else cn)
        if config.SHOW_TRACK_ID:
            parts.append(f"#{display_id}")
        if config.SHOW_CONFIDENCE:
            parts.append(f"{conf:.2f}")
        label = " ".join(parts)

        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, 1)
        ly = max(y1 - lh - 8, 0)
        cv2.rectangle(annotated, (x1, ly), (x1 + lw + 8, ly + lh + 8), color, -1)
        cv2.putText(annotated, label, (x1 + 4, ly + lh + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA)
    return annotated


def draw_counter_overlay(frame, unique_counts: dict, frame_num: int, total_frames: int):
    h, w = frame.shape[:2]

    lines   = sorted(unique_counts.items())
    total   = sum(c for _, c in lines)
    row_h   = 28
    hdr_h   = 44
    ftr_h   = 36
    pad     = 8
    panel_w = 280
    panel_h = hdr_h + len(lines) * row_h + ftr_h + pad
    x0, y0  = w - panel_w - 8, 8

    # Semi-transparent dark panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 4, y0), (w - 4, y0 + panel_h), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    # Accent top bar
    cv2.rectangle(frame, (x0 - 4, y0), (w - 4, y0 + 3), (0, 160, 255), -1)

    # Header
    cv2.putText(frame, "INVENTORY COUNT", (x0 + 4, y0 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (x0, y0 + hdr_h - 4), (w - 8, y0 + hdr_h - 4), (55, 55, 55), 1)

    # One row per class
    for i, (class_name, count) in enumerate(lines):
        y     = y0 + hdr_h + i * row_h + 20
        cid   = _CLASS_NAME_TO_ID.get(class_name, abs(hash(class_name)) % 200)
        color = get_color_for_class(cid)
        cv2.circle(frame, (x0 + 8, y - 6), 5, color, -1)
        short = (class_name[:20] + "..") if len(class_name) > 22 else class_name
        cv2.putText(frame, short, (x0 + 20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1, cv2.LINE_AA)
        cnt_str = str(count)
        (tw, _), _ = cv2.getTextSize(cnt_str, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        cv2.putText(frame, cnt_str, (w - 10 - tw, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

    # Divider + total
    y_div = y0 + hdr_h + len(lines) * row_h + pad
    cv2.line(frame, (x0, y_div), (w - 8, y_div), (55, 55, 55), 1)
    cv2.putText(frame, f"TOTAL  {total}", (x0 + 4, y_div + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 235, 100), 2, cv2.LINE_AA)

    # Progress bar at bottom
    progress = min(frame_num / max(total_frames, 1), 1.0)
    cv2.rectangle(frame, (0, h - 7), (w, h), (35, 35, 35), -1)
    cv2.rectangle(frame, (0, h - 7), (int(w * progress), h), (0, 150, 255), -1)

    return frame


def make_contact_sheet(entries: list, title: str) -> np.ndarray:
    if not entries:
        blank = np.full((100, 400, 3), _BG_COLOR, dtype=np.uint8)
        cv2.putText(blank, f"{title}: no objects", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        return blank

    cols = min(CONTACT_COLS, len(entries))
    rows = (len(entries) + cols - 1) // cols
    cell_w  = CONTACT_THUMB_W
    cell_h  = CONTACT_THUMB_H + LABEL_H
    title_h = 40

    sheet_w = cols * cell_w
    sheet_h = rows * cell_h + title_h
    sheet = np.full((sheet_h, sheet_w, 3), _BG_COLOR, dtype=np.uint8)

    cv2.rectangle(sheet, (0, 0), (sheet_w, title_h), (20, 60, 100), -1)
    cv2.putText(sheet, title, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 220), 2)

    for i, entry in enumerate(entries):
        row = i // cols
        col = i % cols
        x   = col * cell_w
        y   = title_h + row * cell_h

        img = entry["img"]
        if img is not None and img.size > 0:
            thumb = cv2.resize(img, (CONTACT_THUMB_W, CONTACT_THUMB_H))
        else:
            thumb = np.full((CONTACT_THUMB_H, CONTACT_THUMB_W, 3), (60, 60, 60), dtype=np.uint8)

        sheet[y:y + CONTACT_THUMB_H, x:x + cell_w] = thumb
        cv2.rectangle(sheet,
                      (x, y + CONTACT_THUMB_H), (x + cell_w, y + CONTACT_THUMB_H + LABEL_H),
                      (15, 15, 15), -1)
        cv2.putText(sheet, entry["label"],
                    (x + 4, y + CONTACT_THUMB_H + LABEL_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        cv2.rectangle(sheet, (x, y), (x + cell_w - 1, y + cell_h - 1), _BORDER_COLOR, 1)

    return sheet


def make_timeline_strip(
    unique_objects: list,
    canonical_first_seen: dict,
    canonical_last_seen: dict,
    total_frames: int,
    fps: float,
    title: str,
) -> np.ndarray:
    """Horizontal bar chart showing when each unique object was first/last detected."""
    if not unique_objects:
        return None

    strip_w = 1280
    row_h   = 30
    label_w = 250
    bar_w   = strip_w - label_w - 16
    hdr_h   = 44
    ftr_h   = 28

    img_h = hdr_h + len(unique_objects) * row_h + ftr_h
    img   = np.full((img_h, strip_w, 3), (18, 18, 18), dtype=np.uint8)

    # Header
    cv2.rectangle(img, (0, 0), (strip_w, hdr_h), (20, 55, 90), -1)
    cv2.putText(img, title, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 210, 255), 2, cv2.LINE_AA)

    # Time-axis ticks at 0 / 25 / 50 / 75 / 100 %
    duration_s = total_frames / max(fps, 1)
    for pct in range(0, 101, 25):
        x = label_w + int(bar_w * pct / 100)
        cv2.line(img, (x, hdr_h), (x, img_h - ftr_h), (45, 45, 45), 1)
        t_s  = duration_s * pct / 100
        tick = f"{int(t_s)}s" if t_s < 60 else f"{t_s / 60:.1f}m"
        cv2.putText(img, tick, (x - 14, img_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (100, 100, 100), 1)

    for i, obj in enumerate(unique_objects):
        uid   = obj["unique_id"]
        cid   = obj.get("class_id", 0)
        color = get_color_for_class(int(cid))

        y = hdr_h + i * row_h
        if i % 2 == 0:
            cv2.rectangle(img, (0, y), (strip_w, y + row_h), (26, 26, 26), -1)

        short = obj["class_name"]
        if len(short) > 30:
            short = short[:28] + ".."
        cv2.putText(img, short, (6, y + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, (185, 185, 185), 1, cv2.LINE_AA)

        f0 = canonical_first_seen.get(uid, 0)
        f1 = canonical_last_seen.get(uid, f0 + 1)
        bx0 = label_w + int(bar_w * f0 / max(total_frames, 1))
        bx1 = label_w + int(bar_w * f1 / max(total_frames, 1))
        bx1 = max(bx1, bx0 + 6)
        cv2.rectangle(img, (bx0, y + 7), (bx1, y + row_h - 7), color, -1)
        cv2.circle(img, (bx0 + 3, y + row_h // 2), 4, (255, 255, 255), -1)

    return img


def process_video(video_path: str, model: YOLO, output_dir: str):
    video_name = Path(video_path).stem

    if not os.path.exists(video_path):
        print(f"  ERROR: Video not found: {video_path}")
        return None

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Resolution  : {width}x{height}")
    print(f"  FPS         : {fps:.1f}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Frame Skip  : {config.FRAME_SKIP}")
    print(f"  Frames to Process: ~{total_frames // config.FRAME_SKIP}")

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

    out_fps = config.OUTPUT_VIDEO_FPS or max(1, fps / config.FRAME_SKIP)
    output_video_path = os.path.join(output_dir, f"{video_name}_detected.mp4")
    fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)
    writer = cv2.VideoWriter(output_video_path, fourcc, out_fps, (width, height))
    print(f"  Output FPS  : {out_fps}  (only annotated frames written)")

    track_class_map       = {}
    track_confidence_map  = {}
    track_confidence_sum  = {}   # for computing mean confidence per track
    track_detection_count = {}   # number of processed frames where track was detected
    track_first_seen      = {}
    track_last_seen       = {}
    track_first_box       = {}
    track_last_box        = {}
    canonical_map         = {}
    peak_counts           = {}   # max simultaneous distinct tracks per class in any frame
    padded_crops          = {}   # tid -> {"crop", "score", "frame_jpg", "box", "class_id"}

    frame_idx      = 0
    processed_count = 0
    start_time     = time.time()

    pbar = tqdm(total=total_frames, desc="  Processing", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        pbar.update(1)

        if frame_idx % config.FRAME_SKIP != 0:
            continue

        processed_count += 1

        results = model.track(
            frame,
            persist=True,
            conf=config.YOLO_CONFIDENCE,
            iou=config.YOLO_IOU,
            imgsz=config.YOLO_IMG_SIZE,
            tracker=config.TRACKER_CONFIG_PATH,
            verbose=False,
        )

        result    = results[0]
        boxes      = []
        track_ids  = []
        class_ids  = []
        confidences = []

        if result.boxes is not None and result.boxes.id is not None:
            boxes       = result.boxes.xyxy.cpu().numpy()
            track_ids   = result.boxes.id.cpu().numpy().astype(int)
            class_ids   = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()

            if reid:
                reid.register_frame_tracks(track_ids, frame, frame_idx)

            # Peak simultaneous count — hard lower bound on distinct object count.
            frame_class_tracks: dict = defaultdict(set)
            for tid, cid in zip(track_ids, class_ids):
                frame_class_tracks[int(cid)].add(int(tid))
            for cid, tids in frame_class_tracks.items():
                class_name = config.CLASS_NAMES.get(cid, f"class_{cid}")
                peak_counts[class_name] = max(peak_counts.get(class_name, 0), len(tids))

            for box, tid, cid, conf in zip(boxes, track_ids, class_ids, confidences):
                track_class_map[tid]       = cid
                track_confidence_map[tid]  = max(track_confidence_map.get(tid, 0), conf)
                track_confidence_sum[tid]  = track_confidence_sum.get(tid, 0.0) + float(conf)
                track_detection_count[tid] = track_detection_count.get(tid, 0) + 1
                if tid not in track_first_seen:
                    track_first_seen[tid] = frame_idx
                track_last_seen[tid] = frame_idx
                if tid not in track_first_box:
                    track_first_box[tid] = tuple(map(float, box))
                track_last_box[tid] = tuple(map(float, box))

                if reid and processed_count % config.REID_CHECK_INTERVAL == 0:
                    x1c, y1c, x2c, y2c = map(int, box)
                    x1c, y1c = max(0, x1c), max(0, y1c)
                    x2c, y2c = min(width, x2c), min(height, y2c)
                    crop = frame[y1c:y2c, x1c:x2c]
                    if crop.size > 0:
                        reid.update_track(tid, cid, crop, (x1c, y1c, x2c, y2c), frame=frame)

                # Padded display crop — quality-scored, best kept per track
                x1b, y1b, x2b, y2b = map(int, box)
                x1t, y1t = max(0, x1b), max(0, y1b)
                x2t, y2t = min(width, x2b), min(height, y2b)
                tight = frame[y1t:y2t, x1t:x2t]
                x1p   = max(0, x1b - CROP_PAD)
                y1p   = max(0, y1b - CROP_PAD)
                x2p   = min(width,  x2b + CROP_PAD)
                y2p   = min(height, y2b + CROP_PAD)
                pcrop = frame[y1p:y2p, x1p:x2p]
                if tight.size > 0 and pcrop.size > 0:
                    _gray     = cv2.cvtColor(tight, cv2.COLOR_BGR2GRAY)
                    _sharp    = cv2.Laplacian(_gray, cv2.CV_64F).var()
                    _area     = (x2t - x1t) * (y2t - y1t)
                    _complete = 0.6 if (x1b <= 3 or y1b <= 3
                                        or x2b >= width - 3 or y2b >= height - 3) else 1.0
                    _score    = math.log1p(_sharp) * math.log1p(_area) * _complete
                    if int(tid) not in padded_crops or _score > padded_crops[int(tid)]["score"]:
                        ok, fbuf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        padded_crops[int(tid)] = {
                            "crop":      pcrop.copy(),
                            "score":     _score,
                            "frame_jpg": fbuf.tobytes() if ok else None,
                            "box":       (x1b, y1b, x2b, y2b),
                            "class_id":  int(cid),
                        }

        annotated = draw_detections(frame, boxes, track_ids, class_ids, confidences,
                                     canonical_map=canonical_map)

        if reid:
            current_counts = reid.get_unique_counts()
        else:
            current_counts = defaultdict(int)
            seen_canonical = set()
            for tid, cid in track_class_map.items():
                c_id = canonical_map.get(tid, tid)
                if c_id not in seen_canonical:
                    seen_canonical.add(c_id)
                    current_counts[config.CLASS_NAMES.get(cid, f"class_{cid}")] += 1
            current_counts = dict(current_counts)

        writer.write(annotated)

    pbar.close()
    cap.release()
    writer.release()
    elapsed = time.time() - start_time

    # ── Track quality filter ───────────────────────────────────────
    # Remove ghost tracks: too few detections or too-low mean confidence.
    # Must run before pre-ReID crop saving AND before reid collector sees them.
    min_track_frames = getattr(config, "MIN_TRACK_FRAMES", 1)
    min_track_conf   = getattr(config, "MIN_TRACK_CONFIDENCE", 0.0)
    dead_tids: set = set()
    for tid in list(track_class_map.keys()):
        if track_detection_count.get(tid, 0) < min_track_frames:
            dead_tids.add(tid)
        elif (track_confidence_sum.get(tid, 0.0)
              / max(track_detection_count.get(tid, 1), 1)) < min_track_conf:
            dead_tids.add(tid)
    if dead_tids:
        for tid in dead_tids:
            track_class_map.pop(tid, None)
            track_confidence_map.pop(tid, None)
            track_first_seen.pop(tid, None)
            track_last_seen.pop(tid, None)
            track_first_box.pop(tid, None)
            track_last_box.pop(tid, None)
            padded_crops.pop(int(tid), None)
        if reid:
            reid.collector.remove_tracks(dead_tids)
        print(f"  [Filter] Removed {len(dead_tids)} ghost tracks "
              f"(min_frames={min_track_frames}, min_conf={min_track_conf})")

    # ── Track stitching (motion continuity) ────────────────────────
    if reid and getattr(config, "ENABLE_TRACK_STITCH", False):
        from track_stitch import stitch_tracks
        meta = {
            int(tid): {
                "class_id":   int(track_class_map[tid]),
                "first_seen": track_first_seen[tid],
                "last_seen":  track_last_seen[tid],
                "first_box":  track_first_box[tid],
                "last_box":   track_last_box[tid],
            }
            for tid in track_class_map
            if tid in track_first_box
        }
        stitch_map = stitch_tracks(meta, config.TRACK_STITCH_MAX_GAP,
                                   config.TRACK_STITCH_MIN_IOU)
        n_chains = sum(1 for t, r in stitch_map.items() if t != r)
        if n_chains:
            print(f"  [Stitch] {n_chains} track fragments stitched by motion continuity")
        reid.apply_stitch_map(stitch_map, padded_crops)

    # ── Pre-ReID crops (optional) ─────────────────────────────────
    if getattr(config, "SAVE_PRE_REID_CROPS", False):
        raw_crops_dir  = os.path.join(output_dir, f"{video_name}_crops_raw")
        raw_sheet_path = os.path.join(output_dir, f"{video_name}_contact_sheet_raw.jpg")
        os.makedirs(raw_crops_dir, exist_ok=True)
        raw_entries = []
        for tid in sorted(track_class_map.keys()):
            cid        = track_class_map[tid]
            class_name = config.CLASS_NAMES.get(cid, f"class_{cid}")
            entry      = padded_crops.get(int(tid))
            crop       = entry["crop"] if entry else None
            safe_cls   = class_name.replace("/", "_").replace(" ", "_")
            if crop is not None and crop.size > 0:
                cv2.imwrite(os.path.join(raw_crops_dir, f"{safe_cls}_id{tid}.jpg"),
                            crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            raw_entries.append({"img": crop, "label": f"{class_name} | raw:{tid}"})
        raw_entries.sort(key=lambda e: e["label"])
        if raw_entries:
            raw_sheet = make_contact_sheet(
                raw_entries,
                f"{video_name} — {len(track_class_map)} raw tracks (pre-ReID)")
            cv2.imwrite(raw_sheet_path, raw_sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"  Raw crops (pre-ReID): {raw_crops_dir}")
            print(f"  Raw contact sheet   : {raw_sheet_path}")

    # ── Post-hoc deduplication ─────────────────────────────────
    if reid:
        reid.finalize()

    # ── Compute final results ──────────────────────────────────
    if reid:
        unique_counts  = reid.get_unique_counts()
        unique_objects = reid.get_unique_objects_detail()
    else:
        unique_counts  = defaultdict(int)
        unique_objects = []
        seen = set()
        for tid, cid in track_class_map.items():
            c_id = canonical_map.get(tid, tid)
            if c_id not in seen:
                seen.add(c_id)
                class_name = config.CLASS_NAMES.get(cid, f"class_{cid}")
                unique_counts[class_name] += 1
                unique_objects.append({"unique_id": int(c_id), "class_id": int(cid),
                                        "class_name": class_name})
        unique_counts = dict(unique_counts)

    # ── Peak count floor ───────────────────────────────────────
    peak_overrides = {}
    for class_name, peak in peak_counts.items():
        reid_count = unique_counts.get(class_name, 0)
        if peak > reid_count:
            unique_counts[class_name] = peak
            peak_overrides[class_name] = (reid_count, peak)

    if peak_overrides:
        print(f"\n  [Peak floor applied]")
        for cls, (before, after) in sorted(peak_overrides.items()):
            print(f"    {cls}: ReID={before} → peak={after}")

    # ── Save individual crops & contact sheet ─────────────────────
    crops_dir  = os.path.join(output_dir, f"{video_name}_crops")
    sheet_path = os.path.join(output_dir, f"{video_name}_contact_sheet.jpg")
    os.makedirs(crops_dir, exist_ok=True)

    c_map = reid._canonical_map if reid else {}

    # First/last frame seen per canonical object (used for timeline strip)
    canonical_first_seen: dict = {}
    canonical_last_seen:  dict = {}
    for tid, f_first in track_first_seen.items():
        canon  = c_map.get(tid, tid)
        f_last = track_last_seen.get(tid, f_first)
        if canon not in canonical_first_seen or f_first < canonical_first_seen[canon]:
            canonical_first_seen[canon] = f_first
        if canon not in canonical_last_seen or f_last > canonical_last_seen[canon]:
            canonical_last_seen[canon] = f_last

    canonical_to_tids: dict = defaultdict(list)
    for tid in padded_crops:
        canon = c_map.get(tid, tid)
        canonical_to_tids[canon].append(tid)

    post_entries    = []
    class_counters: dict = defaultdict(int)
    for obj in sorted(unique_objects, key=lambda o: (o["class_name"], o["unique_id"])):
        uid        = obj["unique_id"]
        class_name = obj["class_name"]
        class_counters[class_name] += 1
        display_idx = class_counters[class_name]

        best_score = -1.0
        best_entry = None
        for tid in canonical_to_tids.get(uid, [uid]):
            entry = padded_crops.get(int(tid))
            if entry and entry["score"] > best_score:
                best_score = entry["score"]
                best_entry = entry

        safe_cls  = class_name.replace("/", "_").replace(" ", "_")
        best_crop = best_entry["crop"] if best_entry else None

        if best_crop is not None and best_crop.size > 0:
            cv2.imwrite(
                os.path.join(crops_dir, f"{safe_cls}_{display_idx}_crop.jpg"),
                best_crop, [cv2.IMWRITE_JPEG_QUALITY, 92])

        if best_entry:
            frame_jpg = best_entry.get("frame_jpg")
            box       = best_entry.get("box")
            cid       = obj.get("class_id", 0)
            if frame_jpg and box:
                fbuf      = np.frombuffer(frame_jpg, dtype=np.uint8)
                ctx_frame = cv2.imdecode(fbuf, cv2.IMREAD_COLOR)
                if ctx_frame is not None:
                    x1b, y1b, x2b, y2b = box
                    color = get_color_for_class(cid)
                    cv2.rectangle(ctx_frame, (x1b, y1b), (x2b, y2b), color, 3)
                    lbl = f"{class_name} #{display_idx}"
                    (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(ctx_frame,
                                  (x1b, y1b - lh - 12), (x1b + lw + 6, y1b),
                                  color, -1)
                    cv2.putText(ctx_frame, lbl, (x1b + 3, y1b - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.imwrite(
                        os.path.join(crops_dir, f"{safe_cls}_{display_idx}_context.jpg"),
                        ctx_frame, [cv2.IMWRITE_JPEG_QUALITY, 88])

        post_entries.append({"img": best_crop,
                              "label": f"{class_name} #{display_idx}"})

    if post_entries:
        contact_sheet = make_contact_sheet(
            post_entries, f"{video_name} — {len(unique_objects)} unique objects")
        cv2.imwrite(sheet_path, contact_sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # Per-class collages
    per_class_collage_paths: dict = {}
    by_class_entries: dict = defaultdict(list)
    for entry in post_entries:
        cls = entry["label"].rsplit(" #", 1)[0]
        by_class_entries[cls].append(entry)
    for cls, entries in sorted(by_class_entries.items()):
        safe_cls  = cls.replace("/", "_").replace(" ", "_")
        cls_sheet = make_contact_sheet(entries, f"{cls}  ×{len(entries)}")
        cls_path  = os.path.join(output_dir, f"{video_name}_collage_{safe_cls}.jpg")
        cv2.imwrite(cls_path, cls_sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        per_class_collage_paths[cls] = cls_path

    # Timeline strip
    timeline_path = os.path.join(output_dir, f"{video_name}_timeline.jpg")
    tl_img = make_timeline_strip(
        unique_objects, canonical_first_seen, canonical_last_seen,
        total_frames, fps, f"{video_name} — Detection Timeline",
    )
    if tl_img is not None:
        cv2.imwrite(timeline_path, tl_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    else:
        timeline_path = ""

    total_unique     = sum(unique_counts.values())
    total_raw_tracks = len(track_class_map)

    # ── Output JSON (clean client-facing) ─────────────────────
    # Build per-object list with just class, instance number, and file paths
    clean_objects = []
    _cls_ctr: dict = defaultdict(int)
    for obj in sorted(unique_objects, key=lambda o: (o["class_name"], o["unique_id"])):
        cls = obj["class_name"]
        _cls_ctr[cls] += 1
        idx      = _cls_ctr[cls]
        safe_cls = cls.replace("/", "_").replace(" ", "_")
        clean_objects.append({
            "class_name":        cls,
            "instance_number":   idx,
            "crop_path":         os.path.join(crops_dir, f"{safe_cls}_{idx}_crop.jpg"),
            "context_frame_path": os.path.join(crops_dir, f"{safe_cls}_{idx}_context.jpg"),
        })

    output_data = {
        "summary": {
            "total_unique_objects":    total_unique,
            "total_raw_tracks":        total_raw_tracks,
            "duplicates_removed":      total_raw_tracks - total_unique,
            "processing_time_seconds": round(elapsed, 2),
            "frames_processed":        processed_count,
            "total_frames":            total_frames,
            "reid_enabled":            config.ENABLE_CLIP_REID,
        },
        "counts_by_class": dict(sorted(unique_counts.items())),
        "objects":         clean_objects,
        "outputs": {
            "annotated_video":     output_video_path,
            "crops_folder":        crops_dir,
            "collage":             sheet_path,
            "timeline":            timeline_path,
            "per_class_collages":  per_class_collage_paths,
        },
        # kept for internal/run_wrapper compat
        "counts_by_class":      dict(sorted(unique_counts.items())),
        "output_crops_dir":     crops_dir,
        "output_contact_sheet": sheet_path,
    }

    output_json_path = os.path.join(output_dir, f"{video_name}_counts.json")
    with open(output_json_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # ── Summary ────────────────────────────────────────────────
    print(f"\n  {'─'*50}")
    print(f"  Total Raw Tracks    : {total_raw_tracks}")
    print(f"  Duplicates Removed  : {total_raw_tracks - total_unique}")
    print(f"  Total Unique Objects: {total_unique}")
    for class_name, count in sorted(unique_counts.items()):
        peak = peak_counts.get(class_name, 0)
        print(f"    {class_name:.<40} {count}  (peak={peak})")
    print(f"  Processing Time: {elapsed:.1f}s ({processed_count / max(elapsed, 0.1):.1f} frames/sec)")
    print(f"  Output JSON      : {output_json_path}")
    print(f"  Annotated Video  : {output_video_path}")
    print(f"  Individual Crops : {crops_dir}")
    print(f"  Collage (all)    : {sheet_path}")
    if per_class_collage_paths:
        print(f"  Per-class collages: {len(per_class_collage_paths)} files")
    if timeline_path:
        print(f"  Timeline Strip   : {timeline_path}")
    print(f"  {'─'*50}")

    return output_data


def main():
    input_folder  = config.INPUT_FOLDER
    output_folder = config.OUTPUT_FOLDER

    if not os.path.isdir(input_folder):
        print(f"ERROR: Input folder not found: {input_folder}")
        sys.exit(1)

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

    print(f"\n{'='*60}")
    print(f"  ALL DONE — {len(all_results)}/{len(video_files)} videos processed")
    print(f"{'='*60}")
    for name, res in all_results.items():
        print(f"  {name}: {res['summary']['total_unique_objects']} unique objects")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
