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

CROP_PAD        = 20
CONTACT_THUMB_W = 200
CONTACT_THUMB_H = 200
CONTACT_COLS    = 6
LABEL_H         = 28
_BG_COLOR       = (30, 30, 30)
_BORDER_COLOR   = (80, 80, 80)


def _vlm_filter_unique_objects(vlm_objects: dict) -> dict:
    """
    Send unique objects to AWS Bedrock Claude Haiku for class-specific validation.
    Removes objects where the crop does NOT match the assigned class label.
    One API call per class (batch) for efficiency.
    Returns filtered dict (same structure, subset of input).
    """
    try:
        import boto3
    except ImportError:
        print("  [VLM] boto3 not installed — skipping")
        return vlm_objects

    region   = getattr(config, "VLM_AWS_REGION",  "us-east-1")
    model_id = getattr(config, "VLM_MODEL_ID",    "us.anthropic.claude-haiku-4-5-20251001-v1:0")
    batch_sz = getattr(config, "VLM_BATCH_SIZE",   8)

    try:
        client = boto3.client("bedrock-runtime", region_name=region)
    except Exception as e:
        print(f"  [VLM] Bedrock init failed: {e}")
        return vlm_objects

    by_class: dict = defaultdict(list)
    for uid, obj in vlm_objects.items():
        by_class[obj["class_name"]].append((uid, obj))

    to_remove: set = set()

    strict_classes = set(getattr(config, "VLM_STRICT_CLASSES", []) or [])

    for class_name, items in by_class.items():
        # Only validate classes in strict_classes — everything else is skipped.
        # Non-strict classes have real objects in low-res crops that look ambiguous;
        # running VLM on them causes false removals.
        use_strict = class_name in strict_classes
        if not use_strict:
            continue

        for batch_start in range(0, len(items), batch_sz):
            batch = items[batch_start: batch_start + batch_sz]

            content = []
            uid_order = []
            for uid, obj in batch:
                crop = obj.get("best_crop")
                if crop is None or crop.size == 0:
                    to_remove.add(uid)
                    continue
                ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    continue
                content.append({"image": {"format": "jpeg",
                                           "source": {"bytes": buf.tobytes()}}})
                content.append({"text": f"[ID:{uid}]"})
                uid_order.append(uid)

            if not uid_order:
                continue

            if len(uid_order) == 1:
                uid = uid_order[0]
                if use_strict:
                    prompt = (
                        f"Does this image show a '{class_name}'? "
                        f"Answer ONLY 'yes' or 'no'. "
                        f"Answer 'no' ONLY if you are confident this is NOT a '{class_name}'. "
                        f"When in doubt, answer 'yes'."
                    )
                else:
                    prompt = (
                        "Does this image contain any recognizable office object or furniture? "
                        "Answer ONLY 'yes' or 'no'. "
                        "Answer 'no' ONLY if this is clearly an empty wall, bare floor, "
                        "ceiling, or completely blank/black image. When in doubt, answer 'yes'."
                    )
                content.append({"text": prompt})
                try:
                    resp = client.converse(
                        modelId=model_id,
                        messages=[{"role": "user", "content": content}],
                        inferenceConfig={"maxTokens": 5, "temperature": 0.0},
                        system=[{"text": "Answer only yes or no."}],
                    )
                    ans = resp["output"]["message"]["content"][0]["text"].strip().lower()
                    if not ans.startswith("yes"):
                        print(f"  [VLM] Removed ID:{uid} ({class_name})")
                        to_remove.add(uid)
                except Exception as e:
                    print(f"  [VLM] Error ID:{uid}: {e}")
                continue

            if use_strict:
                batch_prompt = (
                    f"I have {len(uid_order)} crops detected as '{class_name}' (IDs: {uid_order}).\n"
                    f"List ONLY the IDs that are definitely NOT a '{class_name}'.\n"
                    f"Be conservative: only flag if very confident it is wrong.\n"
                    f"Respond ONLY as JSON: {{\"wrong_class\": []}}"
                )
                remove_key = "wrong_class"
            else:
                batch_prompt = (
                    f"I have {len(uid_order)} object crops (IDs: {uid_order}).\n"
                    f"List ONLY the IDs where the image contains absolutely nothing "
                    f"(empty wall, bare floor, or blank frame — no objects at all).\n"
                    f"Be VERY conservative. When in doubt do NOT include the ID.\n"
                    f"Respond ONLY as JSON: {{\"no_object\": []}}"
                )
                remove_key = "no_object"

            content.append({"text": batch_prompt})
            try:
                resp = client.converse(
                    modelId=model_id,
                    messages=[{"role": "user", "content": content}],
                    inferenceConfig={"maxTokens": 100, "temperature": 0.0},
                    system=[{"text": "Respond ONLY with valid JSON. No markdown."}],
                )
                raw = resp["output"]["message"]["content"][0]["text"].strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                result = json.loads(raw)
                for uid in result.get(remove_key, []):
                    if uid in vlm_objects:
                        print(f"  [VLM] Removed ID:{uid} ({class_name})")
                        to_remove.add(uid)
            except Exception as e:
                print(f"  [VLM] Batch error for '{class_name}': {e}")

    kept = {uid: obj for uid, obj in vlm_objects.items() if uid not in to_remove}
    removed = len(vlm_objects) - len(kept)
    if removed:
        print(f"  [VLM] Total removed: {removed} objects")
    return kept


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
        if config.SHOW_TRACK_ID:
            parts.append(f"ID:{display_id}")
        if config.SHOW_CLASS_NAME:
            class_name = config.CLASS_NAMES.get(int(cls_id), f"cls_{int(cls_id)}")
            parts.append(class_name[:18] + ".." if len(class_name) > 20 else class_name)
        if config.SHOW_CONFIDENCE:
            parts.append(f"{conf:.2f}")
        label = " | ".join(parts)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, 1)
        cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w + 5, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, (255, 255, 255), 2)
    return annotated


def draw_counter_overlay(frame, unique_counts: dict, frame_num: int, total_frames: int):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    panel_h = 30 + len(unique_counts) * 25 + 10
    panel_w = 320
    cv2.rectangle(overlay, (w - panel_w - 10, 5), (w - 5, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, f"Unique Objects (Frame {frame_num}/{total_frames})",
                (w - panel_w - 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    y_offset = 50
    total = 0
    for class_name, count in sorted(unique_counts.items()):
        cv2.putText(frame, f"{class_name}: {count}", (w - panel_w - 5, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y_offset += 25
        total += count
    cv2.putText(frame, f"TOTAL: {total}", (w - panel_w - 5, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
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

    track_class_map      = {}
    track_confidence_map = {}
    track_first_seen     = {}
    track_last_seen      = {}
    canonical_map        = {}
    peak_counts          = {}    # max simultaneous distinct tracks per class in any frame
    padded_crops         = {}    # tid -> {"crop": ndarray, "score": float}

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
                track_class_map[tid] = cid
                track_confidence_map[tid] = max(track_confidence_map.get(tid, 0), conf)
                if tid not in track_first_seen:
                    track_first_seen[tid] = frame_idx
                track_last_seen[tid] = frame_idx

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
                        padded_crops[int(tid)] = {"crop": pcrop.copy(), "score": _score}

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

        annotated = draw_counter_overlay(annotated, current_counts, frame_idx, total_frames)
        writer.write(annotated)

    pbar.close()
    cap.release()
    writer.release()
    elapsed = time.time() - start_time

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

    # ── VLM false-positive filter ──────────────────────────────
    if config.ENABLE_VLM_VALIDATION and reid:
        print("\n  [VLM] Validating with AWS Bedrock...")
        vlm_input = {}
        for obj in unique_objects:
            canon_id  = obj["unique_id"]
            best_crop = reid.deduplicator.get_best_crop_for_canonical(
                reid.collector, reid._canonical_map, canon_id
            )
            vlm_input[canon_id] = {**obj, "best_crop": best_crop}

        vlm_output  = _vlm_filter_unique_objects(vlm_input)
        removed_ids = set(vlm_input.keys()) - set(vlm_output.keys())

        for obj in unique_objects:
            if obj["unique_id"] in removed_ids:
                class_name = obj["class_name"]
                if unique_counts.get(class_name, 0) > 0:
                    unique_counts[class_name] -= 1
                    if unique_counts[class_name] == 0:
                        del unique_counts[class_name]

        unique_objects = [o for o in unique_objects if o["unique_id"] not in removed_ids]

    # ── Save individual crops & contact sheet ─────────────────────
    crops_dir  = os.path.join(output_dir, f"{video_name}_crops")
    sheet_path = os.path.join(output_dir, f"{video_name}_contact_sheet.jpg")
    os.makedirs(crops_dir, exist_ok=True)

    c_map = reid._canonical_map if reid else {}
    canonical_to_tids: dict = defaultdict(list)
    for tid in padded_crops:
        canon = c_map.get(tid, tid)
        canonical_to_tids[canon].append(tid)

    post_entries = []
    for obj in sorted(unique_objects, key=lambda o: (o["class_name"], o["unique_id"])):
        uid        = obj["unique_id"]
        class_name = obj["class_name"]
        best_score = -1.0
        best_crop  = None
        for tid in canonical_to_tids.get(uid, [uid]):
            entry = padded_crops.get(int(tid))
            if entry and entry["score"] > best_score:
                best_score = entry["score"]
                best_crop  = entry["crop"]
        safe_cls = class_name.replace("/", "_").replace(" ", "_")
        if best_crop is not None and best_crop.size > 0:
            cv2.imwrite(os.path.join(crops_dir, f"{safe_cls}_id{uid}.jpg"),
                        best_crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        post_entries.append({"img": best_crop,
                              "label": f"{class_name} | ID:{uid}"})

    if post_entries:
        contact_sheet = make_contact_sheet(
            post_entries, f"{video_name} — {len(unique_objects)} unique objects")
        cv2.imwrite(sheet_path, contact_sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])

    total_unique     = sum(unique_counts.values())
    total_raw_tracks = len(track_class_map)

    # ── Output JSON ────────────────────────────────────────────
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
        "counts_by_class":      dict(sorted(unique_counts.items())),
        "peak_counts_by_class": dict(sorted(peak_counts.items())),
        "unique_objects":       unique_objects,
        "output_crops_dir":     crops_dir,
        "output_contact_sheet": sheet_path,
        "config": {
            "yolo_model":           config.YOLO_MODEL_PATH,
            "confidence_threshold": config.YOLO_CONFIDENCE,
            "frame_skip":           config.FRAME_SKIP,
            "reid_threshold":       config.REID_SIMILARITY_THRESHOLD if config.ENABLE_CLIP_REID else None,
            "tracker":              config.TRACKER_TYPE,
        },
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
    print(f"  Output JSON    : {output_json_path}")
    print(f"  Output Video   : {output_video_path}")
    print(f"  Individual crops: {crops_dir}")
    print(f"  Contact sheet  : {sheet_path}")
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
