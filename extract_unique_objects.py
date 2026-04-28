"""
Unique Object Extractor
========================
Runs both YOLO models through the full pipeline (ByteTrack + CLIP ReID).
For each unique object, saves ONE crop image using the LARGEST bounding box
seen across all frames for that track.

At the end of each video, also generates a contact sheet (grid image)
showing every unique object per model side by side.

Output structure:
    unique_objects/
        <video_name>/
            <model_name>/
                individual/
                    Desk_id1.jpg
                    Office_Chair_id2.jpg
                    ...
                contact_sheet.jpg
            comparison_sheet.jpg   ← model1 grid vs model2 grid

Usage:
    python extract_unique_objects.py \
        --model1 old.pt \
        --model2 new.pt \
        --videos /path/to/videos \
        --output unique_objects \
        [--model1-name "v1"] [--model2-name "v2"] \
        [--frame-skip 2] [--conf 0.35] [--reid-threshold 0.82] \
        [--crop-pad 10]
"""

import argparse
import math
import os
import textwrap
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

import config
from reid_module import CLIPReIdentifier

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}

import base64

CONTACT_THUMB_W = 200
CONTACT_THUMB_H = 200
CONTACT_COLS    = 6
LABEL_H         = 28
BG_COLOR        = (30, 30, 30)
BORDER_COLOR    = (80, 80, 80)


# ── Quality filters ──────────────────────────────────────────────────────────

def filter_by_sharpness(unique_objects: dict, min_sharpness: float) -> dict:
    """
    Remove objects whose best crop Laplacian variance < min_sharpness.
    Catches motion-blurred walls, partial crops, featureless surfaces.
    """
    if min_sharpness <= 0:
        return unique_objects
    kept = {}
    for uid, obj in unique_objects.items():
        crop = obj.get("best_crop")
        if crop is None or crop.size == 0:
            print(f"  [SharpFilter] Removed ID:{uid} ({obj['class_name']}) — no valid crop")
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        if sharpness < min_sharpness:
            print(f"  [SharpFilter] Removed ID:{uid} ({obj['class_name']}) "
                  f"— sharpness {sharpness:.1f} < {min_sharpness}")
        else:
            kept[uid] = obj
    return kept


def validate_with_vlm(unique_objects: dict) -> dict:
    """
    Validate each unique object with Claude claude-haiku-4-5 (vision).
    Removes objects the VLM says are NOT a valid instance of their class.
    Requires ANTHROPIC_API_KEY env var.
    """
    try:
        import anthropic
    except ImportError:
        print("  [VLM] anthropic package not installed — skipping validation")
        return unique_objects

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  [VLM] ANTHROPIC_API_KEY not set — skipping validation")
        return unique_objects

    client = anthropic.Anthropic(api_key=api_key)
    kept = {}

    for uid, obj in unique_objects.items():
        crop = obj.get("best_crop")
        class_name = obj["class_name"]
        if crop is None or crop.size == 0:
            print(f"  [VLM] Removed ID:{uid} ({class_name}) — no valid crop")
            continue

        # Encode crop as JPEG base64
        success, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            kept[uid] = obj
            continue
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        prompt = (
            f"Does this image clearly show a '{class_name}'? "
            f"Answer only 'yes' or 'no'. "
            f"Answer 'no' if: the image is blurry/noisy, shows only a partial fragment, "
            f"is a wall/floor/ceiling with no furniture, or is clearly not a {class_name}."
        )

        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=10,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        }},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            answer = response.content[0].text.strip().lower()
            if answer.startswith("yes"):
                kept[uid] = obj
            else:
                print(f"  [VLM] Removed ID:{uid} ({class_name}) — VLM: '{answer}'")
        except Exception as e:
            print(f"  [VLM] Error for ID:{uid} — {e}; keeping by default")
            kept[uid] = obj

    return kept


# ── Contact sheet builder ────────────────────────────────────────────────────

def make_contact_sheet(entries: list, title: str) -> np.ndarray:
    """
    entries: list of {"img": np.ndarray BGR, "label": str}
    Returns a single BGR image (contact sheet).
    """
    if not entries:
        blank = np.full((100, 400, 3), BG_COLOR, dtype=np.uint8)
        cv2.putText(blank, f"{title}: no objects", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        return blank

    cols = min(CONTACT_COLS, len(entries))
    rows = (len(entries) + cols - 1) // cols
    cell_w = CONTACT_THUMB_W
    cell_h = CONTACT_THUMB_H + LABEL_H
    title_h = 40

    sheet_w = cols * cell_w
    sheet_h = rows * cell_h + title_h
    sheet = np.full((sheet_h, sheet_w, 3), BG_COLOR, dtype=np.uint8)

    # Title bar
    cv2.rectangle(sheet, (0, 0), (sheet_w, title_h), (20, 60, 100), -1)
    cv2.putText(sheet, title, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 220), 2)

    for i, entry in enumerate(entries):
        row = i // cols
        col = i % cols
        x = col * cell_w
        y = title_h + row * cell_h

        # Resize crop to thumbnail
        img = entry["img"]
        if img is not None and img.size > 0:
            thumb = cv2.resize(img, (CONTACT_THUMB_W, CONTACT_THUMB_H))
        else:
            thumb = np.full((CONTACT_THUMB_H, CONTACT_THUMB_W, 3),
                             (60, 60, 60), dtype=np.uint8)

        sheet[y:y + CONTACT_THUMB_H, x:x + cell_w] = thumb

        # Label background
        cv2.rectangle(sheet,
                      (x, y + CONTACT_THUMB_H),
                      (x + cell_w, y + CONTACT_THUMB_H + LABEL_H),
                      (15, 15, 15), -1)

        # Wrap label text
        label = entry["label"]
        font_scale = 0.38
        cv2.putText(sheet, label,
                    (x + 4, y + CONTACT_THUMB_H + LABEL_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

        # Border
        cv2.rectangle(sheet, (x, y), (x + cell_w - 1, y + cell_h - 1),
                      BORDER_COLOR, 1)

    return sheet


def stack_contact_sheets(sheet1: np.ndarray, _name1: str,
                          sheet2: np.ndarray, _name2: str) -> np.ndarray:
    """Stack two contact sheets vertically with a divider."""
    w = max(sheet1.shape[1], sheet2.shape[1])
    def pad_w(img, target_w):
        dw = target_w - img.shape[1]
        if dw > 0:
            img = cv2.copyMakeBorder(img, 0, 0, 0, dw,
                                     cv2.BORDER_CONSTANT, value=BG_COLOR)
        return img

    divider = np.full((6, w, 3), (100, 100, 100), dtype=np.uint8)
    return np.vstack([pad_w(sheet1, w), divider, pad_w(sheet2, w)])


# ── Core: run one model pass, collect best crops ─────────────────────────────

def run_and_collect(video_path: str, model: YOLO, model_name: str,
                    args) -> tuple:
    """
    Two-pass unique object collection:
      Pass 1: YOLO + ByteTrack, collect quality-scored padded crops per track.
      Pass 2: DINOv2 + DBSCAN deduplication after video ends.

    Returns:
        (unique_objects, frame_detections, canonical_map)
        unique_objects: {canonical_id: {"class_id", "class_name", "best_crop"}}
        frame_detections: {frame_idx: [{"box", "tid", "cid", "conf"}, ...]}
        canonical_map: {track_id: canonical_id}
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: cannot open {video_path}")
        return {}, {}, {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pad    = args.crop_pad

    reid = CLIPReIdentifier(
        similarity_threshold=args.reid_threshold,
        model_name=config.CLIP_MODEL_NAME,
    )

    # Padded crops tracked separately for display (tight crops go to ReID).
    # Best padded crop = largest bbox seen per track.
    padded_crops: dict = {}       # tid -> {"crop": ndarray, "area": int}
    frame_detections: dict = {}   # frame_idx -> [{"box","tid","cid","conf"}, ...]

    # Raw YOLO detections (ignoring track IDs) — top-K quality per class.
    # Bounded memory: prune to 15 per class every 30 detections, keep top-8 at end.
    _YOLO_PRUNE_AT = 30
    _YOLO_KEEP     = 15
    _YOLO_FINAL_K  = 8
    yolo_raw_scored: dict = defaultdict(list)   # class_id -> [(score, crop)]
    yolo_det_counts: dict = defaultdict(int)    # class_id -> total detection count

    processed_count = 0
    frame_idx = -1
    pbar = tqdm(total=total_frames,
                desc=f"    {model_name}", unit="frame", leave=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        pbar.update(1)

        if frame_idx % args.frame_skip != 0:
            continue

        processed_count += 1

        results = model.track(
            frame,
            persist=True,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.img_size,
            tracker=f"{config.TRACKER_TYPE}.yaml",
            verbose=False,
        )

        result = results[0]
        if result.boxes is None or result.boxes.id is None:
            continue

        boxes     = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy().astype(int)
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confs     = result.boxes.conf.cpu().numpy()

        reid.register_frame_tracks(track_ids, frame, frame_idx)

        frame_detections[frame_idx] = []
        for box, tid, cid, conf in zip(boxes, track_ids, class_ids, confs):
            frame_detections[frame_idx].append({
                "box": tuple(box.tolist()),
                "tid": int(tid),
                "cid": int(cid),
                "conf": float(conf),
            })
            x1, y1, x2, y2 = map(int, box)

            # Tight crop → ReID embedding. Full frame → scene fingerprint.
            x1t, y1t = max(0, x1), max(0, y1)
            x2t, y2t = min(width, x2), min(height, y2)
            tight = frame[y1t:y2t, x1t:x2t]
            if tight.size > 0:
                reid.update_track(tid, cid, tight,
                                  (x1t, y1t, x2t, y2t),
                                  frame=frame)

            # Padded crop → display only. Keep the sharpest, most complete crop.
            # Score uses tight crop (not padded) so background padding doesn't
            # dilute the sharpness signal for identical-looking objects.
            x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
            x2p, y2p = min(width, x2 + pad), min(height, y2 + pad)
            pcrop = frame[y1p:y2p, x1p:x2p]
            if tight.size > 0 and pcrop.size > 0:
                _gray = cv2.cvtColor(tight, cv2.COLOR_BGR2GRAY)
                _sharp = cv2.Laplacian(_gray, cv2.CV_64F).var()
                _tight_area = (x2t - x1t) * (y2t - y1t)
                _complete = 0.6 if (x1 <= 3 or y1 <= 3
                                     or x2 >= width - 3 or y2 >= height - 3) else 1.0
                _disp_score = math.log1p(_sharp) * math.log1p(_tight_area) * _complete
                if int(tid) not in padded_crops or _disp_score > padded_crops[int(tid)]["score"]:
                    padded_crops[int(tid)] = {"crop": pcrop.copy(), "score": _disp_score}

            # YOLO raw — collect quality-scored crops per class (tracking-agnostic).
            if pcrop.size > 0:
                gray = cv2.cvtColor(pcrop, cv2.COLOR_BGR2GRAY)
                qs = cv2.Laplacian(gray, cv2.CV_64F).var() * (x2p - x1p) * (y2p - y1p)
                yolo_raw_scored[int(cid)].append((qs, pcrop.copy()))
                yolo_det_counts[int(cid)] += 1
                if len(yolo_raw_scored[int(cid)]) >= _YOLO_PRUNE_AT:
                    yolo_raw_scored[int(cid)].sort(key=lambda x: x[0], reverse=True)
                    yolo_raw_scored[int(cid)] = yolo_raw_scored[int(cid)][:_YOLO_KEEP]

    pbar.close()
    cap.release()

    # ── Post-hoc deduplication ────────────────────────────────
    reid.finalize()
    canonical_map = reid._canonical_map
    collector = reid.collector

    # ── YOLO raw: top-K crops per class, with total detection counts ──
    yolo_raw: dict = {}   # class_id -> {"class_name", "total_count", "top_crops"}
    for cid, scored in yolo_raw_scored.items():
        scored.sort(key=lambda x: x[0], reverse=True)
        yolo_raw[cid] = {
            "class_name":  config.CLASS_NAMES.get(cid, f"cls_{cid}"),
            "total_count": yolo_det_counts[cid],
            "top_crops":   [c for _, c in scored[:_YOLO_FINAL_K]],
        }

    # ── Raw tracks (pre-dedup) for before/after comparison ───────
    raw_tracks: dict = {}
    for tid, info in collector.tracks.items():
        entry = padded_crops.get(int(tid))
        raw_tracks[tid] = {
            "class_id":   info["class_id"],
            "class_name": info["class_name"],
            "best_crop":  entry["crop"] if entry else None,
        }

    # ── Build output: one entry per canonical object ──────────────
    # Display crop = best padded crop across ALL tracks in the canonical cluster.
    unique_objects: dict = {}
    seen_canonicals: set = set()

    for tid, info in collector.tracks.items():
        canon_id = canonical_map.get(tid, tid)
        if canon_id in seen_canonicals:
            continue
        seen_canonicals.add(canon_id)

        best_score = -1.0
        best_display_crop = None
        for t in collector.tracks:
            if canonical_map.get(t, t) == canon_id:
                entry = padded_crops.get(int(t))
                if entry and entry["score"] > best_score:
                    best_score = entry["score"]
                    best_display_crop = entry["crop"]

        unique_objects[canon_id] = {
            "class_id":   info["class_id"],
            "class_name": info["class_name"],
            "best_crop":  best_display_crop,
        }

    return unique_objects, frame_detections, canonical_map, raw_tracks, yolo_raw


# ── Annotated video writer ────────────────────────────────────────────────────

def _class_color(class_id: int) -> tuple:
    np.random.seed(int(class_id) * 42 + 7)
    return tuple(int(c) for c in np.random.randint(50, 255, size=3))


def write_annotated_video(video_path: str, frame_detections: dict,
                           canonical_map: dict, valid_canon_ids: set,
                           output_path: str, args):
    """
    Re-read video. For every processed frame draw bboxes with canonical IDs.
    Valid objects   → colored box, label = "ID:{canon} | {class} | {conf}"
    Filtered objects → gray box,  label += " [filtered]"
    Counter overlay shows final unique counts (post-filter).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: cannot re-open for annotation: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_fps = config.OUTPUT_VIDEO_FPS or max(1, fps_in / args.frame_skip)
    fourcc  = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)
    writer  = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))

    # Pre-compute final unique counts for the static overlay
    unique_counts: dict = defaultdict(int)
    seen_canon: set = set()
    for dets in frame_detections.values():
        for d in dets:
            canon = canonical_map.get(d["tid"], d["tid"])
            if canon in valid_canon_ids and canon not in seen_canon:
                seen_canon.add(canon)
                cname = config.CLASS_NAMES.get(d["cid"], f"cls_{d['cid']}")
                unique_counts[cname] += 1
    total_unique = sum(unique_counts.values())

    frame_idx = -1
    pbar = tqdm(total=total_frames, desc="    Annotating", unit="frame", leave=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        pbar.update(1)

        if frame_idx % args.frame_skip != 0:
            continue

        annotated = frame.copy()
        dets = frame_detections.get(frame_idx, [])

        for d in dets:
            x1, y1, x2, y2 = map(int, d["box"])
            canon = canonical_map.get(d["tid"], d["tid"])
            is_valid = canon in valid_canon_ids
            cid = d["cid"]
            conf = d["conf"]

            color = _class_color(cid) if is_valid else (120, 120, 120)
            cname = config.CLASS_NAMES.get(cid, f"cls_{cid}")
            if len(cname) > 20:
                cname = cname[:18] + ".."
            label = f"ID:{canon} | {cname} | {conf:.2f}"
            if not is_valid:
                label += " [filtered]"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, config.BBOX_THICKNESS)
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 5, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, (255, 255, 255), 2)

        # Counter overlay (static final counts)
        h, w = annotated.shape[:2]
        panel_w = 320
        panel_h = 30 + len(unique_counts) * 25 + 35
        overlay = annotated.copy()
        cv2.rectangle(overlay, (w - panel_w - 10, 5), (w - 5, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
        cv2.putText(annotated, "Unique Objects (post-filter)",
                    (w - panel_w - 5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_off = 50
        for cname, cnt in sorted(unique_counts.items()):
            cv2.putText(annotated, f"{cname}: {cnt}",
                        (w - panel_w - 5, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y_off += 25
        cv2.putText(annotated, f"TOTAL: {total_unique}",
                    (w - panel_w - 5, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        writer.write(annotated)

    pbar.close()
    cap.release()
    writer.release()
    print(f"    Annotated video  : {output_path}")


# ── Per-video ─────────────────────────────────────────────────────────────────

def process_video(video_path: str, model1: YOLO, model2: YOLO,
                   model1_name: str, model2_name: str,
                   output_dir: str, args):
    video_name = Path(video_path).stem
    video_out  = os.path.join(output_dir, video_name)

    print(f"\n  -- {model1_name} pass --")
    objs1, fd1, cmap1, raw1, yolo1 = run_and_collect(video_path, model1, model1_name, args)
    objs1 = filter_by_sharpness(objs1, config.MIN_CROP_SHARPNESS)
    if config.ENABLE_VLM_VALIDATION:
        print(f"  [VLM] Validating {len(objs1)} objects ({model1_name})...")
        objs1 = validate_with_vlm(objs1)
    vid1_dir = os.path.join(video_out, model1_name)
    os.makedirs(vid1_dir, exist_ok=True)
    write_annotated_video(video_path, fd1, cmap1, set(objs1.keys()),
                          os.path.join(vid1_dir, f"{video_name}_annotated.mp4"), args)

    print(f"\n  -- {model2_name} pass --")
    objs2, fd2, cmap2, raw2, yolo2 = run_and_collect(video_path, model2, model2_name, args)
    objs2 = filter_by_sharpness(objs2, config.MIN_CROP_SHARPNESS)
    if config.ENABLE_VLM_VALIDATION:
        print(f"  [VLM] Validating {len(objs2)} objects ({model2_name})...")
        objs2 = validate_with_vlm(objs2)
    vid2_dir = os.path.join(video_out, model2_name)
    os.makedirs(vid2_dir, exist_ok=True)
    write_annotated_video(video_path, fd2, cmap2, set(objs2.keys()),
                          os.path.join(vid2_dir, f"{video_name}_annotated.mp4"), args)

    for model_name, objects, raw_tracks, yolo_data in [
            (model1_name, objs1, raw1, yolo1), (model2_name, objs2, raw2, yolo2)]:
        ind_dir = os.path.join(video_out, model_name, "individual")
        os.makedirs(ind_dir, exist_ok=True)

        # ── Stage 1: YOLO raw sheet ───────────────────────────────
        yolo_total = sum(v["total_count"] for v in yolo_data.values())
        yolo_entries = []
        for cid in sorted(yolo_data, key=lambda c: yolo_data[c]["class_name"]):
            info = yolo_data[cid]
            for i, crop in enumerate(info["top_crops"]):
                label = f"{info['class_name']} (×{info['total_count']}) #{i+1}"
                yolo_entries.append({"img": crop, "label": label})
        yolo_sheet = make_contact_sheet(
            yolo_entries,
            f"Stage 1: YOLO Raw — {video_name} — {yolo_total} total detections")
        yolo_sheet_path = os.path.join(video_out, model_name, "contact_sheet_yolo_raw.jpg")
        cv2.imwrite(yolo_sheet_path, yolo_sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # ── Stage 2: ByteTrack raw sheet ─────────────────────────
        raw_entries = [
            {"img": obj["best_crop"],
             "label": f"{obj['class_name']} | raw:{tid}"}
            for tid, obj in sorted(raw_tracks.items(),
                                   key=lambda kv: (kv[1]["class_name"], kv[0]))
        ]
        raw_sheet = make_contact_sheet(
            raw_entries,
            f"Stage 2: ByteTrack — {video_name} — {len(raw_tracks)} tracks (pre-ReID)")
        raw_sheet_path = os.path.join(video_out, model_name, "contact_sheet_raw.jpg")
        cv2.imwrite(raw_sheet_path, raw_sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # ── Stage 3: Post-filter sheet + individual crops ─────────
        post_entries = []
        for uid, obj in sorted(objects.items(),
                                key=lambda kv: (kv[1]["class_name"], kv[0])):
            safe_cls = obj["class_name"].replace("/", "_").replace(" ", "_")
            crop = obj["best_crop"]
            if crop is not None and crop.size > 0:
                cv2.imwrite(os.path.join(ind_dir, f"{safe_cls}_id{uid}.jpg"),
                            crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            post_entries.append({"img": crop, "label": f"{obj['class_name']} | ID:{uid}"})

        post_sheet = make_contact_sheet(
            post_entries,
            f"Stage 3: Post-filter — {video_name} — {len(objects)} unique objects")
        post_sheet_path = os.path.join(video_out, model_name, "contact_sheet.jpg")
        cv2.imwrite(post_sheet_path, post_sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # ── 3-stage comparison (stacked vertically) ───────────────
        stages_sheet = stack_contact_sheets(
            stack_contact_sheets(yolo_sheet, "", raw_sheet, ""), "",
            post_sheet, "")
        stages_path = os.path.join(video_out, model_name, "contact_sheet_stages.jpg")
        cv2.imwrite(stages_path, stages_sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])

        print(f"\n  [{model_name}]  YOLO: {yolo_total} dets "
              f"→ ByteTrack: {len(raw_tracks)} tracks "
              f"→ Post-filter: {len(objects)} unique")
        print(f"    Individual crops : {ind_dir}")
        print(f"    YOLO raw sheet   : {yolo_sheet_path}")
        print(f"    ByteTrack sheet  : {raw_sheet_path}")
        print(f"    Post-filter sheet: {post_sheet_path}")
        print(f"    Stages comparison: {stages_path}")

    # Comparison sheet: model1 on top, model2 below
    sheet1 = cv2.imread(os.path.join(video_out, model1_name, "contact_sheet.jpg"))
    sheet2 = cv2.imread(os.path.join(video_out, model2_name, "contact_sheet.jpg"))
    if sheet1 is not None and sheet2 is not None:
        comp = stack_contact_sheets(sheet1, model1_name, sheet2, model2_name)
        comp_path = os.path.join(video_out, "comparison_sheet.jpg")
        cv2.imwrite(comp_path, comp, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"\n  Comparison sheet : {comp_path}")

    # Print diff
    cls1 = defaultdict(int)
    cls2 = defaultdict(int)
    for o in objs1.values():
        cls1[o["class_name"]] += 1
    for o in objs2.values():
        cls2[o["class_name"]] += 1
    all_cls = sorted(set(list(cls1.keys()) + list(cls2.keys())))

    print(f"\n  {'CLASS':<35} {model1_name:>14} {model2_name:>14}  DIFF")
    print(f"  {'-'*70}")
    for cls in all_cls:
        c1, c2 = cls1.get(cls, 0), cls2.get(cls, 0)
        diff = c2 - c1
        print(f"  {cls:<35} {c1:>14} {c2:>14}  {diff:+}")
    print(f"  {'-'*70}")
    t1, t2 = sum(cls1.values()), sum(cls2.values())
    print(f"  {'TOTAL UNIQUE':<35} {t1:>14} {t2:>14}  {t2-t1:+}")


# ── Args + main ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract unique object crops from two YOLO models (with ReID)")
    p.add_argument("--model1",       required=True)
    p.add_argument("--model2",       required=True)
    p.add_argument("--videos",       default=config.INPUT_FOLDER)
    p.add_argument("--output",       default="unique_objects")
    p.add_argument("--model1-name",  default="Model-v1")
    p.add_argument("--model2-name",  default="Model-v2")
    p.add_argument("--frame-skip",   type=int,   default=config.FRAME_SKIP)
    p.add_argument("--conf",         type=float, default=config.YOLO_CONFIDENCE)
    p.add_argument("--iou",          type=float, default=config.YOLO_IOU)
    p.add_argument("--img-size",     type=int,   default=config.YOLO_IMG_SIZE)
    p.add_argument("--reid-threshold", type=float,
                   default=config.REID_SIMILARITY_THRESHOLD)
    p.add_argument("--crop-pad",     type=int,   default=60,
                   help="Pixels of padding around each crop bbox")
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
    print(f"  UNIQUE OBJECT EXTRACTOR")
    print(f"{'='*65}")
    print(f"  Model 1  : {args.model1_name}  ({args.model1})")
    print(f"  Model 2  : {args.model2_name}  ({args.model2})")
    print(f"  Videos   : {len(videos)} in {args.videos}")
    print(f"  Output   : {args.output}")
    print(f"  Crop pad : {args.crop_pad}px")
    print(f"{'='*65}\n")

    print("[*] Loading YOLO models...")
    model1 = YOLO(args.model1)
    model2 = YOLO(args.model2)
    print("    Done.\n")

    for idx, vp in enumerate(videos, 1):
        print(f"\n{'='*65}")
        print(f"  [{idx}/{len(videos)}] {vp.name}")
        print(f"{'='*65}")
        process_video(str(vp), model1, model2,
                      args.model1_name, args.model2_name,
                      args.output, args)

    print(f"\n{'='*65}")
    print(f"  DONE → {args.output}/")
    print(f"  <video>/<model>/individual/  ← one JPG per unique object")
    print(f"  <video>/<model>/contact_sheet.jpg  ← all objects in grid")
    print(f"  <video>/comparison_sheet.jpg  ← both models stacked")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
