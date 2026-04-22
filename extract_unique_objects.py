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

CONTACT_THUMB_W = 200
CONTACT_THUMB_H = 200
CONTACT_COLS    = 6
LABEL_H         = 28
BG_COLOR        = (30, 30, 30)
BORDER_COLOR    = (80, 80, 80)


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


def stack_contact_sheets(sheet1: np.ndarray, name1: str,
                          sheet2: np.ndarray, name2: str) -> np.ndarray:
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
                    args) -> dict:
    """
    Two-pass unique object collection:
      Pass 1: YOLO + ByteTrack, collect quality-scored padded crops per track.
      Pass 2: DINOv2 + DBSCAN deduplication after video ends.

    Returns:
        {
          canonical_id: {
              "class_id": int,
              "class_name": str,
              "best_crop": np.ndarray BGR,   # highest-quality crop after dedup
          }, ...
        }
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: cannot open {video_path}")
        return {}

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
    padded_crops: dict = {}   # tid -> {"crop": ndarray, "area": int}

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

        reid.register_frame_tracks(track_ids, frame, frame_idx)

        for box, tid, cid in zip(boxes, track_ids, class_ids):
            x1, y1, x2, y2 = map(int, box)

            # Tight crop → ReID embedding. Full frame → scene fingerprint.
            x1t, y1t = max(0, x1), max(0, y1)
            x2t, y2t = min(width, x2), min(height, y2)
            tight = frame[y1t:y2t, x1t:x2t]
            if tight.size > 0:
                reid.update_track(tid, cid, tight,
                                  (x1t, y1t, x2t, y2t),
                                  frame=frame)

            # Padded crop → display only. Track the largest one seen per track.
            x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
            x2p, y2p = min(width, x2 + pad), min(height, y2 + pad)
            area = (x2p - x1p) * (y2p - y1p)
            if int(tid) not in padded_crops or area > padded_crops[int(tid)]["area"]:
                padded_crops[int(tid)] = {
                    "crop": frame[y1p:y2p, x1p:x2p].copy(),
                    "area": area,
                }

    pbar.close()
    cap.release()

    # ── Post-hoc deduplication ────────────────────────────────
    reid.finalize()
    canonical_map = reid._canonical_map
    collector = reid.collector

    # Build output: one entry per canonical object.
    # Display crop = best padded crop across ALL tracks in the canonical cluster.
    unique_objects: dict = {}
    seen_canonicals: set = set()

    for tid, info in collector.tracks.items():
        canon_id = canonical_map.get(tid, tid)
        if canon_id in seen_canonicals:
            continue
        seen_canonicals.add(canon_id)

        # Find best padded crop across all tracks merged into this canonical
        best_area = -1
        best_display_crop = None
        for t in collector.tracks:
            if canonical_map.get(t, t) == canon_id:
                entry = padded_crops.get(int(t))
                if entry and entry["area"] > best_area:
                    best_area = entry["area"]
                    best_display_crop = entry["crop"]

        unique_objects[canon_id] = {
            "class_id":   info["class_id"],
            "class_name": info["class_name"],
            "best_crop":  best_display_crop,
        }

    return unique_objects


# ── Per-video ─────────────────────────────────────────────────────────────────

def process_video(video_path: str, model1: YOLO, model2: YOLO,
                   model1_name: str, model2_name: str,
                   output_dir: str, args):
    video_name = Path(video_path).stem
    video_out  = os.path.join(output_dir, video_name)

    print(f"\n  -- {model1_name} pass --")
    objs1 = run_and_collect(video_path, model1, model1_name, args)

    print(f"\n  -- {model2_name} pass --")
    objs2 = run_and_collect(video_path, model2, model2_name, args)

    for model_name, objects in [(model1_name, objs1), (model2_name, objs2)]:
        ind_dir = os.path.join(video_out, model_name, "individual")
        os.makedirs(ind_dir, exist_ok=True)

        entries = []
        # Sort by class name then ID for consistent ordering
        for uid, obj in sorted(objects.items(),
                                key=lambda kv: (kv[1]["class_name"], kv[0])):
            safe_cls = obj["class_name"].replace("/", "_").replace(" ", "_")
            filename = f"{safe_cls}_id{uid}.jpg"
            filepath = os.path.join(ind_dir, filename)

            crop = obj["best_crop"]
            if crop is not None and crop.size > 0:
                cv2.imwrite(filepath, crop, [cv2.IMWRITE_JPEG_QUALITY, 92])

            label = f"{obj['class_name']} | ID:{uid}"
            entries.append({"img": crop, "label": label})

        sheet = make_contact_sheet(
            entries, f"{model_name} — {video_name} — {len(objects)} unique objects")
        sheet_path = os.path.join(video_out, model_name, "contact_sheet.jpg")
        cv2.imwrite(sheet_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"\n  [{model_name}] {len(objects)} unique objects")
        print(f"    Individual crops : {ind_dir}")
        print(f"    Contact sheet    : {sheet_path}")

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
    p.add_argument("--crop-pad",     type=int,   default=10,
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
