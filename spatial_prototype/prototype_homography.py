"""
Spatial Dedup Feasibility Prototype (Workstream C, Task C1).

Self-contained script — reads config.py for model path / tracker settings
but does not modify or import anything from main.py / reid_module.py.

Pipeline:
1. Open client_video/client_video.mp4, process every FRAME_SKIP-th frame.
2. Track camera motion via goodFeaturesToTrack + calcOpticalFlowPyrLK between
   consecutive processed frames -> findHomography(RANSAC) -> maintain a
   cumulative homography H_cum mapping current-frame coords -> frame-0 coords.
   If fewer than 30 tracked points or homography estimation fails, carry the
   previous H_cum forward and count the frame as "lost".
3. Run YOLO tracking with the exact conf/iou/imgsz/tracker settings main.py
   uses (read from config, not hardcoded).
4. For every detection, project bbox bottom-center through H_cum into
   panorama coords. Per track, keep the projection from the frame where the
   track had its largest bbox area (best localization).
5. Write out/track_positions.json and out/pair_distances.csv, print the key
   co-occurring vs all-pairs distance statistic.

Run: python3 spatial_prototype/prototype_homography.py
"""
import csv
import json
import os
import sys
import time
from collections import defaultdict
from itertools import combinations

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)
import config  # noqa: E402 — repo config, read-only

VIDEO_PATH = os.path.join(_REPO, "client_video", "client_video.mp4")
OUT_DIR = os.path.join(_HERE, "out")
os.makedirs(OUT_DIR, exist_ok=True)

MIN_FLOW_POINTS = 30  # guard threshold from spec


def compute_homography_chain(video_path, frame_skip):
    """Second pass over the video: cumulative homography per processed frame.
    Returns {frame_idx: H_cum (3x3 np.array mapping this frame -> frame 0)},
    plus (lost_frames, total_frames) stability stats.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    feature_params = dict(maxCorners=400, qualityLevel=0.01, minDistance=8, blockSize=7)
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    H_cum = np.eye(3, dtype=np.float64)
    homographies = {}
    prev_gray = None
    frame_idx = 0
    processed_count = 0
    lost = 0
    total = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % frame_skip != 0:
            continue
        processed_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            homographies[frame_idx] = H_cum.copy()
            prev_gray = gray
            continue

        total += 1
        pts_prev = cv2.goodFeaturesToTrack(prev_gray, **feature_params)
        ok_frame = False
        if pts_prev is not None and len(pts_prev) >= MIN_FLOW_POINTS:
            pts_next, status, _err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts_prev, None, **lk_params)
            status = status.reshape(-1).astype(bool)
            good_prev = pts_prev[status]
            good_next = pts_next[status]
            if len(good_prev) >= MIN_FLOW_POINTS:
                H_step, inlier_mask = cv2.findHomography(good_prev, good_next, cv2.RANSAC, 3.0)
                if H_step is not None:
                    # H_step maps prev-frame pts -> current-frame pts.
                    # We want current -> frame0, i.e. H_cum_new = H_cum_prev @ inv(H_step)
                    try:
                        H_step_inv = np.linalg.inv(H_step)
                        H_cum = H_cum @ H_step_inv
                        ok_frame = True
                    except np.linalg.LinAlgError:
                        ok_frame = False

        if not ok_frame:
            lost += 1
            # carry previous H_cum forward (already is H_cum, unchanged)

        homographies[frame_idx] = H_cum.copy()
        prev_gray = gray

    cap.release()
    return homographies, lost, total


def project_point(H, x, y):
    v = H @ np.array([x, y, 1.0])
    if abs(v[2]) < 1e-9:
        return None
    return float(v[0] / v[2]), float(v[1] / v[2])


def run_tracking(video_path, homographies, frame_skip):
    """Run YOLO tracking with main.py's exact settings; project detections."""
    from ultralytics import YOLO
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None
    for attempt in range(2):
        try:
            model = YOLO(config.YOLO_MODEL_PATH)
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and attempt == 0:
                print("  CUDA OOM loading model, waiting 60s and retrying...")
                time.sleep(60)
                continue
            raise

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    # track_id -> best record {class_id, x, y, frame, area}
    best_by_track = {}
    # frame_idx -> set of track_ids present (for co-occurrence)
    tracks_per_frame = defaultdict(set)
    track_class = {}

    frame_idx = 0
    use_device = device
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % frame_skip != 0:
            continue

        try:
            results = model.track(
                frame,
                persist=True,
                conf=config.YOLO_CONFIDENCE,
                iou=config.YOLO_IOU,
                imgsz=config.YOLO_IMG_SIZE,
                tracker=config.TRACKER_CONFIG_PATH,
                device=use_device,
                verbose=False,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and use_device != "cpu":
                print("  CUDA OOM during tracking, waiting 60s and retrying once...")
                time.sleep(60)
                try:
                    results = model.track(
                        frame, persist=True, conf=config.YOLO_CONFIDENCE,
                        iou=config.YOLO_IOU, imgsz=config.YOLO_IMG_SIZE,
                        tracker=config.TRACKER_CONFIG_PATH, device=use_device, verbose=False,
                    )
                except RuntimeError:
                    print("  Still OOM — falling back to CPU for remainder of run.")
                    use_device = "cpu"
                    model = YOLO(config.YOLO_MODEL_PATH)
                    results = model.track(
                        frame, persist=True, conf=config.YOLO_CONFIDENCE,
                        iou=config.YOLO_IOU, imgsz=config.YOLO_IMG_SIZE,
                        tracker=config.TRACKER_CONFIG_PATH, device=use_device, verbose=False,
                    )
            else:
                raise

        result = results[0]
        if result.boxes is None or result.boxes.id is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy().astype(int)
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        H = homographies.get(frame_idx)
        if H is None:
            # frame wasn't in the homography pass (shouldn't happen, same skip stride)
            continue

        for box, tid, cid in zip(boxes, track_ids, class_ids):
            x1, y1, x2, y2 = box
            area = float((x2 - x1) * (y2 - y1))
            bx, by = (x1 + x2) / 2.0, y2  # bbox bottom-center
            proj = project_point(H, bx, by)
            if proj is None:
                continue
            px, py = proj

            tid = int(tid)
            track_class[tid] = int(cid)
            tracks_per_frame[frame_idx].add(tid)

            prev = best_by_track.get(tid)
            if prev is None or area > prev["area"]:
                best_by_track[tid] = {
                    "class_id": int(cid), "x": px, "y": py,
                    "frame": frame_idx, "area": area,
                    "w": float(x2 - x1), "h": float(y2 - y1),
                }

    cap.release()
    return best_by_track, tracks_per_frame, track_class


def compute_cooccurrence(tracks_per_frame, min_frames=3):
    """same-class co-occurrence pairs seen together >= min_frames times."""
    pair_counts = defaultdict(int)
    for frame_idx, tids in tracks_per_frame.items():
        for a, b in combinations(sorted(tids), 2):
            pair_counts[(a, b)] += 1
    return {pair: n for pair, n in pair_counts.items() if n >= min_frames}


def main():
    print(f"Video: {VIDEO_PATH}")
    print(f"FRAME_SKIP (from config): {config.FRAME_SKIP}")
    print(f"YOLO_MODEL_PATH: {config.YOLO_MODEL_PATH}")

    t0 = time.time()
    print("\n[1/3] Computing cumulative homography via optical flow...")
    homographies, lost, total = compute_homography_chain(VIDEO_PATH, config.FRAME_SKIP)
    lost_pct = 100.0 * lost / total if total else 0.0
    print(f"  Homography frames: {total} total, {lost} lost ({lost_pct:.1f}%)")

    print("\n[2/3] Running YOLO tracking + projecting detections...")
    best_by_track, tracks_per_frame, track_class = run_tracking(VIDEO_PATH, homographies, config.FRAME_SKIP)
    print(f"  Tracks found: {len(best_by_track)}")

    print("\n[3/3] Computing co-occurrence + pairwise distances...")
    min_cooc = getattr(config, "MIN_COOCCURRENCE_FRAMES", 3)
    cooc_pairs = compute_cooccurrence(tracks_per_frame, min_frames=min_cooc)
    print(f"  Co-occurring pairs (>= {min_cooc} shared frames): {len(cooc_pairs)}")

    # ── median bbox diagonal per class (normalization unit) ──────────────
    diag_by_class = defaultdict(list)
    for tid, rec in best_by_track.items():
        diag = float(np.hypot(rec["w"], rec["h"]))
        diag_by_class[rec["class_id"]].append(diag)
    median_diag = {cid: float(np.median(d)) for cid, d in diag_by_class.items() if d}

    # ── track_positions.json ──────────────────────────────────────────────
    positions_out = {
        str(tid): {"class_id": rec["class_id"], "x": rec["x"], "y": rec["y"], "frame": rec["frame"]}
        for tid, rec in best_by_track.items()
    }
    cooc_list = [[a, b, n] for (a, b), n in sorted(cooc_pairs.items())]
    with open(os.path.join(OUT_DIR, "track_positions.json"), "w") as f:
        json.dump({"tracks": positions_out, "cooccurrence_pairs": cooc_list}, f, indent=2)

    # ── pair_distances.csv ──────────────────────────────────────────────
    rows = []
    by_class_tracks = defaultdict(list)
    for tid, rec in best_by_track.items():
        by_class_tracks[rec["class_id"]].append(tid)

    for cid, tids in by_class_tracks.items():
        unit = median_diag.get(cid)
        if not unit or unit <= 0:
            continue
        cname = config.CLASS_NAMES.get(cid, f"class_{cid}")
        for a, b in combinations(sorted(tids), 2):
            ra, rb = best_by_track[a], best_by_track[b]
            d = float(np.hypot(ra["x"] - rb["x"], ra["y"] - rb["y"])) / unit
            is_cooc = (a, b) in cooc_pairs or (b, a) in cooc_pairs
            rows.append({"class": cname, "tid_a": a, "tid_b": b,
                          "distance_objwidths": round(d, 4), "cooccurring": is_cooc})

    with open(os.path.join(OUT_DIR, "pair_distances.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "tid_a", "tid_b", "distance_objwidths", "cooccurring"])
        writer.writeheader()
        writer.writerows(rows)

    cooc_d = [r["distance_objwidths"] for r in rows if r["cooccurring"]]
    all_d = [r["distance_objwidths"] for r in rows]

    def stats(vals):
        if not vals:
            return None
        return {"min": min(vals), "median": float(np.median(vals)), "max": max(vals), "n": len(vals)}

    cooc_stats = stats(cooc_d)
    all_stats = stats(all_d)

    print("\n=== KEY STATISTIC ===")
    print(f"  Co-occurring same-class pairs (n={cooc_stats['n'] if cooc_stats else 0}): "
          f"{cooc_stats if cooc_stats else 'NONE'}")
    print(f"  All same-class pairs        (n={all_stats['n'] if all_stats else 0}): "
          f"{all_stats if all_stats else 'NONE'}")
    print(f"\nElapsed: {time.time() - t0:.1f}s")

    # stash raw stats for FEASIBILITY.md authoring
    with open(os.path.join(OUT_DIR, "summary_stats.json"), "w") as f:
        json.dump({
            "homography": {"lost": lost, "total": total, "lost_pct": lost_pct},
            "n_tracks": len(best_by_track),
            "n_cooccurring_pairs": len(cooc_pairs),
            "cooccurring_stats": cooc_stats,
            "all_pairs_stats": all_stats,
            "median_diag_by_class": {config.CLASS_NAMES.get(k, str(k)): v for k, v in median_diag.items()},
        }, f, indent=2)


if __name__ == "__main__":
    main()
