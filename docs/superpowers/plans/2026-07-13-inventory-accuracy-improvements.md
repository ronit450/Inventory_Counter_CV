# Inventory Counter Accuracy Improvements — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce end-to-end counting error on ground-truth video from abs-error 6 to ≤5 (track stitching), ship a self-contained client HTML report, and produce a go/no-go feasibility verdict on homography-based spatial deduplication.

**Architecture:** Three independent workstreams with disjoint file ownership, executed by parallel agents in the SAME working tree (no worktrees — untracked videos/eval outputs are needed). Workstream A owns `main.py`/`reid_module.py`/`config.py` + new `track_stitch.py`. Workstream B owns new `generate_client_report.py` only. Workstream C owns new `spatial_prototype/` folder only.

**Tech Stack:** Python 3.11 (system `python3` works — conda env rk_yolo), ultralytics YOLO, OpenCV, DINOv2 via transformers, scipy clustering.

---

## Shared Context (read this first, whichever workstream you are)

### The system
`main.py` runs YOLO+BoT-SORT tracking over videos in `config.INPUT_FOLDER`, collects per-track crops, then `reid_module.py` does offline dedup: DINOv2 appearance embeddings + HSV background fingerprints → combined similarity → complete-linkage clustering per class, with a hard co-occurrence veto (tracks seen in the same frame are provably distinct objects). Output: `<video>_counts.json` + crops + contact sheets.

### The eval loop (your source of truth)
- Ground truth: `eval/ground_truth.json` — true counts for `client_video/client_video.mp4`.
- Score any run: `python3 eval_counts.py <results_dir>` (repo root).
- Run the pipeline on the eval video into a fresh dir (≈60–90 s, GPU):
  ```bash
  python3 -c "
  import config
  config.INPUT_FOLDER  = 'client_video'
  config.OUTPUT_FOLDER = 'eval/<YOUR_RUN_NAME>'
  import main
  main.main()
  " > eval/<YOUR_RUN_NAME>_log.txt 2>&1
  ```
- **Current committed-config score (must reproduce before changing code):**
  ```
  exact classes: 6/10  abs error: 6  (over: +2, under: -4)
  Desks 4/6 (-2), Pedestals 1/3 (-2), Filing 3/2 (+1), Monitors 5/4 (+1)
  EXACT (must not regress): Office Chairs 6, Printers 2, Laptops 6, Mouse 2,
                            Cubicles 2, Conference Tables 0
  ```

### Key facts discovered by analysis (do not rediscover)
1. **The under-count mechanism:** YOLO detected 7 desk tracks (true 6) and 4 pedestal tracks (true 3), but tracks whose bboxes only ever appear near the frame edge are skipped by `MIN_BBOX_INSET=15` in `reid_module.py:update_track` and therefore never enter `collector.tracks` — and **only collector tracks are counted**. They vanish silently.
2. **Naive inclusion fails:** removing the inset guard floods clustering with partial-view tracks whose embeddings can't merge back to their identities → Office Chairs exploded 6→17, total error 22. Don't do that.
3. **Appearance can't separate identical furniture:** co-occurring (provably distinct) chairs score up to app_sim 0.917 and bg_sim 0.987. Only co-occurrence and spatial position are trustworthy signals for identical objects.
4. **Bias decision (client-confirmed):** under-counting is worse than over-counting. When in doubt, do not merge.

### Ground rules (all workstreams)
- A **Fact-Forcing Gate** hook may reject your first Bash/Edit/Write with a demand to "present facts". State the requested facts as plain text in your reply, then retry the exact same tool call. This is normal.
- Touch ONLY the files your workstream owns (ownership matrix below). Another agent is editing other files at the same time.
- **Do NOT `git commit`** — three agents share one working tree; the orchestrator commits after review. Do not `git add` either.
- GPU is shared. If you hit CUDA OOM, wait 60 s and retry once; if it persists, use `device='cpu'` for your own prototype runs (Workstream C only).
- Your final message is a report to the orchestrator: what you changed, eval/verification results (before vs after), files touched, and anything you could not finish. Raw data, no prose padding.

### File ownership matrix

| Workstream | Owns (create/modify) | Read-only |
|---|---|---|
| A — Track stitching | `track_stitch.py` (new), `test_track_stitch.py` (new), `main.py`, `reid_module.py`, `config.py` | `eval/*`, `eval_counts.py` |
| B — Client report | `generate_client_report.py` (new) | `eval/final_check/*` |
| C — Spatial prototype | `spatial_prototype/` (new folder, everything inside) | `client_video/`, `config.py`, `eval/final_check/*` |

---

## Workstream A: Track Stitching by Motion Continuity

**Problem:** ID switches + the edge-guard drop real objects. When a track dies at frame N and a same-class track is born ≤ gap frames later with an overlapping bbox, that is one physical object. Stitching such chains lets partial tracks inherit an anchor's identity (fixing duplicates) and lets orphan chains be seeded into the collector from their display crops (fixing silent drops).

### Task A1: Pure stitching function (TDD)

**Files:**
- Create: `track_stitch.py`
- Create: `test_track_stitch.py`

- [ ] **Step 1: Write the failing test**

```python
# test_track_stitch.py
"""Unit tests for track_stitch.stitch_tracks. Run: python3 -m pytest test_track_stitch.py -v"""
from track_stitch import stitch_tracks


def _t(cls, first, last, first_box, last_box):
    return {"class_id": cls, "first_seen": first, "last_seen": last,
            "first_box": first_box, "last_box": last_box}

BOX = (100, 100, 200, 200)          # reference box
NEAR = (110, 105, 210, 205)         # IoU ~0.7 with BOX
FAR = (500, 500, 600, 600)          # IoU 0 with BOX


def test_stitches_adjacent_overlapping_same_class():
    tracks = {1: _t(7, 0, 50, BOX, BOX), 2: _t(7, 60, 100, NEAR, NEAR)}
    m = stitch_tracks(tracks, max_gap=90, min_iou=0.3)
    assert m[1] == m[2] == 1


def test_blocks_different_class():
    tracks = {1: _t(7, 0, 50, BOX, BOX), 2: _t(5, 60, 100, NEAR, NEAR)}
    m = stitch_tracks(tracks, max_gap=90, min_iou=0.3)
    assert m[1] != m[2]


def test_blocks_temporal_overlap():
    # overlapping lifespans = co-occurred = provably distinct
    tracks = {1: _t(7, 0, 70, BOX, BOX), 2: _t(7, 60, 100, NEAR, NEAR)}
    m = stitch_tracks(tracks, max_gap=90, min_iou=0.3)
    assert m[1] != m[2]


def test_blocks_gap_too_large():
    tracks = {1: _t(7, 0, 50, BOX, BOX), 2: _t(7, 200, 300, NEAR, NEAR)}
    m = stitch_tracks(tracks, max_gap=90, min_iou=0.3)
    assert m[1] != m[2]


def test_blocks_low_iou():
    tracks = {1: _t(7, 0, 50, BOX, BOX), 2: _t(7, 60, 100, FAR, FAR)}
    # track 1's LAST box vs track 2's FIRST box is what matters
    m = stitch_tracks(tracks, max_gap=90, min_iou=0.3)
    assert m[1] != m[2]


def test_chain_of_three_shares_root():
    tracks = {1: _t(7, 0, 50, BOX, BOX),
              2: _t(7, 60, 100, NEAR, NEAR),
              3: _t(7, 110, 150, NEAR, NEAR)}
    m = stitch_tracks(tracks, max_gap=90, min_iou=0.3)
    assert m[1] == m[2] == m[3] == 1


def test_each_track_stitches_to_one_successor_only():
    # two candidates born after track 1; only the better (smaller gap) joins
    tracks = {1: _t(7, 0, 50, BOX, BOX),
              2: _t(7, 55, 100, NEAR, NEAR),
              3: _t(7, 58, 100, NEAR, NEAR)}
    m = stitch_tracks(tracks, max_gap=90, min_iou=0.3)
    assert m[2] == 1 and m[3] != 1  # 2 wins (gap 5 < 8); 3 co-occurs with 2 anyway
```

- [ ] **Step 2: Run tests, verify they fail** — `python3 -m pytest test_track_stitch.py -v` → ImportError.

- [ ] **Step 3: Implement**

```python
# track_stitch.py
"""
Stitch fragmented tracks by motion continuity.

A pair (a, b) is stitchable when: same class, a ends strictly before b starts,
the gap is small, and a's last bbox overlaps b's first bbox. Temporally
overlapping tracks are never stitched (co-occurrence = provably distinct).
Greedy matching by (gap, 1-IoU): each track gets at most one successor and
one predecessor. Returns {tid: root_tid} (root = smallest tid in chain).
"""


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = ((ax2 - ax1) * (ay2 - ay1)) + ((bx2 - bx1) * (by2 - by1)) - inter
    return inter / union if union > 0 else 0.0


def stitch_tracks(tracks: dict, max_gap: int, min_iou: float) -> dict:
    tids = sorted(tracks)
    candidates = []
    for a in tids:
        for b in tids:
            if a == b:
                continue
            ta, tb = tracks[a], tracks[b]
            if ta["class_id"] != tb["class_id"]:
                continue
            gap = tb["first_seen"] - ta["last_seen"]
            if gap <= 0 or gap > max_gap:
                continue
            iou = _iou(ta["last_box"], tb["first_box"])
            if iou < min_iou:
                continue
            candidates.append((gap, 1.0 - iou, a, b))
    candidates.sort()

    parent = {t: t for t in tids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    has_successor, has_predecessor = set(), set()
    for _gap, _inv_iou, a, b in candidates:
        if a in has_successor or b in has_predecessor:
            continue
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        lo, hi = min(ra, rb), max(ra, rb)
        parent[hi] = lo
        has_successor.add(a)
        has_predecessor.add(b)

    return {t: find(t) for t in tids}
```

- [ ] **Step 4: Run tests, verify all pass** — `python3 -m pytest test_track_stitch.py -v` → 7 passed.

### Task A2: Capture first/last boxes in main.py

**Files:** Modify `main.py` (detection loop, near the `track_first_seen` bookkeeping ~line 336–343).

- [ ] **Step 1:** Add two dicts next to the existing ones (`track_first_seen` block): `track_first_box = {}` and `track_last_box = {}`. Inside the per-detection loop where `track_first_seen`/`track_last_seen` are set, add:

```python
                if tid not in track_first_box:
                    track_first_box[tid] = tuple(map(float, box))
                track_last_box[tid] = tuple(map(float, box))
```

- [ ] **Step 2:** Also pop these dicts in the ghost-track filter loop (where other track dicts are popped).

### Task A3: Integrate stitching into the ReID pipeline

**Files:** Modify `main.py` (after ghost filter, before `reid.finalize()`), `reid_module.py` (new methods), `config.py` (new params).

- [ ] **Step 1: config.py params** (append near the other ReID params):

```python
# ─── Track Stitching ────────────────────────────────────────────
ENABLE_TRACK_STITCH  = True
TRACK_STITCH_MAX_GAP = 90    # raw-frame gap (~3 s at 30 fps) between death and rebirth
TRACK_STITCH_MIN_IOU = 0.30  # IoU between last box of dead track and first box of new one
STITCH_SEED_ORPHANS  = True  # seed never-registered tracks into clustering via padded crops
```

- [ ] **Step 2: main.py hook** (after the ghost-filter block, before `SAVE_PRE_REID_CROPS` section):

```python
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
```

- [ ] **Step 3: reid_module.py — `TrackCollector.absorb` + `CLIPReIdentifier.apply_stitch_map`.**

`absorb(child, parent)` on TrackCollector: move child's `scored_crops` into parent (extend list, keep parent's class info), keep the better `best_score`/crop histogram, keep the bg fingerprint with larger `bbox_area`, and **remap co-occurrence**: every `frozenset({child, X})` count adds onto `frozenset({parent, X})` (skip X == parent). Then delete child from all dicts.

`apply_stitch_map(stitch_map, padded_crops)` on CLIPReIdentifier:
1. Group tracks by root. For each chain (len > 1):
   - If the root is NOT in `collector.tracks` but some member is, re-root the chain on the smallest member that IS in collector.
   - If NO member is in collector (orphan chain — the silently-dropped desks/pedestals): seed the root via `collector.add_detection(root, class_id, class_name, best_crop)` where `best_crop` is the highest-`score` entry among chain members in `padded_crops` (entries have keys `crop`, `score`, `class_id`). Skip the chain if no member has a padded crop.
   - Absorb every other collector-resident member into the root.
2. If `STITCH_SEED_ORPHANS`: also seed single never-registered tracks (chains of length 1 not in collector) from their padded crop the same way — they enter clustering with their best display crop, so clustering can still merge them away (unlike the failed blind-count experiment).
3. Store `self._stitch_map = {tid: final_root}` so `finalize()` can compose: after `deduplicate()` returns `canon`, set `self._canonical_map = {tid: canon.get(root, root) for tid, root in self._stitch_map.items()}` merged over the plain canon map for unstitched tracks.

- [ ] **Step 4:** Run `python3 -m pytest test_track_stitch.py -v` again (still 7 passed) and `python3 -c "import main"` (no syntax errors).

### Task A4: Eval-driven validation (the actual acceptance test)

- [ ] **Step 1: Reproduce baseline.** Run the pipeline into `eval/ws_a_baseline` with `config.ENABLE_TRACK_STITCH = False` forced in the runner snippet. Score must be abs error 6 as documented above. If not, STOP and report.
- [ ] **Step 2: Stitching on.** Run into `eval/ws_a_stitch`. Score it.
- [ ] **Step 3: Iterate** on `TRACK_STITCH_MAX_GAP` (try 60/90/150), `TRACK_STITCH_MIN_IOU` (0.2/0.3/0.4), and `STITCH_SEED_ORPHANS` on/off. Keep the debug log (`[Stitch]`, `[REID_DEBUG]`, merge lines) as your microscope. Each run gets its own eval dir.
- [ ] **Step 4: Acceptance:** abs error ≤ 5 AND under-count ≤ 4 AND every currently-exact class stays exact (Office Chairs 6, Printers 2, Laptops 6, Mouse 2, Cubicles 2, Conference Tables 0). If unreachable, set `ENABLE_TRACK_STITCH = False` (and/or `STITCH_SEED_ORPHANS = False`) in config.py, keep the code, and report honestly with the best table you achieved.
- [ ] **Step 5:** Leave the winning parameter values in `config.py` with a one-line comment stating the score they achieved.

---

## Workstream B: Self-Contained Client HTML Report

**Problem:** Clients get loose JPEGs + JSON. Deliverable: one double-clickable HTML file per video — counts table, per-class object gallery with crop + context images embedded as base64, confidence badges when present.

### Task B1: Report generator CLI

**Files:** Create `generate_client_report.py` (this workstream touches NOTHING else).

- [ ] **Step 1: Implement.** CLI: `python3 generate_client_report.py <results_dir> [--video NAME]` (default: infer from the single `*_counts.json` in the dir). Reads `<video>_counts.json` — relevant fields:

```json
{
  "summary": {"total_unique_objects": 29, "total_raw_tracks": 65,
               "duplicates_removed": 36, "frames_processed": 1167},
  "counts_by_class": {"Desks": 4, "Office Chairs": 6},
  "objects": [{"class_name": "Desks", "instance_number": 1,
                "crop_path": ".../Desks_1_crop.jpg",
                "context_frame_path": ".../Desks_1_context.jpg",
                "confidence": "confirmed"}]
}
```

Requirements:
- `confidence` may be ABSENT (Workstream A adds it later) — tolerate missing key, omit badge.
- `crop_path`/`context_frame_path` may be absolute paths from another machine — resolve by basename inside `<results_dir>/<video>_crops/` if the literal path doesn't exist.
- Embed images as base64 `data:` URIs. Resize crops to max-width 320 px and context frames to max-width 900 px with cv2 before encoding (JPEG quality 80) to keep the file well under 15 MB.
- Layout (inline CSS only, no external requests, no JS frameworks): header (video name, generation date, total objects, per-class totals as pill chips) → counts table → one `<section>` per class → card grid (crop thumbnail, "Class #N" caption, `<details><summary>Show in room</summary><img context></details>`).
- Clean, professional, client-facing: neutral background, one accent color, readable at a glance. A missing crop file renders a gray placeholder div, never a broken image icon.
- Write output to `<results_dir>/<video>_report.html` and print the path.

- [ ] **Step 2: Verify on real data.** Run against `eval/final_check` (contains a full 29-object run). Checks: exit 0; output HTML exists; `grep -c "data:image" <html>` ≥ 29; file size < 15 MB; number of `<section` occurrences equals number of classes with count > 0 (9).

- [ ] **Step 3: Edge test.** Run against a temp dir containing a minimal counts JSON with one object whose crop file doesn't exist → report still generates with a placeholder.

---

## Workstream C: Spatial Dedup Feasibility Prototype (homography)

**Problem:** Appearance cannot distinguish identical chairs (fact 3 above). Hypothesis: cumulative frame-to-frame homography places every track at rough "panorama" coordinates; identical objects far apart in panorama space are distinct. Deliverable is a **feasibility verdict**, NOT pipeline integration.

### Task C1: Prototype script

**Files:** Create `spatial_prototype/prototype_homography.py`, outputs in `spatial_prototype/out/`.

- [ ] **Step 1: Implement.** Single script that:
  1. Opens `client_video/client_video.mp4`, processes every 2nd frame (match `FRAME_SKIP=2`).
  2. Camera motion: `cv2.goodFeaturesToTrack` + `cv2.calcOpticalFlowPyrLK` between consecutive processed frames → `cv2.findHomography(..., cv2.RANSAC)` → maintain cumulative homography `H_cum` mapping current frame → frame-0 coordinates. Guard: if fewer than 30 tracked points or homography fails, carry previous `H_cum` and count a "lost" frame.
  3. Object tracks: run `model.track` exactly like main.py does (same conf/iou/imgsz/tracker settings from `config`; model path from `config.YOLO_MODEL_PATH`) in the same loop.
  4. For every detection, project the bbox bottom-center point through `H_cum` into panorama coords. Per track, store the projected point from the frame where the track had its LARGEST bbox (best localization), plus class id.
  5. Save `spatial_prototype/out/track_positions.json`: `{tid: {"class_id": c, "x": px, "y": py, "frame": f}}` plus co-occurrence pairs `[[tid_a, tid_b, n_frames], ...]` for same-class pairs seen in the same frame ≥ 3 times.
- [ ] **Step 2: Diagnostic.** For each same-class track pair compute panorama distance, normalized by the median bbox diagonal of that class (so distance ≈ "object widths"). Emit `spatial_prototype/out/pair_distances.csv` with columns `class,tid_a,tid_b,distance_objwidths,cooccurring`. Print the key statistic: min/median/max distance for CO-OCCURRING pairs (known distinct) vs all pairs. **Hypothesis confirmed if co-occurring same-class pairs sit consistently ≥ ~1.5 object-widths apart in panorama space.**
- [ ] **Step 3: Verdict.** Write `spatial_prototype/FEASIBILITY.md`: homography stability (lost frames / total), the distance statistics table, a GO/NO-GO verdict, and if GO — a 10-line integration design (where a spatial veto/bonus slots into `PostHocDeduplicator.deduplicate`, e.g. `dist[i,j] = 2.0 if panorama_distance > K`).
- GPU note: you share the GPU with Workstream A. On CUDA OOM, wait 60 s and retry once, then fall back to CPU inference — slower but acceptable for a one-video prototype.

---

## Self-review notes
- Spec coverage: stitching (A), client outputs (B), spatial dedup (C) — roadmap items 1, 4, 2. Detection-training work (roadmap 3) is deliberately out of scope: it needs new labeled data, not code.
- Interfaces: `stitch_tracks` signature in A1 test == A1 impl == A3 hook. `padded_crops` entry keys (`crop`, `score`, `class_id`) verified against `main.py:370-378`. `confidence` field contract between A and B: optional, string, values `confirmed`/`review`.
- The A-acceptance gate (exact classes must stay exact) encodes the no-regression rule; the fallback (flag off, report honestly) keeps main.py releasable regardless of outcome.
