# Inventory Counter — Automated Office Asset Counting via Computer Vision

A computer vision pipeline that walks through a video of an office space and counts every unique piece of furniture and equipment it sees — even when the camera revisits the same area multiple times.

---

## The Problem

Manually counting office inventory (chairs, desks, monitors, laptops, etc.) is slow, error-prone, and expensive. A person with a phone camera can walk through an office and record it, but counting from a video is harder than it sounds:

- The camera **pans and rotates**, so the same desk appears multiple times
- Objects enter and leave the frame repeatedly, causing a naive tracker to count each re-entry as a new object
- Identical objects (e.g. 10 identical office chairs) must each be counted separately
- Objects partially occlude each other or are briefly obscured

This pipeline solves all of these.

---

## How It Works

The pipeline is a three-stage system that runs on every video in an input folder:

### Stage 1 — Detection (YOLO)

A custom-trained **YOLOv11** model (`yolo11lg_custom_12142025.pt`) detects 20 classes of office objects in each frame:

| Category | Items |
|----------|-------|
| Furniture | Desk, Office Chair, Conference Table, Filing Cabinet, Pedestal, Reception Desk, Bookshelf/Cabinet, Breakroom Table, Breakroom Chair, Couch/Lounge Chair, Cubicle/Partition |
| Electronics | Laptop, Desktop Computer, Monitor, Keyboard, Mouse, Printer/Scanner, Projector, Telephone/VoIP Phone, Video Conferencing Equipment |

Every detection gets a bounding box, class label, and confidence score. Frames are sampled every N frames (configurable via `FRAME_SKIP`) for speed.

### Stage 2 — Tracking (ByteTrack)

**ByteTrack** assigns persistent track IDs to detections across frames. As long as an object stays in frame, it keeps the same ID. When it leaves and returns, ByteTrack tries to re-link it — but after a long pan, it assigns a new ID.

This is the core challenge: one physical chair might accumulate 5–10 different track IDs over the course of a video.

### Stage 3 — Re-Identification (CLIP)

When a new track appears, the pipeline checks whether it has been seen before using **CLIP** (OpenAI's vision-language model). It:

1. Crops the detected object from the frame
2. Computes a CLIP embedding (a 512-dimensional visual fingerprint)
3. Compares it against stored embeddings of all existing tracks of the **same class** using cosine similarity
4. If similarity exceeds the threshold (`REID_SIMILARITY_THRESHOLD = 0.82`), the new track is merged into the existing one

A critical safety mechanism prevents false merges: if two tracks ever **appeared in the same frame simultaneously**, they are guaranteed to be different physical objects and will never be merged — even if they look identical. This handles rooms full of matching chairs correctly.

---

## Pipeline Diagram

```
Input Video
     │
     ▼
┌─────────────┐
│  Frame Skip │  Process every Nth frame
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  YOLO Det.  │  Detect objects → bounding boxes + class IDs
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  ByteTrack  │  Assign persistent track IDs across frames
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  CLIP ReID  │  Compare new tracks to known tracks via embeddings
│             │  Merge duplicates → single canonical ID per object
└──────┬──────┘
       │
       ▼
┌─────────────────────────┐
│  Output                 │
│  ├── _counts.json       │  Unique count per class + full detail
│  └── _detected.mp4      │  Annotated video with live counter
└─────────────────────────┘
```

---

## Results Example

On a real client office walkthrough video (2335 frames, ~52 seconds of processing):

| Class | Count |
|-------|-------|
| Office Chair | 6 |
| Desk | 4 |
| Bookshelf / Cabinet | 3 |
| Monitor | 2 |
| Laptop | 2 |
| Mouse | 2 |
| Pedestal | 2 |
| Printer / Scanner | 2 |
| Telephone / VoIP Phone | 1 |
| **Total Unique** | **24** |

Raw tracker produced **83 tracks** — CLIP ReID collapsed them to 24 unique objects (**59 duplicates removed**).

---

## Setup

### Requirements

```bash
pip install ultralytics opencv-python torch torchvision transformers scipy tqdm pillow numpy
```

GPU is strongly recommended for CLIP embedding speed. The pipeline falls back to CPU automatically.

### Configuration

Edit `config.py` before running:

```python
# Point to your YOLO weights
YOLO_MODEL_PATH = "/path/to/yolo11lg_custom_12142025.pt"

# Input/output folders
INPUT_FOLDER  = "/path/to/videos"
OUTPUT_FOLDER = "results_analysis"
```

All other parameters have sensible defaults. Key ones to tune:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FRAME_SKIP` | `2` | Process every Nth frame. Higher = faster, may miss objects |
| `YOLO_CONFIDENCE` | `0.35` | Detection confidence threshold |
| `REID_SIMILARITY_THRESHOLD` | `0.82` | CLIP cosine similarity to merge tracks. Higher = stricter |
| `ENABLE_CLIP_REID` | `True` | Disable if camera never revisits the same area |
| `TRACK_BUFFER` | `60` | Frames to keep a lost track alive before dropping it |
| `REID_CHECK_INTERVAL` | `5` | Run ReID every N processed frames |

### Running

```bash
python main.py
```

The script processes all `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`, `.flv` files found in `INPUT_FOLDER` and writes results to `OUTPUT_FOLDER`.

---

## Output Format

### JSON (`<video_name>_counts.json`)

```json
{
  "summary": {
    "total_unique_objects": 24,
    "total_raw_tracks": 83,
    "duplicates_removed": 59,
    "processing_time_seconds": 52.8,
    "frames_processed": 1167,
    "total_frames": 2335,
    "reid_enabled": true
  },
  "counts_by_class": {
    "Desk": 4,
    "Office Chair": 6,
    ...
  },
  "unique_objects": [
    {
      "unique_id": 7,
      "class_name": "Office Chair",
      "num_embeddings": 10,
      "merged_track_ids": [31, 63, 100, 167]
    },
    ...
  ]
}
```

`merged_track_ids` shows exactly which tracker IDs were identified as the same physical object — useful for debugging and verifying ReID decisions.

### Annotated Video (`<video_name>_detected.mp4`)

Each processed frame is written to the output video with:
- Bounding boxes colored by class
- Label showing `ID | Class Name | Confidence`
- Live counter overlay (top-right) showing current unique counts per class

Output plays at `OUTPUT_VIDEO_FPS` (default 5 FPS) regardless of input FPS, making it easy to inspect detections frame by frame.

---

## Project Structure

```
Inventory_counter/
├── main.py                  # Pipeline entry point + video processing loop
├── reid_module.py           # CLIPReIdentifier — embedding management + dedup logic
├── config.py                # All configuration parameters
├── yolo11lg_custom_12142025.pt  # Custom-trained YOLO weights (not tracked in git)
├── videos_to_test/          # Input videos
├── results_analysis/        # Output JSONs + annotated videos
└── results/                 # Sample results
```

---

## Tuning Guide

**Too many duplicates (under-counting):** Lower `REID_SIMILARITY_THRESHOLD` (e.g. 0.75). This makes merging more aggressive.

**Objects being incorrectly merged (over-counting is the goal, but false merges appear):** Raise `REID_SIMILARITY_THRESHOLD` (e.g. 0.88+). Also check that `TRACK_BUFFER` is high enough — if ByteTrack drops a track too early, ReID has to do more work.

**Missing objects:** Lower `YOLO_CONFIDENCE`, or decrease `FRAME_SKIP` to process more frames.

**Slow performance:** Increase `FRAME_SKIP`, increase `REID_CHECK_INTERVAL`, or set `ENABLE_CLIP_REID = False` if ReID is not needed.

**Camera pans slowly / long revisits:** Increase `TRACK_BUFFER` so ByteTrack keeps tracks alive longer and CLIP ReID has fewer new tracks to process.
