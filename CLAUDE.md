# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Pipeline

```bash
python main.py
```

Before running, update `config.py` with the correct paths for your environment:
- `YOLO_MODEL_PATH` — path to the `.pt` weights file (the repo includes `yolo11lg_custom_12142025.pt`)
- `INPUT_FOLDER` — directory containing input video files
- `OUTPUT_FOLDER` — where JSON results and annotated videos are written (default: `results_analysis/`)

## Architecture

The pipeline has three layers:

**Detection + Tracking (`main.py`)** — Loads a custom-trained YOLO model and runs it with ByteTrack on every Nth frame (`FRAME_SKIP`). Produces bounding boxes with persistent track IDs across frames. The model is loaded once and shared across all videos in the input folder.

**Re-Identification (`reid_module.py` → `CLIPReIdentifier`)** — When the camera pans away and returns, ByteTrack assigns new IDs to previously seen objects. CLIP embeddings (from HuggingFace `openai/clip-vit-base-patch32`) are computed for each crop and compared against stored embeddings of the same class. If cosine similarity exceeds `REID_SIMILARITY_THRESHOLD`, the new track is merged into the canonical track via `merge_map`. A co-occurrence guard (`_cooccurring` set) prevents merging tracks that appeared in the same frame — critical for counting identical objects (e.g. 5 identical chairs).

**Configuration (`config.py`)** — All tunable parameters live here: model path, class ID→name mapping, thresholds, frame skip, tracker type, output options. The `CLASS_NAMES` dict must match the trained model's class order.

**Outputs per video:**
- `<video_name>_counts.json` — summary with unique counts per class, raw track count, duplicates removed
- `<video_name>_detected.mp4` — annotated video with bounding boxes and live counter overlay

## Key Tuning Parameters

| Parameter | Effect |
|-----------|--------|
| `FRAME_SKIP` | Higher = faster but may miss objects |
| `REID_SIMILARITY_THRESHOLD` | Higher = stricter (fewer false merges); lower = more aggressive (risk over-merging identical items) |
| `TRACK_BUFFER` | Frames to keep a lost track alive; increase for slow camera pans |
| `REID_CHECK_INTERVAL` | Run ReID every N processed frames |
| `ENABLE_CLIP_REID` | Disable for speed when camera doesn't revisit areas |
