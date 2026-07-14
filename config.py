"""
Configuration for the Unique Object Counter Pipeline.
Adjust these parameters based on your YOLO model and use case.
"""

import os
_HERE = os.path.dirname(os.path.abspath(__file__))

# ─── YOLO Model ────────────────────────────────────────────────
# Override at runtime: MODEL_PATH=/path/to/model.pt python main.py
YOLO_MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "/home/ronit/Downloads/inventory_train-3/inventory_counter_new_model_9_july.pt",
)
YOLO_CONFIDENCE = 0.35
YOLO_IOU = 0.45
YOLO_IMG_SIZE = 640

# ─── Class Mapping ──────────────────────────────────────────────
# NOTE: matches the new 10-class model being trained in Training-codes/
# (final_train.py / final_config.yaml). Rk_trained_model.pt above was
# trained on the OLD 20-class taxonomy — swap YOLO_MODEL_PATH once the
# new weights exist, or detections will carry these labels but the old
# model's class ids underneath.
CLASS_NAMES = {
    0: "Conference Tables",
    1: "Cubicles / Partitions",
    2: "Desks",
    3: "Filing Cabinets / Storage Units",
    4: "Laptops",
    5: "Monitors",
    6: "Mouse",
    7: "Office Chairs",
    8: "Pedestals",
    9: "Printers Scanners",
}

# ─── Frame Sampling ─────────────────────────────────────────────
FRAME_SKIP = 2

# ─── Tracker ────────────────────────────────────────────────────
TRACKER_TYPE = "botsort"
TRACKER_CONFIG_PATH = os.path.join(_HERE, "botsort_reid.yaml")
TRACK_HIGH_THRESH = 0.35
TRACK_LOW_THRESH = 0.1
TRACK_MATCH_THRESH = 0.8
TRACK_BUFFER = 150

# ─── Re-Identification (DINOv2 two-pass) ────────────────────────
ENABLE_CLIP_REID = True
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"  # unused — kept for backwards compat
REID_SIMILARITY_THRESHOLD = 0.72        # Global default — overridden per class below.

# Per-class thresholds tuned on client video ground truth.
# Higher threshold = only merge if objects are extremely similar (same object re-seen).
# Use for classes where many identical items exist in the same room.
CLASS_REID_THRESHOLDS = {
    # Tuned via eval_counts.py on client_video ground truth (2026-07-13, Exp D):
    # abs error 10 -> 6, under-count -7 -> -4. Under-counting is worse for the
    # client, so thresholds bias toward NOT merging when in doubt.
    "Office Chairs":         0.90,  # identical chairs: known-distinct pairs score up to 0.89 combined
    "Cubicles / Partitions": 0.89,  # 2 distinct partitions have combined=0.880; don't merge
    "Desks":                 0.85,  # blocks merges; desks are under-counted, never fuse in doubt
    "Monitors":              0.74,  # dark screens have low app_sim at different angles/distances
    "Printers Scanners":     0.78,  # 2 identical printers score ~0.74 cross-pair; 0.72 fused them
}

REID_BACKGROUND_WEIGHT = 0.30   # 0.50 was too heavy — bg context shifts with camera angle even for same location

MIN_COOCCURRENCE_FRAMES = 3     # frames a pair must co-occur before hard-blocking merge
COLOR_VETO_THRESHOLD    = 0.0   # disabled — padded crops include background, histograms unreliable
MIN_TRACK_FRAMES        = 2     # discard tracks detected in fewer processed frames than this
MIN_TRACK_CONFIDENCE    = 0.32  # discard tracks whose mean YOLO confidence is below this

# ─── Quality Filtering ──────────────────────────────────────────
MIN_CROP_SHARPNESS = 20         # Only removes near-blank images. Real furniture ~50+.

# ─── Partial Detection Guard ────────────────────────────────────
MIN_BBOX_INSET = 15             # Skip crops where bbox edge within N px of frame edge.

# ─── VLM Validation (AWS Bedrock) ───────────────────────────────
ENABLE_VLM_VALIDATION = True    # Remove false positives via AWS Bedrock
VLM_AWS_REGION  = "us-east-1"
VLM_MODEL_ID    = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
VLM_BATCH_SIZE  = 8
# Only validate these classes with strict class-matching prompt.
# All other classes use the conservative "is there any object?" prompt.
# Set to None to validate all classes with strict prompt.
VLM_STRICT_CLASSES = []
# Old strict-class list (Video Conferencing Equipment, Couch / Lounge Chair,
# Keyboard) no longer exist in the new 10-class taxonomy — reassess which of
# the new classes need strict VLM validation once the new model has runs to
# look at.
# Classes NOT in VLM_STRICT_CLASSES are skipped entirely — low-res crops cause false removals.

# ─── Pre-ReID crop saving ────────────────────────────────────────
SAVE_PRE_REID_CROPS = True  # Save raw per-track crops + sheet BEFORE deduplication

# ─── Debug ──────────────────────────────────────────────────────
REID_DEBUG = True
REID_CHECK_INTERVAL = 1  # collect crops every processed frame; =3 starved short tracks of crops (Laptops 7->6 exact with =1)
REID_MIN_CROP_SIZE = 20
REID_TOP_K_CROPS = 5

# ─── Folders ────────────────────────────────────────────────────
# Override at runtime via env vars (required in Docker/CI).
INPUT_FOLDER  =  r"/home/ronit/Ronit-Personal/Personal/Inventory_counter/unseen_videos_testing"
OUTPUT_FOLDER = r"/home/ronit/Ronit-Personal/Personal/Inventory_counter/unseen_videos_testing_output"

# ─── Output ─────────────────────────────────────────────────────
OUTPUT_VIDEO_FPS = 5
VIDEO_CODEC = "mp4v"

# ─── Track Stitching ────────────────────────────────────────────
ENABLE_TRACK_STITCH  = True
TRACK_STITCH_MAX_GAP = 90    # raw-frame gap (~3 s at 30 fps) between death and rebirth
TRACK_STITCH_MIN_IOU = 0.30  # IoU between last box of dead track and first box of new one
STITCH_SEED_ORPHANS  = True  # seed never-registered tracks into clustering via padded crops
# Restrict orphan-singleton seeding to classes proven to silently under-count
# (edge-of-frame drops via MIN_BBOX_INSET). Seeding ALL classes floods clustering
# with partial-view crops that can't merge back (Office Chairs 6->17, abs err 22).
# None = seed every class (do not use — regresses badly, see eval/ws_a_stitch_log.txt).
# Winning config (2026-07-13, eval/ws_a_stitch_scoped): abs error 6->4, exact
# classes 6/10->7/10 (Pedestals now exact), no exact-class regressions.
STITCH_SEED_CLASSES  = {"Desks", "Pedestals"}

# ─── Visualization ──────────────────────────────────────────────
BBOX_THICKNESS = 2
FONT_SCALE = 0.6
SHOW_TRACK_ID = True
SHOW_CLASS_NAME = True
SHOW_CONFIDENCE = True
