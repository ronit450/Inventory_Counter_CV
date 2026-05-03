"""
Configuration for the Unique Object Counter Pipeline.
Adjust these parameters based on your YOLO model and use case.
"""

import os
_HERE = os.path.dirname(os.path.abspath(__file__))

# ─── YOLO Model ────────────────────────────────────────────────
YOLO_MODEL_PATH = "/home/ronit/Downloads/inventory_new_model.pt"
YOLO_CONFIDENCE = 0.35
YOLO_IOU = 0.45
YOLO_IMG_SIZE = 640

# ─── Class Mapping ──────────────────────────────────────────────
CLASS_NAMES = {
    0: "Desk",
    1: "Office Chair",
    2: "Conference Table",
    3: "Filing Cabinet / Storage Unit",
    4: "Pedestal",
    5: "Reception Desk",
    6: "Bookshelf / Cabinet",
    7: "Breakroom Table",
    8: "Breakroom Chair",
    9: "Couch / Lounge Chair",
    10: "Cubicle / Partition",
    11: "Laptop",
    12: "Desktop Computer",
    13: "Monitor",
    14: "Keyboard",
    15: "Mouse",
    16: "Printer / Scanner",
    17: "Projector",
    18: "Telephone / VoIP Phone",
    19: "Video Conferencing Equipment",
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
    "Office Chair":        0.93,  # identical chairs: only {15,53}=0.932 should merge
    "Cubicle / Partition": 0.89,  # 2 distinct partitions have combined=0.880; don't merge
    "Desk":                0.76,  # 3-track over-merge at 0.72; correct pairs survive at 0.76
}

REID_BACKGROUND_WEIGHT = 0.50   # combined_sim = 0.50*appearance + 0.50*location

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
VLM_STRICT_CLASSES = [
    "Video Conferencing Equipment",  # almost always a misclassified monitor/projector
    "Couch / Lounge Chair",          # rarely in standard offices, often a misclassified chair
    "Keyboard",                      # low-confidence detections often desktops/table surfaces
]
# Classes NOT in VLM_STRICT_CLASSES are skipped entirely — low-res crops cause false removals.

# ─── Pre-ReID crop saving ────────────────────────────────────────
SAVE_PRE_REID_CROPS = True  # Save raw per-track crops + sheet BEFORE deduplication

# ─── Debug ──────────────────────────────────────────────────────
REID_DEBUG = False
REID_CHECK_INTERVAL = 3
REID_MIN_CROP_SIZE = 20
REID_TOP_K_CROPS = 5

# ─── Folders ────────────────────────────────────────────────────
INPUT_FOLDER  = "/home/ronit/Ronit-Personal/Personal/Inventory_counter/client_video"
OUTPUT_FOLDER = "/home/ronit/Ronit-Personal/Personal/Inventory_counter/results_tuning"

# ─── Output ─────────────────────────────────────────────────────
OUTPUT_VIDEO_FPS = 5
VIDEO_CODEC = "mp4v"

# ─── Visualization ──────────────────────────────────────────────
BBOX_THICKNESS = 2
FONT_SCALE = 0.6
SHOW_TRACK_ID = True
SHOW_CLASS_NAME = True
SHOW_CONFIDENCE = True
