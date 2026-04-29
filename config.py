"""
Configuration for the Unique Object Counter Pipeline.
Adjust these parameters based on your YOLO model and use case.
"""

import os
_HERE = os.path.dirname(os.path.abspath(__file__))

# ─── YOLO Model ────────────────────────────────────────────────
YOLO_MODEL_PATH = r"C:\Users\Administrator\Downloads\Inventory_counter\Inventory_counter\yolo11lg_custom_12142025.pt" 
          # Path to your trained YOLO model weights
YOLO_CONFIDENCE = 0.35       # Minimum confidence threshold for detections
YOLO_IOU = 0.45              # NMS IoU threshold
YOLO_IMG_SIZE = 640          # Inference image size

# ─── Class Mapping ──────────────────────────────────────────────
# Map your YOLO class indices to human-readable names.
# UPDATE THIS to match your trained model's class order.
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
FRAME_SKIP = 2             # Process every Nth frame (1 = every frame, 2 = every other, etc.)
                            # Higher = faster but might miss objects

# ─── Tracker ────────────────────────────────────────────────────
TRACKER_TYPE = "botsort"    # botsort has Camera Motion Compensation (CMC) built in —
                             # compensates for camera pan/tilt so tracks survive rotation.
                             # bytetrack = no CMC, IDs break on pan; use only if botsort unavailable.
TRACKER_CONFIG_PATH = os.path.join(_HERE, "botsort_reid.yaml")
                             # Custom tracker config with with_reid=True and track_buffer=150.
                             # Edit botsort_reid.yaml to tune appearance/proximity thresholds.
TRACK_HIGH_THRESH = 0.35    # High detection threshold for tracking
TRACK_LOW_THRESH = 0.1      # Low detection threshold for second association
TRACK_MATCH_THRESH = 0.8    # Matching threshold for tracking
TRACK_BUFFER = 150          # Frames to keep lost tracks alive (FRAME_SKIP=2, 30fps → ~10s)
                             # 150 handles slow office walks without losing tracks

# ─── Re-Identification (DINOv2 two-pass) ────────────────────────
ENABLE_CLIP_REID = True                     # Enable two-pass ReID (DINOv2 + DBSCAN)
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"  # Unused — kept for backwards compat
REID_SIMILARITY_THRESHOLD = 0.60            # Combined-similarity threshold for merging.
                                             # Raise toward 0.68 if unrelated objects merge.
                                             # Lower toward 0.55 if same object still not merging.
REID_BACKGROUND_WEIGHT = 0.50               # Location signal weight in merge decision.
                                             # combined_sim = 0.50*appearance + 0.50*location
                                             # Higher = location dominates (protects against
                                             # identical furniture false-merges).
                                             # Set 0.0 to disable location signal entirely.

# ─── Quality Filtering ──────────────────────────────────────────
MIN_CROP_SHARPNESS = 20                     # Laplacian variance floor.
                                             # Only removes near-blank images (wall, floor, black frame).
                                             # Real furniture scores 50+; featureless wall scores 0-15.
                                             # Keep LOW — padded crops score less than tight crops.
                                             # 0 = disabled.

# ─── Partial Detection Guard ────────────────────────────────────
MIN_BBOX_INSET = 15                         # Skip crops where any bbox edge is within N px of
                                             # frame boundary. Prevents partially-visible objects
                                             # (chair entering frame) from becoming separate IDs.
                                             # 0 = disabled. 15 is a safe default.

# ─── VLM Validation (AWS Bedrock) ───────────────────────────────
ENABLE_VLM_VALIDATION = True             # Batch-validate unique objects via Claude Haiku on Bedrock.
                                             # Requires AWS credentials (boto3) in environment.
                                             # Sends all same-class crops in ONE call — VLM compares
                                             # them side-by-side for false positives AND missed merges.
VLM_AWS_REGION  = "us-east-1"
VLM_MODEL_ID    = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
VLM_BATCH_SIZE  = 8                         # Max crops per Bedrock call (stays within token limits)

# ─── Debug ──────────────────────────────────────────────────────
REID_DEBUG = False                          # Print pairwise similarity scores during deduplication.
                                             # Use to diagnose why specific objects don't merge.
REID_CHECK_INTERVAL = 3                     # Collect a crop every N processed frames
                                             # Lower = more crops stored; higher = less memory
REID_MIN_CROP_SIZE = 20                     # Minimum crop dimension (px) to store
REID_TOP_K_CROPS = 5                        # Keep top-K quality crops per track
                                             # Embeddings averaged over these for robustness

# ─── Folders ───────────────────────────────────────────────────
INPUT_FOLDER =  r"C:\Users\Administrator\Downloads\Inventory_counter\Inventory_counter\videos_to_test"         # Folder containing inpu  video files
OUTPUT_FOLDER = "results_analysis"                   # Folder for output JSON + annotated videos

# ─── Output ─────────────────────────────────────────────────────
OUTPUT_VIDEO_FPS = 5                        # FPS for output video (lower = slower playback,
                                             # easier to inspect detections; None = input FPS / frame_skip)
VIDEO_CODEC = "mp4v"                        # Video codec (mp4v for .mp4)

# ─── Visualization ──────────────────────────────────────────────
BBOX_THICKNESS = 2
FONT_SCALE = 0.6
SHOW_TRACK_ID = True        # Show track ID on bounding boxes
SHOW_CLASS_NAME = True      # Show class name on bounding boxes
SHOW_CONFIDENCE = True      # Show confidence score
