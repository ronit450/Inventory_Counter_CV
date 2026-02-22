"""
Configuration for the Unique Object Counter Pipeline.
Adjust these parameters based on your YOLO model and use case.
"""

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

# ─── ByteTrack Tracker ──────────────────────────────────────────
TRACKER_TYPE = "bytetrack"  # Options: "bytetrack", "botsort"
TRACK_HIGH_THRESH = 0.35    # High detection threshold for tracking
TRACK_LOW_THRESH = 0.1      # Low detection threshold for second association
TRACK_MATCH_THRESH = 0.8    # Matching threshold for tracking
TRACK_BUFFER = 60           # Frames to keep lost tracks alive
                             # Higher = better re-id after camera rotation

# ─── Re-Identification (CLIP Embeddings) ────────────────────────
ENABLE_CLIP_REID = True                     # Enable CLIP-based deduplication
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"  # CLIP model from HuggingFace
REID_SIMILARITY_THRESHOLD = 0.82            # Cosine similarity threshold for same object
                                             # Higher = stricter matching (fewer merges)
                                             # Lower = more aggressive merging (risk false merges)
REID_CHECK_INTERVAL = 5                     # Check re-id every N frames
REID_MIN_CROP_SIZE = 20                     # Minimum crop dimension (px) to compute embedding

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
