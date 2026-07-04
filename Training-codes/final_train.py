#!/usr/bin/env python3
"""
final_train.py — Office Inventory: Convert → Augment → Split → Train

Input: flat directory with images + matching .txt annotation files.
Annotation format (.txt, one file per image):
    [Image #1]
    { "category": "Desks", "x1": 0.10, "y1": 0.20, "x2": 0.30, "y2": 0.45 }
    { "category": "Office Chairs", "x1": 0.50, "y1": 0.30, "x2": 0.70, "y2": 0.80 }

Coordinates x1,y1,x2,y2 are normalized [0,1] (left, top, right, bottom).

Workflow:
    1. Convert custom .txt  -> YOLO .txt (class_id cx cy w h) in-place
    2. Augment: horizontal flip, vertical flip, rotate 180 (via Augment.py)
    3. Copy into yolo_dataset/{images,labels}/{train,val}
    4. Write data.yaml
    5. Train YOLO

Usage:
    python final_train.py                          # uses final_config.yaml
    python final_train.py --config /path/cfg.yaml
    python final_train.py --data-dir /path/data
    python final_train.py --convert-only           # just convert, no aug/train
    python final_train.py --skip-train             # prep dataset, skip training
    python final_train.py --no-augment             # skip manual augmentation step
"""
import json
import os
import re
import shutil
import random
import sys
import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

# Augment.py lives in the same directory as this script
sys.path.insert(0, str(Path(__file__).parent))
from Augment import Augment


# ── Class definitions ─────────────────────────────────────────────────────────
# Position in this list = class_id written into YOLO .txt labels.
# Must match the names list in final_config.yaml exactly.
CLASS_NAMES = [
    "Conference Tables",                # 0
    "Cubicles / Partitions",            # 1
    "Desks",                            # 2
    "Filing Cabinets / Storage Units",  # 3
    "Laptops",                          # 4
    "Monitors",                         # 5
    "Mouse",                            # 6
    "Office Chairs",                    # 7
    "Other",                            # 8
    "Pedestals",                        # 9
    "Printers Scanners",                # 10
    "Telephones VoIP Phones",           # 11
]

# Exact-match lookup
_CLASS_INDEX: dict = {name: i for i, name in enumerate(CLASS_NAMES)}

# Normalized (lowercase + collapsed whitespace) fallback for typos/case mismatches
_NORM_INDEX: dict = {
    re.sub(r"\s+", " ", n.strip().lower()): i for i, n in enumerate(CLASS_NAMES)
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


# ── Annotation helpers ────────────────────────────────────────────────────────

def _resolve_class(category):
    """Return class_id for a category string, or None if unrecognised."""
    if category in _CLASS_INDEX:
        return _CLASS_INDEX[category]
    return _NORM_INDEX.get(re.sub(r"\s+", " ", category.strip().lower()))


def _is_custom_format(txt_path):
    """Return True if the .txt file is in the [Image #N] + JSON-lines format."""
    try:
        with open(txt_path) as f:
            first = f.readline().strip()
        # Custom format starts with "[Image" header or a "{" JSON object.
        # YOLO format starts with a digit (class_id).
        return first.startswith("[") or first.startswith("{")
    except OSError:
        return False


def _parse_custom_txt(txt_path):
    """Parse [Image #N] + JSON-lines annotations into a list of dicts."""
    annotations = []
    with open(txt_path) as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("["):
                continue
            try:
                annotations.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  Warning {txt_path.name}:{lineno}: skipping bad line: {line!r}")
    return annotations


# ── Step 1: Convert ───────────────────────────────────────────────────────────

def convert_annotations(data_dir):
    """
    Convert every custom-format .txt in data_dir to YOLO format in-place.

    Returns:
        converted  - number of files rewritten
        skipped    - files already in YOLO format
        unknown    - category strings not in CLASS_NAMES
    """
    converted = 0
    skipped = 0
    unknown = set()

    for img_path in sorted(data_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        txt_path = img_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        if not _is_custom_format(txt_path):
            skipped += 1
            continue

        annotations = _parse_custom_txt(txt_path)
        yolo_lines = []

        for ann in annotations:
            cat = ann.get("category", "").strip()
            class_id = _resolve_class(cat)
            if class_id is None:
                unknown.add(cat)
                continue

            x1 = float(ann["x1"])
            y1 = float(ann["y1"])
            x2 = float(ann["x2"])
            y2 = float(ann["y2"])

            # Normalise order so x1 < x2, y1 < y2
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

            # Clamp; enforce non-zero box size
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w = max(0.001, min(1.0, w))
            h = max(0.001, min(1.0, h))

            yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        txt_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
        converted += 1

    return converted, skipped, unknown


# ── Step 3: Split ─────────────────────────────────────────────────────────────

def split_dataset(data_dir, out_dir, val_fraction=0.2, seed=42):
    """
    Copy image+label pairs from data_dir into:
        out_dir/images/{train,val}/
        out_dir/labels/{train,val}/

    Wipes out_dir first so reruns are clean.
    """
    pairs = []
    for img_path in sorted(data_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            pairs.append((img_path, txt_path))

    if not pairs:
        print("  ERROR: no image+label pairs found in data_dir.")
        sys.exit(1)

    random.seed(seed)
    random.shuffle(pairs)

    n_val = max(1, int(len(pairs) * val_fraction))
    splits = {"val": pairs[:n_val], "train": pairs[n_val:]}

    if out_dir.exists():
        shutil.rmtree(out_dir)

    for split_name, split_pairs in splits.items():
        img_dir = out_dir / "images" / split_name
        lbl_dir = out_dir / "labels" / split_name
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for img_path, txt_path in split_pairs:
            shutil.copy2(img_path, img_dir / img_path.name)
            shutil.copy2(txt_path, lbl_dir / txt_path.name)

    n_train = len(splits["train"])
    n_val_actual = len(splits["val"])
    print(f"  {n_train} train / {n_val_actual} val  ({len(pairs)} total pairs)")
    return n_train, n_val_actual


# ── Step 4: data.yaml ─────────────────────────────────────────────────────────

def write_data_yaml(out_dir, class_names):
    data = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": class_names,
    }
    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    return yaml_path


# ── Step 5: Train ─────────────────────────────────────────────────────────────

def train_yolo(cfg, data_yaml):
    size_map = {"nano": "n", "small": "s", "medium": "m", "large": "l", "xlarge": "x"}
    version = cfg.get("model_version", "yolo11")
    size = size_map.get(cfg.get("model_size", "large"), "l")

    pretrained = cfg.get("pretrained_weights")
    if pretrained and os.path.exists(pretrained):
        model = YOLO(pretrained)
        print(f"  Loaded pretrained weights: {pretrained}")
    else:
        model_name = f"{version}{size}.pt"
        model = YOLO(model_name)
        print(f"  Base model: {model_name}")

    model.train(
        data=data_yaml,
        epochs=cfg.get("epochs", 100),
        batch=cfg.get("batch_size", 8),
        imgsz=cfg.get("img_size", 640),
        device=cfg.get("device", "0"),
        optimizer=cfg.get("optimizer", "auto"),
        lr0=cfg.get("lr0", 0.01),
        lrf=cfg.get("lrf", 0.001),
        cos_lr=cfg.get("cos_lr", True),
        warmup_epochs=cfg.get("warmup_epochs", 5),
        warmup_momentum=cfg.get("warmup_momentum", 0.9),
        warmup_bias_lr=cfg.get("warmup_bias_lr", 0.1),
        weight_decay=cfg.get("weight_decay", 0.0005),
        momentum=cfg.get("momentum", 0.937),
        box=cfg.get("box", 6.5),
        cls=cfg.get("cls", 1.5),
        dfl=cfg.get("dfl", 4.5),
        hsv_h=cfg.get("hsv_h", 0.015),
        hsv_s=cfg.get("hsv_s", 0.7),
        hsv_v=cfg.get("hsv_v", 0.4),
        degrees=cfg.get("degrees", 0.0),
        translate=cfg.get("translate", 0.1),
        scale=cfg.get("scale", 0.5),
        shear=cfg.get("shear", 0.0),
        perspective=cfg.get("perspective", 0.0),
        flipud=cfg.get("flipud", 0.0),
        fliplr=cfg.get("fliplr", 0.5),
        mosaic=cfg.get("mosaic", 1.0),
        mixup=cfg.get("mixup", 0.0),
        copy_paste=cfg.get("copy_paste", 0.0),
        patience=cfg.get("patience", 50),
        iou=cfg.get("iou", 0.2),
        amp=cfg.get("amp", True),
        single_cls=cfg.get("single_cls", False),
        resume=cfg.get("resume", False),
        workers=cfg.get("workers", 6),
        save_period=cfg.get("save_period", 10),
        plots=cfg.get("plots", True),
        project=cfg.get("project", "runs"),
        name=cfg.get("experiment", "inventory_train"),
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert custom annotations -> augment -> split -> train YOLO"
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "final_config.yaml"),
        help="Path to final_config.yaml (default: same directory as this script)",
    )
    parser.add_argument(
        "--data-dir",
        help="Override dataset_dir from config",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Skip manual augmentation (h-flip / v-flip / rotate-180)",
    )
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Only convert annotations to YOLO format, then exit",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Build dataset (convert + augment + split) but skip YOLO training",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(args.data_dir or cfg.get("dataset_dir", ""))
    if not data_dir.is_dir():
        print(f"ERROR: dataset_dir '{data_dir}' does not exist.")
        print("  Set dataset_dir in final_config.yaml or pass --data-dir /path/to/data")
        sys.exit(1)

    # ── 1. Convert ────────────────────────────────────────────────────────────
    print(f"\n[1/4] Converting annotations in: {data_dir}")
    has_custom = any(
        _is_custom_format(p.with_suffix(".txt"))
        for p in data_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTS and p.with_suffix(".txt").exists()
    )
    if not has_custom:
        print("  Skipped (all .txt files already in YOLO format)")
        converted, skipped, unknown = 0, 0, set()
    else:
        converted, skipped, unknown = convert_annotations(data_dir)
        print(f"  Converted {converted} file(s) | Already YOLO: {skipped}")
    if unknown:
        print(f"  WARNING - unrecognised categories (skipped): {sorted(unknown)}")
        print(f"  Valid names: {CLASS_NAMES}")

    if args.convert_only:
        print("\nDone (--convert-only).")
        return

    # ── 2. Augment ────────────────────────────────────────────────────────────
    already_augmented = any("_new_h_flip" in f.name for f in data_dir.iterdir())
    if args.no_augment:
        print("\n[2/4] Augmentation skipped (--no-augment)")
    elif already_augmented:
        print("\n[2/4] Augmentation skipped (augmented files already present)")
    else:
        print(f"\n[2/4] Augmenting images in: {data_dir}")
        Augment(str(data_dir)).process_images()
        print("  Done - dataset is now 4x original (orig + h-flip + v-flip + rot-180)")

    # ── 3. Split ──────────────────────────────────────────────────────────────
    out_dir = data_dir / "yolo_dataset"
    print(f"\n[3/4] Splitting dataset -> {out_dir}")
    split_dataset(
        data_dir,
        out_dir,
        val_fraction=cfg.get("val_fraction", 0.2),
        seed=cfg.get("split_seed", 42),
    )

    # ── 4. data.yaml ──────────────────────────────────────────────────────────
    data_yaml = write_data_yaml(out_dir, CLASS_NAMES)
    print(f"  data.yaml -> {data_yaml}")

    if args.skip_train:
        print("\nDataset ready. To train, run:")
        print(f"  python final_train.py --data-dir {data_dir}")
        print(f"  # or: yolo train data={data_yaml} model=yolo11l.pt epochs=100")
        return

    # ── 5. Train ──────────────────────────────────────────────────────────────
    print("\n[4/4] Training YOLO...")
    train_yolo(cfg, str(data_yaml))
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
