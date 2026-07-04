"""
YOLO Training Script — Pre-labeled .txt Dataset
================================================
Use this when you already have YOLO-format .txt label files alongside images.

Your flat folder structure:
    dataset_dir/
        image1.jpg  image1.txt
        image2.jpg  image2.txt
        ...

The script will split images+labels into train/val and generate data.yaml,
then launch YOLO training. You provide nc and names in the config.

Usage:
    python train_txt.py                          # uses config.txt.yaml
    python train_txt.py --config config.txt.yaml
    python train_txt.py --epochs 50 --batch_size 16
"""

import argparse
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import random
import shutil
import sys

import yaml
import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_IMG = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

SIZE_ABBREV = {
    "nano": "n", "small": "s", "medium": "m", "large": "l", "xlarge": "x",
}

TASK_SUFFIX = {
    "detection": "", "segmentation": "-seg",
}

VALID_VERSIONS = {"yolov8", "yolo11", "yolo26"}
VALID_SIZES = {"nano", "small", "medium", "large", "xlarge"}
VALID_TASKS = {"detection", "segmentation"}


# ---------------------------------------------------------------------------
# Dataset Preparer — splits flat image+txt folder into YOLO structure
# ---------------------------------------------------------------------------

class TxtDatasetPreparer:
    """Prepares a flat folder of images + YOLO .txt labels for training.

    Input layout (flat folder):
        dataset_dir/
            img1.jpg   img1.txt
            img2.png   img2.txt
            ...

    Output layout (YOLO training structure):
        dataset_dir/
            images/
                train/   <- images moved here
                val/     <- images moved here
            labels/
                train/   <- .txt files moved here
                val/     <- .txt files moved here
            data.yaml    <- generated from nc + names in config
    """

    def __init__(self, dataset_dir, task, nc, names, val_fraction=0.2, seed=42):
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.task = task
        self.nc = nc
        self.names = names
        self.val_fraction = val_fraction
        self.seed = seed

    def prepare(self):
        """Run dataset preparation. Skips if data.yaml already exists."""
        if not os.path.isdir(self.dataset_dir):
            print(f"ERROR: dataset_dir does not exist: {self.dataset_dir}")
            sys.exit(1)

        data_yaml = os.path.join(self.dataset_dir, "data.yaml")
        if os.path.isfile(data_yaml):
            print(f"[AUTO-PREPARE] data.yaml already exists, skipping conversion.")
            return

        pairs = self._scan_pairs()

        if len(pairs) < 2:
            print(f"[AUTO-PREPARE] Need at least 2 image+txt pairs, found {len(pairs)}.")
            sys.exit(1)

        print(f"\n{'=' * 60}")
        print(f"[AUTO-PREPARE] Pre-labeled .txt dataset detected!")
        print(f"  Task:        {self.task}")
        print(f"  Pairs:       {len(pairs)} image+txt pairs")
        print(f"  Classes:     {self.nc} — {self.names}")
        print(f"  Val split:   {self.val_fraction:.0%}")
        print(f"  Seed:        {self.seed}")
        print(f"{'=' * 60}\n")

        self._split_and_move(pairs)
        self._write_data_yaml()

        print(f"\n[AUTO-PREPARE] Dataset preparation complete!")

    def _scan_pairs(self):
        """Find all image+txt pairs in the flat dataset_dir."""
        all_files = os.listdir(self.dataset_dir)
        img_files = sorted(f for f in all_files if f.lower().endswith(SUPPORTED_IMG))

        pairs = []
        missing_txt = []

        for img_file in img_files:
            stem = os.path.splitext(img_file)[0]
            txt_file = stem + ".txt"
            txt_path = os.path.join(self.dataset_dir, txt_file)

            if os.path.isfile(txt_path):
                pairs.append((img_file, txt_file))
            else:
                missing_txt.append(img_file)

        if missing_txt:
            print(f"[AUTO-PREPARE] WARNING: {len(missing_txt)} images have no matching .txt "
                  f"(will be skipped): {missing_txt[:5]}"
                  + (" ..." if len(missing_txt) > 5 else ""))

        print(f"[AUTO-PREPARE] Found {len(pairs)} image+txt pairs.")
        return pairs

    def _split_and_move(self, pairs):
        """Split pairs into train/val and MOVE files into YOLO structure."""
        random.seed(self.seed)
        shuffled = list(pairs)
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * (1.0 - self.val_fraction))
        splits = {
            "train": shuffled[:split_idx],
            "val": shuffled[split_idx:],
        }

        for split_name, split_pairs in splits.items():
            img_dir = os.path.join(self.dataset_dir, "images", split_name)
            lbl_dir = os.path.join(self.dataset_dir, "labels", split_name)
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)

            for img_file, txt_file in split_pairs:
                src_img = os.path.join(self.dataset_dir, img_file)
                src_txt = os.path.join(self.dataset_dir, txt_file)
                dst_img = os.path.join(img_dir, img_file)
                dst_txt = os.path.join(lbl_dir, txt_file)

                if os.path.isfile(src_img):
                    shutil.move(src_img, dst_img)
                if os.path.isfile(src_txt):
                    shutil.move(src_txt, dst_txt)

            print(f"  {split_name}: {len(split_pairs)} images/labels "
                  f"moved to images/{split_name}/ and labels/{split_name}/")

    def _write_data_yaml(self):
        """Generate data.yaml using nc and names from config."""
        data = {
            "path": self.dataset_dir.replace("\\", "/"),
            "train": "images/train",
            "val": "images/val",
            "nc": self.nc,
            "names": self.names,
        }

        yaml_path = os.path.join(self.dataset_dir, "data.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        print(f"  data.yaml written to {yaml_path}")


# ---------------------------------------------------------------------------
# YOLO Trainer (same as train.py, detection/segmentation only)
# ---------------------------------------------------------------------------

class YOLOTrainer:

    def __init__(self, cfg):
        self.cfg = cfg
        self._validate_config()

    def _validate_config(self):
        task = self.cfg.get("task", "detection")
        if task not in VALID_TASKS:
            print(f"ERROR: Unknown task '{task}'. Choose from: {sorted(VALID_TASKS)}")
            sys.exit(1)

        version = self.cfg.get("model_version", "yolo11")
        if version not in VALID_VERSIONS:
            print(f"ERROR: Unknown model_version '{version}'. "
                  f"Choose from: {sorted(VALID_VERSIONS)}")
            sys.exit(1)

        size = self.cfg.get("model_size", "medium")
        if size not in VALID_SIZES:
            print(f"ERROR: Unknown model_size '{size}'. "
                  f"Choose from: {sorted(VALID_SIZES)}")
            sys.exit(1)

    def _build_model_name(self):
        version = self.cfg["model_version"]
        size = SIZE_ABBREV[self.cfg["model_size"]]
        suffix = TASK_SUFFIX[self.cfg["task"]]
        return f"{version}{size}{suffix}.pt"

    def _print_summary(self):
        task = self.cfg["task"]
        model_name = self.cfg.get("pretrained_weights") or self._build_model_name()
        print(f"\n{'=' * 60}")
        print(f"YOLO {task.upper()} TRAINING (pre-labeled .txt)")
        print(f"{'=' * 60}")
        print(f"  Task:           {task}")
        print(f"  Model:          {model_name}")
        print(f"  Dataset:        {self.cfg.get('dataset_dir', 'N/A')}")
        print(f"  Classes (nc):   {self.cfg.get('nc')} — {self.cfg.get('names')}")
        print(f"  Epochs:         {self.cfg.get('epochs', 100)}")
        print(f"  Batch size:     {self.cfg.get('batch_size', 8)}")
        print(f"  Image size:     {self.cfg.get('img_size', 640)}")
        print(f"  Optimizer:      {self.cfg.get('optimizer', 'auto')}")
        print(f"  LR:             {self.cfg.get('lr0', 0.01)}")
        print(f"  Device:         {self.cfg.get('device', '0')}")
        print(f"  Output:         {self.cfg.get('project', 'runs')}/{self.cfg.get('experiment', 'train')}")
        print(f"{'=' * 60}\n")

    def train(self):
        from ultralytics import YOLO

        self._print_summary()

        task = self.cfg["task"]
        model_name = self.cfg.get("pretrained_weights") or self._build_model_name()
        model = YOLO(model_name)

        dataset_dir = self.cfg.get("dataset_dir", "")
        data_yaml = os.path.join(dataset_dir, "data.yaml")
        if not os.path.isfile(data_yaml):
            print(f"ERROR: data.yaml not found at {data_yaml}")
            sys.exit(1)

        train_kwargs = {
            "data": data_yaml,
            "imgsz": self.cfg.get("img_size", 640),
            "epochs": self.cfg.get("epochs", 100),
            "batch": self.cfg.get("batch_size", 8),
            "device": self.cfg.get("device", "0"),
            "optimizer": self.cfg.get("optimizer", "auto"),
            "lr0": self.cfg.get("lr0", 0.01),
            "lrf": self.cfg.get("lrf", 0.001),
            "weight_decay": self.cfg.get("weight_decay", 0.0005),
            "momentum": self.cfg.get("momentum", 0.937),
            "cos_lr": self.cfg.get("cos_lr", True),
            "workers": self.cfg.get("workers", 6),
            "warmup_epochs": self.cfg.get("warmup_epochs", 5),
            "warmup_momentum": self.cfg.get("warmup_momentum", 0.9),
            "warmup_bias_lr": self.cfg.get("warmup_bias_lr", 0.1),
            "patience": self.cfg.get("patience", 50),
            "amp": self.cfg.get("amp", True),
            "single_cls": self.cfg.get("single_cls", False),
            "resume": self.cfg.get("resume", False),
            "save_period": self.cfg.get("save_period", 10),
            "plots": self.cfg.get("plots", True),
            "project": self.cfg.get("project", "runs"),
            "name": self.cfg.get("experiment", "train"),
            "iou": self.cfg.get("iou", 0.2),
            "box": self.cfg.get("box", 6.5),
            "cls": self.cfg.get("cls", 1.5),
            "dfl": self.cfg.get("dfl", 4.5),
            "mosaic": self.cfg.get("mosaic", 1.0),
            "mixup": self.cfg.get("mixup", 0.0),
            "copy_paste": self.cfg.get("copy_paste", 0.0),
            "hsv_h": self.cfg.get("hsv_h", 0.015),
            "hsv_s": self.cfg.get("hsv_s", 0.7),
            "hsv_v": self.cfg.get("hsv_v", 0.4),
            "degrees": self.cfg.get("degrees", 0.0),
            "translate": self.cfg.get("translate", 0.1),
            "scale": self.cfg.get("scale", 0.5),
            "shear": self.cfg.get("shear", 0.0),
            "perspective": self.cfg.get("perspective", 0.0),
            "flipud": self.cfg.get("flipud", 0.0),
            "fliplr": self.cfg.get("fliplr", 0.5),
        }

        if task == "segmentation":
            train_kwargs["overlap_mask"] = self.cfg.get("overlap_mask", True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model.train(**train_kwargs)

        print(f"\n{'=' * 60}")
        print(f"{task.capitalize()} training complete!")
        print(f"Results saved to: {self.cfg.get('project', 'runs')}/{self.cfg.get('experiment', 'train')}/")
        print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Config loading & CLI override
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Training — Pre-labeled .txt Dataset")
    parser.add_argument("--config", type=str, default="config.txt.yaml",
                        help="Path to YAML config file")
    args, unknown = parser.parse_known_args()

    if not os.path.isfile(args.config):
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Parse CLI overrides (e.g. --epochs 50 --batch_size 16)
    i = 0
    while i < len(unknown):
        key = unknown[i].lstrip("-")
        if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
            val = unknown[i + 1]
            if val.lower() in ("true", "false"):
                val = val.lower() == "true"
            elif val.replace(".", "", 1).replace("-", "", 1).replace("e", "", 1).isdigit():
                val = float(val) if "." in val or "e" in val.lower() else int(val)
            elif val.lower() in ("null", "none"):
                val = None
            cfg[key] = val
            i += 2
        else:
            cfg[key] = True
            i += 1

    return cfg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    try:
        cfg = parse_args()
    except SystemExit:
        raise
    except Exception as e:
        print(f"[YOLO ERROR] Failed to parse config: {e}", flush=True)
        sys.exit(1)

    # Validate required fields
    nc = cfg.get("nc")
    names = cfg.get("names")
    if nc is None or not names:
        print("ERROR: 'nc' and 'names' must be set in config.")
        print("  nc:    number of classes (integer)")
        print("  names: list of class names, e.g. [\"cat\", \"dog\"]")
        sys.exit(1)
    if len(names) != nc:
        print(f"ERROR: nc={nc} but names has {len(names)} entries. They must match.")
        sys.exit(1)

    task = cfg.get("task", "detection")
    dataset_dir = cfg.get("dataset_dir", "")

    # Auto-prepare: split flat image+txt folder into YOLO structure
    if cfg.get("auto_prepare", True):
        if dataset_dir and os.path.isdir(dataset_dir):
            try:
                preparer = TxtDatasetPreparer(
                    dataset_dir=dataset_dir,
                    task=task,
                    nc=nc,
                    names=names,
                    val_fraction=cfg.get("val_fraction", 0.2),
                    seed=cfg.get("split_seed", 42),
                )
                preparer.prepare()
            except Exception as e:
                print(f"[YOLO ERROR] Dataset preparation failed: {e}", flush=True)
                sys.exit(1)
        else:
            print(f"[WARN] auto_prepare is true but dataset_dir "
                  f"'{dataset_dir}' does not exist.", flush=True)
            sys.exit(1)

    # Train
    try:
        trainer = YOLOTrainer(cfg)
        trainer.train()
    except KeyboardInterrupt:
        print("\n[YOLO] Training interrupted by user.", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"\n[YOLO ERROR] Training crashed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
