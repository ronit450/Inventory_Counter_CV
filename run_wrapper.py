"""
S3 orchestration wrapper for the inventory counting pipeline.

Reads VIDEO_URL (S3 URI or HTTPS) from environment, runs our YOLO +
ByteTrack + DINOv2 ReID pipeline, uploads results and crops to S3, and
writes a structured JSON result to OUTPUT_FILE.

Environment variables:
  VIDEO_URL          — s3://bucket/key, https://..., or local path
  INPUT_S3_URI       — fallback for VIDEO_URL (Step Functions legacy)
  MODEL_PATH         — override YOLO model path (default: models/Rk_trained_model.pt)
  RESULTS_S3_BUCKET  — S3 bucket for uploads (default: same bucket as input)
  OUTPUT_FILE        — where to write results JSON (default: /tmp/results.json)
  EXECUTION_NAME     — Step Functions execution name for S3 key organisation
"""

import glob
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import boto3
import cv2
import requests
from botocore.exceptions import BotoCoreError, ClientError


# ── S3 helpers ────────────────────────────────────────────────────────────────

def _parse_s3_uri(s3_uri: str):
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _download_from_s3(s3_uri: str, local_path: str) -> None:
    bucket, key = _parse_s3_uri(s3_uri)
    print(f"Downloading s3://{bucket}/{key} → {local_path}")
    boto3.client("s3").download_file(bucket, key, local_path)
    print(f"Downloaded: {local_path}")


def _download_from_url(url: str, local_path: str) -> None:
    print(f"Downloading {url} → {local_path}")
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
    print(f"Downloaded: {local_path}")


def _upload_to_s3(local_path: str, bucket: str, key: str) -> str:
    print(f"Uploading {local_path} → s3://{bucket}/{key}")
    boto3.client("s3").upload_file(local_path, bucket, key)
    return f"s3://{bucket}/{key}"


def _upload_dir_to_s3(local_dir: str, bucket: str, base_key: str) -> list:
    """Upload all images in a directory to S3 in parallel."""
    images = [
        f for ext in ("*.jpg", "*.jpeg", "*.png")
        for f in glob.glob(os.path.join(local_dir, ext))
    ]
    if not images:
        return []
    urls = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(
                _upload_to_s3, img, bucket, f"{base_key}/{os.path.basename(img)}"
            ): img
            for img in images
        }
        for fut in as_completed(futures):
            try:
                urls.append(fut.result())
            except Exception as e:
                print(f"  Upload error {futures[fut]}: {e}")
    print(f"  Uploaded {len(urls)}/{len(images)} images")
    return urls


def _results_bucket(video_url: str) -> str:
    bucket = os.getenv("RESULTS_S3_BUCKET", "")
    if bucket:
        return bucket
    if video_url.startswith("s3://"):
        return urlparse(video_url).netloc
    return "inventory-counter-results"


# ── Pipeline adapter ──────────────────────────────────────────────────────────

def _run_pipeline(input_path: str) -> dict:
    """
    Run our YOLO + ReID pipeline on a single video/image.
    Returns {"items": [...], "video_url": "..."} matching the expected contract.
    """
    import config
    from ultralytics import YOLO

    output_dir = os.getenv("OUTPUT_FOLDER", "/tmp/rk_output")
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.getenv("MODEL_PATH", config.YOLO_MODEL_PATH)
    print(f"[Pipeline] Loading model: {model_path}")
    model = YOLO(model_path)

    ext = Path(input_path).suffix.lower()
    is_image = ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    if is_image:
        return _run_image(input_path, model, output_dir, config)
    return _run_video(input_path, model, output_dir, config)


def _run_video(input_path: str, model, output_dir: str, config) -> dict:
    from main import process_video

    result = process_video(input_path, model, output_dir)
    if not result:
        return {"items": [], "video_url": ""}

    crops_dir = result.get("output_crops_dir", "")
    counts = result.get("counts_by_class", {})
    video_name = Path(input_path).stem
    output_video = os.path.join(output_dir, f"{video_name}_detected.mp4")

    # Organise crops into per-class subdirs for S3 upload
    items = []
    for class_name, quantity in sorted(counts.items()):
        safe_cls = class_name.replace("/", "_").replace(" ", "_")
        class_dir = os.path.join(output_dir, f"crops_{safe_cls}")
        os.makedirs(class_dir, exist_ok=True)
        for src in glob.glob(os.path.join(crops_dir, f"{safe_cls}_*.jpg")):
            shutil.copy2(src, class_dir)
        items.append({
            "itemName": class_name,
            "quantity": quantity,
            "image_folder": class_dir,
            "review": [],
        })

    return {"items": items, "video_url": output_video}


def _run_image(input_path: str, model, output_dir: str, config) -> dict:
    """Single-image detection (no tracking/ReID — just count detections)."""
    results = model(
        input_path,
        conf=config.YOLO_CONFIDENCE,
        iou=config.YOLO_IOU,
        imgsz=config.YOLO_IMG_SIZE,
        verbose=False,
    )
    counts: dict = {}
    if results and results[0].boxes is not None:
        for cls_id in results[0].boxes.cls.cpu().numpy().astype(int):
            name = config.CLASS_NAMES.get(int(cls_id), f"class_{cls_id}")
            counts[name] = counts.get(name, 0) + 1
    items = [
        {"itemName": k, "quantity": v, "image_folder": "", "review": []}
        for k, v in sorted(counts.items())
    ]
    return {"items": items, "video_url": ""}


# ── Headless OpenCV patch ─────────────────────────────────────────────────────

def _patch_headless():
    noop = lambda *a, **k: None
    cv2.imshow = noop
    cv2.destroyAllWindows = noop
    cv2.waitKey = lambda *a, **k: -1


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    video_url = os.getenv("VIDEO_URL", "").strip() or os.getenv("INPUT_S3_URI", "").strip()
    output_file = os.getenv("OUTPUT_FILE", "/tmp/results.json")

    # Resolve input path
    if video_url.startswith("s3://"):
        _, key = _parse_s3_uri(video_url)
        ext = Path(key).suffix or ".mp4"
        input_path = f"/tmp/downloaded_video{ext}"
        _download_from_s3(video_url, input_path)
    elif video_url.startswith(("http://", "https://")):
        input_path = "/tmp/downloaded_video.mp4"
        _download_from_url(video_url, input_path)
    elif video_url:
        input_path = video_url
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Video not found: {input_path}")
    else:
        input_path = "input_videos/office_1.mp4"
        print(f"No VIDEO_URL set — using default: {input_path}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    os.environ["NO_DISPLAY"] = "1"
    _patch_headless()

    print(f"\n{'='*70}")
    print(f"  INVENTORY COUNTER — {input_path}")
    print(f"{'='*70}\n")

    try:
        pipeline_result = _run_pipeline(input_path)
        items_list = pipeline_result.get("items", [])
        video_output = pipeline_result.get("video_url", "")

        if not items_list:
            final = {
                "status": "completed",
                "processed_at": datetime.utcnow().isoformat() + "Z",
                "video_url": video_url or input_path,
                "detections": [],
                "total_items_detected": 0,
                "total_item_types": 0,
            }
        else:
            execution_id = os.getenv(
                "EXECUTION_NAME",
                datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
            )
            results_bucket = _results_bucket(video_url)
            base_key = f"results/{execution_id}"

            detections = []
            total_items = 0
            for item in items_list:
                name = item.get("itemName", "Unknown")
                qty = item.get("quantity", 0)
                img_folder = item.get("image_folder", "")
                total_items += qty
                print(f"\n  {name}: qty={qty}  folder={img_folder}")

                safe_name = name.lower().replace(" ", "_")
                img_s3_key = f"{base_key}/images/{safe_name}"
                image_urls = []
                if img_folder and os.path.isdir(img_folder):
                    image_urls = _upload_dir_to_s3(img_folder, results_bucket, img_s3_key)

                detections.append({
                    "itemName": name,
                    "quantity": qty,
                    "image_folder": img_folder,
                    "image_folder_s3": f"s3://{results_bucket}/{img_s3_key}" if img_folder else None,
                    "image_urls": image_urls,
                    "review": item.get("review", []),
                    "processed_at": datetime.utcnow().isoformat() + "Z",
                    "execution_id": execution_id,
                    "original_video_url": video_url or input_path,
                })

            video_s3_url = None
            if video_output and os.path.exists(video_output):
                try:
                    video_s3_url = _upload_to_s3(
                        video_output, results_bucket,
                        f"{base_key}/video/{Path(video_output).name}",
                    )
                    print(f"\n  Video uploaded: {video_s3_url}")
                except Exception as e:
                    print(f"  Video upload failed: {e}")

            final = {
                "status": "completed",
                "processed_at": datetime.utcnow().isoformat() + "Z",
                "video_url": video_url or input_path,
                "execution_id": execution_id,
                "detections": detections,
                "total_items_detected": total_items,
                "total_item_types": len(items_list),
                "video_url_s3": video_s3_url,
            }

        print(f"\n{'='*70}")
        print("  Results:")
        print(f"{'='*70}")
        print(json.dumps(final, indent=2))

        with open(output_file, "w") as f:
            json.dump(final, f, indent=2)
        print(f"\n  Saved: {output_file}")
        print(f"JSON_OUTPUT:{json.dumps(final)}")

    except Exception as e:
        error = {
            "status": "error",
            "error_message": str(e),
            "processed_at": datetime.utcnow().isoformat() + "Z",
            "video_url": video_url or input_path,
        }
        try:
            with open(output_file, "w") as f:
                json.dump(error, f, indent=2)
        except Exception:
            pass
        print(f"\n  ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
