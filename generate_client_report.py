#!/usr/bin/env python3
"""
Generate a single self-contained HTML report from a pipeline results directory.

Usage:
    python3 generate_client_report.py <results_dir> [--video NAME]

Reads <results_dir>/<video>_counts.json and produces
<results_dir>/<video>_report.html: header with totals, counts table, and a
per-class gallery of object cards (base64-embedded crop + context images).
Tolerates a missing `confidence` key and missing/absolute image paths.
"""
import argparse
import base64
import glob
import html
import json
import os
import sys
from datetime import datetime

import cv2

CROP_MAX_W = 320
CONTEXT_MAX_W = 900
JPEG_QUALITY = 80

ACCENT = "#2f6f4f"


def find_counts_json(results_dir, video):
    if video:
        path = os.path.join(results_dir, f"{video}_counts.json")
        if not os.path.exists(path):
            sys.exit(f"error: {path} not found")
        return path
    matches = glob.glob(os.path.join(results_dir, "*_counts.json"))
    if not matches:
        sys.exit(f"error: no *_counts.json found in {results_dir}")
    if len(matches) > 1:
        sys.exit(f"error: multiple *_counts.json found in {results_dir}, use --video NAME")
    return matches[0]


def resolve_image_path(path, results_dir, video):
    """Resolve a possibly-absolute/foreign path by basename fallback."""
    if not path:
        return None
    if os.path.exists(path):
        return path
    candidate = os.path.join(results_dir, f"{video}_crops", os.path.basename(path))
    if os.path.exists(candidate):
        return candidate
    return None


def encode_image(path, max_width):
    """Resize (if needed) and base64-encode an image as a JPEG data URI. None if unreadable."""
    if not path:
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


PLACEHOLDER_DIV = '<div class="placeholder">Image not available</div>'


def img_or_placeholder(data_uri, alt, css_class):
    if data_uri:
        return f'<img class="{css_class}" src="{data_uri}" alt="{html.escape(alt)}">'
    return PLACEHOLDER_DIV


CONFIDENCE_LABELS = {"confirmed": "Confirmed", "review": "Needs review"}


def confidence_badge(obj):
    conf = obj.get("confidence")
    if not conf:
        return ""
    label = CONFIDENCE_LABELS.get(conf, conf.title())
    css = "badge-confirmed" if conf == "confirmed" else "badge-review"
    return f'<span class="badge {css}">{html.escape(label)}</span>'


def build_html(data, results_dir, video):
    summary = data.get("summary", {})
    counts_by_class = data.get("counts_by_class", {})
    objects = data.get("objects", [])

    total_objects = summary.get("total_unique_objects", sum(counts_by_class.values()))
    total_raw = summary.get("total_raw_tracks", "-")
    duplicates_removed = summary.get("duplicates_removed", "-")
    frames_processed = summary.get("frames_processed", "-")

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Group objects by class, preserving counts_by_class order (only classes with count > 0)
    by_class = {}
    for obj in objects:
        by_class.setdefault(obj.get("class_name", "Unknown"), []).append(obj)

    pills = "".join(
        f'<span class="pill">{html.escape(cls)}: <strong>{count}</strong></span>'
        for cls, count in counts_by_class.items()
        if count > 0
    )

    rows = "".join(
        f"<tr><td>{html.escape(cls)}</td><td>{count}</td></tr>"
        for cls, count in counts_by_class.items()
    )

    sections = []
    for cls, count in counts_by_class.items():
        if count <= 0:
            continue
        cls_objects = sorted(by_class.get(cls, []), key=lambda o: o.get("instance_number", 0))
        cards = []
        for obj in cls_objects:
            n = obj.get("instance_number", "?")
            crop_path = resolve_image_path(obj.get("crop_path"), results_dir, video)
            ctx_path = resolve_image_path(obj.get("context_frame_path"), results_dir, video)
            crop_uri = encode_image(crop_path, CROP_MAX_W)
            ctx_uri = encode_image(ctx_path, CONTEXT_MAX_W)

            crop_html = img_or_placeholder(crop_uri, f"{cls} #{n}", "crop-img")
            details_html = ""
            if ctx_uri:
                details_html = (
                    "<details><summary>Show in room</summary>"
                    f'<img class="context-img" src="{ctx_uri}" alt="{html.escape(cls)} #{n} in context"></details>'
                )
            elif obj.get("context_frame_path"):
                details_html = (
                    f"<details><summary>Show in room</summary>{PLACEHOLDER_DIV}</details>"
                )

            cards.append(
                f'<div class="card">{crop_html}'
                f'<div class="card-body"><span class="caption">{html.escape(cls)} #{n}</span>'
                f'{confidence_badge(obj)}</div>{details_html}</div>'
            )
        sections.append(
            f'<section><h2>{html.escape(cls)} <span class="count-tag">{count}</span></h2>'
            f'<div class="card-grid">{"".join(cards)}</div></section>'
        )

    return f"""<title>Inventory Report — {html.escape(video)}</title>
<style>
  :root {{ --accent: {ACCENT}; }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    background: #f6f7f8; color: #1c1e21; margin: 0; padding: 0 0 3rem;
    line-height: 1.5;
  }}
  header {{
    background: #1c2b24; color: #fff; padding: 2rem 2.5rem;
  }}
  header h1 {{ margin: 0 0 0.25rem; font-size: 1.6rem; font-weight: 600; }}
  header .meta {{ color: #c9d3ce; font-size: 0.9rem; margin-bottom: 1rem; }}
  .stat-row {{ display: flex; gap: 2rem; flex-wrap: wrap; margin-top: 1rem; }}
  .stat {{ min-width: 120px; }}
  .stat .value {{ font-size: 1.8rem; font-weight: 700; }}
  .stat .label {{ font-size: 0.8rem; color: #c9d3ce; text-transform: uppercase; letter-spacing: 0.03em; }}
  .pills {{ margin-top: 1.25rem; display: flex; flex-wrap: wrap; gap: 0.5rem; }}
  .pill {{
    background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2);
    border-radius: 999px; padding: 0.3rem 0.8rem; font-size: 0.85rem;
  }}
  main {{ max-width: 1100px; margin: 0 auto; padding: 0 2.5rem; }}
  table.counts {{ width: 100%; border-collapse: collapse; margin: 2rem 0; background: #fff;
    border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  table.counts th, table.counts td {{ text-align: left; padding: 0.7rem 1rem; border-bottom: 1px solid #eee; }}
  table.counts th {{ background: var(--accent); color: #fff; font-weight: 600; }}
  table.counts tr:last-child td {{ border-bottom: none; }}
  section {{ margin-bottom: 2.5rem; }}
  section h2 {{ font-size: 1.2rem; border-bottom: 2px solid var(--accent); padding-bottom: 0.4rem;
    display: flex; align-items: center; gap: 0.6rem; }}
  .count-tag {{ background: var(--accent); color: #fff; font-size: 0.8rem; border-radius: 999px;
    padding: 0.1rem 0.6rem; }}
  .card-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 1rem; margin-top: 1rem; }}
  .card {{ background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    overflow: hidden; display: flex; flex-direction: column; }}
  .crop-img {{ width: 100%; height: 180px; object-fit: cover; display: block; background: #eee; }}
  .placeholder {{ width: 100%; height: 180px; background: #e2e5e8; color: #8a8f96;
    display: flex; align-items: center; justify-content: center; font-size: 0.85rem; text-align: center; }}
  .card-body {{ padding: 0.6rem 0.8rem; display: flex; align-items: center; justify-content: space-between; gap: 0.5rem; }}
  .caption {{ font-weight: 600; font-size: 0.9rem; }}
  .badge {{ font-size: 0.7rem; border-radius: 999px; padding: 0.15rem 0.5rem; white-space: nowrap; }}
  .badge-confirmed {{ background: #dff3e6; color: #1e7a45; }}
  .badge-review {{ background: #fdecd2; color: #9a5b00; }}
  details {{ border-top: 1px solid #eee; }}
  details summary {{ cursor: pointer; padding: 0.5rem 0.8rem; font-size: 0.8rem; color: var(--accent); }}
  .context-img {{ width: 100%; display: block; }}
  footer {{ text-align: center; color: #8a8f96; font-size: 0.8rem; margin-top: 2rem; }}
</style>
<header>
  <h1>Inventory Report — {html.escape(video)}</h1>
  <div class="meta">Generated {html.escape(generated)}</div>
  <div class="stat-row">
    <div class="stat"><div class="value">{total_objects}</div><div class="label">Unique Objects</div></div>
    <div class="stat"><div class="value">{total_raw}</div><div class="label">Raw Tracks</div></div>
    <div class="stat"><div class="value">{duplicates_removed}</div><div class="label">Duplicates Removed</div></div>
    <div class="stat"><div class="value">{frames_processed}</div><div class="label">Frames Processed</div></div>
  </div>
  <div class="pills">{pills}</div>
</header>
<main>
  <table class="counts">
    <thead><tr><th>Class</th><th>Count</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  {"".join(sections)}
  <footer>Generated automatically from pipeline output.</footer>
</main>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir")
    parser.add_argument("--video", default=None, help="video name (default: infer from *_counts.json)")
    args = parser.parse_args()

    results_dir = args.results_dir
    counts_path = find_counts_json(results_dir, args.video)
    video = args.video or os.path.basename(counts_path)[: -len("_counts.json")]

    with open(counts_path) as f:
        data = json.load(f)

    report_html = build_html(data, results_dir, video)
    out_path = os.path.join(results_dir, f"{video}_report.html")
    with open(out_path, "w") as f:
        f.write(report_html)

    print(out_path)


if __name__ == "__main__":
    main()
