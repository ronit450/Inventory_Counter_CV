"""
Score pipeline count outputs against eval/ground_truth.json.

Usage:
    python eval_counts.py <results_dir> [<results_dir> ...]

For every video in ground_truth.json, looks for <video>_counts.json in each
results dir and prints a per-class over/under table plus MAE.
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
GT_PATH = os.path.join(_HERE, "eval", "ground_truth.json")

# Old 20-class model names -> new 10-class taxonomy (so old runs are scorable)
ALIASES = {
    "Office Chair": "Office Chairs",
    "Desk": "Desks",
    "Monitor": "Monitors",
    "Laptop": "Laptops",
    "Pedestal": "Pedestals",
    "Cubicle / Partition": "Cubicles / Partitions",
    "Printer / Scanner": "Printers Scanners",
    "Bookshelf / Cabinet": "Filing Cabinets / Storage Units",
    "Conference Table": "Conference Tables",
}


def normalize(counts: dict) -> dict:
    out = {}
    for name, c in counts.items():
        key = ALIASES.get(name, name)
        out[key] = out.get(key, 0) + c
    return out


def score_video(video: str, truth: dict, results_dir: str) -> bool:
    path = os.path.join(results_dir, f"{video}_counts.json")
    if not os.path.exists(path):
        return False
    data = json.load(open(path))
    pred = normalize(data.get("counts_by_class", {}))

    classes = sorted(set(truth) | set(pred))
    abs_err = over = under = 0
    print(f"\n{video}  [{results_dir}]")
    print(f"  {'class':<34} {'true':>4} {'pred':>4} {'err':>5}")
    for cls in classes:
        t, p = truth.get(cls, 0), pred.get(cls, 0)
        e = p - t
        abs_err += abs(e)
        over += max(e, 0)
        under += max(-e, 0)
        mark = "" if e == 0 else ("  <- over" if e > 0 else "  <- under")
        print(f"  {cls:<34} {t:>4} {p:>4} {e:>+5}{mark}")
    n_exact = sum(1 for c in classes if truth.get(c, 0) == pred.get(c, 0))
    print(f"  {'-'*50}")
    print(f"  exact classes: {n_exact}/{len(classes)}  "
          f"abs error: {abs_err}  (over: +{over}, under: -{under})")
    return True


def main():
    dirs = sys.argv[1:]
    if not dirs:
        print(__doc__)
        sys.exit(1)
    gt = json.load(open(GT_PATH))
    found = 0
    for video, truth in gt.items():
        for d in dirs:
            if score_video(video, truth, d):
                found += 1
    if not found:
        print("No matching <video>_counts.json found in given dirs "
              f"for videos: {list(gt)}")


if __name__ == "__main__":
    main()
