# track_stitch.py
"""
Stitch fragmented tracks by motion continuity.

A pair (a, b) is stitchable when: same class, a ends strictly before b starts,
the gap is small, and a's last bbox overlaps b's first bbox. Temporally
overlapping tracks are never stitched (co-occurrence = provably distinct).
Greedy matching by (gap, 1-IoU): each track gets at most one successor and
one predecessor. Returns {tid: root_tid} (root = smallest tid in chain).
"""


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = ((ax2 - ax1) * (ay2 - ay1)) + ((bx2 - bx1) * (by2 - by1)) - inter
    return inter / union if union > 0 else 0.0


def stitch_tracks(tracks: dict, max_gap: int, min_iou: float) -> dict:
    tids = sorted(tracks)
    candidates = []
    for a in tids:
        for b in tids:
            if a == b:
                continue
            ta, tb = tracks[a], tracks[b]
            if ta["class_id"] != tb["class_id"]:
                continue
            gap = tb["first_seen"] - ta["last_seen"]
            if gap <= 0 or gap > max_gap:
                continue
            iou = _iou(ta["last_box"], tb["first_box"])
            if iou < min_iou:
                continue
            candidates.append((gap, 1.0 - iou, a, b))
    candidates.sort()

    parent = {t: t for t in tids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    has_successor, has_predecessor = set(), set()
    for _gap, _inv_iou, a, b in candidates:
        if a in has_successor or b in has_predecessor:
            continue
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        lo, hi = min(ra, rb), max(ra, rb)
        parent[hi] = lo
        has_successor.add(a)
        has_predecessor.add(b)

    return {t: find(t) for t in tids}
