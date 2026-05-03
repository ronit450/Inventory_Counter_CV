"""
Two-Pass Re-Identification Module
==================================
Pass 1 (online):  ByteTrack produces tracks; collect quality-scored crops +
                  scene/background fingerprints per track.
Pass 2 (offline): Combined similarity (appearance + background) drives
                  complete-linkage hierarchical clustering.

Core insight — why appearance alone fails for identical furniture:
  DINOv2/CLIP ask "does object A LOOK like object B?"
  For 5 identical office chairs this is always "yes" — unusable.

The fix — background fingerprinting:
  Ask "was object A in the SAME PHYSICAL LOCATION as object B?"
  For each track, store the full frame (64×64 grayscale) with the object
  blacked out. When the camera returns to the same spot, the background
  (walls, windows, other furniture) is similar regardless of object identity.
  Two different identical chairs in different locations → different backgrounds
  → blocked from merging even if appearance is the same.

Combined merge distance:
  dist = 1 - (app_weight * appearance_sim + bg_weight * background_sim)

Hard constraints (distance = 2.0, never merged):
  - Tracks visible in the same frame (co-occurrence guard)
  - No hard background threshold; background weight handles it continuously
"""

import math
import cv2
import numpy as np
import torch
from collections import defaultdict
from PIL import Image

import config

# ── Crop quality scoring ──────────────────────────────────────────────────────

def crop_quality_score(crop: np.ndarray, bbox=None, frame_shape=None) -> float:
    """Score a crop: higher = better quality. Sharpness × area × completeness."""
    if crop is None or crop.size == 0:
        return 0.0
    h, w = crop.shape[:2]
    if h < 10 or w < 10:
        return 0.0
    area = h * w
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    completeness = 1.0
    if frame_shape is not None and bbox is not None:
        x1, y1, x2, y2 = bbox
        fh, fw = frame_shape[:2]
        if x1 <= 3 or y1 <= 3 or x2 >= fw - 3 or y2 >= fh - 3:
            completeness = 0.6
    return math.log1p(area) * math.log1p(sharpness + 1.0) * completeness


def _context_fingerprint(frame: np.ndarray, bbox: tuple) -> np.ndarray:
    """
    HSV color histogram of the local context AROUND the object (object blacked out).
    Context region = object + one object-size of margin on each side.

    Why this works as a location fingerprint:
      - The wall color / floor color / nearby furniture near an object is distinctive
        to that physical location.
      - Same chair revisited: similar local color palette → high similarity.
      - Different identical chair elsewhere: different colors nearby → low similarity.
      - Empirically: same-location similarity ~0.95-1.0, different-location ~0.30-0.50.
        Gap of 0.45-0.65 is far more discriminative than full-frame blurred (~0.05 gap).

    Returns 96-float L2-normalised vector (3 channels × 32 bins).
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)

    margin = max(bw, bh)   # one object-size of context
    cx1, cy1 = max(0, x1 - margin), max(0, y1 - margin)
    cx2, cy2 = min(w, x2 + margin), min(h, y2 + margin)

    context = frame[cy1:cy2, cx1:cx2].copy()
    # Black out the object itself so its appearance doesn't bleed into location signal
    ox1, oy1 = x1 - cx1, y1 - cy1
    ox2, oy2 = ox1 + bw, oy1 + bh
    context[max(0, oy1):min(context.shape[0], oy2),
            max(0, ox1):min(context.shape[1], ox2)] = 0

    hsv = cv2.cvtColor(context, cv2.COLOR_BGR2HSV)
    bins = 32
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256]).flatten()

    vec = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-6 else vec


# ── Track collector (online, Pass 1) ─────────────────────────────────────────

class TrackCollector:
    """
    Collects per-track data during a single video processing pass.
    Two signals stored per track:
      1. Top-k quality-scored crops (for DINOv2 embedding).
      2. Best scene fingerprint (for location-based gate).
    No DINOv2 calls during collection — fast and cheap.
    """

    def __init__(self, top_k_crops: int = 5):
        self.top_k = top_k_crops
        # track_id -> {"class_id", "class_name", "scored_crops": [(score, crop)]}
        self.tracks: dict = {}
        # frozenset({tid1, tid2}) for any pair visible in the same frame
        self.cooccurrence: set = set()
        # track_id -> {"fp": np.ndarray, "bbox_area": int}
        # Stores the fingerprint from the frame where the object had the LARGEST bbox
        # (camera closest = most object context visible in scene).
        self.bg_fingerprints: dict = {}
        self._prune_interval = top_k_crops * 3

    def register_frame(self, track_ids, frame_idx: int = 0):
        ids = [int(t) for t in track_ids]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                self.cooccurrence.add(frozenset({ids[i], ids[j]}))

    def add_detection(self, track_id: int, class_id: int, class_name: str,
                      crop: np.ndarray, bbox=None, frame_shape=None):
        """Store quality-scored crop for DINOv2 embedding at dedup time."""
        tid = int(track_id)
        if tid not in self.tracks:
            self.tracks[tid] = {
                "class_id": int(class_id),
                "class_name": class_name,
                "scored_crops": [],
            }
        if crop is None or crop.size == 0:
            return
        score = crop_quality_score(crop, bbox, frame_shape)
        t = self.tracks[tid]
        t["scored_crops"].append((score, crop.copy()))
        if len(t["scored_crops"]) > self._prune_interval:
            t["scored_crops"].sort(key=lambda x: x[0], reverse=True)
            t["scored_crops"] = t["scored_crops"][:self.top_k]

    def add_background(self, track_id: int, frame: np.ndarray, bbox: tuple):
        """
        Store context fingerprint for the frame where this track had its largest bbox.
        Large bbox = camera closest = most surrounding context captured.
        """
        tid = int(track_id)
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        bbox_area = max(0, x2 - x1) * max(0, y2 - y1)
        existing = self.bg_fingerprints.get(tid)
        if existing is None or bbox_area > existing["bbox_area"]:
            self.bg_fingerprints[tid] = {
                "fp": _context_fingerprint(frame, bbox),
                "bbox_area": bbox_area,
            }

    def get_best_crops(self, track_id: int, k: int = None) -> list:
        k = k or self.top_k
        t = self.tracks.get(int(track_id))
        if not t or not t["scored_crops"]:
            return []
        sorted_crops = sorted(t["scored_crops"], key=lambda x: x[0], reverse=True)
        return [c for _, c in sorted_crops[:k] if c is not None and c.size > 0]

    def get_best_crop(self, track_id: int) -> np.ndarray | None:
        crops = self.get_best_crops(track_id, k=1)
        return crops[0] if crops else None

    def get_best_crop_score(self, track_id: int) -> float:
        t = self.tracks.get(int(track_id))
        if not t or not t["scored_crops"]:
            return 0.0
        return max(s for s, _ in t["scored_crops"])


# ── Post-hoc deduplicator (offline, Pass 2) ──────────────────────────────────

class PostHocDeduplicator:
    """
    Offline deduplication via complete-linkage hierarchical clustering.

    Distance between two tracks:
        dist = 1 - (app_weight * appearance_sim + bg_weight * background_sim)

    This means:
      - Same object, same location:  high app_sim AND high bg_sim → low dist → merged
      - Same object, diff angle:     medium app_sim, high bg_sim → still low dist → merged
      - Diff identical objects:      high app_sim, LOW bg_sim → higher dist → NOT merged
      - Hard vetoes (dist = 2.0):    co-occurring tracks (guaranteed different objects)

    Complete-linkage clustering: ALL pairs in a cluster must be within max_d.
    No single-linkage chaining → no false merges through intermediates.
    """

    def __init__(self, similarity_threshold: float = 0.60,
                 top_k_crops: int = 5,
                 model_name: str = "facebook/dinov2-base"):
        self.threshold = similarity_threshold
        self.top_k = top_k_crops
        self._model_name = model_name
        self._model = None
        self._processor = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_loaded(self):
        if self._model is not None:
            return
        print(f"  [ReID] Loading DINOv2 ({self._model_name}) on {self._device}...")
        from transformers import AutoModel, AutoImageProcessor
        self._processor = AutoImageProcessor.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name).to(self._device)
        self._model.eval()
        print("  [ReID] DINOv2 ready.")

    @torch.no_grad()
    def _embed_crops(self, crops: list) -> np.ndarray | None:
        """Average DINOv2 CLS-token embedding over multiple crops. L2-normalised."""
        self._ensure_loaded()
        embs = []
        for crop in crops:
            if crop is None or crop.size == 0:
                continue
            try:
                pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                inp = self._processor(images=pil, return_tensors="pt").to(self._device)
                out = self._model(**inp)
                feat = out.last_hidden_state[:, 0, :].cpu().numpy()[0]
                n = np.linalg.norm(feat)
                if n > 1e-6:
                    embs.append(feat / n)
            except Exception:
                continue
        if not embs:
            return None
        avg = np.mean(embs, axis=0)
        n = np.linalg.norm(avg)
        return avg / n if n > 1e-6 else avg

    def deduplicate(self, collector: TrackCollector) -> dict:
        """
        Cluster tracks per class into unique physical objects.
        Returns {track_id: canonical_id}. canonical_id = lowest track_id in cluster.
        """
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        bg_weight = float(getattr(config, "REID_BACKGROUND_WEIGHT", 0.0))
        app_weight = 1.0 - bg_weight

        canonical_map = {tid: tid for tid in collector.tracks}

        by_class: dict = defaultdict(list)
        for tid, info in collector.tracks.items():
            by_class[info["class_id"]].append(tid)

        for class_id, track_ids in by_class.items():
            if len(track_ids) < 2:
                continue

            # ── Appearance embeddings ─────────────────────────────────────
            embeddings = {}
            for tid in track_ids:
                crops = collector.get_best_crops(tid, k=self.top_k)
                if crops:
                    emb = self._embed_crops(crops)
                    if emb is not None:
                        embeddings[tid] = emb

            valid_ids = [tid for tid in track_ids if tid in embeddings]
            if len(valid_ids) < 2:
                continue

            n = len(valid_ids)

            # ── Appearance distance matrix ────────────────────────────────
            M = np.stack([embeddings[tid] for tid in valid_ids])
            app_sim = M @ M.T   # cosine similarity (L2-normalised)
            np.clip(app_sim, -1.0, 1.0, out=app_sim)

            # ── Background similarity matrix ──────────────────────────────
            if bg_weight > 0:
                bg_fps = {}
                for tid in valid_ids:
                    entry = collector.bg_fingerprints.get(tid)
                    if entry is not None:
                        bg_fps[tid] = entry["fp"]

                bg_sim = np.full((n, n), 0.5)   # neutral default (no fingerprint)
                np.fill_diagonal(bg_sim, 1.0)
                for i, tid1 in enumerate(valid_ids):
                    for j in range(i + 1, n):
                        tid2 = valid_ids[j]
                        if tid1 in bg_fps and tid2 in bg_fps:
                            s = float(np.dot(bg_fps[tid1], bg_fps[tid2]))
                            bg_sim[i, j] = max(0.0, s)
                            bg_sim[j, i] = bg_sim[i, j]
            else:
                bg_sim = np.ones((n, n))

            # ── Combined distance ─────────────────────────────────────────
            combined_sim = app_weight * app_sim + bg_weight * bg_sim
            dist = np.clip(1.0 - combined_sim, 0.0, 2.0).astype(np.float64)
            np.fill_diagonal(dist, 0.0)

            # ── Hard co-occurrence veto ───────────────────────────────────
            for i, tid1 in enumerate(valid_ids):
                for j in range(i + 1, n):
                    tid2 = valid_ids[j]
                    if frozenset({tid1, tid2}) in collector.cooccurrence:
                        dist[i, j] = 2.0
                        dist[j, i] = 2.0

            # ── Debug pairwise scores ─────────────────────────────────────
            if getattr(config, "REID_DEBUG", False):
                cls_name = config.CLASS_NAMES.get(class_id, f"cls_{class_id}")
                print(f"\n  [REID_DEBUG] Class: {cls_name}  ({n} tracks: {valid_ids})")
                for i, tid1 in enumerate(valid_ids):
                    for j in range(i + 1, n):
                        tid2 = valid_ids[j]
                        a_sim = float(app_sim[i, j])
                        b_sim = float(bg_sim[i, j])
                        c_sim = float(combined_sim[i, j])
                        d = float(dist[i, j])
                        cooc = "CO-OCC" if frozenset({tid1, tid2}) in collector.cooccurrence else ""
                        print(f"    {tid1}↔{tid2}: app={a_sim:.3f} bg={b_sim:.3f} "
                              f"combined={c_sim:.3f} dist={d:.3f} {cooc}")

            # ── Complete-linkage hierarchical clustering ──────────────────
            # max_d = 1 - threshold in COMBINED similarity space.
            # All pairs within a cluster satisfy combined_sim > threshold.
            cls_name_for_thresh = config.CLASS_NAMES.get(class_id, f"cls_{class_id}")
            _class_thresholds = getattr(config, "CLASS_REID_THRESHOLDS", {})
            effective_threshold = _class_thresholds.get(cls_name_for_thresh, self.threshold)
            max_d = max(0.01, 1.0 - effective_threshold)
            condensed = squareform(dist, checks=False)
            Z = linkage(condensed, method="complete")
            labels = fcluster(Z, t=max_d, criterion="distance")

            clusters: dict = defaultdict(list)
            for tid, label in zip(valid_ids, labels):
                clusters[label].append(tid)

            for cluster_tracks in clusters.values():
                canon = min(cluster_tracks)
                for tid in cluster_tracks:
                    canonical_map[tid] = canon
                if len(cluster_tracks) > 1:
                    cls_name = config.CLASS_NAMES.get(class_id, f"cls_{class_id}")
                    merged = [t for t in cluster_tracks if t != canon]
                    print(f"  [ReID] Merged {merged} → {canon}  ({cls_name})")

        return canonical_map

    def get_unique_counts(self, collector: TrackCollector,
                          canonical_map: dict) -> dict:
        seen: set = set()
        counts: dict = defaultdict(int)
        for tid, info in collector.tracks.items():
            canon = canonical_map.get(tid, tid)
            if canon not in seen:
                seen.add(canon)
                counts[info["class_name"]] += 1
        return dict(counts)

    def get_unique_objects_detail(self, collector: TrackCollector,
                                  canonical_map: dict) -> list:
        seen: set = set()
        result = []
        for tid in sorted(collector.tracks):
            canon = canonical_map.get(tid, tid)
            if canon not in seen:
                seen.add(canon)
                info = collector.tracks[tid]
                merged = sorted(
                    t for t in collector.tracks
                    if canonical_map.get(t, t) == canon and t != canon
                )
                result.append({
                    "unique_id": int(canon),
                    "class_id": info["class_id"],
                    "class_name": info["class_name"],
                    "merged_track_ids": [int(t) for t in merged],
                })
        return result

    def get_best_crop_for_canonical(self, collector: TrackCollector,
                                    canonical_map: dict,
                                    canonical_id: int) -> np.ndarray | None:
        best_score = -1.0
        best_crop = None
        for tid in collector.tracks:
            if canonical_map.get(tid, tid) != canonical_id:
                continue
            score = collector.get_best_crop_score(tid)
            if score > best_score:
                c = collector.get_best_crop(tid)
                if c is not None:
                    best_score = score
                    best_crop = c
        return best_crop


# ── Backwards-compatible wrapper ──────────────────────────────────────────────

class CLIPReIdentifier:
    """
    Drop-in wrapper around the two-pass deduplication system.
    Public API unchanged; add frame= to update_track and call finalize() after loop.

    Usage:
        reid = CLIPReIdentifier(similarity_threshold=0.60)

        # Per processed frame:
        reid.register_frame_tracks(track_ids, frame, frame_idx)
        reid.update_track(tid, cid, crop, bbox, frame=frame)   # pass full frame

        # After video loop:
        reid.finalize()

        counts = reid.get_unique_counts()
        detail = reid.get_unique_objects_detail()
    """

    def __init__(self, similarity_threshold=None, model_name=None):
        thresh = similarity_threshold if similarity_threshold is not None \
            else config.REID_SIMILARITY_THRESHOLD
        top_k = getattr(config, "REID_TOP_K_CROPS", 5)
        self.collector = TrackCollector(top_k_crops=top_k)
        self.deduplicator = PostHocDeduplicator(
            similarity_threshold=thresh,
            top_k_crops=top_k,
            model_name="facebook/dinov2-base",
        )
        self._canonical_map: dict | None = None
        self._frame_shape = None

    def register_frame_tracks(self, track_ids, frame=None, frame_idx: int = 0):
        if frame is not None and self._frame_shape is None:
            self._frame_shape = frame.shape
        self.collector.register_frame(track_ids, frame_idx)

    def update_track(self, track_id: int, class_id: int,
                     crop: np.ndarray, bbox=None, frame=None) -> int:
        """
        Collect crop + scene fingerprint. No merge during video.
        Pass frame= for background fingerprinting (strongly recommended).
        Returns track_id unchanged until finalize() is called.
        """
        if crop is None or crop.size == 0:
            return int(track_id)
        if (crop.shape[0] < config.REID_MIN_CROP_SIZE or
                crop.shape[1] < config.REID_MIN_CROP_SIZE):
            return int(track_id)

        # Partial-detection guard: skip crops where bbox touches frame boundary.
        # Chair entering frame 10% visible → bad embedding → double-count if allowed in.
        # ByteTrack still tracks it normally; we just don't register it for ReID.
        inset = getattr(config, "MIN_BBOX_INSET", 0)
        if inset > 0 and bbox is not None and self._frame_shape is not None:
            fh, fw = self._frame_shape[:2]
            bx1, by1, bx2, by2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            if bx1 <= inset or by1 <= inset or bx2 >= fw - inset or by2 >= fh - inset:
                return int(track_id)

        cls_name = config.CLASS_NAMES.get(int(class_id), f"cls_{int(class_id)}")
        self.collector.add_detection(
            int(track_id), int(class_id), cls_name,
            crop, bbox, self._frame_shape,
        )

        # Background / scene fingerprint — the key spatial signal
        if frame is not None and bbox is not None:
            self.collector.add_background(int(track_id), frame, bbox)

        return int(track_id)

    def finalize(self) -> dict:
        """
        Run post-hoc deduplication. Call ONCE after the video loop ends.
        Returns canonical_map {track_id: canonical_id}.
        """
        print("\n  [ReID] Running post-hoc deduplication...")
        self._canonical_map = self.deduplicator.deduplicate(self.collector)
        n_raw = len(self.collector.tracks)
        n_unique = len(set(self._canonical_map.values()))
        n_bg = sum(1 for t in self.collector.tracks
                   if t in self.collector.bg_fingerprints)
        print(f"  [ReID] {n_raw} raw tracks → {n_unique} unique objects "
              f"({n_raw - n_unique} merged, {n_bg}/{n_raw} tracks have scene fingerprint)\n")
        return self._canonical_map

    def get_canonical_id(self, track_id: int) -> int:
        if self._canonical_map is None:
            return int(track_id)
        return self._canonical_map.get(int(track_id), int(track_id))

    def get_unique_counts(self) -> dict:
        if self._canonical_map is None:
            counts: dict = defaultdict(int)
            for info in self.collector.tracks.values():
                counts[info["class_name"]] += 1
            return dict(counts)
        return self.deduplicator.get_unique_counts(self.collector, self._canonical_map)

    def get_unique_objects_detail(self) -> list:
        if self._canonical_map is None:
            self.finalize()
        return self.deduplicator.get_unique_objects_detail(self.collector, self._canonical_map)
