"""
CLIP-based Re-Identification Module.

Uses CLIP embeddings to compute visual similarity between object crops.
When the camera rotates and revisits the same area, this module identifies
objects that were already counted and prevents double-counting.
"""

import numpy as np
import torch
from PIL import Image
from collections import defaultdict
from scipy.spatial.distance import cosine

import config


class CLIPReIdentifier:
    """
    Manages object embeddings and deduplication using CLIP features.
    
    Strategy:
    - For each tracked object, we store CLIP embeddings of its crops over time.
    - When a new track appears, we compare its embedding against all existing
      tracks of the SAME class.
    - If similarity > threshold, we merge them (mark new track as duplicate).
    """

    def __init__(self, similarity_threshold=None, model_name=None):
        self.similarity_threshold = similarity_threshold or config.REID_SIMILARITY_THRESHOLD
        self.model_name = model_name or config.CLIP_MODEL_NAME
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self._load_model()

        # track_id -> list of embeddings (numpy arrays)
        self.track_embeddings = defaultdict(list)
        # track_id -> class_id
        self.track_classes = {}
        # track_id -> merged_into_track_id (dedup mapping)
        self.merge_map = {}
        # Track how many crops we've stored per track (limit memory)
        self.max_embeddings_per_track = 10
        # Pairs of track IDs that appeared in the same frame at least once.
        # If two tracks co-occurred, they are guaranteed to be different
        # physical objects and must NEVER be merged (even if CLIP says they
        # look identical — e.g. 5 identical office chairs).
        self._cooccurring = set()  # set of frozenset({id_a, id_b})

    def _load_model(self):
        """Load CLIP model from HuggingFace."""
        print(f"[ReID] Loading CLIP model: {self.model_name}")
        from transformers import CLIPProcessor, CLIPModel
        
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        print(f"[ReID] CLIP loaded on {self.device}")

    @torch.no_grad()
    def get_embedding(self, crop_image: Image.Image) -> np.ndarray:
        """
        Compute CLIP image embedding for a crop.
        
        Args:
            crop_image: PIL Image of the object crop
            
        Returns:
            Normalized embedding vector (numpy array)
        """
        inputs = self.processor(images=crop_image, return_tensors="pt").to(self.device)
        features = self.model.get_image_features(**inputs)
        # L2 normalize
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return 1.0 - cosine(emb1, emb2)

    def get_average_embedding(self, track_id: int) -> np.ndarray:
        """Get the average embedding for a track."""
        embeddings = self.track_embeddings[track_id]
        if not embeddings:
            return None
        avg = np.mean(embeddings, axis=0)
        avg = avg / np.linalg.norm(avg)  # re-normalize
        return avg

    def get_canonical_id(self, track_id: int) -> int:
        """Follow the merge chain to get the canonical (original) track ID."""
        visited = set()
        current = track_id
        while current in self.merge_map and current not in visited:
            visited.add(current)
            current = self.merge_map[current]
        return current

    def register_frame_tracks(self, track_ids):
        """Record that all given track IDs were visible in the same frame.

        Any pair of tracks that co-occur in a frame are guaranteed to be
        different physical objects and will never be merged by CLIP re-id.
        """
        ids = list(track_ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                self._cooccurring.add(frozenset((ids[i], ids[j])))

    def _have_cooccurred(self, id_a: int, id_b: int) -> bool:
        """Check whether id_a ever appeared in the same frame as id_b,
        or as any track that was already merged into id_b."""
        # Direct check
        if frozenset((id_a, id_b)) in self._cooccurring:
            return True
        # Also check against all tracks that were merged into id_b,
        # because those shared a frame with id_b's group.
        for tid, target in self.merge_map.items():
            if self.get_canonical_id(tid) == id_b:
                if frozenset((id_a, tid)) in self._cooccurring:
                    return True
        return False

    def update_track(self, track_id: int, class_id: int, crop: np.ndarray) -> int:
        """
        Update track with a new crop and check for duplicates.
        
        Args:
            track_id: Track ID from the tracker
            class_id: Detected class ID
            crop: Object crop as numpy array (BGR from OpenCV)
            
        Returns:
            Canonical track ID (may differ from input if merged)
        """
        # If already merged, return canonical ID
        if track_id in self.merge_map:
            return self.get_canonical_id(track_id)

        # Convert crop to PIL
        crop_rgb = crop[:, :, ::-1]  # BGR -> RGB
        pil_crop = Image.fromarray(crop_rgb)

        # Skip tiny crops
        if pil_crop.width < config.REID_MIN_CROP_SIZE or pil_crop.height < config.REID_MIN_CROP_SIZE:
            self.track_classes[track_id] = class_id
            return track_id

        # Compute embedding
        embedding = self.get_embedding(pil_crop)

        # Store class
        self.track_classes[track_id] = class_id

        # If this is a new track, check against existing tracks of same class
        if track_id not in self.track_embeddings:
            best_match_id = None
            best_similarity = 0.0

            for existing_id, existing_class in self.track_classes.items():
                if existing_id == track_id:
                    continue
                if existing_class != class_id:
                    continue
                if existing_id in self.merge_map:
                    continue
                if not self.track_embeddings[existing_id]:
                    continue
                # CRITICAL: never merge tracks that were seen in the
                # same frame — they are different physical objects.
                if self._have_cooccurred(track_id, existing_id):
                    continue

                avg_emb = self.get_average_embedding(existing_id)
                if avg_emb is None:
                    continue

                sim = self.compute_similarity(embedding, avg_emb)
                if sim > best_similarity:
                    best_similarity = sim
                    best_match_id = existing_id

            if best_match_id is not None and best_similarity >= self.similarity_threshold:
                # Merge: this track is a duplicate of best_match_id
                self.merge_map[track_id] = best_match_id
                # Also add this embedding to the canonical track
                if len(self.track_embeddings[best_match_id]) < self.max_embeddings_per_track:
                    self.track_embeddings[best_match_id].append(embedding)
                print(f"[ReID] Track {track_id} merged into Track {best_match_id} "
                      f"(class={config.CLASS_NAMES.get(class_id, class_id)}, sim={best_similarity:.3f})")
                return best_match_id

        # Store embedding for this track
        if len(self.track_embeddings[track_id]) < self.max_embeddings_per_track:
            self.track_embeddings[track_id].append(embedding)

        return track_id

    def get_unique_counts(self) -> dict:
        """
        Get the final unique object counts per class.
        
        Returns:
            dict: {class_name: count}
        """
        # Find all canonical (non-merged) tracks
        canonical_tracks = set()
        for track_id in self.track_classes:
            canonical = self.get_canonical_id(track_id)
            canonical_tracks.add(canonical)

        # Count per class
        counts = defaultdict(int)
        for track_id in canonical_tracks:
            class_id = self.track_classes.get(track_id)
            if class_id is not None:
                class_name = config.CLASS_NAMES.get(class_id, f"class_{class_id}")
                counts[class_name] += 1

        return dict(counts)

    def get_unique_objects_detail(self) -> list:
        """
        Get detailed info about each unique object.
        
        Returns:
            list of dicts with object details
        """
        canonical_tracks = set()
        for track_id in self.track_classes:
            canonical = self.get_canonical_id(track_id)
            canonical_tracks.add(canonical)

        objects = []
        for track_id in sorted(canonical_tracks):
            class_id = self.track_classes.get(track_id)
            if class_id is not None:
                # Find all tracks merged into this one
                merged_from = [tid for tid, target in self.merge_map.items()
                               if self.get_canonical_id(tid) == track_id]
                objects.append({
                    "unique_id": int(track_id),
                    "class_id": int(class_id),
                    "class_name": config.CLASS_NAMES.get(class_id, f"class_{class_id}"),
                    "num_embeddings": len(self.track_embeddings.get(track_id, [])),
                    "merged_track_ids": [int(t) for t in merged_from],
                })

        return objects
