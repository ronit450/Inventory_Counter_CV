# Spatial Dedup Feasibility (Workstream C, Task C1)

Prototype: `spatial_prototype/prototype_homography.py`
Outputs: `spatial_prototype/out/track_positions.json`, `spatial_prototype/out/pair_distances.csv`, `spatial_prototype/out/summary_stats.json`

Ran against `client_video/client_video.mp4`, `FRAME_SKIP=2`, YOLO tracking with the exact conf/iou/imgsz/tracker settings from `config.py` (model: `config.YOLO_MODEL_PATH`), on GPU (CUDA), full run in 49.7s.

## Homography stability

| Metric | Value |
|---|---|
| Total inter-frame homography steps | 1166 |
| Lost frames (fallback to previous H_cum) | 0 |
| Lost % | 0.0% |

Homography estimation was **fully stable** on this footage — every consecutive frame pair had ≥30 tracked optical-flow points and a valid RANSAC homography. This is a handheld phone pan, but it's a comparatively slow, mostly-lateral pan over a warehouse/office floor — not the rotation-heavy case the task worried about. So the "does homography survive this footage" half of the hypothesis is confirmed GO. The dedup-signal half is not (see below).

## Distance statistics (normalized by median class bbox diagonal, "object-widths")

67 tracks total, 312 same-class co-occurring pairs (≥3 shared frames = provably distinct objects per the existing hard-veto rule), 319 same-class pairs total with valid panorama projections.

| Group | n | min | median | max |
|---|---|---|---|---|
| Co-occurring (provably distinct) | 39 | 0.159 | 3.278 | 204.6 |
| All same-class pairs | 319 | 0.087 | 2.365 | 209.1 |
| Non-co-occurring (candidate duplicates) | 280 | 0.087 | 2.306 | 209.1 |

Key breakdown — fraction of co-occurring (**known-distinct**) pairs at each distance band, which is what a veto/bonus threshold would have to clear without also vetoing real duplicates:

- **12 / 39 (31%)** of co-occurring pairs sit **< 1.0 object-widths** apart.
- **15 / 39 (38%)** sit **< 1.5 object-widths** apart (the hypothesis's proposed threshold).
- Lowest-distance co-occurring pairs by class: Office Chairs 11↔12 = 0.159, Desks 83↔95 = 0.240, Filing Cabinets 68↔75 = 0.443, Office Chairs 28↔29 = 0.458, Office Chairs 11↔15 = 0.493.

For comparison, the non-co-occurring (candidate-duplicate) pool has **101/280 (36%)** of pairs also under 1.5 object-widths — almost the identical rate as the known-distinct pool. The two distributions are not separated: median 3.28 vs 2.31, but full overlap 0.09–209 in both, and no distance cutoff sits with known-distinct pairs cleanly above it and candidate-duplicate pairs cleanly below it.

## Verdict: NO-GO

**Panorama distance from cumulative homography is not a usable dedup signal on this video**, despite homography itself being stable. Root cause: many genuinely distinct, co-occurring objects of the same class sit physically adjacent in the room (chairs pushed under neighboring desks, desks in a row, filing cabinets against the same wall) — adjacency in real-world floor coordinates is common for office furniture, so "close in panorama space" does not imply "same object." A third of provably-distinct pairs land under 1 object-width apart, which is tighter than the proposed 1.5-object-width separation hypothesis required. Using panorama distance as a hard veto would incorrectly suppress real distinct objects (worse under-counting, the exact failure mode the client already flagged as costliest); using it as a soft merge bonus would incorrectly encourage merging genuinely separate but nearby objects, with no clean threshold to tune against. The bottom-center-of-largest-bbox point estimate also has no depth/perspective correction, so two objects at different depths but similar image position can project to very different panorama distances even when physically adjacent, and vice versa — adding noise on top of the adjacency problem.

No integration design is provided per the GO/NO-GO branching in the task (this section intentionally omitted — see task spec: "if GO — a 10-line integration design").

## What would change the verdict

- A video with more physical separation between identical-class objects (rare in dense office layouts) would show a real gap.
- True 3D localization (depth from stereo/monocular depth or ground-plane calibration) instead of raw panorama-pixel distance would remove the depth-confound noise, but that's a materially bigger effort than this prototype's scope.
- Not attempted here since the primary blocker (co-located distinct furniture) is a property of the room, not the projection method.

## Files created

- `spatial_prototype/prototype_homography.py`
- `spatial_prototype/out/track_positions.json`
- `spatial_prototype/out/pair_distances.csv`
- `spatial_prototype/out/summary_stats.json`
- `spatial_prototype/FEASIBILITY.md`
