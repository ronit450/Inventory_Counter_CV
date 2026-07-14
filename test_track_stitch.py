# test_track_stitch.py
"""Unit tests for track_stitch.stitch_tracks. Run: python3 -m pytest test_track_stitch.py -v"""
from track_stitch import stitch_tracks


def _t(cls, first, last, first_box, last_box):
    return {"class_id": cls, "first_seen": first, "last_seen": last,
            "first_box": first_box, "last_box": last_box}

BOX = (100, 100, 200, 200)          # reference box
NEAR = (110, 105, 210, 205)         # IoU ~0.7 with BOX
FAR = (500, 500, 600, 600)          # IoU 0 with BOX


def test_stitches_adjacent_overlapping_same_class():
    tracks = {1: _t(7, 0, 50, BOX, BOX), 2: _t(7, 60, 100, NEAR, NEAR)}
    m = stitch_tracks(tracks, max_gap=90, min_iou=0.3)
    assert m[1] == m[2] == 1


def test_blocks_different_class():
    tracks = {1: _t(7, 0, 50, BOX, BOX), 2: _t(5, 60, 100, NEAR, NEAR)}
    m = stitch_tracks(tracks, max_gap=90, min_iou=0.3)
    assert m[1] != m[2]


def test_blocks_temporal_overlap():
    # overlapping lifespans = co-occurred = provably distinct
    tracks = {1: _t(7, 0, 70, BOX, BOX), 2: _t(7, 60, 100, NEAR, NEAR)}
    m = stitch_tracks(tracks, max_gap=90, min_iou=0.3)
    assert m[1] != m[2]


def test_blocks_gap_too_large():
    tracks = {1: _t(7, 0, 50, BOX, BOX), 2: _t(7, 200, 300, NEAR, NEAR)}
    m = stitch_tracks(tracks, max_gap=90, min_iou=0.3)
    assert m[1] != m[2]


def test_blocks_low_iou():
    tracks = {1: _t(7, 0, 50, BOX, BOX), 2: _t(7, 60, 100, FAR, FAR)}
    # track 1's LAST box vs track 2's FIRST box is what matters
    m = stitch_tracks(tracks, max_gap=90, min_iou=0.3)
    assert m[1] != m[2]


def test_chain_of_three_shares_root():
    tracks = {1: _t(7, 0, 50, BOX, BOX),
              2: _t(7, 60, 100, NEAR, NEAR),
              3: _t(7, 110, 150, NEAR, NEAR)}
    m = stitch_tracks(tracks, max_gap=90, min_iou=0.3)
    assert m[1] == m[2] == m[3] == 1


def test_each_track_stitches_to_one_successor_only():
    # two candidates born after track 1; only the better (smaller gap) joins
    tracks = {1: _t(7, 0, 50, BOX, BOX),
              2: _t(7, 55, 100, NEAR, NEAR),
              3: _t(7, 58, 100, NEAR, NEAR)}
    m = stitch_tracks(tracks, max_gap=90, min_iou=0.3)
    assert m[2] == 1 and m[3] != 1  # 2 wins (gap 5 < 8); 3 co-occurs with 2 anyway
