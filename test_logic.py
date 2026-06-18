#!/usr/bin/env python3
"""
Unit tests for the pure matching / calibration / routing logic in face_sorter.

These use synthetic embeddings and need only numpy — no cv2, insightface, or a
GPU — because face_sorter imports the heavy deps lazily. Run with:

    .venv/bin/python -m pytest test_logic.py        # if pytest is installed
    .venv/bin/python test_logic.py                  # plain-python fallback
"""
import numpy as np

import face_sorter as fs


def _unit(*vals):
    v = np.asarray(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


# Three clearly-separated "people" in 4-D space, plus a stranger.
ANA = _unit(1, 0, 0, 0)
JUAN = _unit(0, 1, 0, 0)
EVA = _unit(0, 0, 1, 0)
STRANGER = _unit(0, 0, 0, 1)
# Near-but-distinct variations of each person (high within-person similarity).
ANA2 = _unit(0.97, 0.05, 0.05, 0)
JUAN2 = _unit(0.05, 0.97, 0, 0.05)


def make_gallery():
    return {
        "Ana": np.stack([ANA, ANA2]),
        "Juan": np.stack([JUAN, JUAN2]),
        "Eva": np.stack([EVA]),
    }


def test_cosine_basic():
    assert abs(fs.cosine(ANA, ANA) - 1.0) < 1e-6
    assert abs(fs.cosine(ANA, JUAN) - 0.0) < 1e-6
    # Robust to non-normalised input.
    assert abs(fs.cosine([2, 0, 0, 0], [5, 0, 0, 0]) - 1.0) < 1e-6


def test_single_known_face_matches_one_person():
    g = make_gallery()
    matched = fs.image_matches([ANA], g, threshold=0.5)
    assert set(matched) == {"Ana"}
    assert matched["Ana"] > 0.99


def test_multi_person_photo_matches_all_present():
    g = make_gallery()
    matched = fs.image_matches([ANA, JUAN], g, threshold=0.5)
    assert set(matched) == {"Ana", "Juan"}


def test_known_plus_unknown_routes_to_known_only():
    g = make_gallery()
    matched = fs.image_matches([ANA, STRANGER], g, threshold=0.5)
    assert set(matched) == {"Ana"}


def test_only_unknown_matches_nobody():
    g = make_gallery()
    matched = fs.image_matches([STRANGER], g, threshold=0.5)
    assert matched == {}


def test_no_faces_matches_nobody():
    g = make_gallery()
    assert fs.image_matches([], g, threshold=0.5) == {}


def test_threshold_is_respected():
    g = make_gallery()
    # ANA2 vs ANA similarity is high (~0.99); a 0.999 cutoff should still match
    # via the exact ANA reference, but a stranger never should.
    assert "Ana" in fs.image_matches([ANA2], g, threshold=0.9)
    assert fs.image_matches([STRANGER], g, threshold=0.3) == {}


def test_calibration_separable():
    cal = fs.calibrate(make_gallery())
    assert cal["within_min"] is not None
    assert cal["cross_max"] is not None
    assert cal["separable"] is True
    # Suggested threshold sits strictly between the two classes.
    assert cal["cross_max"] < cal["suggested"] < cal["within_min"]


def test_calibration_overlap_warns_and_falls_back():
    # Two "people" that are actually almost identical -> overlap.
    g = {
        "A": np.stack([_unit(1, 0, 0, 0), _unit(0, 1, 0, 0)]),   # very low within-sim
        "B": np.stack([_unit(0.99, 0.14, 0, 0), _unit(0.14, 0.99, 0, 0)]),
    }
    cal = fs.calibrate(g)
    assert cal["separable"] is False
    assert cal["suggested"] == fs.DEFAULT_THRESHOLD
    assert any("OVERLAP" in n for n in cal["notes"])


def test_calibration_single_person():
    cal = fs.calibrate({"Ana": np.stack([ANA, ANA2])})
    assert cal["cross_max"] is None
    assert cal["suggested"] == fs.DEFAULT_THRESHOLD


def test_unique_destination(tmp_path=None):
    import tempfile
    from pathlib import Path
    d = Path(tempfile.mkdtemp())
    p1 = fs.unique_destination(d, "photo.jpg")
    assert p1.name == "photo.jpg"
    p1.write_text("x")
    p2 = fs.unique_destination(d, "photo.jpg")
    assert p2.name == "photo__1.jpg"
    p2.write_text("x")
    p3 = fs.unique_destination(d, "photo.jpg")
    assert p3.name == "photo__2.jpg"


def test_resolve_destination_idempotent():
    import tempfile, shutil
    from pathlib import Path
    src_dir = Path(tempfile.mkdtemp())
    dst_dir = Path(tempfile.mkdtemp())
    src = src_dir / "photo.jpg"
    src.write_bytes(b"AAAA")

    # First placement: fresh name, not already there.
    p1, already1 = fs.resolve_destination(dst_dir, src)
    assert p1.name == "photo.jpg" and already1 is False
    shutil.copy2(src, p1)

    # Re-run with the SAME content: must reuse the existing copy, not duplicate.
    p2, already2 = fs.resolve_destination(dst_dir, src)
    assert p2 == p1 and already2 is True

    # A DIFFERENT file that merely shares the name: fresh __1 name.
    other = src_dir / "other.jpg"
    other.write_bytes(b"BBBBBB")
    # Simulate it arriving as "photo.jpg" by copying then resolving its twin.
    twin = src_dir / "photo.jpg"  # same name, different bytes
    twin.write_bytes(b"BBBBBB")
    p3, already3 = fs.resolve_destination(dst_dir, twin)
    assert p3.name == "photo__1.jpg" and already3 is False


def _run_all():
    """Plain-python runner so the file works without pytest installed."""
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {exc!r}")
        else:
            passed += 1
            print(f"ok   {fn.__name__}")
    print(f"\n{passed}/{len(fns)} tests passed")
    return passed == len(fns)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_all() else 1)
