# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-purpose CLI (`face_sorter.py`) that sorts a folder of photos by who
appears in them, using InsightFace (`buffalo_l`: RetinaFace detector + ArcFace
embeddings) on GPU via `onnxruntime-gpu`. There is no package, no framework —
just the script, its tests, and an installer.

## Commands

```bash
bash setup.sh                       # create .venv, install deps in the correct
                                    # order, run the GPU self-check + unit tests
.venv/bin/python test_logic.py      # run all logic tests (no pytest needed;
                                    # has a built-in plain runner, exits non-zero on fail)
.venv/bin/python -c "import test_logic as t; t.test_calibration_separable()"   # one test

# Run the tool (always via the venv python; there is no global install):
.venv/bin/python face_sorter.py --known ./known --input ./to_sort --output ./results --dry-run
```

There is no build or lint step. `requirements.txt` is reference-only — **use
`setup.sh` to install**, because the onnxruntime install order is load-bearing
(see GPU section).

## Architecture (the parts that span files / aren't obvious)

**Pipeline** (`main` → `build_app` → `build_gallery`/cache → `calibrate` →
`classify`): embed labelled reference folders into a gallery, derive a match
threshold from that labelled data, then detect+embed every face in each input
image and route the image to each matched person's folder.

**Lazy heavy imports are a deliberate constraint.** `cv2`, `insightface`, and
`onnxruntime` are imported *inside* the functions that use them, never at module
top. This keeps the pure decision logic — `cosine`, `image_matches`,
`calibrate`, `resolve_destination`, `unique_destination` — importable with only
numpy, which is why `test_logic.py` runs with no GPU and no model download.
**When adding matching/calibration/routing logic, keep it free of top-level
heavy imports** so it stays unit-testable.

**GPU setup is fragile and the failure mode is silent** (it runs on CPU ~50x
slower). Two traps, both encoded in `setup.sh` and guarded in `build_app`:
1. `insightface` pulls in the CPU `onnxruntime`, which shares the `onnxruntime/`
   package dir with `onnxruntime-gpu` and corrupts it. The install must end with
   `pip uninstall -y onnxruntime onnxruntime-gpu` then
   `pip install --no-deps onnxruntime-gpu`.
2. The CUDA 12 runtime libs aren't present on WSL2 by default (only the driver).
   The `nvidia-*-cu12` wheels supply them, and `build_app` calls
   `onnxruntime.preload_dlls()` before creating any session.
   `build_app` then checks each model's `session.get_providers()` and **warns
   loudly** if GPU was requested but CUDA didn't actually initialise — listing a
   provider as *available* is not the same as it being *active*.

**Multi-scale detection.** `--det-size` is a comma-separated list (default
`640,320,1024`) passed to `prepare()` as a list of sizes; insightface runs
detection at every scale and NMS-merges. This exists because a single det_size
misses faces that fill the frame (large/close) — empirically a 640-only pass
detected 0 faces on full-frame portraits.

**Threshold / "90% confidence."** ArcFace yields a cosine similarity, not a
probability. `calibrate` measures within-person min vs cross-person max
similarity across the labelled gallery and suggests a max-margin midpoint when
they separate (else warns and falls back to `DEFAULT_THRESHOLD`). Calibration
cannot measure out-of-gallery strangers — precision against impostors is the
user's `--threshold` knob.

**Routing buckets** (`classify`): a photo goes to *every* matched person's
folder; faces-but-no-match → `unknown/`; no face → `_no_face/`; unreadable →
`_failed/`. Recursive input scan is the default.

**Re-run safety / idempotency.** `resolve_destination` byte-compares against any
same-named file already at the destination and skips the copy if identical, so
re-runs don't grow `name__1.jpg`, `__2.jpg`. The input scan skips anything under
the resolved `--output` dir (so a nested output isn't re-ingested), and
`output == input` is rejected. `copy` is the default; `move` copies to N
destinations then deletes the source exactly once.

**Gallery cache.** Reference embeddings are cached to `<output>/gallery.npz`,
keyed by a signature of reference file paths/sizes/mtimes + model name +
det_sizes. Change any of those (or pass `--rebuild-gallery`) to force a rebuild.

**Image loading** (`load_bgr`) normalises everything to BGR for insightface:
`np.fromfile` + `cv2.imdecode` (handles non-ASCII paths), HEIC via `pillow-heif`
converted RGB→BGR. Returns `None` on failure rather than raising.

## Environment

Python 3.12 venv at `.venv/`. Target machine is WSL2 with an RTX 4090; the
tested stack is numpy 2.4.6 / insightface 1.0.1 / onnxruntime-gpu 1.26.0 /
CUDA 12.9 + cuDNN 9.23. The first run downloads the `buffalo_l` model (~280 MB)
to `~/.insightface/models/`.
