#!/usr/bin/env python3
"""
face_sorter.py — Sort a folder of photos by who appears in them, using face recognition.

Pipeline
--------
1. Build a "gallery" of face embeddings from labelled reference folders
   (one subfolder per person, named after the person).
2. Calibrate a similarity threshold from the labelled data itself, so the
   match cutoff is grounded in *your* faces rather than a blind default.
3. Scan the input folder (recursively, including subfolders), detect every
   face in every image, and place each image into the folder of every known
   person it contains.

Routing rules
-------------
- A photo with Ana and Juan  -> copied to both  Ana/  and Juan/
- A photo with a known + an unknown face -> goes to the known person's folder(s)
- A photo with faces but none matching   -> unknown/
- A photo with no detectable face         -> _no_face/
- A photo that fails to load (corrupt/unreadable) -> _failed/

Engine
------
InsightFace 'buffalo_l' (RetinaFace detector + ArcFace w600k_r50 embeddings)
run via onnxruntime. Uses the GPU (CUDAExecutionProvider) when available and
*loudly warns* if it silently falls back to CPU.

Heavy dependencies (cv2, insightface) are imported lazily inside the functions
that need them, so the matching / calibration logic can be unit-tested with
numpy alone.

Usage
-----
    python face_sorter.py \
        --known   /path/to/known      # subfolders: known/Ana, known/Juan, ...
        --input   /path/to/to_sort    # scanned recursively
        --output  /path/to/results    # results/Ana, results/unknown, ...
        [--threshold 0.50]            # omit to use the calibrated suggestion
        [--action copy|move]          # default: copy (non-destructive)
        [--det-size 640,320,1024]     # multi-scale detector sizes (big + small faces)
        [--min-det-score 0.5]         # ignore low-confidence face detections
        [--cpu]                       # force CPU even if a GPU is present
        [--no-recursive]              # scan only the top level of --input
        [--dry-run]                   # report only; copy/move nothing
        [--rebuild-gallery]           # ignore the cached gallery.npz
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import warnings
from pathlib import Path

import numpy as np

# Silence a noisy internal insightface deprecation warning (skimage estimate()).
warnings.filterwarnings("ignore", category=FutureWarning, module=r"insightface.*")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
HEIC_EXTS = {".heic", ".heif"}
DEFAULT_THRESHOLD = 0.50           # precision-leaning fallback for w600k_r50
THRESHOLD_CLAMP = (0.20, 0.75)     # sane bounds for any auto-suggested threshold
MODEL_NAME = "buffalo_l"
UNKNOWN_DIR = "unknown"
NO_FACE_DIR = "_no_face"
FAILED_DIR = "_failed"
GALLERY_CACHE = "gallery.npz"
REPORT_CSV = "report.csv"


# ===========================================================================
# Pure logic (numpy only) — unit-testable without cv2 / insightface
# ===========================================================================
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Robust to non-normalised input."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def image_matches(face_embs: list[np.ndarray],
                  gallery: dict[str, np.ndarray],
                  threshold: float) -> dict[str, float]:
    """
    Given the embeddings of every face in one image, return the persons whose
    best similarity to any face meets the threshold, mapped to that best score.

    `gallery[person]` is an (k, d) matrix of L2-normalised reference embeddings.
    Face embeddings are assumed L2-normalised too, so similarity is a dot product.
    """
    if not face_embs:
        return {}
    faces = np.asarray(np.stack(face_embs), dtype=np.float32)   # (n, d)
    matched: dict[str, float] = {}
    for person, refs in gallery.items():
        refs = np.asarray(refs, dtype=np.float32)
        if refs.ndim == 1:
            refs = refs[None, :]
        sims = faces @ refs.T                                   # (n, k)
        best = float(sims.max()) if sims.size else 0.0
        if best >= threshold:
            matched[person] = best
    return matched


def calibrate(gallery: dict[str, np.ndarray]) -> dict:
    """
    Derive a threshold from the labelled references.

    Returns a dict with:
        cross_max   : highest similarity between two *different* people
        within_min  : lowest similarity within the *same* person (needs >=2 refs)
        suggested   : recommended threshold (max-margin midpoint when separable)
        separable   : True when the two classes don't overlap
        notes       : human-readable diagnostics
    """
    persons = sorted(gallery)
    notes: list[str] = []

    # within-person minimum similarity (over persons with >= 2 references)
    within_min = None
    multi_ref = 0
    for p in persons:
        refs = np.asarray(gallery[p], dtype=np.float32)
        if refs.shape[0] < 2:
            continue
        multi_ref += 1
        sims = refs @ refs.T
        iu = np.triu_indices(refs.shape[0], k=1)               # upper triangle only
        m = float(sims[iu].min())
        within_min = m if within_min is None else min(within_min, m)

    # cross-person maximum similarity (over all person pairs)
    cross_max = None
    for i in range(len(persons)):
        ri = np.asarray(gallery[persons[i]], dtype=np.float32)
        for j in range(i + 1, len(persons)):
            rj = np.asarray(gallery[persons[j]], dtype=np.float32)
            m = float((ri @ rj.T).max())
            cross_max = m if cross_max is None else max(cross_max, m)

    separable = False
    if within_min is not None and cross_max is not None:
        if within_min > cross_max:
            suggested = (within_min + cross_max) / 2.0
            separable = True
            notes.append(
                f"Classes separable: within-person min {within_min:.3f} > "
                f"cross-person max {cross_max:.3f}.")
        else:
            suggested = DEFAULT_THRESHOLD
            notes.append(
                f"OVERLAP: within-person min {within_min:.3f} <= cross-person max "
                f"{cross_max:.3f}. The references for some people are not cleanly "
                f"separable — check for mislabelled or low-quality reference photos. "
                f"Falling back to default {DEFAULT_THRESHOLD:.2f}.")
    else:
        suggested = DEFAULT_THRESHOLD
        if cross_max is None:
            notes.append("Only one person in the gallery — no cross-person data; "
                         f"using default {DEFAULT_THRESHOLD:.2f}.")
        if within_min is None:
            notes.append("No person has >=2 reference photos — cannot measure "
                         f"within-person spread; using default {DEFAULT_THRESHOLD:.2f}. "
                         "Add 2-3 varied photos per person for a calibrated threshold.")

    suggested = float(min(max(suggested, THRESHOLD_CLAMP[0]), THRESHOLD_CLAMP[1]))
    return {
        "cross_max": cross_max,
        "within_min": within_min,
        "suggested": suggested,
        "separable": separable,
        "multi_ref_persons": multi_ref,
        "notes": notes,
    }


def unique_destination(dest_dir: Path, filename: str) -> Path:
    """Return a non-colliding path inside dest_dir, suffixing __1, __2, ... ."""
    candidate = dest_dir / filename
    if not candidate.exists():
        return candidate
    stem, suffix = candidate.stem, candidate.suffix
    i = 1
    while True:
        candidate = dest_dir / f"{stem}__{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def resolve_destination(dest_dir: Path, src: Path):
    """
    Decide where `src` goes inside `dest_dir`, idempotently.

    Returns (path, already_there). If a byte-identical copy of `src` is already
    present under its name-family (name, name__1, ...), returns that path with
    already_there=True so the caller can skip the copy — making re-runs safe
    (no a_flip__1.jpg, __2.jpg growth). A different photo that merely shares a
    filename still gets a fresh __N name.
    """
    import filecmp

    base = dest_dir / src.name
    if not base.exists():
        return base, False
    try:
        src_size = src.stat().st_size
    except OSError:
        return unique_destination(dest_dir, src.name), False

    stem, suffix = base.stem, base.suffix
    family = [base]
    i = 1
    while (dest_dir / f"{stem}__{i}{suffix}").exists():
        family.append(dest_dir / f"{stem}__{i}{suffix}")
        i += 1
    for cand in family:
        try:
            if cand.stat().st_size == src_size and \
                    filecmp.cmp(str(src), str(cand), shallow=False):
                return cand, True            # identical copy already here
        except OSError:
            continue
    return dest_dir / f"{stem}__{i}{suffix}", False


# ===========================================================================
# Image loading (lazy cv2 / pillow-heif) — everything normalised to BGR
# ===========================================================================
def load_bgr(path: Path):
    """
    Load any supported image as a BGR uint8 numpy array (what InsightFace wants),
    or return None if it cannot be read. Uses np.fromfile + cv2.imdecode so
    non-ASCII paths work; converts HEIC (RGB via pillow-heif) to BGR.
    """
    import cv2  # lazy

    ext = path.suffix.lower()
    try:
        if ext in HEIC_EXTS:
            try:
                import pillow_heif  # noqa: F401
                from PIL import Image
                pillow_heif.register_heif_opener()
                rgb = np.array(Image.open(path).convert("RGB"))
                return np.ascontiguousarray(rgb[:, :, ::-1])     # RGB -> BGR
            except Exception:
                return None
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)               # decodes to BGR
        return img                                               # may be None
    except Exception:
        return None


# ===========================================================================
# Engine
# ===========================================================================
def build_app(use_gpu: bool, det_sizes: list, det_thresh: float):
    """
    Construct and prepare an InsightFace FaceAnalysis app. Returns
    (app, gpu_active: bool). Warns loudly if GPU was requested but the
    CUDA provider did not actually initialise (the classic WSL2 cuDNN trap).

    `det_sizes` is a list of (w, h) detector input sizes. insightface runs
    detection at every size and NMS-merges the results, so a single pass
    catches both large/close faces (small sizes) and tiny faces in group
    photos (large sizes).
    """
    import onnxruntime as ort  # lazy
    from insightface.app import FaceAnalysis  # lazy

    if use_gpu:
        # Load the CUDA 12 / cuDNN 9 runtime libraries shipped as nvidia-*-cu12
        # wheels. Without this, onnxruntime-gpu can't find libcublasLt.so.12 etc.
        # on a clean WSL2 box and silently falls back to CPU.
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception as exc:
                print(f"  (onnxruntime.preload_dlls failed: {exc})", file=sys.stderr)
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ctx_id = 0
    else:
        providers = ["CPUExecutionProvider"]
        ctx_id = -1

    app = FaceAnalysis(name=MODEL_NAME, providers=providers)
    app.prepare(ctx_id=ctx_id, det_thresh=det_thresh, det_size=det_sizes)

    # Inspect what onnxruntime *actually* loaded, not what we asked for.
    active = set()
    try:
        for model in app.models.values():
            sess = getattr(model, "session", None)
            if sess is not None:
                active.update(sess.get_providers())
    except Exception:
        pass
    gpu_active = "CUDAExecutionProvider" in active

    if use_gpu and not gpu_active:
        print(
            "\n*** WARNING: GPU was requested but CUDAExecutionProvider is NOT "
            "active — running on CPU (much slower).\n"
            "    Active providers: " + (", ".join(sorted(active)) or "unknown") + "\n"
            "    Likely cause on WSL2: onnxruntime-gpu cannot find the CUDA 12 / "
            "cuDNN 9 runtime libraries, or CPU 'onnxruntime' is shadowing "
            "'onnxruntime-gpu'.\n"
            "    Fix: ensure ONLY onnxruntime-gpu is installed and the CUDA/cuDNN "
            "libs are on the loader path. Use --cpu to silence this.\n",
            file=sys.stderr,
        )
    elif gpu_active:
        print("GPU active: CUDAExecutionProvider is in use.")
    else:
        print("Running on CPU.")
    return app, gpu_active


def largest_face(faces):
    """Pick the most prominent face (largest bbox area) from a detection list."""
    def area(f):
        x1, y1, x2, y2 = f.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return max(faces, key=area)


# ===========================================================================
# Gallery (build + cache)
# ===========================================================================
def iter_images(root: Path, recursive: bool):
    exts = IMAGE_EXTS | HEIC_EXTS
    it = root.rglob("*") if recursive else root.glob("*")
    for p in sorted(it):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _gallery_signature(known_dir: Path, det_sizes: list) -> str:
    """Hash of all reference files (path/size/mtime) + model + det sizes."""
    h = hashlib.sha256()
    h.update(f"{MODEL_NAME}|{det_sizes}".encode())
    for person_dir in sorted(p for p in known_dir.iterdir() if p.is_dir()):
        for img in iter_images(person_dir, recursive=True):
            st = img.stat()
            h.update(f"{img.relative_to(known_dir)}|{st.st_size}|{int(st.st_mtime)}".encode())
    return h.hexdigest()


def load_gallery_cache(output_dir: Path, signature: str):
    cache = output_dir / GALLERY_CACHE
    if not cache.exists():
        return None
    try:
        data = np.load(cache, allow_pickle=False)
        if str(data["__signature__"]) != signature:
            return None
        gallery = {k[3:]: data[k] for k in data.files if k.startswith("P::")}
        return gallery or None
    except Exception:
        return None


def save_gallery_cache(output_dir: Path, signature: str, gallery: dict[str, np.ndarray]):
    arrays = {f"P::{p}": v for p, v in gallery.items()}
    arrays["__signature__"] = np.array(signature)
    try:
        np.savez(output_dir / GALLERY_CACHE, **arrays)
    except Exception as exc:
        print(f"  (could not write gallery cache: {exc})", file=sys.stderr)


def build_gallery(app, known_dir: Path, min_det_score: float) -> dict[str, np.ndarray]:
    """
    Embed each person's reference photos. For a reference image, the most
    prominent face is used (the folder is labelled as that one person).
    Returns {person: (k, d) float32 matrix of normalised embeddings}.
    """
    gallery: dict[str, list[np.ndarray]] = {}
    person_dirs = sorted(p for p in known_dir.iterdir() if p.is_dir())
    if not person_dirs:
        raise SystemExit(f"No person subfolders found in {known_dir}")

    for person_dir in person_dirs:
        person = person_dir.name
        embs: list[np.ndarray] = []
        n_imgs = n_noface = n_multi = 0
        for img_path in iter_images(person_dir, recursive=True):
            n_imgs += 1
            img = load_bgr(img_path)
            if img is None:
                print(f"  [{person}] could not read {img_path.name}", file=sys.stderr)
                continue
            faces = [f for f in app.get(img) if f.det_score >= min_det_score]
            if not faces:
                n_noface += 1
                continue
            if len(faces) > 1:
                n_multi += 1
            embs.append(largest_face(faces).normed_embedding.astype(np.float32))
        if not embs:
            print(f"  WARNING: no usable face in any reference for '{person}' "
                  f"— this person will never be matched.", file=sys.stderr)
            continue
        gallery[person] = np.stack(embs)
        extra = []
        if n_noface:
            extra.append(f"{n_noface} with no face")
        if n_multi:
            extra.append(f"{n_multi} multi-face (used largest)")
        suffix = f"  [{'; '.join(extra)}]" if extra else ""
        print(f"  {person}: {len(embs)} reference embedding(s) from {n_imgs} image(s){suffix}")

    if not gallery:
        raise SystemExit("Gallery is empty — no usable reference faces found.")
    return gallery


# ===========================================================================
# Classification
# ===========================================================================
def place_image(src: Path, dest_dirs: list[Path], action: str, dry_run: bool) -> list[Path]:
    """Copy src into each dest dir (idempotent); if action=='move', remove src once."""
    import shutil  # lazy-ish; cheap but keep top tidy

    finals: list[Path] = []
    for d in dest_dirs:
        if not dry_run:
            d.mkdir(parents=True, exist_ok=True)
        dst, already_there = resolve_destination(d, src)
        if not already_there and not dry_run:
            shutil.copy2(src, dst)
        finals.append(dst)
    if action == "move" and not dry_run and finals:
        try:
            src.unlink()
        except OSError as exc:
            print(f"  (could not remove source {src}: {exc})", file=sys.stderr)
    return finals


def classify(app, gallery, input_dir, output_dir, threshold, action,
             min_det_score, recursive, dry_run):
    from tqdm import tqdm  # lazy

    output_dir.mkdir(parents=True, exist_ok=True)
    # If --output is nested inside --input, never re-ingest already-sorted
    # images (recursive scan is the default, so a re-run would otherwise pick
    # up results/PersonA/*.jpg and re-copy them as __1.jpg, __2.jpg, ...).
    out_resolved = output_dir.resolve()
    images = [p for p in iter_images(input_dir, recursive)
              if not p.resolve().is_relative_to(out_resolved)]
    if not images:
        print(f"No images found in {input_dir}", file=sys.stderr)
        return
    report_rows = [("image", "n_faces", "matched_persons", "best_score",
                    "status", "destinations")]
    stats = {"matched": 0, "unknown": 0, "no_face": 0, "failed": 0}

    for img_path in tqdm(images, desc="Classifying", unit="img"):
        img = load_bgr(img_path)
        if img is None:
            dests = place_image(img_path, [output_dir / FAILED_DIR], action, dry_run)
            stats["failed"] += 1
            report_rows.append((str(img_path), 0, "", "", "failed",
                                ";".join(str(d) for d in dests)))
            continue

        faces = [f for f in app.get(img) if f.det_score >= min_det_score]
        if not faces:
            dests = place_image(img_path, [output_dir / NO_FACE_DIR], action, dry_run)
            stats["no_face"] += 1
            report_rows.append((str(img_path), 0, "", "", "no_face",
                                ";".join(str(d) for d in dests)))
            continue

        embs = [f.normed_embedding.astype(np.float32) for f in faces]
        matched = image_matches(embs, gallery, threshold)

        if matched:
            persons = sorted(matched, key=lambda p: matched[p], reverse=True)
            dests = place_image(img_path, [output_dir / p for p in persons], action, dry_run)
            stats["matched"] += 1
            best = max(matched.values())
            report_rows.append((str(img_path), len(faces),
                                "|".join(persons), f"{best:.3f}", "matched",
                                ";".join(str(d) for d in dests)))
        else:
            dests = place_image(img_path, [output_dir / UNKNOWN_DIR], action, dry_run)
            stats["unknown"] += 1
            report_rows.append((str(img_path), len(faces), "", "", "unknown",
                                ";".join(str(d) for d in dests)))

    # Write report
    import csv
    with open(output_dir / REPORT_CSV, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(report_rows)

    print("\n=== Summary ===")
    print(f"  Total images   : {len(images)}")
    print(f"  Matched a person: {stats['matched']}")
    print(f"  Unknown (faces, no match): {stats['unknown']}")
    print(f"  No face detected: {stats['no_face']}")
    print(f"  Failed to load  : {stats['failed']}")
    print(f"  Report          : {output_dir / REPORT_CSV}")
    if dry_run:
        print("  (DRY RUN — no files were copied or moved.)")


# ===========================================================================
# Main
# ===========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Sort photos by recognised person using InsightFace.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--known", required=True, type=Path,
                    help="Folder of reference subfolders (one per person, named after them).")
    ap.add_argument("--input", required=True, type=Path,
                    help="Folder of images to classify (scanned recursively by default).")
    ap.add_argument("--output", required=True, type=Path,
                    help="Destination root for per-person / unknown / _no_face folders.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Cosine-similarity match cutoff. Omit to use the calibrated value.")
    ap.add_argument("--action", choices=["copy", "move"], default="copy")
    ap.add_argument("--det-size", type=str, default="640,320,1024",
                    help="Comma-separated detector input sizes for multi-scale "
                         "detection (large + small faces). Use one value (e.g. 640) "
                         "to force a single, faster scale.")
    ap.add_argument("--min-det-score", type=float, default=0.5,
                    help="Detector confidence floor; ignore faces below this.")
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if a GPU exists.")
    ap.add_argument("--no-recursive", action="store_true",
                    help="Scan only the top level of --input (default is recursive).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report only; do not copy or move any files.")
    ap.add_argument("--rebuild-gallery", action="store_true",
                    help="Ignore any cached gallery.npz and re-embed references.")
    args = ap.parse_args(argv)

    for d, label in [(args.known, "--known"), (args.input, "--input")]:
        if not d.is_dir():
            raise SystemExit(f"{label} is not a directory: {d}")
    if args.output.resolve() == args.input.resolve():
        raise SystemExit("--output must not be the same folder as --input.")
    args.output.mkdir(parents=True, exist_ok=True)
    recursive = not args.no_recursive

    try:
        det_sizes = [(int(s), int(s)) for s in args.det_size.split(",") if s.strip()]
        if not det_sizes:
            raise ValueError
    except ValueError:
        raise SystemExit(f"--det-size must be comma-separated integers, got: {args.det_size!r}")

    print(f"Initialising InsightFace '{MODEL_NAME}' (det_size={det_sizes}) ...")
    app, _ = build_app(use_gpu=not args.cpu, det_sizes=det_sizes,
                       det_thresh=args.min_det_score)

    # --- Gallery (cached) ---
    signature = _gallery_signature(args.known, det_sizes)
    gallery = None if args.rebuild_gallery else load_gallery_cache(args.output, signature)
    if gallery is not None:
        print(f"Loaded cached gallery for {len(gallery)} person(s).")
    else:
        print("Building gallery from reference folders ...")
        gallery = build_gallery(app, args.known, args.min_det_score)
        save_gallery_cache(args.output, signature, gallery)

    # --- Calibration ---
    cal = calibrate(gallery)
    print("\n=== Threshold calibration ===")
    cm = "n/a" if cal["cross_max"] is None else f"{cal['cross_max']:.3f}"
    wm = "n/a" if cal["within_min"] is None else f"{cal['within_min']:.3f}"
    print(f"  cross-person max similarity : {cm}")
    print(f"  within-person min similarity: {wm}")
    print(f"  suggested threshold         : {cal['suggested']:.3f} "
          f"({'separable' if cal['separable'] else 'NOT cleanly separable'})")
    for note in cal["notes"]:
        print(f"  - {note}")

    if args.threshold is None:
        threshold = cal["suggested"]
        print(f"\nUsing calibrated threshold: {threshold:.3f} "
              f"(override with --threshold).")
    else:
        threshold = args.threshold
        print(f"\nUsing user threshold: {threshold:.3f} "
              f"(calibration suggested {cal['suggested']:.3f}).")

    # --- Classify ---
    print()
    classify(app, gallery, args.input, args.output, threshold, args.action,
             args.min_det_score, recursive, args.dry_run)


if __name__ == "__main__":
    main()
