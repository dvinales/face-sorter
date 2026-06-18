# face-sorter

[![Repo](https://img.shields.io/badge/GitHub-dvinales%2Fface--sorter-181717?logo=github)](https://github.com/dvinales/face-sorter)

Sort a folder of photos by **who appears in them**, using GPU face recognition
(InsightFace `buffalo_l` — RetinaFace detector + ArcFace embeddings).

## 1. Install (one time)

```bash
cd ~/face-sorter
bash setup.sh
```

This creates `.venv/`, installs the dependencies, and runs a **GPU self-check**
that confirms your RTX 4090 is actually used (and warns if it silently falls
back to CPU). If insightface fails to compile, run
`sudo apt-get install -y build-essential python3-dev` and re-run `setup.sh`.

## 2. Arrange your folders

```
known/                 # reference faces — one subfolder per person, named after them
  Ana/   ana1.jpg ana2.jpg ...      (2–3 varied photos per person = best results)
  Juan/  juan1.jpg ...
to_sort/               # the photos to classify (subfolders are scanned too)
  ... lots of images, possibly in nested folders ...
```

## 3. Run

Start with a **dry run** to see how things sort without touching any files:

```bash
.venv/bin/python face_sorter.py \
    --known   ./known \
    --input   ./to_sort \
    --output  ./results \
    --dry-run
```

Inspect `results/report.csv` (and the printed calibration) and tune. When happy:

```bash
.venv/bin/python face_sorter.py \
    --known ./known --input ./to_sort --output ./results
```

### Output layout

```
results/
  Ana/         photos containing Ana   (a photo with Ana+Juan appears in both)
  Juan/
  unknown/     photos with faces but no known match
  _no_face/    photos where no face was detected
  _failed/     photos that could not be read (corrupt / unsupported)
  report.csv   per-image: faces found, matched persons, best score, destination
  gallery.npz  cached reference embeddings (auto; speeds up re-runs)
```

## The threshold ("90% confident")

ArcFace outputs a **cosine similarity**, not a calibrated probability. The script
calibrates a cutoff from *your own* labelled references:

- **cross-person max** — how similar two *different* people's photos get (a lower bound).
- **within-person min** — how dissimilar the *same* person's photos get (an upper bound).

If those don't overlap, it prints a `suggested threshold` at the maximum-margin
midpoint and uses it automatically. To be stricter (fewer false matches) raise it:

```bash
--threshold 0.55      # or 0.60 for very high precision
```

If the two ranges **overlap**, the script warns loudly — that usually means a
mislabelled or poor reference photo for someone. Fix the references and re-run
(use `--rebuild-gallery` after changing reference photos).

**Two limits to know (and how to handle them):**

1. Calibration only measures the people *in your gallery*. A **stranger who
   happens to resemble someone** is never measured, so a too-low threshold can
   leak strangers into a person's folder. **Always `--dry-run` first and scan
   `report.csv`** — if a person's folder contains photos of someone else, raise
   the cutoff: `--threshold 0.55` (or `0.60` for very high precision).
2. The more *varied* your reference photos (different days, angles, lighting,
   glasses), the more honest the calibration. A couple of near-identical refs
   will report an over-optimistic separation.

## Useful flags

| Flag | Meaning |
|------|---------|
| `--action move` | move instead of copy (destructive; source folder is emptied) |
| `--threshold X` | force a cutoff instead of the calibrated one |
| `--det-size 640,320,1024` | multi-scale detection (default): big/close faces *and* tiny faces in group photos. Pass one value (e.g. `640`) for a single, faster scale. |
| `--min-det-score 0.6` | be stricter about what counts as a face |
| `--cpu` | force CPU |
| `--no-recursive` | scan only the top level of `--input` |
| `--rebuild-gallery` | re-embed references after you change them |

## Tests

```bash
.venv/bin/python test_logic.py        # matching / calibration / routing logic
```
