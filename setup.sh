#!/usr/bin/env bash
# Set up an isolated environment for face_sorter.py and verify the GPU path.
# Ubuntu 24.04 is PEP-668 "externally managed", so a venv is required.
#
# Install order matters and is deliberate:
#   1. insightface pulls in the *CPU* onnxruntime + numpy as deps.
#   2. We install the CUDA 12 / cuDNN 9 runtime libraries as nvidia-*-cu12 wheels.
#   3. We then remove BOTH onnxruntime builds and reinstall ONLY onnxruntime-gpu
#      with --no-deps. The CPU and GPU builds share the same 'onnxruntime'
#      package directory, so leaving both installed (or uninstalling just one)
#      corrupts the import. This guarantees a single, intact GPU build.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
VENV="$HERE/.venv"
PY="$VENV/bin/python"

echo "==> Creating virtualenv at $VENV"
python3 -m venv "$VENV"

echo "==> Upgrading pip tooling"
"$PY" -m pip install --upgrade pip wheel setuptools

echo "==> Installing insightface and friends (pulls numpy, onnx, etc.)"
"$PY" -m pip install insightface opencv-python-headless tqdm pillow pillow-heif

echo "==> Installing CUDA 12 / cuDNN 9 runtime libraries (for the RTX 4090)"
"$PY" -m pip install \
    nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 \
    nvidia-curand-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12

echo "==> Pinning onnxruntime to the GPU build only (removing any CPU build)"
"$PY" -m pip uninstall -y onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
"$PY" -m pip install --no-deps onnxruntime-gpu

echo
echo "==> GPU / runtime self-check"
"$PY" - <<'PYEOF'
import sys
import onnxruntime as ort
if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()          # load the nvidia-*-cu12 CUDA/cuDNN libraries
print("onnxruntime version :", ort.__version__)
print("available providers :", ort.get_available_providers())

import numpy as np
print("numpy version       :", np.__version__)
from insightface.app import FaceAnalysis
app = FaceAnalysis(name="buffalo_l",
                   providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_thresh=0.5, det_size=[(640, 640), (320, 320), (1024, 1024)])
active = set()
for m in app.models.values():
    s = getattr(m, "session", None)
    if s is not None:
        active.update(s.get_providers())
print("active providers    :", sorted(active))
faces = app.get(np.zeros((640, 640, 3), dtype=np.uint8))
print("smoke inference ok  : detected", len(faces), "faces on a blank image (0 expected)")
if "CUDAExecutionProvider" in active:
    print("\nSUCCESS: GPU pipeline is live on the RTX 4090.")
else:
    print("\nWARNING: CUDA did not initialise — running on CPU (still works, ~50x slower).")
    print("         Check that the nvidia-*-cu12 wheels installed cleanly.")
PYEOF

echo
echo "==> Running logic unit tests"
"$PY" test_logic.py

echo
echo "==> Done. Example run:"
echo "      $PY face_sorter.py --known ./known --input ./to_sort --output ./results --dry-run"
