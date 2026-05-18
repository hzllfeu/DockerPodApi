#!/usr/bin/env bash
# Build the tar layers consumed by Bazel oci_image. Run this once on a
# Koyeb GPU pod (or any Linux box with python3.11 + pip + git) BEFORE
# running `bazel build //:image`.
#
# Outputs in the working dir (next to MODULE.bazel):
#   site_packages.tar    pre-installed pip deps from requirements.txt
#   vendored.tar         DiariZen (with vendored pyannote-audio) + DiCoW
#   models.tar           HF cache pre-populated with all model weights
#
# Total size of the three tarballs: ~7–8 GB.

set -euo pipefail

# Resolve paths
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"            # models/echo-l4
BAZEL_DIR="$ROOT/bazel"
REQ="$ROOT/requirements.txt"
WORK="${BUILD_WORK:-/workspace/build/echo-l4}"

mkdir -p "$WORK"
echo "[build] WORK=$WORK"

# -- 1. site-packages -----------------------------------------------------
echo "[build] downloading wheels..."
mkdir -p "$WORK/wheels"
pip download --quiet -r "$REQ" \
    --dest "$WORK/wheels" \
    --extra-index-url https://download.pytorch.org/whl/cu121
pip download --quiet --dest "$WORK/wheels" setuptools wheel

echo "[build] pre-installing wheels into portable site-packages..."
rm -rf "$WORK/site-packages"
mkdir -p "$WORK/site-packages"
pip install --quiet --no-index \
    --target="$WORK/site-packages" \
    --find-links="$WORK/wheels" \
    -r "$REQ"

echo "[build] tarring site-packages → site_packages.tar"
tar -C "$WORK" --transform 's,^site-packages,opt/python-runtime/site-packages,' \
    -cf "$BAZEL_DIR/site_packages.tar" site-packages
du -h "$BAZEL_DIR/site_packages.tar" | head -1

# -- 2. vendored DiariZen + DiCoW ----------------------------------------
echo "[build] cloning DiariZen + DiCoW..."
mkdir -p "$WORK/vendored"
cd "$WORK/vendored"
[ -d DiariZen ] || git clone --depth 1 https://github.com/BUTSpeechFIT/DiariZen.git
[ -d DiCoW ] || git clone --depth 1 https://github.com/BUTSpeechFIT/DiCoW.git

echo "[build] stub gradio module (DiCoW pipeline imports it)"
mkdir -p "$WORK/vendored/_stubs"
cat > "$WORK/vendored/_stubs/gradio.py" <<'EOF'
def Info(*a, **k): pass
def Warning(*a, **k): pass
__version__ = "stub"
EOF

echo "[build] tarring vendored → vendored.tar"
tar -C "$WORK/vendored" \
    --transform 's,^DiariZen,opt/DiariZen,' \
    --transform 's,^DiCoW,opt/DiCoW,' \
    --transform 's,^_stubs/gradio.py,opt/python-runtime/site-packages/gradio.py,' \
    -cf "$BAZEL_DIR/vendored.tar" DiariZen DiCoW _stubs/gradio.py
du -h "$BAZEL_DIR/vendored.tar" | head -1

# -- 3. HF model weights -------------------------------------------------
# Pre-populate an HF_HOME pointing at $WORK/hf_cache so we ship a tarball
# the runtime can use with HF_HUB_OFFLINE=1.
echo "[build] pre-fetching HF weights..."
export HF_HOME="$WORK/hf_cache"
mkdir -p "$HF_HOME"

# Required: an HF token with read access to Hellfeu/echo-dia.
if [ -z "${HF_TOKEN:-}" ]; then
    echo "[build] ERROR: set HF_TOKEN env var (read access to Hellfeu/echo-dia)" >&2
    exit 1
fi

python3 - <<'PYEOF'
import os
from huggingface_hub import hf_hub_download, snapshot_download

token = os.environ["HF_TOKEN"]

# Echo Dia fine-tuned weights (Hellfeu/echo-dia, private repo, gated by token)
hf_hub_download("Hellfeu/echo-dia", "pytorch_model.bin", token=token)

# DiariZen V2 base (public)
snapshot_download("BUT-FIT/diarizen-wavlm-large-s80-md-v2")

# SE-DiCoW ASR + custom code (public, trust_remote_code)
snapshot_download("BUT-FIT/SE-DiCoW", allow_patterns=["*"])

# pyannote/wespeaker speaker embedder (pulled transitively by DiariZen)
snapshot_download("pyannote/wespeaker-voxceleb-resnet34-LM")

print("HF weights ready under", os.environ["HF_HOME"])
PYEOF

echo "[build] tarring models → models.tar"
tar -C "$WORK" --transform 's,^hf_cache,opt/hf_cache,' \
    -cf "$BAZEL_DIR/models.tar" hf_cache
du -h "$BAZEL_DIR/models.tar" | head -1

echo "[build] done. Layers ready in $BAZEL_DIR"
echo "       next: cd $BAZEL_DIR && bazel build //:image"
