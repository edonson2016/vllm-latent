#!/usr/bin/env bash
set -euo pipefail
export UV_CACHE_DIR=/tmp/uv-cache
repo=/home/edonson/vllm-latent
uv venv --python /usr/bin/python3.12 /tmp/swir/.venv
uv pip install --python /tmp/swir/.venv/bin/python \
  -r "$repo/results/scaling/swir-requirements.txt" \
  --extra-index-url https://download.pytorch.org/whl/cu128
cd /tmp
/tmp/swir/.venv/bin/python "$repo/benchmarks/latent/comparator_suite.py" --engine swir
