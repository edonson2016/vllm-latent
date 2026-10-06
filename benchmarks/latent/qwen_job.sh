#!/usr/bin/env bash
set -euo pipefail
export UV_CACHE_DIR=/tmp/uv-cache
repo=/home/edonson/vllm-latent
uv venv --python /usr/bin/python3.12 /tmp/qwen/.venv
uv pip install --python /tmp/qwen/.venv/bin/python \
  -r "$repo/results/scaling/qwen-requirements.txt"
cd /tmp
/tmp/qwen/.venv/bin/python "$repo/benchmarks/latent/comparator_suite.py" --engine qwen
