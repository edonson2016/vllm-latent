#!/usr/bin/env bash
set -euo pipefail
export UV_CACHE_DIR=/tmp/uv-cache
repo=/home/edonson/vllm-latent
uv venv --system-site-packages --python /usr/bin/python3.12 /tmp/latent/.venv
cd /tmp
/tmp/latent/.venv/bin/python "$repo/benchmarks/latent/scaling_verify.py"
