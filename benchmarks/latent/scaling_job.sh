#!/usr/bin/env bash
set -euo pipefail
export UV_CACHE_DIR=/tmp/uv-cache
uv venv --system-site-packages --python /usr/bin/python3.12 /tmp/latent/.venv
uv pip install --python /tmp/latent/.venv/bin/python pytest
cd /tmp
/tmp/latent/.venv/bin/python /home/edonson/vllm-latent/benchmarks/latent/scaling_suite.py
