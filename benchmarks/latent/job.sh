#!/usr/bin/env bash
set -euo pipefail
export UV_CACHE_DIR=/tmp/uv-cache
uv venv --system-site-packages --python /usr/bin/python3.12 /tmp/latent/.venv
uv pip install --python /tmp/latent/.venv/bin/python pytest
cd /tmp
/tmp/latent/.venv/bin/python -c \
  'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3-8B", revision="b968826d9c46dd6066d109eabc6255188de91218")'
bash /home/edonson/vllm-latent/benchmarks/latent/run_suite.sh
bash /home/edonson/vllm-latent/benchmarks/latent/verify_suite.sh
