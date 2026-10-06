#!/usr/bin/env bash
set -euo pipefail
export UV_CACHE_DIR=/tmp/uv-cache
repo=/home/edonson/vllm-latent
py=/tmp/latent/.venv/bin/python
uv venv --system-site-packages --python /usr/bin/python3.12 /tmp/latent/.venv
cd /tmp
"$py" "$repo/benchmarks/latent/overlay.py" "$repo" /tmp/latent-overlay
export HF_HUB_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1
for mode in embed_control token; do
  if [[ "$mode" == token ]]; then
    export PYTHONPATH=/tmp/latent-overlay
  else
    export PYTHONPATH=''
  fi
  "$py" "$repo/benchmarks/latent/run.py" --mode "$mode" --repeats 1 \
    --output "$repo/results/scaling/invariant-control.jsonl" \
    > "$repo/results/scaling/invariant-$mode.log" 2>&1
done
