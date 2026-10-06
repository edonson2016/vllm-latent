#!/usr/bin/env bash
set -euo pipefail
export UV_CACHE_DIR=/tmp/uv-cache
repo=/home/edonson/vllm-latent
py=/tmp/latent/.venv/bin/python
uv venv --system-site-packages --python /usr/bin/python3.12 /tmp/latent/.venv
cd /tmp
"$py" "$repo/benchmarks/latent/overlay.py" "$repo" /tmp/latent-overlay
export PYTHONPATH="/tmp/latent-overlay:$repo/benchmarks/latent"
export HF_HUB_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0
for phase in capture reference; do
  "$py" "$repo/benchmarks/latent/hidden_audit.py" --phase "$phase" \
    --output "$repo/results/scaling/hidden-audit-trace.json" \
    > "$repo/results/scaling/hidden-audit-$phase.log" 2>&1
done
