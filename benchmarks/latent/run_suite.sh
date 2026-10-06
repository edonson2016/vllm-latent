#!/usr/bin/env bash
set -euo pipefail
repo=/home/edonson/vllm-latent
py=/tmp/latent/.venv/bin/python
results="$repo/results/latent"
export HF_HUB_OFFLINE=1
export VLLM_USE_V2_MODEL_RUNNER=0
cd /tmp
"$py" "$repo/benchmarks/latent/overlay.py" "$repo" /tmp/latent-overlay
PYTHONPATH=/tmp/latent-overlay "$py" -m pytest "$repo/tests/v1/latent" \
  --confcutdir="$repo/tests/v1/latent" -q > "$results/tests.log" 2>&1
"$py" "$repo/benchmarks/latent/run.py" --mode stock \
  --output "$results/native.jsonl" > "$results/stock.log" 2>&1
for mode in disabled token soft hidden norm_hidden entropy mixed; do
  PYTHONPATH=/tmp/latent-overlay "$py" "$repo/benchmarks/latent/run.py" \
    --mode "$mode" --output "$results/native.jsonl" > "$results/$mode.log" 2>&1
done
PYTHONPATH=/tmp/latent-overlay "$py" "$repo/benchmarks/latent/run.py" \
  --mode soft --batches 1 --steps 8 --repeats 3 \
  --output "$results/short.jsonl" > "$results/short.log" 2>&1
for kind in token soft; do
  "$py" "$repo/benchmarks/latent/stock_replay.py" --kind "$kind" \
    --output "$results/replay-$kind.jsonl" > "$results/replay-$kind.log" 2>&1
done
date -u > "$results/SUITE_COMPLETE"
