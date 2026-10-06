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
  --confcutdir="$repo/tests/v1/latent" -q > "$results/tests-final.log" 2>&1
"$py" "$repo/benchmarks/latent/run.py" --mode embed_control \
  --output "$results/native.jsonl" > "$results/embed_control.log" 2>&1
for blocks in 32 default; do
  opts=()
  if [[ "$blocks" != default ]]; then opts=(--blocks "$blocks"); fi
  PYTHONPATH=/tmp/latent-overlay "$py" "$repo/benchmarks/latent/check_engine.py" \
    "${opts[@]}" --output "$results/preemption-$blocks.json" \
    > "$results/preemption-$blocks.log" 2>&1
done
"$py" "$repo/benchmarks/latent/hf_reference.py" --output "$results/hf-reference.json" \
  > "$results/hf-reference.log" 2>&1
for mode in token hidden; do
  PYTHONPATH=/tmp/latent-overlay "$py" "$repo/benchmarks/latent/run.py" \
    --mode "$mode" --batches 1 --steps 8 --repeats 3 \
    --output "$results/short.jsonl" > "$results/short-$mode.log" 2>&1
done
PYTHONPATH=/tmp/latent-overlay "$py" "$repo/benchmarks/latent/run.py" \
  --mode entropy --batches 8 --entropy-threshold 0.5 --trace-masks \
  --output "$results/adaptive.jsonl" > "$results/adaptive.log" 2>&1
PYTHONPATH=/tmp/latent-overlay "$py" "$repo/benchmarks/latent/transition_cost.py" \
  --output "$results/transition-cost.json" > "$results/transition-cost.log" 2>&1
for mode in stock disabled token; do
  if [[ "$mode" == stock ]]; then export PYTHONPATH=; else export PYTHONPATH=/tmp/latent-overlay; fi
  "$py" "$repo/benchmarks/latent/run.py" --mode "$mode" --repeats 3 \
    --output "$results/final-controls.jsonl" > "$results/final-$mode.log" 2>&1
done
date -u > "$results/VERIFY_COMPLETE"

for blocks in 32 11000; do
  PYTHONPATH=/tmp/latent-overlay "$py" "$repo/benchmarks/latent/check_engine.py" \
    --invariant --blocks "$blocks" --output "$results/invariant-$blocks.json" \
    > "$results/invariant-$blocks.log" 2>&1
done
PYTHONPATH='' "$py" "$repo/benchmarks/latent/stock_replay.py" --kind soft --prefix-cache \
  --output "$results/replay-soft-apc.jsonl" > "$results/replay-soft-apc.log" 2>&1
cp "$results/preemption-32.json" "$results/preemption-counted.json"
"$py" "$repo/benchmarks/latent/verify_results.py"
date -u > "$results/EXPERIMENTS_COMPLETE"
