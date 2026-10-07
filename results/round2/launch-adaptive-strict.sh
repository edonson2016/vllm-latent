#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -e
cd /tmp
export HF_HUB_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1
for variant in base gated; do
 overlay=/tmp/final-overlay
 if [ "$variant" = gated ]; then overlay=/tmp/gated-overlay; fi
 PYTHONPATH="$overlay:/home/edonson/vllm-latent/benchmarks/latent" /tmp/final/.venv/bin/python /home/edonson/vllm-latent/benchmarks/latent/round2_bench.py --model-index 0 --fixed-norm --label adaptive-strict-"$variant" --opts prune,fast_input,staging,skip_inactive,share,skip_head,conditional_expect,union_gate,async_metadata,capture_head,fuse_state,parameterize --modes adaptive --batches 32 --steps 2048 --temperature .6 --quality /home/edonson/vllm-latent/results/round2/gsm8k-128.json --output /home/edonson/vllm-latent/results/round2/adaptive-strict-"$variant".jsonl > /home/edonson/vllm-latent/results/round2/adaptive-strict-"$variant".log 2>&1
done
