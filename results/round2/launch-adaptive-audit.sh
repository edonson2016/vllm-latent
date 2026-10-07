#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -e
cd /tmp
export HF_HUB_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0
PYTHONPATH=/tmp/gated-overlay /tmp/final/.venv/bin/python /home/edonson/vllm-latent/benchmarks/latent/audit_approximation.py > /home/edonson/vllm-latent/results/round2/approximation-arithmetic.log 2>&1
for variant in base gated; do
 overlay=/tmp/final-overlay
 if [ "$variant" = gated ]; then overlay=/tmp/gated-overlay; fi
 PYTHONPATH="$overlay:/home/edonson/vllm-latent/benchmarks/latent" /tmp/final/.venv/bin/python /home/edonson/vllm-latent/benchmarks/latent/round2_bench.py --model-index 0 --label adaptive-isolated-"$variant" --opts prune,fast_input,staging,skip_inactive,share,skip_head,conditional_expect,union_gate,async_metadata,capture_head,fuse_state,parameterize --modes adaptive --batches 32 --steps 2048 --temperature .6 --quality /home/edonson/vllm-latent/results/round2/gsm8k-128.json --output /home/edonson/vllm-latent/results/round2/adaptive-isolated-"$variant".jsonl > /home/edonson/vllm-latent/results/round2/adaptive-isolated-"$variant".log 2>&1
done

PYTHONPATH=/tmp/gated-overlay:/home/edonson/vllm-latent/benchmarks/latent /tmp/final/.venv/bin/python /home/edonson/vllm-latent/benchmarks/latent/round2_bench.py --model-index 0 --label exact-nohead --opts prune,fast_input,staging,skip_inactive,share,skip_head,conditional_expect,union_gate,async_metadata,tail_sample,fuse_state,parameterize,exact_compact --modes swi512,swi8 --batches 1,8,32 --repeats 3 --output /home/edonson/vllm-latent/results/round2/exact-nohead0.jsonl > /home/edonson/vllm-latent/results/round2/exact-nohead0.log 2>&1
