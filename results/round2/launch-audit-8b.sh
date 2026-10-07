#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -e
cd /tmp
export HF_HUB_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0 UV_CACHE_DIR=/tmp/uv-cache
if [ ! -x /tmp/final/.venv/bin/python ]; then
 uv venv --system-site-packages /tmp/final/.venv
fi
/tmp/final/.venv/bin/python /home/edonson/vllm-latent/benchmarks/latent/overlay.py /home/edonson/latent-round2-final /tmp/final-overlay

export PYTHONPATH=/tmp/final-overlay:/home/edonson/vllm-latent/benchmarks/latent
/tmp/final/.venv/bin/python /home/edonson/vllm-latent/benchmarks/latent/round2_bench.py --model-index 3 --label exact-quality --opts prune,fast_input,staging,skip_inactive,share,skip_head,conditional_expect,union_gate,async_metadata,capture_head,fuse_state,parameterize,exact_compact --modes swi512,dense32 --batches 32 --steps 2048 --temperature .6 --quality /home/edonson/vllm-latent/results/round2/gsm8k-128.json --output /home/edonson/vllm-latent/results/round2/exact-quality3.jsonl > /home/edonson/vllm-latent/results/round2/exact-quality3.log 2>&1
