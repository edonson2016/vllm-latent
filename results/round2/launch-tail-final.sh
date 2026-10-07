#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -e
cd /tmp
export HF_HUB_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0 UV_CACHE_DIR=/tmp/uv-cache
uv venv --system-site-packages /tmp/final/.venv
/tmp/final/.venv/bin/python /home/edonson/vllm-latent/benchmarks/latent/overlay.py /home/edonson/latent-round2-final /tmp/final-overlay

export PYTHONPATH=/tmp/final-overlay:/home/edonson/vllm-latent/benchmarks/latent
for i in 2 3; do
 /tmp/final/.venv/bin/python /home/edonson/vllm-latent/benchmarks/latent/round2_bench.py --model-index "$i" --opts prune,fast_input,staging,skip_inactive,share,skip_head,conditional_expect,union_gate,async_metadata,capture_head,fuse_state,parameterize --modes dense32,bf16,topk,adaptive,lowrank --batches 1,32 --repeats 7 --constants /home/edonson/latent-round2-factors/model"$i"-rank128.safetensors --output /home/edonson/vllm-latent/results/round2/approx-timing"$i".jsonl > /home/edonson/vllm-latent/results/round2/approx-timing"$i".log 2>&1
done
