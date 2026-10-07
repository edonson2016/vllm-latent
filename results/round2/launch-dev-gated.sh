#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -e
cd /tmp
export HF_HUB_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0 UV_CACHE_DIR=/tmp/uv-cache
/tmp/final/.venv/bin/python /home/edonson/vllm-latent/benchmarks/latent/overlay.py /home/edonson/latent-round2-gated /tmp/gated-overlay
mkdir -p /tmp/gated-overlay/tests/v1
cp -r /home/edonson/latent-round2-gated/tests/v1/latent /tmp/gated-overlay/tests/v1/
export PYTHONPATH=/tmp/gated-overlay:/home/edonson/vllm-latent/benchmarks/latent
/tmp/final/.venv/bin/python -m pytest /tmp/gated-overlay/tests/v1/latent --confcutdir=/tmp/gated-overlay/tests/v1/latent -q > /home/edonson/vllm-latent/results/round2/unit-v6.txt 2>&1
for i in 0 1; do
 /tmp/final/.venv/bin/python /home/edonson/vllm-latent/benchmarks/latent/round2_bench.py --model-index "$i" --label gated-approximations --opts prune,fast_input,staging,skip_inactive,share,skip_head,conditional_expect,union_gate,async_metadata,capture_head,fuse_state,parameterize --modes dense32,bf16,topk,adaptive,lowrank --batches 1,32 --repeats 7 --constants /home/edonson/latent-round2-factors/model"$i"-rank128.safetensors --output /home/edonson/vllm-latent/results/round2/approx-gated"$i".jsonl > /home/edonson/vllm-latent/results/round2/approx-gated"$i".log 2>&1
done
/tmp/final/.venv/bin/python /home/edonson/vllm-latent/benchmarks/latent/round2_bench.py --model-index 0 --label gated-quality --opts prune,fast_input,staging,skip_inactive,share,skip_head,conditional_expect,union_gate,async_metadata,capture_head,fuse_state,parameterize --modes topk,adaptive,lowrank --batches 32 --steps 2048 --temperature .6 --constants /home/edonson/latent-round2-factors/model0-rank128.safetensors --quality /home/edonson/vllm-latent/results/round2/gsm8k-128.json --output /home/edonson/vllm-latent/results/round2/gated-quality0.jsonl > /home/edonson/vllm-latent/results/round2/gated-quality0.log 2>&1
