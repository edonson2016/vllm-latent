#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -e
cd /tmp
export HF_HUB_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0 UV_CACHE_DIR=/tmp/uv-cache
uv venv --system-site-packages /tmp/final/.venv
/tmp/final/.venv/bin/python /home/edonson/vllm-latent/benchmarks/latent/overlay.py /home/edonson/latent-round2-final /tmp/final-overlay

PYTHONPATH=/tmp/qwen-overlay /tmp/qwen/.venv/bin/python /home/edonson/vllm-latent/benchmarks/latent/swi_oracle.py --output /home/edonson/vllm-latent/results/round2/swi-oracle-grid.json > /home/edonson/vllm-latent/results/round2/swi-oracle-grid.log 2>&1

for i in 0 1 2 3; do
  for variant in qwen fork exact stock; do
    engine="$variant"
    label=paired-final
    extra_opts=
    if [ "$variant" = exact ]; then engine=fork; label=paired-exact; extra_opts=,exact_compact; fi
    py=/tmp/final/.venv/bin/python
    overlay=/tmp/final-overlay
    modes=token,swi512,swi8
    if [ "$engine" = qwen ]; then py=/tmp/qwen/.venv/bin/python; overlay=/tmp/qwen-overlay; fi
    if [ "$engine" = stock ]; then overlay=; modes=token; fi
    PYTHONPATH="$overlay:/home/edonson/vllm-latent/benchmarks/latent" "$py" /home/edonson/vllm-latent/benchmarks/latent/round2_bench.py --engine "$engine" --label "$label" --model-index "$i" --opts prune,fast_input,staging,skip_inactive,share,skip_head,conditional_expect,union_gate,async_metadata,capture_head,fuse_state,parameterize"$extra_opts" --modes "$modes" --batches 1,8,32 --repeats 7 --output /home/edonson/vllm-latent/results/round2/paired-model"$i".jsonl > /home/edonson/vllm-latent/results/round2/paired-model"$i"-"$variant".log 2>&1
  done
done
