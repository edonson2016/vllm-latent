# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA-event calibration of transition-only execution at Qwen3-8B dimensions."""

import argparse
import json
from pathlib import Path

import torch

from vllm.v1.latent.graph import GraphProgram
from vllm.v1.latent.program import Program, preset

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
e = torch.randn(151936, 4096, device="cuda", dtype=torch.bfloat16)
results = []
for kind in ["token", "soft", "hidden", "norm_hidden", "entropy"]:
    graph = GraphProgram(Program(preset(kind, 128), 4096, 151936), e, 32)
    for batch in [1, 8, 32]:
        args = (
            torch.randn(batch, 151936, device="cuda"),
            torch.randn(batch, 4096, device="cuda", dtype=torch.bfloat16),
            torch.randn(batch, 4096, device="cuda", dtype=torch.bfloat16),
            torch.zeros(batch, 1, device="cuda", dtype=torch.long),
            torch.zeros(batch, 8, device="cuda"),
            e,
        )
        for _ in range(5):
            graph(*args)
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(100):
            graph.graphs[batch][0].replay()
        end.record()
        end.synchronize()
        results.append(
            {
                "kind": kind,
                "batch": batch,
                "graph_device_ms": start.elapsed_time(end) / 100,
            }
        )
    del graph
a.output.write_text(json.dumps(results, indent=2) + "\n")
