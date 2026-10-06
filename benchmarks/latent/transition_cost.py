# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA-event calibration of transition-only execution at supplied dimensions."""

import argparse
import json
from pathlib import Path

import torch

from vllm.v1.latent.graph import GraphProgram
from vllm.v1.latent.program import Program, preset

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
p.add_argument("--width", type=int, default=4096)
p.add_argument("--vocab", type=int, default=151936)
a = p.parse_args()
e = torch.randn(a.vocab, a.width, device="cuda", dtype=torch.bfloat16)
results = []
for kind in ["token", "soft", "hidden", "norm_hidden", "entropy"]:
    torch.accelerator.synchronize()
    before = torch.accelerator.memory_allocated()
    graph = GraphProgram(Program(preset(kind, 128), a.width, a.vocab), e, 32)
    graph_bytes = torch.accelerator.memory_allocated() - before
    for batch in [1, 8, 32]:
        args = (
            torch.randn(batch, a.vocab, device="cuda"),
            torch.randn(batch, a.width, device="cuda", dtype=torch.bfloat16),
            torch.randn(batch, a.width, device="cuda", dtype=torch.bfloat16),
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
                "all_capture_sizes_allocated_bytes": graph_bytes,
                "width": a.width,
                "vocab": a.vocab,
            }
        )
    del graph
a.output.write_text(json.dumps(results, indent=2) + "\n")
