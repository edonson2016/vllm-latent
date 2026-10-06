# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolate CUDA-graph expectation latency and numerical error by shape/gating."""

import argparse
import json
from pathlib import Path

import torch

from vllm.v1.latent.kernels import (
    conditional_expect,
    fused_soft_expect,
    gated_expect,
    masked_expect,
    soft_stats,
)

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
torch.manual_seed(19)
records = []
for width in [1024, 2048, 2560, 4096]:
    e = torch.randn(151936, width, device="cuda", dtype=torch.bfloat16)
    for batch in [1, 8, 32]:
        logits = torch.randn(batch, 151936, device="cuda") * 4
        reference = logits.softmax(-1).to(e.dtype) @ e
        entropy = -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)
        torch.testing.assert_close(
            soft_stats(logits)[:, 2], entropy, atol=2e-5, rtol=2e-5
        )
        for active in [True, False]:
            gate = torch.full((batch,), active, device="cuda", dtype=torch.bool)
            methods = {
                "cublas": lambda x=logits, w=e: x.softmax(-1).to(w.dtype) @ w,
                "masked": lambda x=logits, w=e, g=gate: masked_expect(
                    x.softmax(-1), w, g
                ),
                "fused": lambda x=logits, w=e, g=gate: fused_soft_expect(x, w, g),
                "conditional": lambda x=logits, w=e, g=gate: conditional_expect(
                    x.softmax(-1), w, g
                ),
                "gated_gemm": lambda x=logits, w=e, g=gate: gated_expect(
                    x.softmax(-1), w, g
                ),
            }
            for name, fn in methods.items():
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        fn()
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                before = torch.accelerator.memory_allocated()
                torch.accelerator.reset_peak_memory_stats()
                with torch.cuda.graph(graph, stream=stream):
                    actual = fn()
                torch.cuda.current_stream().wait_stream(stream)
                peak = torch.accelerator.max_memory_allocated() - before
                graph.replay()
                torch.accelerator.synchronize()
                expected = reference if active or name == "cublas" else reference * 0
                torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.03)
                error = actual.float() - expected.float()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                for _ in range(30):
                    graph.replay()
                end.record()
                end.synchronize()
                row = {
                    "width": width,
                    "batch": batch,
                    "active": active,
                    "method": name,
                    "ms": start.elapsed_time(end) / 30,
                    "max_abs_error": error.abs().max().item(),
                    "relative_l2": (
                        error.norm() / expected.float().norm().clamp_min(1e-9)
                    ).item(),
                    "active_allocation_bytes": torch.accelerator.memory_allocated()
                    - before,
                    "capture_peak_increment_bytes": peak,
                }
                records.append(row)
                print(json.dumps(row), flush=True)
                del graph, actual
        del logits, reference
    del e
a.output.write_text(json.dumps(records, indent=2) + "\n")
