# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compaction crossover and embedding reconstruction experiments on real weights."""

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vllm.transformers_utils.repo_utils import hf_api
from vllm.v1.latent.kernels import compact_expect, conditional_expect


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-index", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--factors", type=Path)
    p.add_argument("--rank", type=int, default=128)
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[2]
    model = json.loads((repo / "results/scaling/models.json").read_text())[
        a.model_index
    ]
    root = Path(
        hf_api().snapshot_download(
            model["model"], revision=model["revision"], local_files_only=True
        )
    )
    index = root / "model.safetensors.index.json"
    name = "model.embed_tokens.weight"
    shard = (
        json.loads(index.read_text())["weight_map"][name]
        if index.exists()
        else "model.safetensors"
    )
    with safe_open(root / shard, framework="pt", device="cpu") as f:
        embedding = f.get_tensor(name).to("cuda")
    torch.manual_seed(20261006)
    records = []
    if a.factors:
        start = time.perf_counter()
        e = embedding.float()
        _, _, basis = torch.svd_lowrank(e, q=a.rank + 16, niter=2)
        right = basis[:, : a.rank].T.contiguous()
        left = e @ right.T
        reconstruction = left @ right
        error = (reconstruction - e).norm() / e.norm()
        row = dict(
            model=model,
            rank=a.rank,
            seed=20261006,
            reconstruction_relative_l2=error.item(),
            seconds=time.perf_counter() - start,
        )
        a.factors.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "embedding.left": left.to(torch.bfloat16).cpu().contiguous(),
                "embedding.right": right.to(torch.bfloat16).cpu().contiguous(),
            },
            str(a.factors),
            metadata={"model": model["model"], "revision": model["revision"]},
        )
        a.output.write_text(json.dumps(row, indent=2) + "\n")
        return
    for dtype in [torch.bfloat16, torch.float32]:
        table = embedding.to(dtype)
        for batch in [1, 8, 32]:
            prob = (torch.randn(batch, table.shape[0], device="cuda") * 4).softmax(-1)
            reference = prob.to(dtype) @ table
            gate = torch.ones(batch, 1, device="cuda", dtype=torch.bool)
            graphs = {}
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            for name, fn in [
                ("conditional", conditional_expect),
                ("compact", compact_expect),
            ]:
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        fn(prob, table, gate)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    result = fn(prob, table, gate)
                torch.cuda.current_stream().wait_stream(stream)
                graphs[name] = (graph, result)
            for count in sorted({0, 1, max(1, batch // 4), batch // 2, batch}):
                gate.zero_()
                gate[:count] = True
                for name, (graph, result) in graphs.items():
                    graph.replay()
                    torch.accelerator.synchronize()
                    error = result[:count].float() - reference[:count].float()
                    times = []
                    for _ in range(7):
                        start, end = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        start.record()
                        for _ in range(50):
                            graph.replay()
                        end.record()
                        end.synchronize()
                        times.append(start.elapsed_time(end) / 50)
                    records.append(
                        dict(
                            model=model,
                            dtype=str(dtype),
                            batch=batch,
                            active=count,
                            method=name,
                            ms=times,
                            max_abs_error=error.abs().max().item() if count else 0,
                            relative_l2=(
                                error.norm()
                                / reference[:count].float().norm().clamp_min(1e-12)
                            ).item(),
                        )
                    )
            del graphs, result, graph, reference, prob
    a.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
