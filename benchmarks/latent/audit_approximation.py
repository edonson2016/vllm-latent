# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare conditional and unconditional mixtures on identical realistic shapes."""

import json
from pathlib import Path

import torch
from safetensors import safe_open

from vllm.transformers_utils.repo_utils import hf_api
from vllm.v1.latent.swireasoning import SwiProgram

root = Path(__file__).resolve().parents[2]
model = json.loads((root / "results/scaling/models.json").read_text())[0]
checkpoint = Path(
    hf_api().snapshot_download(
        model["model"], revision=model["revision"], local_files_only=True
    )
)
with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as f:
    embedding = f.get_tensor("model.embed_tokens.weight").to("cuda")
torch.manual_seed(20261006)
records = []
for batch in [1, 3, 8, 32]:
    logits = torch.zeros(batch, embedding.shape[0], device="cuda")
    sampled = torch.zeros(batch, 1, device="cuda", dtype=torch.long)
    valid = torch.ones(batch, 1, device="cuda", dtype=torch.bool)
    args = (
        logits,
        None,
        embedding[:batch],
        torch.zeros(batch, 1, device="cuda"),
        torch.zeros(batch, 8, device="cuda"),
        embedding,
        sampled,
    )
    for expectation in ["topk", "adaptive"]:
        graphs = []
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        for gated in [False, True]:
            program = SwiProgram(
                dict(
                    policy="swireasoning",
                    start_id=60,
                    end_id=61,
                    linebreak_id=62,
                    stop_id=63,
                    expectation=expectation,
                ),
                embedding.shape[1],
                embedding.shape[0],
            )
            program.prepare(embedding)
            program.conditional_expect = gated
            with torch.cuda.stream(stream):
                for _ in range(3):
                    program.execute(*args, valid=valid)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                result = program.execute(*args, valid=valid)
            torch.cuda.current_stream().wait_stream(stream)
            graphs.append((graph, result))
        for trial in range(50):
            logits.copy_((torch.randn_like(logits) * 8).bfloat16().float())
            valid.zero_()
            valid[: [0, 1, 13, 32][trial % 4]] = True
            for graph, _ in graphs:
                graph.replay()
            left, right = graphs[0][1], graphs[1][1]
            records.append(
                dict(
                    expectation=expectation,
                    batch=batch,
                    trial=trial,
                    equal=[bool(torch.equal(a, b)) for a, b in zip(left, right)],
                    embedding_max_error=(left[0].float() - right[0].float())
                    .abs()
                    .max()
                    .item(),
                )
            )
(root / "results/round2/approximation-arithmetic.json").write_text(
    json.dumps(records, indent=2) + "\n"
)
assert all(all(row["equal"]) for row in records), "Conditional arithmetic differs"
