# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Materialize a validated program as CUDA graphs before its first decode."""

import torch


class GraphProgram:
    def __init__(self, program, embedding, max_batch):
        self.graphs = {}
        self.embedding = embedding
        vocab, width = embedding.shape
        from vllm.v1.latent.compiler import dependencies

        live = (
            dependencies(program)[1]
            if hasattr(program, "live")
            else {"logits", "hidden"}
        )
        batch = 1
        stream = torch.cuda.Stream(device=embedding.device)
        stream.wait_stream(torch.cuda.current_stream())
        while batch < 2 * max_batch:
            inputs = (
                torch.zeros(batch, vocab, device=embedding.device)
                if "logits" in live
                else None,
                torch.zeros(
                    batch, width, device=embedding.device, dtype=embedding.dtype
                )
                if "hidden" in live
                else None,
                torch.zeros(
                    batch, width, device=embedding.device, dtype=embedding.dtype
                ),
                torch.zeros(batch, 1, device=embedding.device, dtype=torch.long),
                torch.zeros(batch, 8, device=embedding.device),
            )
            with torch.cuda.stream(stream):
                for _ in range(3):
                    program(*inputs, embedding)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs = program(*inputs, embedding)
            self.graphs[batch] = (graph, inputs, outputs)
            batch *= 2
        torch.cuda.current_stream().wait_stream(stream)

    def __call__(self, logits, hidden, token, step, state, embedding):
        n = token.shape[0]
        size = 1 << (n - 1).bit_length()
        graph, inputs, outputs = self.graphs[size]
        for dst, src in zip(inputs, (logits, hidden, token, step, state)):
            if dst is not None:
                dst[:n].copy_(src)
        graph.replay()
        return tuple(out[:n] for out in outputs)
