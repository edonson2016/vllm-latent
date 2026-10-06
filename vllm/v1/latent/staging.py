# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent batch inputs with captured gather, execution, and state commit."""

import torch

from vllm.v1.latent.compiler import SharedProgram, can_share, dependencies
from vllm.v1.latent.kernels import commit


class StagedTransitions:
    def __init__(self, owner):
        self.owner = owner
        runner = owner.runner
        self.batch = 1 << (runner.max_num_reqs - 1).bit_length()
        device, dtype = runner.device, runner.dtype
        self.device, self.dtype = device, dtype
        self.logits = None
        self.hidden = None
        self.sampled = torch.zeros(self.batch, 1, device=device, dtype=torch.long)
        self.metadata_cpu = torch.full(
            (self.batch, 4), -1, dtype=torch.long, pin_memory=True
        )
        self.metadata = self.metadata_cpu.to(device)
        self.graphs = {}
        self.shared_keys = None
        # Keep one capture stream for this runner. CUDA streams come from a
        # finite pool; repeatedly acquiring them can reuse a model capture's
        # stream and disturb its library workspace lifetime.
        self.capture_stream = torch.cuda.Stream(device=device)

    def register(self, key):
        self.capture((key,))
        if "share" not in self.owner.optimizations:
            return
        keys = tuple(sorted(self.owner.definitions))
        self.graphs = {k: v for k, v in self.graphs.items() if len(k) == 1}
        self.shared_keys = (
            keys if can_share([self.owner.definitions[k] for k in keys]) else None
        )
        if self.shared_keys is not None:
            self.capture(self.shared_keys)

    def capture(self, keys):
        if keys in self.graphs:
            return
        # Admission may occur while older requests are live. Warmup must never
        # commit stale metadata into their histories or state registers.
        self.metadata.fill_(-1)
        owner = self.owner
        programs = [owner.definitions[key] for key in keys]
        combined = SharedProgram(programs, share="share" in owner.optimizations)
        live = combined.live
        if "logits" in live and self.logits is None:
            self.logits = torch.zeros(
                self.batch, owner.runner.input_batch.vocab_size, device=self.device
            )
        if "hidden" in live and self.hidden is None:
            self.hidden = torch.zeros(
                self.batch,
                owner.history.shape[-1],
                device=self.device,
                dtype=self.dtype,
            )
        shapes = {}
        size = 1
        while size < 2 * self.batch:
            # Captures beyond max_num_reqs are unnecessary when it is not a power of 2.
            if size > self.batch:
                break

            def run(n=size):
                meta = self.metadata[:n]
                rows = meta[:, 0].clamp_min(0)
                slots = meta[:, 1].clamp_min(0)
                state = owner.state[slots]
                token = owner.embedding[self.sampled[rows, 0]]
                logits = None
                hidden = None
                if "logits" in live:
                    assert self.logits is not None
                    logits = self.logits[rows]
                if "hidden" in live:
                    assert self.hidden is not None
                    hidden = self.hidden[rows]
                output, latent, updated = combined(
                    logits,
                    hidden,
                    token,
                    meta[:, 2:3],
                    state,
                    owner.embedding,
                    meta[:, 3:4],
                )
                commit(
                    output,
                    latent,
                    updated,
                    meta,
                    owner.history,
                    owner.masks,
                    owner.state,
                    self.sampled,
                )

            stream = self.capture_stream
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    run()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                run()
            torch.cuda.current_stream().wait_stream(stream)
            shapes[size] = graph
            size *= 2
        self.graphs[keys] = (shapes, live)

    def execute(self, groups, logits, hidden, sampled):
        count = sampled.shape[0]
        self.sampled[:count].copy_(sampled)
        shared = self.shared_keys is not None and len(groups) > 1
        if shared:
            assert self.shared_keys is not None
            keys = self.shared_keys
        else:
            keys = tuple(sorted(groups))
        packs = [keys] if shared else [(key,) for key in keys]
        live = set().union(*(dependencies(self.owner.definitions[k])[1] for k in keys))
        if "logits" in live:
            assert self.logits is not None
            self.logits[:count].copy_(logits)
        if "hidden" in live:
            assert self.hidden is not None
            self.hidden[:count].copy_(hidden)
        for pack in packs:
            shapes, _ = self.graphs[pack]
            entries = [
                (*entry, i)
                for i, key in enumerate(pack)
                for entry in groups.get(key, [])
            ]
            n = len(entries)
            size = 1 << (n - 1).bit_length()
            self.metadata_cpu.fill_(-1)
            self.metadata_cpu[:, 2].fill_(self.owner.max_steps)
            self.metadata_cpu.numpy()[:n] = entries
            # The synchronous copy protects the pinned host buffer from reuse
            # while a previous pack's asynchronous transfer is still in flight.
            self.metadata[:size].copy_(self.metadata_cpu[:size])
            shapes[size].replay()
        return self.sampled[:count]
