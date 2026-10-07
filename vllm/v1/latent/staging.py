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
        self.fuse_sample = torch.zeros((), device=device, dtype=torch.bool)
        self.metadata_cpu = torch.full(
            (self.batch, 4), -1, dtype=torch.long, pin_memory=True
        )
        self.metadata = self.metadata_cpu.to(device)
        self.metadata_ring = None
        if "async_metadata" in owner.optimizations:
            from vllm.v1.latent.transfer import MetadataRing

            self.metadata_ring = MetadataRing(self.metadata_cpu)
        self.graphs = {}
        self.head_graphs = {}
        self.shared_keys = None
        # Keep one capture stream for this runner. CUDA streams come from a
        # finite pool; repeatedly acquiring them can reuse a model capture's
        # stream and disturb its library workspace lifetime.
        self.capture_stream = torch.cuda.Stream(device=device)

    def register(self, key):
        self.capture((key,))
        if (
            "capture_head" in self.owner.optimizations
            and "logits" in dependencies(self.owner.definitions[key])[1]
        ):
            self.capture((key,), head=True)
        if "share" not in self.owner.optimizations:
            return
        keys = tuple(sorted(self.owner.definitions))
        self.graphs = {k: v for k, v in self.graphs.items() if len(k) == 1}
        self.shared_keys = (
            keys if can_share([self.owner.definitions[k] for k in keys]) else None
        )
        if self.shared_keys is not None:
            self.capture(self.shared_keys)

    def capture(self, keys, head=False):
        registry = self.head_graphs if head else self.graphs
        if keys in registry:
            return
        # Admission may occur while older requests are live. Warmup must never
        # commit stale metadata into their histories or state registers.
        self.metadata.fill_(-1)
        owner = self.owner
        programs = [owner.definitions[key] for key in keys]
        need_parameters = any(op == "PARAM" for p in programs for _, op, _ in p.ops)
        combined = SharedProgram(programs, share="share" in owner.optimizations)
        live = combined.live
        fuse_sample = (
            bool({"tail_sample", "capture_head"} & owner.optimizations)
            and "logits" in live
        )
        record_tokens = fuse_sample or any(
            getattr(p, "builtin", False) for p in programs
        )
        if "logits" in live and self.logits is None:
            self.logits = torch.zeros(
                self.batch, owner.runner.input_batch.vocab_size, device=self.device
            )
        if head:
            live = (live - {"logits"}) | {"hidden"}
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
                sampled = self.sampled[rows]
                logits = None
                hidden = None
                if "logits" in live:
                    assert self.logits is not None
                    logits = self.logits[rows]
                if "hidden" in live:
                    assert self.hidden is not None
                    hidden = self.hidden[rows]
                if head:
                    logits = owner.runner.get_model().compute_logits(hidden)
                if fuse_sample:
                    assert logits is not None
                    greedy = logits.argmax(-1, keepdim=True)
                    sampled = (
                        greedy
                        if head
                        else torch.where(self.fuse_sample, greedy, sampled)
                    )
                token = owner.embedding[sampled[:, 0]]
                outputs = combined(
                    logits,
                    hidden,
                    token,
                    meta[:, 2:3],
                    state,
                    owner.embedding,
                    meta[:, 3:4],
                    sampled=sampled if record_tokens else None,
                    parameters=owner.parameters[slots] if need_parameters else None,
                )
                output, latent, updated = outputs[:3]
                recorded = outputs[3] if record_tokens else None
                commit(
                    output,
                    latent,
                    updated,
                    meta,
                    owner.history,
                    owner.masks,
                    owner.state,
                    self.sampled,
                    recorded,
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
        registry[keys] = (shapes, live)

    def execute(self, groups, logits, hidden, sampled):
        count = sampled.shape[0]
        self.sampled[:count].copy_(sampled)
        head = getattr(self.owner, "capture_head_pending", False)
        shared = self.shared_keys is not None and len(groups) > 1 and not head
        if shared:
            assert self.shared_keys is not None
            keys = self.shared_keys
        else:
            keys = tuple(sorted(groups))
        packs = [keys] if shared else [(key,) for key in keys]
        registry = self.head_graphs if head else self.graphs
        live = set().union(*(registry[pack][1] for pack in packs))
        if "logits" in live:
            assert self.logits is not None
            self.logits[:count].copy_(logits)
        if "hidden" in live:
            assert self.hidden is not None
            self.hidden[:count].copy_(hidden)
        for pack in packs:
            shapes, _ = registry[pack]
            entries = [
                (*entry, i)
                for i, key in enumerate(pack)
                for entry in groups.get(key, [])
            ]
            n = len(entries)
            size = 1 << (n - 1).bit_length()
            if self.metadata_ring is not None:
                self.metadata_cpu = self.metadata_ring.acquire()
            self.metadata_cpu.fill_(-1)
            self.metadata_cpu[:, 2].fill_(self.owner.max_steps)
            self.metadata_cpu.numpy()[:n] = entries
            # The synchronous copy protects the pinned host buffer from reuse
            # while a previous pack's asynchronous transfer is still in flight.
            if self.metadata_ring is None:
                self.metadata[:size].copy_(self.metadata_cpu[:size])
            else:
                self.metadata_ring.copy(self.metadata[:size], self.metadata_cpu[:size])
            shapes[size].replay()
        return self.sampled[:count]
