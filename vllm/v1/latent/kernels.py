# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Captured token/latent input selection and optional masked expectation kernels."""

import torch

from vllm import envs
from vllm.triton_utils import tl, triton


@triton.jit
def _input(
    ids,
    weight,
    history,
    masks,
    indices,
    output,
    D: tl.constexpr,
    V: tl.constexpr,
    HISTORY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    index = tl.load(indices + row)
    tl.device_assert(
        (index >= -1) & (index < HISTORY), "latent history index out of bounds"
    )
    latent = tl.load(masks + tl.maximum(index, 0)) & (index >= 0)
    if latent:
        values = tl.load(history + index * D + cols, cols < D, 0)
    else:
        token = tl.load(ids + row)
        tl.device_assert((token >= 0) & (token < V), "latent token index out of bounds")
        values = tl.load(weight + token * D + cols, cols < D, 0)
    tl.store(output + row * D + cols, values, cols < D)


@torch.library.custom_op("vllm::latent_input", mutates_args=())
def latent_input(
    ids: torch.Tensor,
    weight: torch.Tensor,
    history: torch.Tensor,
    masks: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty(
        (ids.numel(), weight.shape[1]), device=weight.device, dtype=weight.dtype
    )
    _input[(ids.numel(),)](
        ids,
        weight,
        history,
        masks,
        indices,
        output,
        weight.shape[1],
        weight.shape[0],
        masks.numel(),
        triton.next_power_of_2(weight.shape[1]),
    )
    return output


@latent_input.register_fake
def _fake_input(ids, weight, history, masks, indices):
    return weight.new_empty((ids.numel(), weight.shape[1]))


class LatentEmbedding(torch.nn.Module):
    def __init__(self, original, history, masks, indices):
        super().__init__()
        self.weight = original.weight
        self.register_buffer("history", history, persistent=False)
        self.register_buffer("masks", masks, persistent=False)
        self.register_buffer("indices", indices, persistent=False)

    def forward(self, ids):
        return latent_input(ids, self.weight, self.history, self.masks, self.indices)


@triton.jit
def _commit(
    result,
    latent,
    state,
    metadata,
    history,
    masks,
    states,
    sampled,
    recorded,
    HAS_RECORDED: tl.constexpr,
    D: tl.constexpr,
    STEPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(metadata + row * 4 + 1)
    if slot >= 0:
        source = tl.load(metadata + row * 4)
        step = tl.load(metadata + row * 4 + 2)
        index = slot * STEPS + step
        cols = tl.arange(0, BLOCK)
        value = tl.load(result + row * D + cols, cols < D, 0)
        tl.store(history + index * D + cols, value, cols < D)
        mask = tl.load(latent + row)
        tl.store(masks + index, mask)
        if HAS_RECORDED:
            tl.store(sampled + source, tl.load(recorded + row))
        elif mask:
            tl.store(sampled + source, 0)
        regs = tl.arange(0, 8)
        values = tl.load(state + row * 8 + regs)
        tl.store(states + slot * 8 + regs, values)


def commit(
    result, latent, state, metadata, history, masks, states, sampled, recorded=None
):
    _commit[(result.shape[0],)](
        result,
        latent,
        state,
        metadata,
        history,
        masks,
        states,
        sampled,
        recorded,
        recorded is not None,
        result.shape[1],
        history.shape[1],
        triton.next_power_of_2(result.shape[1]),
    )


@triton.jit
def _expect_parts(
    prob,
    embedding,
    gate,
    parts,
    V: tl.constexpr,
    D: tl.constexpr,
    CHUNKS: tl.constexpr,
    BV: tl.constexpr,
    BD: tl.constexpr,
):
    row, vc, dc = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    cols = dc * BD + tl.arange(0, BD)
    value = tl.full((BD,), 0, tl.float32)
    if tl.load(gate + row):
        vocab = vc * BV + tl.arange(0, BV)
        p = tl.load(prob + row * V + vocab, vocab < V, 0)
        e = tl.load(
            embedding + vocab[:, None] * D + cols[None, :],
            (vocab[:, None] < V) & (cols[None, :] < D),
            0,
        )
        # Match EXPECT's BF16 probability rounding before FP32 accumulation.
        p = p.to(e.dtype).to(tl.float32)
        value = tl.sum(p[:, None] * e.to(tl.float32), axis=0)
    tl.store(parts + (row * CHUNKS + vc) * D + cols, value, cols < D)


@triton.jit
def _expect_reduce(
    parts,
    output,
    D: tl.constexpr,
    CHUNKS: tl.constexpr,
    BC: tl.constexpr,
    BD: tl.constexpr,
):
    row, dc = tl.program_id(0), tl.program_id(1)
    chunks, cols = tl.arange(0, BC), dc * BD + tl.arange(0, BD)
    values = tl.load(
        parts + (row * CHUNKS + chunks[:, None]) * D + cols[None, :],
        (chunks[:, None] < CHUNKS) & (cols[None, :] < D),
        0,
    )
    tl.store(output + row * D + cols, tl.sum(values, axis=0), cols < D)


def masked_expect(prob, embedding, gate=None):
    batch, vocab = prob.shape
    width = embedding.shape[1]
    chunks = triton.cdiv(vocab, 512)
    if gate is None:
        gate = torch.ones(batch, device=prob.device, dtype=torch.bool)
    parts = torch.empty((batch, chunks, width), device=prob.device)
    output = torch.empty((batch, width), device=prob.device, dtype=embedding.dtype)
    _expect_parts[(batch, chunks, triton.cdiv(width, 32))](
        prob, embedding, gate, parts, vocab, width, chunks, 512, 32
    )
    _expect_reduce[(batch, triton.cdiv(width, 32))](
        parts, output, width, chunks, triton.next_power_of_2(chunks), 32
    )
    return output


@triton.jit
def _soft_stats(logits, partial, V: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr):
    row, chunk = tl.program_id(0), tl.program_id(1)
    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(logits + row * V + cols, cols < V, -float("inf"))
    maximum = tl.max(x, 0)
    p = tl.exp(x - maximum)
    total = tl.sum(p, 0)
    weighted = tl.sum(p * tl.where(cols < V, x, 0.0), 0)
    offset = (row * C + chunk) * 3
    tl.store(partial + offset, maximum)
    tl.store(partial + offset + 1, total)
    tl.store(partial + offset + 2, weighted)


@triton.jit
def _merge_stats(partial, stats, C: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    base = (row * C + cols) * 3
    maximum = tl.load(partial + base, cols < C, -float("inf"))
    total = tl.load(partial + base + 1, cols < C, 0.0)
    weighted = tl.load(partial + base + 2, cols < C, 0.0)
    m = tl.max(maximum, 0)
    correction = tl.exp(maximum - m)
    s = tl.sum(total * correction, 0)
    w = tl.sum(weighted * correction, 0)
    tl.store(stats + row * 3, m)
    tl.store(stats + row * 3 + 1, 1.0 / s)
    tl.store(stats + row * 3 + 2, tl.log(s) + m - w / s)


def soft_stats(logits):
    batch, vocab = logits.shape
    chunks = triton.cdiv(vocab, 2048)
    partial = torch.empty((batch, chunks, 3), device=logits.device)
    stats = torch.empty((batch, 3), device=logits.device)
    _soft_stats[(batch, chunks)](logits, partial, vocab, chunks, 2048)
    _merge_stats[(batch,)](partial, stats, chunks, triton.next_power_of_2(chunks))
    return stats


@triton.jit
def _soft_expect_parts(
    logits,
    embedding,
    gate,
    stats,
    parts,
    V: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    BV: tl.constexpr,
    BD: tl.constexpr,
):
    row, vc, dc = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    cols = dc * BD + tl.arange(0, BD)
    value = tl.full((BD,), 0, tl.float32)
    if tl.load(gate + row):
        vocab = vc * BV + tl.arange(0, BV)
        x = tl.load(logits + row * V + vocab, vocab < V, -float("inf"))
        m = tl.load(stats + row * 3)
        inv = tl.load(stats + row * 3 + 1)
        e = tl.load(
            embedding + vocab[:, None] * D + cols[None, :],
            (vocab[:, None] < V) & (cols[None, :] < D),
            0,
        )
        p = (tl.exp(x - m) * inv).to(e.dtype).to(tl.float32)
        value = tl.sum(p[:, None] * e.to(tl.float32), 0)
    tl.store(parts + (row * C + vc) * D + cols, value, cols < D)


def fused_soft_expect(logits, embedding, gate=None, stats=None):
    batch, vocab = logits.shape
    width = embedding.shape[1]
    chunks = triton.cdiv(vocab, 512)
    if stats is None:
        stats = soft_stats(logits)
    if gate is None:
        gate = torch.ones(batch, device=logits.device, dtype=torch.bool)
    parts = torch.empty((batch, chunks, width), device=logits.device)
    output = torch.empty((batch, width), device=logits.device, dtype=embedding.dtype)
    _soft_expect_parts[(batch, chunks, triton.cdiv(width, 32))](
        logits, embedding, gate, stats, parts, vocab, width, chunks, 512, 32
    )
    _expect_reduce[(batch, triton.cdiv(width, 32))](
        parts, output, width, chunks, triton.next_power_of_2(chunks), 32
    )
    return output


@triton.jit
def _expect_gemm(
    prob,
    embedding,
    gate,
    partial,
    B: tl.constexpr,
    V: tl.constexpr,
    D: tl.constexpr,
    SPLITS: tl.constexpr,
    KPART: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    mb, nb, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = mb * BM + tl.arange(0, BM)
    cols = nb * BN + tl.arange(0, BN)
    active = tl.load(gate + rows, rows < B, False)
    accum = tl.full((BM, BN), 0, tl.float32)
    if tl.sum(active.to(tl.int32), 0) > 0:
        for offset in range(KPART // BK):
            vocab = split * KPART + offset * BK + tl.arange(0, BK)
            p = tl.load(
                prob + rows[:, None] * V + vocab[None, :],
                active[:, None] & (vocab[None, :] < V),
                0,
            )
            e = tl.load(
                embedding + vocab[:, None] * D + cols[None, :],
                (vocab[:, None] < V) & (cols[None, :] < D),
                0,
            )
            accum = tl.dot(p.to(e.dtype), e, accum)
    tl.store(
        partial + (rows[:, None] * SPLITS + split) * D + cols[None, :],
        accum,
        (rows[:, None] < B) & (cols[None, :] < D),
    )


def gated_expect(prob, embedding, gate=None):
    batch, vocab = prob.shape
    width = embedding.shape[1]
    splits = 16
    if gate is None:
        gate = torch.ones(batch, device=prob.device, dtype=torch.bool)
    partial = torch.empty((batch, splits, width), device=prob.device)
    output = torch.empty((batch, width), device=prob.device, dtype=embedding.dtype)
    _expect_gemm[(triton.cdiv(batch, 16), triton.cdiv(width, 64), splits)](
        prob,
        embedding,
        gate,
        partial,
        batch,
        vocab,
        width,
        splits,
        triton.cdiv(vocab, splits * 64) * 64,
        16,
        64,
        64,
    )
    _expect_reduce[(batch, triton.cdiv(width, 32))](
        partial, output, width, splits, splits, 32
    )
    return output


def conditional_expect(prob, embedding, gate=None):
    """Skip an all-inactive GEMM with a GPU conditional CUDA graph node.

    Active batches retain the same cuBLAS computation as ordinary EXPECT.
    Warmup outside capture evaluates it eagerly without inspecting a GPU value.
    """
    if gate is None or not torch.cuda.is_current_stream_capturing():
        return prob.to(embedding.dtype) @ embedding
    output = torch.zeros(
        (prob.shape[0], embedding.shape[1]),
        device=embedding.device,
        dtype=embedding.dtype,
    )
    weights = prob.to(embedding.dtype)
    graph = torch.cuda.CUDAGraph.get_currently_capturing_graph()
    graph.begin_capture_to_if_node(gate.any())
    if envs.VLLM_BATCH_INVARIANT:
        # mm.out bypasses vLLM's invariant matmul dispatch. Match the eager
        # warmup's backend so capture neither changes arithmetic nor creates
        # a first-use cuBLAS handle inside the conditional node.
        output.copy_(weights @ embedding)
    else:
        torch.mm(weights, embedding, out=output)
    graph.end_capture_to_conditional_node()
    return output


def compact_expect(prob, embedding, gate, exact=False):
    """Experimental device compaction with statically captured GEMM buckets.

    Every branch has a fixed shape; active count only selects a CUDA graph
    conditional node. Changing GEMM shape can change floating-point rounding.
    """
    active = gate.reshape(-1)
    order = active.to(torch.int32).argsort(descending=True, stable=True)
    weights = prob[order].to(embedding.dtype)
    output = torch.zeros(
        (prob.shape[0], embedding.shape[1]), device=prob.device, dtype=embedding.dtype
    )
    count = active.sum()
    previous, size = 0, 1
    capturing = torch.cuda.is_current_stream_capturing()
    graph = torch.cuda.CUDAGraph.get_currently_capturing_graph() if capturing else None
    while previous < prob.shape[0]:
        size = min(size, prob.shape[0])
        selected = (count > previous) & (count <= size)
        if graph is not None:
            graph.begin_capture_to_if_node(selected)
        mixture = weights[:size] @ embedding
        values = torch.where(active[order[:size], None] & selected, mixture, 0)
        output.index_add_(0, order[:size], values)
        if graph is not None:
            graph.end_capture_to_conditional_node()
        previous, size = size, size + 1 if exact else size * 2
    return output
