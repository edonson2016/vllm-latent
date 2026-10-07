# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded GPU state machine matching QwenReasoning's SwiReasoning policy.

State registers hold mode, dwell, entropy reference, lock, switch count,
queue cursor, queue kind and termination budget. Queues are admission constants;
there is no host decision, callback, or device-to-host transfer in a transition.
"""

import math
from typing import Any

import torch


class SwiProgram:
    alpha: float
    beta: float
    mass: float
    window: int
    max_tokens: int
    termination_max_tokens: int
    topk: int
    start_id: int
    end_id: int
    stop_id: int
    linebreak_id: int
    convergence_ids: list[int]
    termination_ids: list[int]
    math_ids: list[int]
    builtin = True
    output = "token"
    latent = "active"
    fallback = "token"
    ops: list[tuple[str, str, list[Any]]] = []
    updates = {"s0": "s0"}  # A stateful policy must never be statically skipped.
    live = {"logits", "token", "sampled"}

    def __init__(self, spec, width, vocab, constants=None):
        allowed = {
            "policy",
            "alpha",
            "beta",
            "window",
            "max_switch_count",
            "convergence_ids",
            "termination_ids",
            "termination_max_tokens",
            "math_ids",
            "linebreak_id",
            "start_id",
            "end_id",
            "stop_id",
            "max_tokens",
            "expectation",
            "topk",
            "mass",
            "lowrank",
        }
        if set(spec) - allowed:
            raise ValueError("Unknown SwiReasoning field")
        self.spec = dict(spec)
        for key in ("alpha", "beta", "mass"):
            value = spec.get(key, {"alpha": 1.0, "beta": 0.7, "mass": 0.99}[key])
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{key} must be between zero and one")
            setattr(self, key, value)
        for key, default in (
            ("window", 512),
            ("max_tokens", 512),
            ("termination_max_tokens", 32),
            ("topk", 64),
        ):
            value = spec.get(key, default)
            if type(value) is not int or not 1 <= value <= 32768:
                raise ValueError(f"Invalid {key}")
            setattr(self, key, value)
        self.max_switch_count = spec.get("max_switch_count")
        if self.max_switch_count is not None and (
            type(self.max_switch_count) is not int or self.max_switch_count < 1
        ):
            raise ValueError("max_switch_count must be positive or null")
        for key in ("start_id", "end_id", "linebreak_id", "stop_id"):
            value = spec.get(key)
            if type(value) is not int or not 0 <= value < vocab:
                raise ValueError(f"Invalid {key}")
            setattr(self, key, value)
        for key in ("convergence_ids", "termination_ids", "math_ids"):
            value = spec.get(key, [])
            if (
                not isinstance(value, list)
                or len(value) > 256
                or any(type(x) is not int or not 0 <= x < vocab for x in value)
            ):
                raise ValueError(f"Invalid {key}")
            setattr(self, key, value)
        self.expectation = spec.get("expectation", "fp32")
        if self.expectation not in {"fp32", "bf16", "topk", "adaptive", "lowrank"}:
            raise ValueError("Unknown expectation implementation")
        if self.topk > vocab:
            raise ValueError("topk exceeds vocabulary")
        self.constants = constants or {}
        self.lowrank = spec.get("lowrank", "embedding")
        if self.expectation == "lowrank":
            left = self.constants.get(self.lowrank + ".left")
            right = self.constants.get(self.lowrank + ".right")
            if (
                left is None
                or right is None
                or left.ndim != 2
                or right.ndim != 2
                or left.shape[0] != vocab
                or right.shape != (left.shape[1], width)
            ):
                raise ValueError("lowrank requires compatible registered factors")
        self.conditional_expect = False

    def prepare(self, embedding, fp32_table=None):
        self.table = (
            (embedding.float() if fp32_table is None else fp32_table)
            if self.expectation == "fp32"
            else embedding
        )
        device = embedding.device
        self.queues = [
            torch.tensor(ids or [0], device=device, dtype=torch.long)
            for ids in (self.convergence_ids, self.termination_ids)
        ]
        self.math = torch.tensor(self.math_ids, device=device, dtype=torch.long)

    def decide(self, entropy, sampled, step, state):
        mode, stay, ref, locked, count, cursor, queue, budget = state.split(1, dim=1)
        first = step == 0
        budget = torch.where(first, -1, budget)
        locked = (locked != 0) | (sampled == self.end_id)
        forced = torch.full_like(sampled, -1)
        for kind, (ids, values) in enumerate(
            zip((self.convergence_ids, self.termination_ids), self.queues), 1
        ):
            pending = (queue == kind) & (cursor < len(ids))
            value = values[cursor.long().clamp(0, len(values) - 1)]
            forced = torch.where(pending, value, forced)
        tok = torch.where(forced >= 0, forced, sampled)
        cursor = cursor + (forced >= 0)
        budget = torch.where(budget >= 0, budget - 1, budget)
        forced = torch.where(budget == 0, self.stop_id, forced)
        stay = torch.where(first, 0, stay + 1)
        ref = torch.where(first, entropy, ref)
        to_normal = ~first & (mode == 0) & (entropy < ref)
        to_soft = (
            ~first & (mode == 1) & (entropy > ref) & (stay >= self.window) & ~locked
        )
        changed = to_normal | to_soft
        mode = torch.where(to_normal, 1, torch.where(to_soft, 0, mode))
        stay = torch.where(changed, 0, stay)
        ref = torch.where(changed, entropy, ref)
        count = count + to_normal
        if self.max_switch_count is not None:
            threshold = self.max_switch_count
            converge = to_normal & (count >= threshold) & (count <= 2 * threshold)
            terminate = to_normal & (count > 2 * threshold)
            if self.convergence_ids:
                queue = torch.where(converge, 1, queue)
                cursor = torch.where(converge, 0, cursor)
            if self.termination_ids:
                queue = torch.where(terminate, 2, queue)
                cursor = torch.where(terminate, 0, cursor)
                budget = torch.where(terminate, self.termination_max_tokens - 1, budget)
        exempt = (tok == self.math.reshape(1, -1)).any(-1, keepdim=True)
        soft = (mode == 0) & ~locked & ~exempt
        active = to_normal | soft
        # The reference constructs blend weights with Python doubles, then
        # uploads float32 weights. Preserve that rounding order on the GPU.
        ratio = step.double() / self.max_tokens
        weight = torch.where(
            to_normal,
            self.beta + (1 - self.beta) * ratio,
            torch.where(
                first,
                0.9,
                torch.where(to_soft, self.alpha + (1 - self.alpha) * ratio, 1.0),
            ),
        )
        weight = weight.float()
        anchor = torch.where(
            to_normal, self.end_id, torch.where(first, self.linebreak_id, self.start_id)
        )
        next_state = torch.cat(
            [
                x.float()
                for x in (mode, stay, ref, locked, count, cursor, queue, budget)
            ],
            1,
        )
        recorded = torch.where(forced >= 0, forced, sampled)
        return active, weight, anchor, next_state, recorded

    def execute(
        self,
        logits,
        hidden,
        token,
        step,
        state,
        embedding,
        sampled,
        cache=None,
        parameters=None,
        valid=None,
    ):
        p = logits.float().softmax(-1)
        entropy = -(p * p.clamp_min(1e-12).log()).sum(-1, keepdim=True)
        active, weight, anchor, updated, recorded = self.decide(
            entropy, sampled, step, state
        )
        updated = torch.where(torch.isfinite(updated), updated, state)
        if valid is not None:
            active = active & valid
        graph = None
        if (
            self.expectation in {"topk", "adaptive", "lowrank"}
            and self.conditional_expect
            and p.is_cuda
            and torch.cuda.is_current_stream_capturing()
        ):
            # These explicit approximations need the same all-inactive skip
            # as the dense backend. The branch and allocation are captured;
            # no Python callback or host predicate runs during decoding.
            gated_mixture = torch.zeros_like(token)
            graph = torch.cuda.CUDAGraph.get_currently_capturing_graph()
            graph.begin_capture_to_if_node(active.any())
        if self.expectation in {"topk", "adaptive"}:
            values, indices = p.topk(self.topk, dim=-1)
            if self.expectation == "adaptive":
                keep = values.cumsum(-1) - values < self.mass
                values = values * keep
            values = values / values.sum(-1, keepdim=True)
            mixture = (
                (embedding[indices].float() * values.unsqueeze(-1))
                .sum(1)
                .to(embedding.dtype)
            )
        elif self.expectation == "lowrank":
            left = self.constants[self.lowrank + ".left"]
            right = self.constants[self.lowrank + ".right"]
            mixture = ((p.to(left.dtype) @ left) @ right).to(embedding.dtype)
        else:
            from vllm.v1.latent.kernels import conditional_expect

            if getattr(self, "compact_expect", False) or getattr(
                self, "exact_compact", False
            ):
                from vllm.v1.latent.kernels import compact_expect

                mixture = compact_expect(
                    p, self.table, active, exact=getattr(self, "exact_compact", False)
                ).to(embedding.dtype)
            else:
                mixture = conditional_expect(
                    p, self.table, active if self.conditional_expect else None
                ).to(embedding.dtype)
        if graph is not None:
            gated_mixture.copy_(mixture)
            graph.end_capture_to_conditional_node()
            mixture = gated_mixture
        eased = (
            weight * mixture.float() + (1 - weight) * embedding[anchor[:, 0]].float()
        )
        output = torch.where(
            active, eased.to(embedding.dtype), embedding[recorded[:, 0]]
        )
        finite = torch.isfinite(output).all(-1, keepdim=True)
        output = torch.where(finite, output, embedding[recorded[:, 0]])
        return output, active & finite, updated, recorded


def swi_preset(tokenizer, max_tokens=512, **overrides):
    def encode(text):
        value = tokenizer.encode(text, add_special_tokens=False)
        return value if isinstance(value, list) else value.ids

    spec = dict(
        policy="swireasoning",
        start_id=encode("<think>")[-1],
        end_id=encode("</think>")[-1],
        linebreak_id=encode("\n")[-1],
        stop_id=tokenizer.eos_token_id,
        convergence_ids=encode("</think>"),
        termination_ids=encode("</think>\n\nThe final answer is"),
        max_tokens=max_tokens,
    )
    spec.update(overrides)
    return spec
