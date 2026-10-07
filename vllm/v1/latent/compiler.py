# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static dependency analysis and common-subexpression sharing at admission."""

import copy
from typing import Any

import torch


def dependencies(program):
    if getattr(program, "builtin", False):
        return [], program.live
    live = {program.output, program.latent, *program.updates.values()}
    kept = []
    for dst, op, args in reversed(program.ops):
        if dst not in live:
            continue
        kept.append((dst, op, args))
        if op not in {"CONST", "PARAM"}:
            live.update(args[:-1] if op in {"PROJECT", "TOPK_EXPECT"} else args)
    return list(reversed(kept)), live


def optimize(program):
    if getattr(program, "builtin", False):
        return program
    result = copy.copy(program)
    result.ops, result.live = dependencies(program)
    nodes = {dst: (dst, op, args) for dst, op, args in result.ops}
    ordered = []
    seen = set()

    def emit(name):
        if name in seen or name not in nodes:
            return
        dst, op, args = nodes[name]
        if op not in {"CONST", "PARAM"}:
            for arg in args[:-1] if op in {"PROJECT", "TOPK_EXPECT"} else args:
                emit(arg)
        seen.add(name)
        ordered.append((dst, op, args))

    for name in [result.latent, result.output, *result.updates.values()]:
        emit(name)
    result.ops = ordered
    return result


def inactive(program, step):
    """Prove a stateless transition inactive using only scheduler position.

    Unknown device values propagate conservatively. This never reads a tensor
    or substitutes a host decision for a data-dependent predicate.
    """
    if program.updates:
        return False
    values = {"step": step}
    for dst, op, args in program.ops:
        x = values.get(args[0]) if args else None
        y = values.get(args[1]) if len(args) > 1 else None
        if op == "CONST":
            values[dst] = args[0]
        elif op == "AND" and (x is False or y is False):
            values[dst] = False
        elif op == "OR" and (x is True or y is True):
            values[dst] = True
        elif x is not None and y is not None:
            if op == "LT":
                values[dst] = x < y
            elif op == "GT":
                values[dst] = x > y
            elif op == "EQ":
                values[dst] = x == y
            elif op == "AND":
                values[dst] = x and y
            elif op == "OR":
                values[dst] = x or y
    return values.get(program.latent) is False


def active_without_logits(program, step):
    if program.fallback != "zero" or {"logits", "token"} & dependencies(program)[1]:
        return False
    # Conservative proof for the common fixed hidden-prefix contract only.
    nodes = {dst: (op, args) for dst, op, args in program.ops}
    op, args = nodes.get(program.latent, (None, []))
    if op == "CONST":
        return args == [True]
    if op == "LT" and args[0] == "step":
        limit_op, limit_args = nodes.get(args[1], (None, []))
        return limit_op == "CONST" and step < limit_args[0]
    return False


class SharedProgram:
    """Evaluate compatible requests together, sharing identical tensor nodes."""

    def __init__(self, programs, share=True):
        self.programs = [copy.copy(p) for p in programs]
        self.union = (
            share
            and len(programs) > 1
            and all(getattr(p, "union_gate", False) for p in programs)
        )
        self.gates = []
        if self.union:
            for p in self.programs:
                gate = copy.copy(p)
                gate.output, gate.updates = "token", {}
                gate = optimize(gate)
                safe = not p.updates and not any(
                    op == "EXPECT" for _, op, _ in gate.ops
                )
                self.gates.append(gate if safe else None)
                p.conditional_expect = True
        if share and len(programs) > 1 and not self.union:
            # One shared dense GEMM serves all policies. A per-policy gate
            # cannot guard that common result without a union-of-consumers mask.
            for p in self.programs:
                p.conditional_expect = False
        self.share = share
        self.live = set().union(*(dependencies(p)[1] for p in programs))

    def __call__(
        self,
        logits,
        hidden,
        token,
        step,
        state,
        embedding,
        ids,
        sampled=None,
        parameters=None,
    ):
        cache: dict[Any, torch.Tensor] | None = {} if self.share else None
        if self.union:
            assert cache is not None
            for i, (program, gate) in enumerate(zip(self.programs, self.gates)):
                selected = ids == i
                if gate is not None:
                    mask = gate(
                        logits,
                        hidden,
                        token,
                        step,
                        state,
                        embedding,
                        cache=cache,
                        parameters=parameters,
                        predicate_only=True,
                    )
                    selected = selected & mask
                expressions: dict[str, Any] = {
                    name: ("input", name)
                    for name in ["logits", "hidden", "token", "step"]
                    + [f"s{i}" for i in range(8)]
                }
                for dst, op, args in program.ops:
                    key = (
                        op,
                        tuple((type(a).__name__, a) for a in args)
                        if op in {"CONST", "PARAM"}
                        else tuple(expressions[a] for a in args),
                        None,
                    )
                    expressions[dst] = key
                    if op == "EXPECT":
                        consumer = ("GATE", key)
                        cache[consumer] = (
                            cache.get(consumer, torch.zeros_like(selected)) | selected
                        )
        result, mask, next_state = (
            token,
            torch.zeros_like(step, dtype=torch.bool),
            state,
        )
        recorded = sampled
        for i, program in enumerate(self.programs):
            selected = ids == i
            if sampled is None:
                output, latent, updated = program(
                    logits,
                    hidden,
                    token,
                    step,
                    state,
                    embedding,
                    cache=cache,
                    parameters=parameters,
                    valid=selected,
                )
            else:
                output, latent, updated, tokens = program.execute(
                    logits,
                    hidden,
                    token,
                    step,
                    state,
                    embedding,
                    sampled,
                    cache=cache,
                    parameters=parameters,
                    valid=selected,
                )
                recorded = torch.where(ids == i, tokens, recorded)
            result = torch.where(selected, output, result)
            mask = torch.where(selected, latent, mask)
            next_state = torch.where(selected, updated, next_state)
        if sampled is None:
            return result, mask, next_state
        return result, mask, next_state, recorded


def fuse_expectations(program):
    result = copy.copy(program)
    sources = {dst: args[0] for dst, op, args in program.ops if op == "SOFTMAX"}
    result.ops = [
        (dst, "SOFT_EXPECT", [sources[args[0]]])
        if op == "EXPECT" and args[0] in sources
        else (dst, op, args)
        for dst, op, args in program.ops
    ]
    result.fused_expect = True
    return optimize(result)


def can_share(programs):
    """Share a batch only when it contains a repeated dense expectation.

    Request-specific projections stay grouped; running each projection over
    every request would multiply their work even if it saves a graph launch.
    """
    seen: set[Any] = set()
    repeated = False
    for program in programs:
        if getattr(program, "builtin", False):
            return False
        expressions: dict[str, Any] = {
            name: name for name in ["logits", "hidden", "token", "step"]
        }
        expressions.update({f"s{i}": f"s{i}" for i in range(8)})
        current = set()
        for dst, op, args in dependencies(program)[0]:
            if op in {"PROJECT", "TOPK_EXPECT", "SOFT_EXPECT"}:
                return False
            key = (
                op,
                tuple((type(x).__name__, x) for x in args)
                if op in {"CONST", "PARAM"}
                else tuple(expressions[x] for x in args),
            )
            expressions[dst] = key
            if op == "EXPECT":
                current.add(key)
        repeated |= bool(seen & current)
        seen.update(current)
    return repeated


def needs_device_gate(program):
    if getattr(program, "builtin", False):
        return True
    gate = copy.copy(program)
    gate.output = "token"
    gate.updates = {}
    live = dependencies(gate)[1] - {"token", "step"}
    return not program.updates and bool(
        live & {"logits", "hidden", *[f"s{i}" for i in range(8)]}
    )
