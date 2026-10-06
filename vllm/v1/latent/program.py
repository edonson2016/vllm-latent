# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Typed straight-line tensor programs. No user code, loops, or host predicates.

Inputs: logits [B,V], hidden/token [B,D], step [B,1], state [B,8].
Outputs: embedding [B,D], latent [B,1], next state [B,8]. Programs are
validated at admission; all data-dependent decisions use tensor operations.
"""

import math

import torch


class Program:
    def __init__(self, spec, width, vocab, constants=None):
        self.constants = constants or {}
        if not isinstance(spec, dict) or set(spec) - {
            "ops",
            "embedding",
            "latent",
            "updates",
        }:
            raise ValueError("Invalid decode program fields")
        ops = spec.get("ops", [])
        if not isinstance(ops, list) or len(ops) > 64:
            raise ValueError("A program may contain at most 64 instructions")
        shapes = {"logits": vocab, "hidden": width, "token": width, "step": 1}
        shapes.update({f"s{i}": 1 for i in range(8)})
        booleans = set()
        self.ops = []
        for inst in ops:
            if not isinstance(inst, list) or len(inst) < 3:
                raise ValueError("Instructions are [destination, opcode, operands...]")
            dst, op, *args = inst
            if not isinstance(dst, str) or dst in shapes:
                raise ValueError("Destinations must be unique register names")
            if op == "CONST":
                if len(args) != 1 or type(args[0]) not in (int, float, bool):
                    raise ValueError("CONST requires one finite scalar")
                if not math.isfinite(args[0]):
                    raise ValueError("Nonfinite constants are forbidden")
                out = 1
                if isinstance(args[0], bool):
                    booleans.add(dst)
            else:
                refs = args[:-1] if op == "PROJECT" else args
                if any(not isinstance(a, str) or a not in shapes for a in refs):
                    raise ValueError(f"Unknown register in {inst}")
                dims = [shapes[a] for a in refs]
                if op not in {"SELECT", "AND", "OR", "EQ"} and any(
                    a in booleans for a in refs
                ):
                    raise ValueError(f"{op} requires numeric operands")
                if op in {"SOFTMAX", "ENTROPY", "EXPECT", "NORM", "SILU"}:
                    if len(args) != 1:
                        raise ValueError(f"{op} requires one register")
                    if op in {"SOFTMAX", "ENTROPY", "EXPECT"} and dims[0] != vocab:
                        raise ValueError(f"{op} requires a vocabulary vector")
                    out = 1 if op == "ENTROPY" else width if op == "EXPECT" else dims[0]
                elif op == "PROJECT":
                    if len(args) != 2 or args[1] not in self.constants:
                        raise ValueError("PROJECT requires a registered matrix")
                    matrix = self.constants[args[1]]
                    if matrix.ndim != 2 or matrix.shape[0] != dims[0]:
                        raise ValueError("Projection shape mismatch")
                    out = matrix.shape[1]
                elif op in {"ADD", "MUL", "DIV", "GT", "LT", "EQ", "AND", "OR"}:
                    if len(args) != 2 or (dims[0] != dims[1] and 1 not in dims):
                        raise ValueError(f"{op} operands must broadcast")
                    out = max(dims)
                    if op in {"GT", "LT", "EQ", "AND", "OR"}:
                        if out != 1:
                            raise ValueError("Predicates must be row scalars")
                        booleans.add(dst)
                    if op in {"AND", "OR"} and not set(args) <= booleans:
                        raise ValueError("Boolean operation requires predicates")
                elif op == "SELECT":
                    if len(args) != 3 or args[0] not in booleans:
                        raise ValueError("SELECT requires predicate, true, false")
                    if dims[1] != dims[2]:
                        raise ValueError("SELECT branches must have equal shapes")
                    if (args[1] in booleans) != (args[2] in booleans):
                        raise ValueError("SELECT branches must have equal types")
                    out = dims[1]
                    if args[1] in booleans and args[2] in booleans:
                        booleans.add(dst)
                else:
                    raise ValueError(f"Unknown opcode {op}")
            shapes[dst] = out
            self.ops.append((dst, op, args))
        self.output = spec.get("embedding", "")
        self.latent = spec.get("latent", "")
        self.updates = spec.get("updates", {})
        if (
            not isinstance(self.output, str)
            or not isinstance(self.latent, str)
            or shapes.get(self.output) != width
            or self.latent not in booleans
        ):
            raise ValueError("Program must return a model-width vector and predicate")
        if not isinstance(self.updates, dict) or any(
            k not in {f"s{i}" for i in range(8)} or shapes.get(v) != 1
            for k, v in self.updates.items()
        ):
            raise ValueError("State updates must target s0..s7 with row scalars")

    def __call__(self, logits, hidden, token, step, state, embedding):
        r = {"logits": logits, "hidden": hidden, "token": token, "step": step}
        r.update({f"s{i}": state[:, i : i + 1] for i in range(8)})
        for dst, op, args in self.ops:
            if op == "CONST":
                r[dst] = torch.full_like(
                    step,
                    args[0],
                    dtype=(torch.bool if isinstance(args[0], bool) else torch.float32),
                )
                continue
            x = r[args[0]]
            if op == "SOFTMAX":
                y = torch.softmax(x.float(), dim=-1)
            elif op == "ENTROPY":
                p = torch.softmax(x.float(), dim=-1)
                y = -(p * p.clamp_min(1e-30).log()).sum(-1, keepdim=True)
            elif op == "EXPECT":
                y = x.to(embedding.dtype) @ embedding
            elif op == "NORM":
                y = (
                    x.float()
                    * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
                ).to(x.dtype)
            elif op == "SILU":
                y = torch.nn.functional.silu(x)
            elif op == "PROJECT":
                matrix = self.constants[args[1]]
                y = x.to(matrix.dtype) @ matrix
            elif op == "SELECT":
                y = torch.where(x, r[args[1]], r[args[2]])
            else:
                z = r[args[1]]
                if op == "ADD":
                    y = x + z
                elif op == "MUL":
                    y = x * z
                elif op == "DIV":
                    y = x / z
                elif op == "GT":
                    y = x > z
                elif op == "LT":
                    y = x < z
                elif op == "EQ":
                    y = x == z
                elif op == "AND":
                    y = x & z
                elif op == "OR":
                    y = x | z
            r[dst] = y
        next_state = torch.cat(
            [r[self.updates.get(f"s{i}", f"s{i}")].float() for i in range(8)], dim=1
        )
        result = r[self.output].to(token.dtype)
        finite = torch.isfinite(result).all(-1, keepdim=True)
        latent = r[self.latent] & finite
        # A bad numeric result falls back to the sampled token, never poisons KV.
        result = torch.where(latent, result, token)
        next_state = torch.where(torch.isfinite(next_state), next_state, state)
        return result, latent, next_state


def preset(kind="soft", latent_steps=32, threshold=2.0):
    """Finite latent prefix followed by normal decoding; entropy is a toy gate.

    The entropy preset is not the full SwiReasoning policy.
    """
    ops = [["limit", "CONST", latent_steps], ["active", "LT", "step", "limit"]]
    if kind == "token":
        ops = [["active", "CONST", False]]
        out = "token"
    elif kind == "hidden":
        out = "hidden"
    elif kind == "norm_hidden":
        ops += [["norm", "NORM", "hidden"]]
        out = "norm"
    elif kind in {"soft", "entropy"}:
        ops += [["p", "SOFTMAX", "logits"], ["soft", "EXPECT", "p"]]
        out = "soft"
        if kind == "entropy":
            ops += [
                ["h", "ENTROPY", "logits"],
                ["tau", "CONST", threshold],
                ["uncertain", "GT", "h", "tau"],
                ["gate", "AND", "active", "uncertain"],
            ]
    else:
        raise ValueError(f"Unknown preset {kind}")
    return {
        "ops": ops,
        "embedding": out,
        "latent": "gate" if kind == "entropy" else "active",
    }
