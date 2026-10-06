# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Assert numerical and lifecycle invariants against persisted experiment data."""

import json
from pathlib import Path

root = Path(__file__).resolve().parents[2] / "results" / "latent"


def read(name):
    return json.loads((root / name).read_text())


def rows(name):
    return [json.loads(line) for line in (root / name).read_text().splitlines()]


main = {(r["mode"], r["batch"]): r for r in rows("native.jsonl")}
final = {(r["mode"], r["batch"]): r for r in rows("final-controls.jsonl")}
for batch in [1, 8, 32]:
    assert main["stock", batch]["tokens"] == main["disabled", batch]["tokens"]
    assert main["embed_control", batch]["tokens"] == main["token", batch]["tokens"]
    assert final["stock", batch]["tokens"] == final["disabled", batch]["tokens"]
    assert final["token", batch]["tokens"] == main["embed_control", batch]["tokens"]
short = {r["mode"]: r for r in rows("short.jsonl")}
hf = read("hf-reference.json")
for kind in ["token", "soft", "hidden"]:
    assert short[kind]["tokens"][0] == hf[kind]
for name in ["replay-soft.jsonl", "replay-soft-apc.jsonl"]:
    assert all(t == short["soft"]["tokens"][0] for t in read(name)["tokens"])
limited = read("invariant-32.json")
unlimited = read("invariant-11000.json")
assert limited["tokens"] == unlimited["tokens"]
assert sum(limited["preemptions"]) > 0
assert sum(unlimited["preemptions"]) == 0
adaptive = read("adaptive.jsonl")
counts = adaptive["latent_counts"][0]
assert 0 < sum(counts) < 8 * 64
ordinary = read("preemption-counted.json")
reference = read("preemption-default.json")
summary = {
    "stock_disabled_exact": True,
    "token_program_stock_embedding_exact": True,
    "hf_short_references_exact": True,
    "stock_replay_references_exact": True,
    "invariant_preemption_exact": True,
    "invariant_preemptions": limited["preemptions"],
    "ordinary_preemptions": ordinary["preemptions"],
    "ordinary_requests_matching": sum(
        a == b for a, b in zip(ordinary["tokens"], reference["tokens"])
    ),
    "adaptive_latent_counts": counts,
}
(root / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
