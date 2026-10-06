# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Report numerical diagnostics separately from standard-mode performance."""

import json
from pathlib import Path

out = Path(__file__).resolve().parents[2] / "results/optimization"


def read(name):
    return [json.loads(s) for s in (out / name).read_text().splitlines()]


def compare(old, new):
    differences = [
        dict(
            request=i,
            first_position=next(j for j, (a, b) in enumerate(zip(x, y)) if a != b),
        )
        for i, (x, y) in enumerate(zip(old["tokens"], new["tokens"]))
        if x != y
    ]
    return dict(
        mode=old["mode"],
        batch=old["batch"],
        exact_requests=len(old["tokens"]) - len(differences),
        requests=len(old["tokens"]),
        differences=differences,
        old_stable=len(set(old["hashes"])) == 1,
        new_stable=len(set(new["hashes"])) == 1,
        old_seconds=old["median_seconds"],
        new_seconds=new["median_seconds"],
        latency_reduction_percent=100
        * (1 - new["median_seconds"] / old["median_seconds"]),
    )


invariant = []
for i, size in enumerate(["0.6B", "1.7B", "4B", "8B"]):
    data = read(f"invariant-model{i}.jsonl")
    old = {r["mode"]: r for r in data if r["label"] == "baseline-invariant"}
    candidates = read("invariant-fixed-model0.jsonl") if i == 0 else data
    new = {
        r["mode"]: r for r in candidates if r["label"].startswith("optimized-invariant")
    }
    assert old.keys() == new.keys()
    assert len(old) == (3 if i < 2 else 5)
    for mode in old:
        invariant.append(dict(model=size, **compare(old[mode], new[mode])))

head_rows = read("head-invariant-8B.jsonl")
head_index = {(r["label"], r["batch"]): r for r in head_rows}
head = [
    compare(
        head_index["head-control-invariant", b], head_index["head-skip-invariant", b]
    )
    for b in [1, 32]
]
preemption = json.loads((out / "preemption-verification.json").read_text())
assert preemption["exact_requests"] == preemption["requests"] == 12
assert all(r["exact_requests"] == r["requests"] for r in head)
result = dict(invariant=invariant, head=head, preemption=preemption)
(out / "diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {
            "invariant": [
                (r["model"], r["mode"], r["exact_requests"]) for r in invariant
            ],
            "head": [(r["batch"], r["exact_requests"]) for r in head],
            "preemption": preemption,
        },
        indent=2,
    )
)
