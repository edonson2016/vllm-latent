# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check completed run coverage and record numerical agreement without hiding drift."""

import hashlib
import json

from summarize_round2 import ROOT, read

required = {f"control-model{i}.jsonl": 15 for i in range(4)}
required.update({f"final-generic{i}.jsonl": 15 for i in range(4)})
required.update({f"paired-model{i}.jsonl": 30 for i in range(4)})
required.update({f"approx-timing{i}.jsonl": 10 for i in range(4)})
required.update({f"approx-gated{i}.jsonl": 10 for i in range(4)})
required.update({f"fork-quality{i}.jsonl": 7 for i in range(4)})
required.update({f"qwen-sampled{i}.jsonl": 3 for i in range(4)})
required.update({f"exact-quality{i}.jsonl": 2 for i in range(1, 4)})
required.update({"exact-compact-valid-quality.jsonl": 2, "gated-quality0.jsonl": 3})
coverage = {
    name: {"expected": count, "actual": len(read(name))}
    for name, count in required.items()
}
assert all(x["actual"] == x["expected"] for x in coverage.values()), coverage


def load(name):
    return json.loads((ROOT / name).read_text())


def matches(a, b):
    assert len(a) == len(b)
    return dict(exact=sum(x == y for x, y in zip(a, b)), total=len(a))


verification = {"coverage": coverage}
for suffix in ["", "-eager"]:
    verification["input_paths" + suffix] = matches(
        [r["tokens"] for r in load(f"drift-legacy{suffix}.json")["rows"]],
        [r["tokens"] for r in load(f"drift-native{suffix}.json")["rows"]],
    )
verification["input_paths_fixed_norm"] = matches(
    [r["tokens"] for r in load("drift-norm-legacy.json")["rows"]],
    [r["tokens"] for r in load("drift-norm-native.json")["rows"]],
)
large, small = load("preemption-11000.json"), load("preemption-32.json")
verification["preemption"] = dict(
    **matches(large["tokens"], small["tokens"]),
    ample_cache_preemptions=sum(large["preemptions"]),
    constrained_preemptions=sum(small["preemptions"]),
)
assert verification["preemption"]["exact"] == 12
assert verification["preemption"]["constrained_preemptions"] > 0
verification["controller_oracle"] = load("swi-oracle-grid.json")
assert verification["controller_oracle"]["assertions_passed"]
verification["quality_trace_comparisons"] = []
for i in range(4):
    reference = {r["mode"]: r for r in read(f"qwen-sampled{i}.jsonl")}
    for filename in [
        f"fork-quality{i}.jsonl",
        f"exact-quality{i}.jsonl" if i else "exact-compact-valid-quality.jsonl",
    ]:
        for row in read(filename):
            if row["mode"] in reference:
                verification["quality_trace_comparisons"].append(
                    dict(
                        file=filename,
                        mode=row["mode"],
                        **matches(
                            row["traces"][0], reference[row["mode"]]["traces"][0]
                        ),
                    )
                )
unstable = []
for path in ROOT.glob("*.jsonl*"):
    for row in read(path.name):
        if row.get("quality"):
            assert len(row["quality"]) == 128
            assert len(row["traces"][0]) == 128
            continue
        hashes = row.get("hashes", row.get("output_hashes", []))
        if len(set(hashes)) > 1:
            unstable.append(
                dict(
                    file=path.name,
                    engine=row.get("engine"),
                    label=row.get("label"),
                    mode=row["mode"],
                    batch=row["batch"],
                    distinct_hashes=len(set(hashes)),
                )
            )
verification["greedy_repeat_variability"] = unstable
strict_base = read("adaptive-strict-base.jsonl")[0]
strict_gated = read("adaptive-strict-gated.jsonl")[0]
verification["adaptive_strict"] = matches(
    strict_base["traces"][0], strict_gated["traces"][0]
)
assert verification["adaptive_strict"]["exact"] == 128
verification["exact_nohead"] = []
for row in read("exact-nohead0.jsonl"):
    reference = next(
        r
        for r in read("paired-model0.jsonl")
        if r["engine"] == "qwen"
        and r["mode"] == row["mode"]
        and r["batch"] == row["batch"]
    )
    agreement = matches(row["traces"][0], reference["traces"][0])
    assert agreement["exact"] == row["batch"]
    verification["exact_nohead"].append(
        dict(mode=row["mode"], batch=row["batch"], **agreement)
    )
verification["unit_results"] = (ROOT / "unit-v6.txt").read_text().splitlines()[-1]
assert "37 passed" in verification["unit_results"]
arithmetic = load("approximation-arithmetic.json")
verification["approximation_arithmetic"] = dict(
    cases=len(arithmetic), exact=sum(all(r["equal"]) for r in arithmetic)
)
(ROOT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")

# A manifest of results and source snapshots; exclude itself.
manifest = {
    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(ROOT.rglob("*"))
    if p.is_file() and p.name != "manifest.json" and p.suffix != ".log"
}
(ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("Verified", len(required), "required result files")
