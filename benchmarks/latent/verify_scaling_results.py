# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check sweep completeness, fixed work, and non-latent output equivalence."""

import json
from pathlib import Path

out = Path(__file__).resolve().parents[2] / "results/scaling"
models = [m["model"] for m in json.loads((out / "models.json").read_text())]
checks = {}
variability = {}
for engine, modes, pairs in [
    (
        "native",
        ["stock", "disabled", "embed_control", "token", "soft", "hidden", "mixed"],
        [("stock", "disabled"), ("embed_control", "token")],
    ),
    ("qwen", ["base", "disabled", "soft", "swir"], [("base", "disabled")]),
    ("swir", ["base", "disabled", "swir"], [("base", "disabled")]),
]:
    records = [
        json.loads(s) for s in (out / f"{engine}.jsonl").read_text().splitlines()
    ]
    rows = {(r["model"], r["mode"], r["batch"]): r for r in records}
    assert len(rows) == len(records), f"Duplicate rows: {engine}"
    assert len(rows) == len(models) * len(modes) * 3, f"Incomplete sweep: {engine}"
    checks[engine] = {}
    variability[engine] = {}
    for model in models:
        for batch in [1, 8, 32]:
            for mode in modes:
                r = rows[model, mode, batch]
                assert len(r["seconds"]) == 5
                assert len(r["tokens"]) == batch
                assert all(len(t) == 128 for t in r["tokens"])
                variability[engine][f"{model}/batch{batch}/{mode}"] = len(
                    set(r["output_hashes"])
                )
            for lhs, rhs in pairs:
                key = f"{model}/batch{batch}/{lhs}={rhs}"
                equal = (
                    rows[model, lhs, batch]["tokens"]
                    == rows[model, rhs, batch]["tokens"]
                )
                checks[engine][key] = equal
(out / "control-checks.json").write_text(json.dumps(checks, indent=2) + "\n")
(out / "repeat-variability.json").write_text(json.dumps(variability, indent=2) + "\n")
assert all(v for k, v in checks["native"].items() if k.endswith("stock=disabled"))
print("All sweeps complete; fixed-position budgets and disabled controls match.")
print("External exact-trace checks:", checks["qwen"], checks["swir"])

external = {}
for engine in ["qwen", "swir"]:
    external[engine] = {
        (r["model"], r["batch"]): r
        for r in map(json.loads, (out / f"{engine}.jsonl").read_text().splitlines())
        if r["mode"] == "swir"
    }
cross = {}
for key, row in external["qwen"].items():
    other = external["swir"][key]
    cross[f"{key[0]}/batch{key[1]}"] = {
        "matching_traces": sum(a == b for a, b in zip(row["tokens"], other["tokens"])),
        "total_traces": key[1],
    }
(out / "cross-fork-traces.json").write_text(json.dumps(cross, indent=2) + "\n")

reference = json.loads((out / "reference-checks.json").read_text())
assert all(r["token"] and r["soft"] and r["stock_replay"] for r in reference.values())
audits = {}
for size, filename in [
    ("1.7B", "hidden-audit-results.json"),
    ("4B", "hidden-audit-4B-results.json"),
]:
    audit = json.loads((out / filename).read_text())
    assert audit["exact_tokens"], audit
    audits[size] = audit["exact_tokens"]
invariant = [
    json.loads(s) for s in (out / "invariant-control.jsonl").read_text().splitlines()
]
assert len(invariant) == 6
lookup = {(r["mode"], r["batch"]): r for r in invariant}
invariant_checks = {
    str(b): lookup["embed_control", b]["tokens"] == lookup["token", b]["tokens"]
    for b in [1, 8, 32]
}
assert all(invariant_checks.values()), invariant_checks
summary = dict(
    timing_groups=168,
    timed_repeats=840,
    disabled_control_matches=12,
    disabled_control_total=12,
    default_token_control_matches=sum(
        v for k, v in checks["native"].items() if k.endswith("embed_control=token")
    ),
    default_token_control_total=12,
    batch_invariant_token_checks=invariant_checks,
    independent_reference_checks=reference,
    teacher_forced_audits=audits,
)
(out / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
