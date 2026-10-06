# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize final fixed-work runs; preliminary ablations remain separate."""

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path(__file__).resolve().parents[2] / "results/optimization"
files = ["release-model0", "release-1.7B", "release-model2", "release-8B"]
models = ["0.6B", "1.7B", "4B", "8B"]
rng = np.random.default_rng(20261006)


def read(name):
    return [json.loads(line) for line in (out / name).read_text().splitlines()]


def interval(a, b):
    """Within-session unpaired bootstrap for median latency reduction."""
    aa = np.median(rng.choice(a, (10000, len(a))), axis=1)
    bb = np.median(rng.choice(b, (10000, len(b))), axis=1)
    return np.percentile(100 * (1 - bb / aa), [2.5, 97.5]).tolist()


summary, quality, references, controls, checks = [], [], [], [], []
quality_changes = []
records = []
for size, stem in zip(models, files):
    data = read(stem + ".jsonl")
    assert len(data) == 36, (stem, len(data))
    records.extend(data)
    rows = {(r["label"], r["mode"], r["batch"]): r for r in data}
    assert len(rows) == 36
    for batch in [1, 8, 32]:
        start = rows["stock-start", "token", batch]
        end = rows["stock-end", "token", batch]
        stock = np.median(start["seconds"] + end["seconds"])
        controls.append(
            dict(
                model=size,
                batch=batch,
                stock_seconds=stock,
                drift_percent=100
                * (end["median_seconds"] / start["median_seconds"] - 1),
            )
        )
        for mode in ["token", "hidden", "soft", "entropy", "mixed"]:
            old = rows["baseline", mode, batch]
            new = rows["optimized", mode, batch]
            assert old["steps"] == new["steps"] == 128
            assert len(old["seconds"]) == len(new["seconds"]) == 9
            a, b = old["median_seconds"], new["median_seconds"]
            lo, hi = interval(old["seconds"], new["seconds"])
            summary.append(
                dict(
                    model=size,
                    mode=mode,
                    batch=batch,
                    stock_seconds=stock,
                    baseline_seconds=a,
                    optimized_seconds=b,
                    latency_reduction_percent=100 * (1 - b / a),
                    reduction_ci_low=lo,
                    reduction_ci_high=hi,
                    speedup=a / b,
                    baseline_overhead_percent=100 * (a / stock - 1),
                    optimized_overhead_percent=100 * (b / stock - 1),
                    optimized_added_ms_per_position=1000 * (b - stock) / 128,
                )
            )
            checks.append(
                dict(
                    model=size,
                    mode=mode,
                    batch=batch,
                    baseline_stable=len(set(old["hashes"])) == 1,
                    optimized_stable=len(set(new["hashes"])) == 1,
                    exact_requests=sum(
                        x == y for x, y in zip(old["tokens"], new["tokens"])
                    ),
                    requests=batch,
                    stock_exact_requests=sum(
                        x == y for x, y in zip(start["tokens"], new["tokens"])
                    )
                    if mode == "token"
                    else None,
                )
            )
    quality_rows = read(stem + ".quality.jsonl")
    quality_index = {(q["label"], q["mode"]): q for q in quality_rows}
    for mode in ["token", "soft", "hidden"]:
        old = quality_index["baseline", mode]["cases"]
        new = quality_index["optimized", mode]["cases"]
        quality_changes.append(
            dict(
                model=size,
                mode=mode,
                exact_sequences=sum(
                    a["tokens"] == b["tokens"] for a, b in zip(old, new)
                ),
                correctness_flips=sum(
                    a["correct"] != b["correct"] for a, b in zip(old, new)
                ),
                total=len(old),
            )
        )
    for q in quality_rows:
        quality.append(
            dict(
                model=size,
                label=q["label"],
                mode=q["mode"],
                correct=q["correct"],
                total=q["total"],
                strict_format_correct=sum(
                    c["strict_format_correct"] for c in q["cases"]
                ),
            )
        )
    for r in read(stem + ".reference.jsonl"):
        references.append(dict(model=size, **r))

with (out / "summary.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(summary)
audit = dict(
    timing_groups=len(records),
    timed_generations=sum(len(r["seconds"]) for r in records),
    hashes={
        label: sorted(
            {r["implementation_hash"] for r in records if r["label"] == label}
        )
        for label in ["baseline", "optimized"]
    },
    controls=controls,
    traces=checks,
    quality=quality,
    quality_changes=quality_changes,
    references=references,
)
(out / "verification.json").write_text(json.dumps(audit, indent=2) + "\n")
lines = [
    "# Final optimization measurements",
    "",
    (
        "128 positions; nine repetitions per engine/policy/batch. Stock is the median "
        "of the 18 bracketing control samples."
    ),
    "",
    (
        "The intervals below are 95% unpaired bootstrap intervals for medians within "
        "one session. They do not cover different GPUs, prompts, "
        "or day-to-day variation."
    ),
    "",
    (
        "| Model | Batch | Policy | Old (s) | Optimized (s) | "
        "Latency reduction [95% interval] | Optimized overhead vs stock |"
    ),
    "| --- | --- | --- | --- | --- | --- | --- |",
]
for r in summary:
    lines.append(
        f"| {r['model']} | {r['batch']} | {r['mode']} | "
        f"{r['baseline_seconds']:.4f} | {r['optimized_seconds']:.4f} | "
        f"{r['latency_reduction_percent']:.1f}% [{r['reduction_ci_low']:.1f}, "
        f"{r['reduction_ci_high']:.1f}] | {r['optimized_overhead_percent']:+.1f}% |"
    )
lines.extend(
    [
        "",
        "## Arithmetic smoke check",
        "",
        (
            "Last integer equals the answer; 64 fixed questions, 64 positions, "
            "four latent-prefix positions. This is not a general reasoning evaluation."
        ),
        "",
        "| Model | Engine | Policy | Correct / 64 | Strict integer format correct |",
        "| --- | --- | --- | --- | --- |",
    ]
)
for q in quality:
    lines.append(
        f"| {q['model']} | {q['label']} | {q['mode']} | {q['correct']} | "
        f"{q['strict_format_correct']} |"
    )
(out / "TABLES.md").write_text("\n".join(lines) + "\n")

fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
colors = {"baseline": "#b65d39", "optimized": "#237ca0"}
for ax, (batch, mode) in zip(
    axes.flat, [(b, m) for b in [1, 32] for m in ["token", "soft", "mixed"]]
):
    for label in colors:
        points = [
            next(
                r
                for r in summary
                if r["model"] == size and r["batch"] == batch and r["mode"] == mode
            )[label + "_overhead_percent"]
            for size in models
        ]
        ax.plot([0.6, 1.7, 4, 8], points, marker="o", label=label, color=colors[label])
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.set(title=f"{mode.capitalize()}, batch {batch}", xticks=[0.6, 1.7, 4, 8])
    ax.grid(alpha=0.2)
for ax in axes[:, 0]:
    ax.set_ylabel("Overhead vs stock (%)")
for ax in axes[1]:
    ax.set_xlabel("Model parameters (billions)")
axes[0, 0].legend()
fig.suptitle("Programmable decode optimizations — Qwen3, BF16, RTX A6000")
fig.tight_layout(rect=(0, 0.055, 1, 0.97))
fig.text(
    0.5,
    0.015,
    "128 generated positions; soft selects 64 latent inputs. "
    "Mixed batch 1 is token-only. Warm fixed-work timings.",
    ha="center",
    fontsize=9,
)
fig.savefig(out / "optimization.png", dpi=180)
fig.savefig(out / "optimization.svg")
print(
    json.dumps(
        {k: audit[k] for k in ["timing_groups", "timed_generations", "hashes"]},
        indent=2,
    )
)
