# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regenerate round-two tables from measured records; never synthesize runs."""

import csv
import gzip
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "results/round2"
RNG = np.random.default_rng(20261006)


def read(name):
    path = ROOT / name
    if not path.exists():
        path = ROOT / (name + ".gz")
    if not path.exists():
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def interval(a, b):
    a, b = np.asarray(a), np.asarray(b)
    left = np.median(RNG.choice(a, (4000, len(a))), axis=1)
    right = np.median(RNG.choice(b, (4000, len(b))), axis=1)
    return np.percentile((left / right - 1) * 100, [2.5, 97.5]).tolist()


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def save_csv(name, records):
    if not records:
        return
    with (ROOT / name).open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def model_name(row):
    model = row["model"]
    return (model["model"] if isinstance(model, dict) else model).split("-")[-1]


def main():
    paired, generic, quality = [], [], []
    sections = [
        "# Generated experiment tables",
        (
            "Reproduce with `.venv/bin/python benchmarks/latent/summarize_round2.py`. "
            "Missing runs are omitted, never imputed."
        ),
    ]
    for i in range(4):
        rows = read(f"paired-model{i}.jsonl")
        for row in rows:
            if row["engine"] != "fork":
                continue
            reference = next(
                (
                    x
                    for x in rows
                    if x["engine"] == "qwen"
                    and x["mode"] == row["mode"]
                    and x["batch"] == row["batch"]
                ),
                None,
            )
            stock = next(
                (
                    x
                    for x in rows
                    if x["engine"] == "stock" and x["batch"] == row["batch"]
                ),
                None,
            )
            if reference is None or stock is None:
                continue
            ci = interval(row["seconds"], reference["seconds"])
            paired.append(
                dict(
                    model=model_name(row),
                    mode=row["mode"],
                    batch=row["batch"],
                    variant=row["label"],
                    fork_seconds=row["median_seconds"],
                    qwen_seconds=reference["median_seconds"],
                    stock_seconds=stock["median_seconds"],
                    change_vs_qwen_pct=(
                        row["median_seconds"] / reference["median_seconds"] - 1
                    )
                    * 100,
                    ci_low=ci[0],
                    ci_high=ci[1],
                    overhead_vs_stock_pct=(
                        row["median_seconds"] / stock["median_seconds"] - 1
                    )
                    * 100,
                    exact_traces=sum(
                        a == b for a, b in zip(row["traces"][0], reference["traces"][0])
                    ),
                )
            )
        controls = read(f"control-model{i}.jsonl")
        final = read(f"final-generic{i}.jsonl")
        for row in final:
            if row["label"] != "optimized":
                continue
            old = next(
                x
                for x in controls
                if x["label"] == "baseline"
                and x["mode"] == row["mode"]
                and x["batch"] == row["batch"]
            )
            stock = [
                s
                for x in controls + final
                if x["label"].startswith("stock") and x["batch"] == row["batch"]
                for s in x["seconds"]
            ]
            generic.append(
                dict(
                    model=model_name(row),
                    mode=row["mode"],
                    batch=row["batch"],
                    previous_seconds=old["median_seconds"],
                    optimized_seconds=row["median_seconds"],
                    change_pct=(row["median_seconds"] / old["median_seconds"] - 1)
                    * 100,
                    overhead_vs_stock_pct=(row["median_seconds"] / np.median(stock) - 1)
                    * 100,
                )
            )
    save_csv("paired.csv", paired)
    save_csv("generic.csv", generic)
    sections += [
        "## Same-GPU matched policy comparison",
        (
            "Negative change means the fork is faster. Intervals bootstrap "
            "within-session repeats, not hardware, prompt, or day variability."
        ),
        table(
            [
                "Model",
                "Policy",
                "B",
                "Variant",
                "Fork s",
                "Qwen fork s",
                "Stock s",
                "Change [95% interval]",
                "Overhead vs stock",
                "Exact traces",
            ],
            [
                [
                    x["model"],
                    x["mode"],
                    x["batch"],
                    x["variant"],
                    f"{x['fork_seconds']:.4f}",
                    f"{x['qwen_seconds']:.4f}",
                    f"{x['stock_seconds']:.4f}",
                    (
                        f"{x['change_vs_qwen_pct']:+.1f}% "
                        f"[{x['ci_low']:+.1f}, {x['ci_high']:+.1f}]"
                    ),
                    f"{x['overhead_vs_stock_pct']:+.1f}%",
                    f"{x['exact_traces']}/{x['batch']}",
                ]
                for x in paired
            ],
        ),
    ]
    sections += [
        "## Generic programs: previous release versus combined profile",
        table(
            [
                "Model",
                "Policy",
                "B",
                "Previous s",
                "Combined s",
                "Change",
                "Overhead vs stock",
            ],
            [
                [
                    x["model"],
                    x["mode"],
                    x["batch"],
                    f"{x['previous_seconds']:.4f}",
                    f"{x['optimized_seconds']:.4f}",
                    f"{x['change_pct']:+.1f}%",
                    f"{x['overhead_vs_stock_pct']:+.1f}%",
                ]
                for x in generic
            ],
        ),
    ]
    for path in sorted(ROOT.glob("*.jsonl*")):
        for row in read(path.name):
            records = row.get("quality", [])
            if not records:
                continue
            quality.append(
                dict(
                    file=path.name,
                    model=model_name(row),
                    engine=row["engine"],
                    mode=row["mode"],
                    seed=row.get("seed", 20261006),
                    temperature=row.get("temperature", 0),
                    budget=row["steps"],
                    correct=sum(x["correct"] for x in records),
                    count=len(records),
                    length_limited=sum(x["finish"] == "length" for x in records),
                    seconds=row["median_seconds"],
                    positions=sum(map(len, row["traces"][0])),
                )
            )
    save_csv("quality.csv", quality)
    sections += [
        "## Natural-termination quality",
        (
            "Same seeded 128-question GSM8K subset. Correct means the final number "
            "after a generated end-think matches the gold integer; unfinished "
            "thinking fails. Timings include different amounts of generated work "
            "and are not fixed-work speedups."
        ),
        table(
            [
                "File",
                "Model",
                "Policy",
                "Seed",
                "Correct/128",
                "Length limited",
                "Positions",
                "Seconds",
            ],
            [
                [
                    x["file"],
                    x["model"],
                    x["mode"],
                    x["seed"],
                    x["correct"],
                    x["length_limited"],
                    x["positions"],
                    f"{x['seconds']:.2f}",
                ]
                for x in quality
            ],
        ),
    ]
    differences = []
    for i in range(4):
        fork = read(f"fork-quality{i}.jsonl")
        ref = read(f"qwen-sampled{i}.jsonl")
        for row in fork:
            other = next((x for x in ref if x["mode"] == row["mode"]), None)
            if other is None and row["mode"] in {"bf16", "topk", "adaptive", "lowrank"}:
                other = next((x for x in fork if x["mode"] == "dense32"), None)
            if other is None:
                continue
            assert [x["index"] for x in row["quality"]] == [
                x["index"] for x in other["quality"]
            ]
            delta = np.asarray(
                [
                    int(a["correct"]) - int(b["correct"])
                    for a, b in zip(row["quality"], other["quality"])
                ]
            )
            ci = np.percentile(
                RNG.choice(delta, (10000, len(delta))).mean(1) * 100, [2.5, 97.5]
            )
            differences.append(
                dict(
                    model=model_name(row),
                    mode=row["mode"],
                    reference=other["engine"] + ":" + other["mode"],
                    difference_pp=delta.mean() * 100,
                    ci_low=ci[0],
                    ci_high=ci[1],
                    gains=int((delta > 0).sum()),
                    losses=int((delta < 0).sum()),
                    exact_traces=sum(
                        a == b for a, b in zip(row["traces"][0], other["traces"][0])
                    ),
                )
            )
    save_csv("quality-differences.csv", differences)
    sections += [
        "## Paired quality differences",
        (
            "Question bootstrap, one seed; no multiplicity correction or equivalence "
            "claim. Approximate variants compare to the fork's FP32 window-32 policy."
        ),
        table(
            [
                "Model",
                "Policy",
                "Reference",
                "Difference pp [95% interval]",
                "Gains/losses",
                "Exact/128",
            ],
            [
                [
                    x["model"],
                    x["mode"],
                    x["reference"],
                    (
                        f"{x['difference_pp']:+.1f} "
                        f"[{x['ci_low']:+.1f}, {x['ci_high']:+.1f}]"
                    ),
                    f"{x['gains']}/{x['losses']}",
                    x["exact_traces"],
                ]
                for x in differences
            ],
        ),
    ]
    for group, pattern in [
        ("Swi ablations", "swi-ablation*.jsonl"),
        ("Approximation fixed-work timings before gating", "approx-timing*.jsonl"),
        ("Approximation fixed-work timings with gating", "approx-gated*.jsonl"),
    ]:
        entries = [
            x for path in sorted(ROOT.glob(pattern + "*")) for x in read(path.name)
        ]
        sections += [
            "## " + group,
            table(
                ["Model", "Variant", "Policy", "B", "Median seconds"],
                [
                    [
                        model_name(x),
                        x["label"],
                        x["mode"],
                        x["batch"],
                        f"{x['median_seconds']:.4f}",
                    ]
                    for x in entries
                ],
            ),
        ]
    (ROOT / "TABLES.md").write_text("\n\n".join(sections) + "\n")
    counts = dict(
        timing_groups=0, measured_generations=0, quality_runs=0, quality_answers=0
    )
    for path in ROOT.glob("*.jsonl*"):
        for row in read(path.name):
            if row.get("quality"):
                counts["quality_runs"] += 1
                counts["quality_answers"] += len(row["quality"])
            elif "seconds" in row:
                counts["timing_groups"] += 1
                counts["measured_generations"] += len(row["seconds"])
    (ROOT / "counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    if len([x for x in paired if x["variant"] == "paired-final"]) == 36:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
        models = ["0.6B", "1.7B", "4B", "8B"]
        for policy, label in [
            ("token", "Token control"),
            ("swi512", "Swi window 512"),
            ("swi8", "Swi window 8 + cutoff"),
        ]:
            values = [
                next(
                    x
                    for x in paired
                    if x["model"] == m
                    and x["mode"] == policy
                    and x["batch"] == 32
                    and x["variant"] == "paired-final"
                )
                for m in models
            ]
            axes[0].plot(
                models,
                [x["overhead_vs_stock_pct"] for x in values],
                marker="o",
                label=label,
            )
            axes[1].plot(
                models,
                [-x["change_vs_qwen_pct"] for x in values],
                marker="o",
                label=label,
            )
        axes[0].set_ylabel("Latency overhead vs stock vLLM (%)")
        axes[1].set_ylabel("Latency reduction vs Qwen fork (%)")
        for ax in axes:
            ax.axhline(0, color="grey", linewidth=0.7)
            ax.grid(alpha=0.2)
            ax.set_xlabel("Qwen3 model size")
        axes[0].legend(fontsize=8)
        fig.suptitle(
            "RTX A6000 · batch 32 · 128 generated positions · 7 warm repetitions"
        )
        fig.savefig(ROOT / "scaling.png", dpi=180)
        fig.savefig(ROOT / "scaling.svg")


if __name__ == "__main__":
    main()
