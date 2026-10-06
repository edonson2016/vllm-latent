# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build size-scaling tables and a plot from raw, fixed-work measurements."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path(__file__).resolve().parents[2] / "results/scaling"
records = [json.loads(s) for s in (out / "native.jsonl").read_text().splitlines()]
rows = {(r["model"], r["mode"], r["batch"]): r for r in records}
models = json.loads((out / "models.json").read_text())
csv_rows = []
for r in records:
    base = rows[r["model"], "stock", r["batch"]]["median_seconds"]
    csv_rows.append(
        dict(
            model=r["model"],
            mode=r["mode"],
            batch=r["batch"],
            seconds=r["median_seconds"],
            min_seconds=min(r["seconds"]),
            max_seconds=max(r["seconds"]),
            overhead_percent=100 * (r["median_seconds"] / base - 1),
            added_ms_per_position=1000 * (r["median_seconds"] - base) / r["steps"],
        )
    )
with (out / "summary.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(csv_rows)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
for ax, batch in zip(axes, [1, 32]):
    for mode in ["token", "soft", "hidden", "mixed"]:
        values = [
            100
            * (
                rows[m["model"], mode, batch]["median_seconds"]
                / rows[m["model"], "stock", batch]["median_seconds"]
                - 1
            )
            for m in models
        ]
        ax.plot([0.6, 1.7, 4, 8], values, marker="o", label=mode)
    ax.set(title=f"Batch {batch}", xlabel="Model size (billions of parameters)")
    ax.set_xticks([0.6, 1.7, 4, 8])
    ax.grid(alpha=0.25)
axes[0].set_ylabel("Latency overhead vs stock vLLM V1 (%)")
axes[1].legend()
fig.suptitle("Smaller models expose transition overhead — RTX A6000, BF16")
fig.tight_layout(rect=(0, 0.10, 1, 1))
fig.text(
    0.5,
    0.025,
    "128 positions/request; five-run medians. Soft/hidden select 64 latent inputs. "
    "Mixed batch 1 is token-only.",
    ha="center",
    fontsize=8,
)
fig.savefig(out / "scaling.png", dpi=180)
fig.savefig(out / "scaling.svg")
svg = out / "scaling.svg"
svg.write_text("\n".join(s.rstrip() for s in svg.read_text().splitlines()) + "\n")
lines = [
    "# Size-scaling latency tables",
    "",
    "Medians of five runs, seconds for 128 generated positions per request.",
    "",
]
for batch in [1, 8, 32]:
    lines += [
        f"## Batch {batch}",
        "",
        "| Model | Stock | Disabled | Token program | Soft | Hidden | Mixed |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for model in models:
        values = [
            rows[model["model"], mode, batch]["median_seconds"]
            for mode in ["stock", "disabled", "token", "soft", "hidden", "mixed"]
        ]
        lines.append(
            "| "
            + model["model"].split("-")[-1]
            + " | "
            + " | ".join(f"{v:.4f}" for v in values)
            + " |"
        )
    lines.append("")
(out / "TABLES.md").write_text("\n".join(lines))

lines = [
    "# Existing-fork latency tables",
    "",
    (
        "Seconds for 128 positions; parenthesized overhead is relative to the "
        "same fork with its latent feature disabled. Policies are different; "
        "see [the comparison protocol](COMPARATORS.md)."
    ),
    "",
]
external_rows = []
for engine in ["qwen", "swir"]:
    records = [
        json.loads(s) for s in (out / f"{engine}.jsonl").read_text().splitlines()
    ]
    lookup = {(r["model"], r["mode"], r["batch"]): r for r in records}
    lines += [
        f"## {engine}",
        "",
        "| Model | Batch | Base wheel | Fork disabled | Soft top-10 | SwiReasoning |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for model in models:
        for batch in [1, 8, 32]:
            base = lookup[model["model"], "disabled", batch]["median_seconds"]
            cells = []
            for mode in ["base", "disabled", "soft", "swir"]:
                r = lookup.get((model["model"], mode, batch))
                if r is None:
                    cells.append("—")
                    continue
                seconds = r["median_seconds"]
                overhead = 100 * (seconds / base - 1)
                cells.append(
                    f"{seconds:.4f}"
                    + (f" ({overhead:+.1f}%)" if mode in {"soft", "swir"} else "")
                )
                external_rows.append(
                    dict(
                        engine=engine,
                        model=model["model"],
                        mode=mode,
                        batch=batch,
                        seconds=seconds,
                        min_seconds=min(r["seconds"]),
                        max_seconds=max(r["seconds"]),
                        overhead_percent=overhead,
                    )
                )
            lines.append(
                f"| {model['model'].split('-')[-1]} | {batch} | "
                + " | ".join(cells)
                + " |"
            )
    lines.append("")
(out / "COMPARATOR_TABLES.md").write_text("\n".join(lines))
with (out / "comparators.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(external_rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(external_rows)
