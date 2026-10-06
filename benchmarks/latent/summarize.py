# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate tables and a standalone chart from measured JSON records."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[2] / "results" / "latent"
rows = [json.loads(x) for x in (root / "native.jsonl").read_text().splitlines()]
records = {(r["mode"], r["batch"]): r for r in rows}
names = {
    "stock": "Stock vLLM",
    "disabled": "Fork disabled",
    "embed_control": "Stock embedding input",
    "token": "Token program",
    "soft": "Soft feedback",
    "hidden": "Hidden feedback",
    "norm_hidden": "Normalized hidden",
    "entropy": "Entropy threshold 2.0",
    "mixed": "Mixed programs",
}
fields = [
    "mode",
    "batch",
    "median_seconds",
    "min_seconds",
    "max_seconds",
    "positions_per_second",
    "overhead_percent",
]
with (root / "summary.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for (mode, batch), r in records.items():
        writer.writerow(
            {
                "mode": mode,
                "batch": batch,
                "median_seconds": r["median_seconds"],
                "min_seconds": min(r["seconds"]),
                "max_seconds": max(r["seconds"]),
                "positions_per_second": r["steps_per_second"],
                "overhead_percent": 100
                * (r["median_seconds"] / records["stock", batch]["median_seconds"] - 1),
            }
        )
table = [
    "| Mode | Batch 1 seconds | Batch 8 seconds | Batch 32 seconds |",
    "| --- | --- | --- | --- |",
]
for mode, name in names.items():
    if (mode, 1) not in records:
        continue
    values = []
    for batch in [1, 8, 32]:
        r = records[mode, batch]
        baseline = records["stock", batch]["median_seconds"]
        overhead = 100 * (r["median_seconds"] / baseline - 1)
        values.append(f"{r['median_seconds']:.4f} ({overhead:+.2f}%)")
    table.append(f"| {name} | " + " | ".join(values) + " |")
(root / "TABLE.md").write_text(
    "# Main latency measurements\n\n" + "\n".join(table) + "\n"
)

plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
modes = ["disabled", "token", "hidden", "norm_hidden", "soft", "entropy", "mixed"]
x = np.arange(len(modes))
for i, batch in enumerate([1, 8, 32]):
    y = [
        100
        * (
            records[m, batch]["median_seconds"]
            / records["stock", batch]["median_seconds"]
            - 1
        )
        for m in modes
    ]
    ax.bar(
        x + (i - 1) * 0.24,
        y,
        0.24,
        label=f"Batch {batch}",
        color=["#3c6478", "#42a998", "#d7a251"][i],
    )
ax.set_xticks(x, [names[m].replace(" ", "\n", 1) for m in modes])
ax.set_ylabel("End-to-end latency overhead versus stock (%)")
ax.set_title("Qwen3-8B BF16 · RTX A6000 · 128 generated positions")
ax.axhline(0, color="#333333", linewidth=0.7)
ax.spines[["right", "top"]].set_visible(False)
ax.legend(frameon=False)
fig.supxlabel(
    "Five warmed runs. Entropy 2.0 selected no latent steps; "
    "mixed batch 1 is token-only.",
    fontsize=8,
)
fig.savefig(root / "latency.svg")
fig.savefig(root / "latency.png", dpi=160)
svg = root / "latency.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
print("\n".join(table))
