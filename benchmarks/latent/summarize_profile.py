# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize recorded profiler events, keeping summed work separate from latency."""

import gzip
import json
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parents[2] / "results/round2"
results = {}
for name in ["core", "combo"]:
    files = list((root / f"profile-{name}").glob("*.json.gz"))
    assert len(files) == 1, files
    with gzip.open(files[0], "rt") as f:
        events = json.load(f)["traceEvents"]
    categories, duration, calls, kernel_duration = (
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    )
    copies = Counter()
    for event in events:
        if event.get("ph") != "X":
            continue
        cat, operation = event.get("cat", ""), event.get("name", "")
        categories[cat] += 1
        duration[cat] += event.get("dur", 0)
        if cat in {"cuda_runtime", "cuda_driver"}:
            calls[operation] += 1
        if cat == "kernel":
            kernel_duration[operation] += event.get("dur", 0)
        if cat == "gpu_memcpy":
            copies[operation] += 1
    results[name] = dict(
        trace=str(files[0].relative_to(root)),
        event_counts=categories,
        summed_duration_ms={k: v / 1000 for k, v in duration.items()},
        runtime_calls=calls,
        copies=copies,
        largest_kernel_totals_ms=[
            [key, value / 1000] for key, value in kernel_duration.most_common(20)
        ],
    )
(root / "profile-summary.json").write_text(json.dumps(results, indent=2) + "\n")
for name, data in results.items():
    print(name, data["event_counts"]["kernel"], data["summed_duration_ms"]["kernel"])
    print({k: v for k, v in data["runtime_calls"].items() if "Synchronize" in k})
