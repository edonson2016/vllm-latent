# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sequential ablation sweep on one GPU; each variant gets a fresh engine."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--model-index", type=int, default=0)
p.add_argument("--plan", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--repeats", type=int, default=7)
p.add_argument("--batches", default="1,32")
p.add_argument("--steps", type=int, default=128)
a = p.parse_args()
repo = Path(__file__).resolve().parents[2]
bench = repo / "benchmarks/latent"
for item in json.loads(a.plan.read_text()):
    label = item["label"]
    env = dict(os.environ, HF_HUB_OFFLINE="1", VLLM_USE_V2_MODEL_RUNNER="0")
    overlay = item.get("overlay", "/tmp/latent-overlay")
    env["PYTHONPATH"] = f"{overlay}:{bench}"
    cmd = [
        sys.executable,
        str(bench / "optimization_bench.py"),
        "--model-index",
        str(a.model_index),
        "--label",
        label,
        "--optimizations",
        item.get("opts", ""),
        "--output",
        str(a.output),
        "--repeats",
        str(a.repeats),
        "--batches",
        a.batches,
        "--steps",
        str(a.steps),
        "--modes",
        item.get("modes", "token,hidden,soft,mixed"),
    ]
    if label.startswith("stock"):
        cmd.append("--stock")
    if item.get("quality"):
        cmd.append("--quality")
    if item.get("invariant"):
        cmd.append("--invariant")
    print("START", a.model_index, label, flush=True)
    with (a.output.parent / f"{a.output.stem}-{label}.log").open("w") as f:
        subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    print("DONE", a.model_index, label, flush=True)
