# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fresh-engine SwiReasoning ablations using a recorded, explicit plan."""

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
a = p.parse_args()
repo = Path(__file__).resolve().parents[2]
bench = repo / "benchmarks/latent"
for item in json.loads(a.plan.read_text()):
    label = item["label"]
    env = dict(os.environ, HF_HUB_OFFLINE="1", VLLM_USE_V2_MODEL_RUNNER="0")
    env["PYTHONPATH"] = f"{item.get('overlay', '/tmp/round2-overlay')}:{bench}"
    command = [
        sys.executable,
        str(bench / "round2_bench.py"),
        "--model-index",
        str(a.model_index),
        "--label",
        label,
        "--opts",
        item["opts"],
        "--modes",
        item.get("modes", "swi512,swi8"),
        "--batches",
        a.batches,
        "--repeats",
        str(a.repeats),
        "--output",
        str(a.output),
    ]
    print("START", a.model_index, label, flush=True)
    with (a.output.parent / f"{a.output.stem}-{label}.log").open("w") as log:
        subprocess.run(
            command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    print("DONE", a.model_index, label, flush=True)
