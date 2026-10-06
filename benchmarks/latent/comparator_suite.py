# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run each external fork and its wheel control in isolated environments."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from vllm.transformers_utils.repo_utils import hf_api

p = argparse.ArgumentParser()
p.add_argument("--engine", choices=["qwen", "swir"], required=True)
a = p.parse_args()
repo = Path(__file__).resolve().parents[2]
out = repo / "results/scaling"
source = repo.parent / "latent-comparators" / (a.engine + "-vllm")
py = f"/tmp/{a.engine}/.venv/bin/python"
overlay = f"/tmp/{a.engine}-overlay"
subprocess.run(
    [py, str(repo / "benchmarks/latent/comparator_overlay.py"), str(source), overlay],
    check=True,
)
for model in json.loads((out / "models.json").read_text()):
    hf_api().snapshot_download(model["model"], revision=model["revision"])
    size = model["model"].split("-")[-1]
    modes = (
        ["base", "disabled", "soft", "swir"]
        if a.engine == "qwen"
        else ["base", "disabled", "swir"]
    )
    for mode in modes:
        output_file = out / f"{a.engine}.jsonl"
        previous = (
            [json.loads(s) for s in output_file.read_text().splitlines()]
            if output_file.exists()
            else []
        )
        if {
            r["batch"]
            for r in previous
            if r["model"] == model["model"] and r["mode"] == mode
        } == {1, 8, 32}:
            continue
        env = dict(os.environ, HF_HUB_OFFLINE="1", VLLM_USE_V2_MODEL_RUNNER="0")
        env["PYTHONPATH"] = "" if mode == "base" else overlay
        command = [
            py,
            str(repo / "benchmarks/latent/comparator_run.py"),
            "--engine",
            a.engine,
            "--model",
            model["model"],
            "--revision",
            model["revision"],
            "--mode",
            mode,
            "--output",
            str(out / f"{a.engine}.jsonl"),
        ]
        print(f"START {a.engine} {size} {mode}", flush=True)
        with (out / f"{a.engine}-{size}-{mode}.log").open("w") as log:
            subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        print(f"DONE {a.engine} {size} {mode}", flush=True)
(out / f"{a.engine.upper()}_COMPLETE").touch()
