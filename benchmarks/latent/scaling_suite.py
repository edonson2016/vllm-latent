# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pinned, sequential model-size sweep; invoke in the release GPU container."""

import json
import os
import subprocess
import sys
from pathlib import Path

from vllm.transformers_utils.repo_utils import hf_api

repo = Path(__file__).resolve().parents[2]
out = repo / "results/scaling"
models = json.loads((out / "models.json").read_text())
subprocess.run(
    [
        sys.executable,
        str(repo / "benchmarks/latent/overlay.py"),
        str(repo),
        "/tmp/latent-overlay",
    ],
    check=True,
)
for model in models:
    hf_api().snapshot_download(model["model"], revision=model["revision"])
    size = model["model"].split("-")[-1]
    for mode in [
        "stock",
        "disabled",
        "embed_control",
        "token",
        "soft",
        "hidden",
        "mixed",
    ]:
        env = dict(os.environ, HF_HUB_OFFLINE="1", VLLM_USE_V2_MODEL_RUNNER="0")
        env["PYTHONPATH"] = (
            "" if mode in {"stock", "embed_control"} else "/tmp/latent-overlay"
        )
        command = [
            sys.executable,
            str(repo / "benchmarks/latent/run.py"),
            "--model",
            model["model"],
            "--revision",
            model["revision"],
            "--mode",
            mode,
            "--output",
            str(out / "native.jsonl"),
        ]
        if mode not in {"stock", "disabled", "embed_control"}:
            command.append("--trace-masks")
        print(f"START {size} {mode}", flush=True)
        with (out / f"{size}-{mode}.log").open("w") as log:
            subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        print(f"DONE {size} {mode}", flush=True)
(out / "SUITE_COMPLETE").touch()
