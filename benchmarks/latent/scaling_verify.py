# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent short references and transition-only model-width calibration."""

import json
import os
import subprocess
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
out = repo / "results/scaling"
env = dict(
    os.environ,
    HF_HUB_OFFLINE="1",
    VLLM_USE_V2_MODEL_RUNNER="0",
    PYTHONPATH="/tmp/latent-overlay",
)
subprocess.run(
    [
        sys.executable,
        str(repo / "benchmarks/latent/overlay.py"),
        str(repo),
        "/tmp/latent-overlay",
    ],
    check=True,
)
checks = {}
for model in json.loads((out / "models.json").read_text()):
    size = model["model"].split("-")[-1]
    base = ["--model", model["model"], "--revision", model["revision"]]
    tasks = [
        (["hf_reference.py", *base, "--output", str(out / f"hf-{size}.json")], "hf")
    ]
    for kind in ["token", "soft", "hidden"]:
        tasks.append(
            (
                [
                    "run.py",
                    *base,
                    "--mode",
                    kind,
                    "--batches",
                    "1",
                    "--steps",
                    "8",
                    "--repeats",
                    "3" if kind == "soft" else "1",
                    *([] if kind == "soft" else ["--eager"]),
                    "--output",
                    str(out / f"short-{size}.jsonl"),
                ],
                kind,
            )
        )
    tasks.append(
        (
            [
                "stock_replay.py",
                *base,
                "--kind",
                "soft",
                "--prefix-cache",
                "--output",
                str(out / f"replay-{size}.json"),
            ],
            "replay",
        )
    )
    tasks.append(
        (
            [
                "transition_cost.py",
                "--width",
                str(model["width"]),
                "--vocab",
                str(model["vocab"]),
                "--output",
                str(out / f"kernels-{size}.json"),
            ],
            "kernels",
        )
    )
    for args, kind in tasks:
        command = [sys.executable, str(repo / "benchmarks/latent" / args[0]), *args[1:]]
        print(f"VERIFY {size} {kind}", flush=True)
        with (out / f"verify-{size}-{kind}.log").open("w") as log:
            subprocess.run(
                command,
                env=(dict(env, PYTHONPATH="") if kind == "replay" else env),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    reference = json.loads((out / f"hf-{size}.json").read_text())
    short = [
        json.loads(s) for s in (out / f"short-{size}.jsonl").read_text().splitlines()
    ]
    checks[size] = {r["mode"]: r["tokens"][0] == reference[r["mode"]] for r in short}
    replay = json.loads((out / f"replay-{size}.json").read_text())
    checks[size]["stock_replay"] = all(t == reference["soft"] for t in replay["tokens"])
    print(checks[size], flush=True)
(out / "reference-checks.json").write_text(json.dumps(checks, indent=2) + "\n")
assert all(k["token"] and k["soft"] and k["stock_replay"] for k in checks.values()), (
    checks
)
print("Hidden free-running trace equality is reported separately; see hidden audits.")
