# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Use release native libraries with this fork's Python sources, without rebuilding.

Run from outside the checkout, using the release container's Python environment.
"""

import argparse
import pathlib
import shutil

import vllm

p = argparse.ArgumentParser()
p.add_argument("source", type=pathlib.Path)
p.add_argument("destination", type=pathlib.Path)
a = p.parse_args()
assert vllm.__version__ == "0.31.0", vllm.__version__
installed = pathlib.Path(vllm.__file__).parent
target = a.destination / "vllm"
shutil.copytree(installed, target, dirs_exist_ok=True)
for rel in [
    "v1/latent",
    "v1/worker/gpu_model_runner.py",
    "v1/worker/gpu_worker.py",
    "config/vllm.py",
    "v1/engine/input_processor.py",
    "v1/core/sched/scheduler.py",
]:
    src = a.source / "vllm" / rel
    dst = target / rel
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
print(target)
