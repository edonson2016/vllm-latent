# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Overlay a pinned fork's complete Python tree on its own base wheel."""

import argparse
import importlib.util
import shutil
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("source", type=Path)
p.add_argument("destination", type=Path)
a = p.parse_args()
installed = Path(importlib.util.find_spec("vllm").origin).parent
target = a.destination / "vllm"
shutil.copytree(installed, target, dirs_exist_ok=True)
shutil.copytree(
    a.source / "vllm",
    target,
    dirs_exist_ok=True,
    ignore=shutil.ignore_patterns("__pycache__"),
)
print(target)
