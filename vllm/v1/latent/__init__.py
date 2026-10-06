# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental, bounded decode-transition programs."""

DEFAULT_OPTIMIZATIONS = (
    "prune",
    "fast_input",
    "staging",
    "skip_inactive",
    "share",
    "skip_head",
    "conditional_expect",
)
