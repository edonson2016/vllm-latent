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

# Opt in for SwiReasoning and workloads with frequent parameter turnover.
# The generic default stays conservative: this combination is not uniformly
# faster for already-warm, fixed generic programs.
COMBINED_OPTIMIZATIONS = DEFAULT_OPTIMIZATIONS + (
    "union_gate",
    "async_metadata",
    "capture_head",
    "fuse_state",
    "parameterize",
)

# Match active-row GEMM shapes more closely to the reference SwiReasoning
# implementation. Capture cost grows with maximum batch size; this is an
# explicit numerical-compatibility tradeoff, not a bitwise guarantee.
REFERENCE_SHAPE_OPTIMIZATIONS = COMBINED_OPTIMIZATIONS + ("exact_compact",)
