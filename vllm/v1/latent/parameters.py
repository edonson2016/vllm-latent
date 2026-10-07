# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Separate numeric admission parameters from reusable straight-line structure."""

import copy


def parameterize(spec):
    if spec.get("policy") is not None:
        return spec, []
    result = copy.deepcopy(spec)
    values: list[int | float] = []
    for inst in result.get("ops", []):
        if len(inst) == 3 and inst[1] == "CONST" and type(inst[2]) in (int, float):
            if len(values) == 16:
                break
            values.append(inst[2])
            inst[1:] = ["PARAM", len(values) - 1]
    return result, values
