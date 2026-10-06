# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small semantic checks independent of transformer/model downloads.

Guard vector shape, mask semantics, request state, and graph-compatible execution
at the cheapest level: direct tensor programs with a tiny vocabulary.
"""

import pytest
import torch

from vllm.v1.latent.program import Program, preset


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_soft_feedback_and_budget_return_to_token(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    logits = torch.tensor([[0.0, 1.0, 2.0], [3.0, 0.0, 0.0]], device=device)
    e = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=device)
    token = e[logits.argmax(-1)]
    program = Program(preset("soft", 1), 2, 3)
    result, mask, _ = program(
        logits,
        token,
        token,
        torch.tensor([[0], [1]], device=device),
        torch.zeros(2, 8, device=device),
        e,
    )
    torch.testing.assert_close(result[0], logits[0].softmax(-1) @ e)
    torch.testing.assert_close(result[1], token[1])
    assert mask.tolist() == [[True], [False]]


def test_state_updates_are_simultaneous_and_isolated():
    spec = {
        "ops": [
            ["yes", "CONST", True],
            ["one", "CONST", 1],
            ["count", "ADD", "s0", "one"],
        ],
        "embedding": "hidden",
        "latent": "yes",
        "updates": {"s0": "count", "s1": "s0"},
    }
    program = Program(spec, 2, 3)
    state = torch.zeros(2, 8)
    state[1, 0] = 7
    _, _, result = program(
        torch.zeros(2, 3),
        torch.ones(2, 2),
        torch.zeros(2, 2),
        torch.zeros(2, 1),
        state,
        torch.zeros(3, 2),
    )
    assert result[:, :2].tolist() == [[1, 0], [8, 7]]


def test_projection_activation_and_numeric_fallback():
    spec = {
        "ops": [
            ["yes", "CONST", True],
            ["p", "PROJECT", "hidden", "w"],
            ["a", "SILU", "p"],
        ],
        "embedding": "a",
        "latent": "yes",
    }
    program = Program(spec, 2, 3, {"w": torch.eye(2)})
    hidden = torch.tensor([[1.0, -1.0], [float("nan"), 0.0]])
    token = torch.ones(2, 2)
    result, mask, _ = program(
        torch.zeros(2, 3),
        hidden,
        token,
        torch.zeros(2, 1),
        torch.zeros(2, 8),
        torch.zeros(3, 2),
    )
    torch.testing.assert_close(result[0], torch.nn.functional.silu(hidden[0]))
    torch.testing.assert_close(result[1], token[1])
    assert mask.tolist() == [[True], [False]]


@pytest.mark.parametrize(
    "spec",
    [
        {"ops": [["x", "CALL", "hidden"]], "embedding": "x", "latent": "x"},
        {"ops": [["x", "CONST", True]], "embedding": "logits", "latent": "x"},
        {"ops": [["x", "SOFTMAX", "missing"]], "embedding": "x", "latent": "x"},
        {"ops": [["x", "CONST", float("inf")]], "embedding": "hidden", "latent": "x"},
    ],
)
def test_invalid_program_rejected_before_execution(spec):
    with pytest.raises(ValueError):
        Program(spec, 2, 3)


def test_compiled_program_is_fullgraph():
    program = torch.compile(
        Program(preset("entropy", 4), 2, 3), fullgraph=True, backend="eager"
    )
    args = (
        torch.randn(2, 3),
        torch.randn(2, 2),
        torch.randn(2, 2),
        torch.zeros(2, 1),
        torch.zeros(2, 8),
        torch.randn(3, 2),
    )
    expected = Program(preset("entropy", 4), 2, 3)(*args)
    actual = program(*args)
    for x, y in zip(expected, actual):
        torch.testing.assert_close(x, y)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_graph_replays_new_rows_and_padded_batch():
    from vllm.v1.latent.graph import GraphProgram

    embedding = torch.randn(7, 4, device="cuda", dtype=torch.bfloat16)
    program = Program(preset("soft", 3), 4, 7)
    graph = GraphProgram(program, embedding, 4)
    for size in [1, 3, 4, 1]:
        args = (
            torch.randn(size, 7, device="cuda"),
            torch.randn(size, 4, device="cuda", dtype=torch.bfloat16),
            torch.randn(size, 4, device="cuda", dtype=torch.bfloat16),
            torch.randint(0, 6, (size, 1), device="cuda"),
            torch.randn(size, 8, device="cuda"),
            embedding,
        )
        expected = program(*args)
        actual = graph(*args)
        for x, y in zip(expected, actual):
            torch.testing.assert_close(x, y)
