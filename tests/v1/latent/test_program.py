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


@pytest.mark.parametrize("kind", ["token", "soft", "hidden", "entropy"])
def test_dependency_pruning_preserves_transition(kind):
    from vllm.v1.latent.compiler import dependencies, inactive, optimize

    original = Program(preset(kind, 3), 4, 7)
    optimized = optimize(original)
    args = (
        torch.randn(4, 7),
        torch.randn(4, 4),
        torch.randn(4, 4),
        torch.arange(4)[:, None],
        torch.randn(4, 8),
        torch.randn(7, 4),
    )
    actual_args = list(args)
    live = dependencies(optimized)[1]
    for i, name in enumerate(["logits", "hidden"]):
        if name not in live:
            actual_args[i] = None
    for a, b in zip(original(*args), optimized(*actual_args)):
        torch.testing.assert_close(a, b)
    assert inactive(optimized, 3)
    assert inactive(optimized, 0) == (kind == "token")


def test_shared_expressions_preserve_per_request_policy_and_state():
    from vllm.v1.latent.compiler import SharedProgram, optimize

    programs = [
        optimize(Program(preset(k, 3), 4, 7))
        for k in ["soft", "entropy", "hidden", "token"]
    ]
    args = (
        torch.randn(4, 7),
        torch.randn(4, 4),
        torch.randn(4, 4),
        torch.arange(4)[:, None],
        torch.randn(4, 8),
        torch.randn(7, 4),
    )
    actual = SharedProgram(programs)(*args, torch.arange(4)[:, None])
    for row, program in enumerate(programs):
        expected = program(*(x[row : row + 1] for x in args[:-1]), args[-1])
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a[row : row + 1], b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_masked_expectation_and_captured_input_selection():
    from vllm.v1.latent.kernels import latent_input, masked_expect

    e = torch.randn(1003, 96, device="cuda", dtype=torch.bfloat16)
    p = torch.randn(3, 1003, device="cuda").softmax(-1)
    mask = torch.tensor([True, False, True], device="cuda")
    result = masked_expect(p, e, mask)
    expected = p.to(e.dtype) @ e
    expected[1].zero_()
    torch.testing.assert_close(result, expected, atol=0.002, rtol=0.02)
    history = result.reshape(1, 3, 96)
    ids = torch.tensor([10, 11, 12], device="cuda")
    indices = torch.tensor([2, -1, 1], device="cuda")
    actual = latent_input(ids, e, history, mask, indices)
    torch.testing.assert_close(actual[0], result[2])
    torch.testing.assert_close(actual[1:], e[ids[1:]])


def test_headless_proof_requires_explicit_finite_fallback_contract():
    from vllm.v1.latent.compiler import active_without_logits

    spec = preset("hidden", 2)
    program = Program(spec, 2, 3)
    assert not active_without_logits(program, 0)
    spec["fallback"] = "zero"
    program = Program(spec, 2, 3)
    assert active_without_logits(program, 0)
    assert not active_without_logits(program, 2)
    result, mask, _ = program(
        None,
        torch.tensor([[float("nan"), 1.0]]),
        torch.ones(1, 2),
        torch.zeros(1, 1),
        torch.zeros(1, 8),
        torch.ones(3, 2),
    )
    assert mask.item()
    assert result.tolist() == [[0.0, 0.0]]


def test_topk_is_explicit_and_k_one_selects_argmax_embedding():
    spec = {
        "ops": [["yes", "CONST", True], ["soft", "TOPK_EXPECT", "logits", 1]],
        "embedding": "soft",
        "latent": "yes",
    }
    e = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    args = (
        torch.tensor([[0.0, 2.0, 1.0]]),
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.zeros(1, 1),
        torch.zeros(1, 8),
        e,
    )
    actual, _, _ = Program(spec, 2, 3)(*args)
    torch.testing.assert_close(actual, e[1:2])


def test_sharing_requires_reused_expectation_and_excludes_custom_projections():
    from vllm.v1.latent.compiler import can_share

    soft = Program(preset("soft"), 2, 3)
    entropy = Program(preset("entropy"), 2, 3)
    hidden = Program(preset("hidden"), 2, 3)
    assert can_share([soft, entropy, hidden])
    assert not can_share([soft, hidden])
    projected = Program(
        {
            "ops": [["yes", "CONST", True], ["out", "PROJECT", "hidden", "w"]],
            "embedding": "out",
            "latent": "yes",
        },
        2,
        3,
        {"w": torch.eye(2)},
    )
    assert not can_share([soft, entropy, projected])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_conditional_expectation_replays_changing_gpu_predicates_exactly():
    from vllm.v1.latent.kernels import conditional_expect

    e = torch.randn(257, 96, device="cuda", dtype=torch.bfloat16)
    p = torch.randn(3, 257, device="cuda").softmax(-1)
    gate = torch.zeros(3, 1, device="cuda", dtype=torch.bool)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        conditional_expect(p, e, gate)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = conditional_expect(p, e, gate)
    torch.cuda.current_stream().wait_stream(stream)
    for mask in [[False] * 3, [True, False, True], [False] * 3, [True] * 3]:
        gate.copy_(torch.tensor(mask, device="cuda")[:, None])
        graph.replay()
        expected = p.to(e.dtype) @ e if any(mask) else torch.zeros_like(actual)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
