# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay must follow request identity, not transient continuous-batch rows."""

from types import SimpleNamespace as NS

import torch

from vllm.v1.latent.program import Program, preset
from vllm.v1.latent.runner import TransitionRunner


def make_runner():
    t = TransitionRunner.__new__(TransitionRunner)
    t.history = torch.zeros(3, 8, 2)
    t.state = torch.zeros(3, 8)
    t.masks = torch.zeros(3, 8, dtype=torch.bool)
    t.max_steps = 8
    t.free = []
    t.requests = {"a": (0, "hidden", 3), "b": (1, "hidden", 2)}
    t.embedding = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    t.programs = {"hidden": Program(preset("hidden", 8), 2, 3)}
    t.runner = NS(
        device="cpu",
        input_batch=NS(req_ids=["b", "a"]),
        requests={
            "a": NS(num_computed_tokens=2, num_tokens=4, output_token_ids=[0]),
            "b": NS(num_computed_tokens=2, num_tokens=3, output_token_ids=[0]),
        },
        model=NS(embed_input_ids=lambda ids: t.embedding[ids]),
    )
    return t


def test_reordered_batch_and_partial_prefill_replay():
    t = make_runner()
    t.history[0, 0] = torch.tensor([10.0, 11.0])
    t.history[1, 0] = torch.tensor([20.0, 21.0])
    embeddings = torch.zeros(3, 2)
    t.inject(NS(num_scheduled_tokens={"b": 1, "a": 2}), embeddings)
    assert embeddings.tolist() == [[20.0, 21.0], [0.0, 0.0], [10.0, 11.0]]


def test_chunked_prefill_does_not_advance_program_state():
    t = make_runner()
    t.runner.requests["a"].num_computed_tokens = 0
    sampled = torch.tensor([[1], [2]])
    t.transition(
        NS(num_scheduled_tokens={"b": 1, "a": 2}),
        torch.zeros(2, 3),
        torch.tensor([[7.0, 8.0], [9.0, 10.0]]),
        sampled,
    )
    torch.testing.assert_close(t.history[1, 1], torch.tensor([7.0, 8.0]))
    assert t.history[0].count_nonzero() == 0
    assert sampled.tolist() == [[0], [2]]


def test_finish_releases_only_completed_request_slot():
    t = make_runner()
    t.finish(["b", "missing"])
    assert set(t.requests) == {"a"}
    assert t.free == [1]


def test_unused_program_is_evicted_without_lifetime_registry_exhaustion():
    from vllm import SamplingParams

    t = make_runner()
    t.finish(["a", "b"])
    t.max_programs = 1
    t.constants = {}
    t.use_graphs = False
    t.compile = False
    t.runner.input_batch.vocab_size = 3
    for kind in ["token", "hidden", "soft"]:
        request = NS(
            req_id=kind,
            prompt_token_ids=[0, 1],
            prompt_embeds=None,
            sampling_params=SamplingParams(
                max_tokens=4,
                ignore_eos=True,
                detokenize=False,
                extra_args={"decode_program": preset(kind, 4)},
            ),
        )
        t.admit(request)
        assert len(t.programs) == 1
        assert kind in t.requests
        t.finish([kind])
