# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drive the new GPU policy and the pinned, unmodified Qwen holder identically."""

import argparse
import importlib.util
import json
import random
from pathlib import Path
from types import SimpleNamespace

import torch
from vllm.v1.sample.swi_reasoning_state import SwiReasoningStateHolder

from vllm import SamplingParams
from vllm.v1.sample.logits_processor.interface import BatchUpdate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    source = Path(__file__).resolve().parents[2] / "vllm/v1/latent/swireasoning.py"
    module_spec = importlib.util.spec_from_file_location("swi_candidate", source)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    rng = random.Random(20261006)
    count = active_count = forced_count = cutoffs = switches = 0
    # Cover the Cartesian product, including a one-position cutoff with
    # switching enabled and math-token exemptions both on and off.
    for trial in range(128):
        window = [1, 2, 8, 512][trial % 4]
        max_switch = [None, 1, 2, 4][(trial // 4) % 4]
        term_max = [1, 3, 10, 32][(trial // 16) % 4]
        math_ids = [11, 12] if trial // 64 else []
        alpha, beta = rng.random(), rng.random()
        kwargs = dict(
            swir=True,
            swir_alpha=alpha,
            swir_beta=beta,
            swir_window=window,
            swir_max_switch_count=max_switch,
            swir_termination_max_tokens=term_max,
            swir_math_token_ids=math_ids,
            swir_convergence_token_ids=[61],
            swir_termination_token_ids=[61, 7, 8],
            swir_linebreak_token_id=62,
            stop_token_ids=[63],
            max_tokens=128,
        )
        holder = SwiReasoningStateHolder(
            SimpleNamespace(
                reasoning_start_token_ids=[60], reasoning_end_token_ids=[61]
            ),
            1,
            torch.device("cpu"),
        )
        holder.sync_batch(
            BatchUpdate(
                batch_size=1,
                removed=[],
                moved=[],
                added=[(0, SamplingParams(**kwargs), [], [])],
            )
        )
        program = module.SwiProgram(
            dict(
                policy="swireasoning",
                alpha=alpha,
                beta=beta,
                window=window,
                max_switch_count=max_switch,
                termination_max_tokens=term_max,
                math_ids=math_ids,
                convergence_ids=[61],
                termination_ids=[61, 7, 8],
                linebreak_id=62,
                start_id=60,
                end_id=61,
                stop_id=63,
                max_tokens=128,
            ),
            4,
            64,
        )
        program.prepare(torch.randn(64, 4))
        state = torch.zeros(1, 8)
        for step in range(128):
            entropy = torch.tensor([[rng.uniform(0.01, 5.0)]])
            sampled = rng.choice([3, 11, 12, 3, 3, 3, 61]) if trial % 5 == 0 else 3
            old = holder._state[0]
            old["entropy"] = entropy.item()
            old["probs"] = torch.full((64,), 1 / 64)
            forced = holder.step(torch.tensor([sampled])).item()
            active, weight, anchor, state, recorded = program.decide(
                entropy, torch.tensor([[sampled]]), torch.tensor([[step]]), state
            )
            assert recorded.item() == (forced if forced >= 0 else sampled), (
                trial,
                step,
            )
            assert active.item() == (old["directive"] is not None), (trial, step)
            expected = [
                old[k] for k in ("mode", "stay", "ref", "locked", "switch_count")
            ]
            torch.testing.assert_close(
                state[0, :5], torch.tensor(expected, dtype=torch.float32)
            )
            assert state[0, 7].item() == old["budget"]
            if active.item():
                assert abs(weight.item() - old["directive"]["blend_w"]) < 2e-7
                if old["directive"]["blend_id"] is not None:
                    assert anchor.item() == old["directive"]["blend_id"]
            count += 1
            active_count += int(active.item())
            forced_count += int(forced >= 0)
            cutoffs += int(forced == 63)
        switches += old["switch_count"]
    a.output.write_text(
        json.dumps(
            dict(
                transitions=count,
                active=active_count,
                forced=forced_count,
                cutoffs=cutoffs,
                switches=switches,
                assertions_passed=True,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
