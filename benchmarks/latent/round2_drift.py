# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolate the 1.7B input-signature discrepancy without any latent transition."""

import argparse
import json
import os
from pathlib import Path

os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
os.environ["VLLM_BATCH_INVARIANT"] = "1"

from vllm import LLM, SamplingParams


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--native", action="store_true")
    p.add_argument("--eager", action="store_true")
    p.add_argument("--fixed-norm", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    opts = ["fast_input"] if a.native else []
    llm = LLM(
        model="Qwen/Qwen3-1.7B",
        revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        dtype="bfloat16",
        max_model_len=2048,
        max_num_seqs=32,
        max_num_batched_tokens=2048,
        enable_prefix_caching=False,
        async_scheduling=False,
        enable_prompt_embeds=not a.native,
        enforce_eager=a.eager,
        compilation_config={"custom_ops": ["+rms_norm"]} if a.fixed_norm else {},
        gpu_memory_utilization=0.85,
        additional_config={
            "latent_decode": dict(max_steps=128, capacity=64, optimizations=opts)
        },
    )
    tokenizer = llm.get_tokenizer()
    prompts = [
        tokenizer.apply_chat_template(
            [
                dict(
                    role="user",
                    content=f"Solve carefully: a train travels {60 + i} "
                    "kilometers in 2 hours and then 90 kilometers in 3 hours. "
                    "What is its average speed? Explain your reasoning.",
                )
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for i in range(32)
    ]
    params = SamplingParams(
        temperature=0, max_tokens=128, ignore_eos=True, detokenize=False, logprobs=2
    )
    outputs = llm.generate(prompts, params, use_tqdm=False)
    rows = []
    for result in outputs:
        output = result.outputs[0]
        rows.append(
            dict(
                tokens=output.token_ids,
                logprobs=[
                    {str(token): value.logprob for token, value in step.items()}
                    for step in output.logprobs
                ],
            )
        )
    a.output.write_text(
        json.dumps(
            dict(native=a.native, eager=a.eager, fixed_norm=a.fixed_norm, rows=rows)
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
