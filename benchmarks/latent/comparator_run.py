# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-position calibration of third-party forks, without changing their code."""

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
from vllm import LLM, SamplingParams


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["qwen", "swir"], required=True)
    p.add_argument(
        "--mode", choices=["base", "disabled", "soft", "swir"], required=True
    )
    p.add_argument("--model", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--batches", default="1,8,32")
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    batches = list(map(int, a.batches.split(",")))
    extra = {}
    enabled = a.mode in {"soft", "swir"}
    if a.engine == "qwen" and enabled:
        extra = dict(enable_prompt_embeds=True, reasoning_parser="qwen3")
    elif a.engine == "swir" and enabled:
        # Keep the compiled input signature consistent before feedback starts.
        extra = dict(enable_prompt_embeds=True)
    llm = LLM(
        model=a.model,
        revision=a.revision,
        tokenizer_revision=a.revision,
        dtype="bfloat16",
        max_model_len=2048,
        max_num_seqs=max(batches),
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=False,
        async_scheduling=False,
        **extra,
    )
    tokenizer = llm.get_tokenizer()

    def encode(s):
        ids = tokenizer.encode(s, add_special_tokens=False)
        return ids.ids if hasattr(ids, "ids") else ids

    kw = {}
    if a.mode == "soft":
        kw = dict(
            soft_thinking=True,
            soft_topk=10,
            soft_entropy_threshold=0.01,
            soft_patience=256,
        )
    elif a.mode == "swir" and a.engine == "qwen":
        kw = dict(
            swir=True,
            swir_alpha=1.0,
            swir_beta=0.7,
            swir_window=512,
            swir_max_switch_count=None,
            swir_math_token_ids=[],
            swir_convergence_token_ids=encode("</think>"),
            swir_termination_token_ids=encode("</think>\n\nThe final answer is"),
            swir_linebreak_token_id=encode("\n")[-1],
        )
    elif a.mode == "swir":
        kw = dict(
            extra_args={
                "swireasoning": dict(
                    think_id=encode("<think>")[-1],
                    end_think_id=encode("</think>")[-1],
                    line_break_id=encode("\n")[-1],
                    eos_token_id=tokenizer.eos_token_id,
                    convergence_ids=encode("</think>"),
                    termination_ids=encode("</think>\n\nThe final answer is"),
                    alpha_0=1.0,
                    beta_0=0.7,
                    window_size=512,
                    max_switch_count=None,
                    max_new_tokens=a.steps,
                    math_ids=[],
                )
            }
        )
    params = SamplingParams(
        temperature=0, max_tokens=a.steps, ignore_eos=True, detokenize=False, **kw
    )
    for batch in batches:
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
            for i in range(batch)
        ]
        llm.generate(prompts, params, use_tqdm=False)
        durations, hashes = [], []
        for _ in range(a.repeats):
            start = time.perf_counter()
            outputs = llm.generate(prompts, params, use_tqdm=False)
            durations.append(time.perf_counter() - start)
            tokens = [o.outputs[0].token_ids for o in outputs]
            assert all(len(t) == a.steps for t in tokens)
            hashes.append(hashlib.sha256(json.dumps(tokens).encode()).hexdigest())
        record = {
            **vars(a),
            "output": str(a.output),
            "batch": batch,
            "seconds": durations,
            "median_seconds": statistics.median(durations),
            "output_hashes": hashes,
            "tokens": tokens,
        }
        with a.output.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(
            json.dumps({k: v for k, v in record.items() if k != "tokens"}), flush=True
        )


if __name__ == "__main__":
    main()
