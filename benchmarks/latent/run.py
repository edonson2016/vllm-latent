# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-work latency calibration. Each invocation loads exactly one engine."""

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

from vllm import LLM, SamplingParams


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=[
            "stock",
            "embed_control",
            "disabled",
            "token",
            "soft",
            "hidden",
            "norm_hidden",
            "entropy",
            "mixed",
        ],
        required=True,
    )
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--revision", default="b968826d9c46dd6066d109eabc6255188de91218")
    p.add_argument("--batches", default="1,8,32")
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--compile-transition", action="store_true")
    p.add_argument("--eager", action="store_true")
    p.add_argument("--entropy-threshold", type=float, default=2.0)
    p.add_argument("--trace-masks", action="store_true")
    args = p.parse_args()
    enabled = args.mode not in {"stock", "disabled", "embed_control"}
    batches = [int(n) for n in args.batches.split(",")]
    extra = {}
    if enabled:
        extra = {
            "enable_prompt_embeds": True,
            "additional_config": {
                "latent_decode": {
                    "max_steps": args.steps,
                    "capacity": max(batches) * 2,
                    "compile": args.compile_transition,
                }
            },
        }
    if args.mode == "embed_control":
        extra = {"enable_prompt_embeds": True}
    llm = LLM(
        model=args.model,
        revision=args.revision,
        tokenizer_revision=args.revision,
        dtype="bfloat16",
        max_model_len=2048,
        max_num_seqs=max(batches),
        max_num_batched_tokens=2048,
        enable_prefix_caching=False,
        async_scheduling=False,
        gpu_memory_utilization=0.85,
        enforce_eager=args.eager,
        **extra,
    )
    tokenizer = llm.get_tokenizer()
    for batch in batches:
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": f"Solve carefully: a train travels {60 + i} "
                        "kilometers in 2 hours and then 90 kilometers in 3 hours. "
                        "What is its average speed? Explain your reasoning.",
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for i in range(batch)
        ]
        params = []
        for i in range(batch):
            kw = {}
            if enabled:
                from vllm.v1.latent.program import preset

                kind = (
                    ["token", "soft", "hidden", "entropy"][i % 4]
                    if (args.mode == "mixed")
                    else args.mode
                )
                kw["extra_args"] = {
                    "decode_program": preset(
                        kind, args.steps // 2, args.entropy_threshold
                    )
                }
            params.append(
                SamplingParams(
                    temperature=0,
                    max_tokens=args.steps,
                    ignore_eos=True,
                    detokenize=False,
                    **kw,
                )
            )
        llm.generate(prompts, params, use_tqdm=False)
        durations, hashes, samples = [], [], []
        for _ in range(args.repeats):
            start = time.perf_counter()
            outputs = llm.generate(prompts, params, use_tqdm=False)
            durations.append(time.perf_counter() - start)
            tokens = [o.outputs[0].token_ids for o in outputs]
            assert all(len(t) == args.steps for t in tokens)
            hashes.append(hashlib.sha256(json.dumps(tokens).encode()).hexdigest())
            samples = tokens
        latent_counts = None
        if enabled and args.trace_masks:
            latent_counts = llm.collective_rpc("get_latent_decode_counts")
        record = {
            **vars(args),
            "output": str(args.output),
            "batch": batch,
            "seconds": durations,
            "median_seconds": statistics.median(durations),
            "steps_per_second": batch * args.steps / statistics.median(durations),
            "output_hashes": hashes,
            "tokens": samples,
            "latent_counts": latent_counts,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(
            json.dumps({k: v for k, v in record.items() if k != "tokens"}), flush=True
        )


if __name__ == "__main__":
    main()
