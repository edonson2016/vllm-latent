# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ablation measurements with identical policies and per-mode registry resets."""

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")


class BenchmarkWorker:
    def reset_latent_programs(self):
        latent = self.model_runner.latent_runner
        # Called only after blocking LLM.generate has completed every request.
        # A final scheduler cleanup can otherwise remain pending until next run.
        latent.finish(list(latent.requests))
        latent.programs.clear()
        if hasattr(latent, "definitions"):
            latent.definitions.clear()
            latent.inactive_steps.clear()
            if hasattr(latent, "headless_steps"):
                latent.headless_steps.clear()
        if getattr(latent, "staged", None) is not None:
            latent.staged.graphs.clear()
            if hasattr(latent.staged, "head_graphs"):
                latent.staged.head_graphs.clear()


def main():
    import vllm
    from vllm import LLM, SamplingParams

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-index", type=int, default=0)
    parser.add_argument("--label", required=True)
    parser.add_argument("--optimizations", default="")
    parser.add_argument("--modes", default="token,hidden,soft,mixed")
    parser.add_argument("--batches", default="1,32")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stock", action="store_true")
    parser.add_argument("--invariant", action="store_true")
    parser.add_argument("--quality", action="store_true")
    args = parser.parse_args()
    package = Path(vllm.__file__).parent
    sources = sorted((package / "v1/latent").glob("*.py")) + [
        package / "v1/worker/gpu_model_runner.py"
    ]
    digest = hashlib.sha256(b"".join(path.read_bytes() for path in sources)).hexdigest()
    if args.invariant:
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
    repo = Path(__file__).resolve().parents[2]
    model = json.loads((repo / "results/scaling/models.json").read_text())[
        args.model_index
    ]
    batches = [int(x) for x in args.batches.split(",")]
    opts = [x for x in args.optimizations.split(",") if x]
    extra = (
        {}
        if args.stock
        else {
            "enable_prompt_embeds": "fast_input" not in opts,
            "worker_extension_cls": "optimization_bench.BenchmarkWorker",
            "additional_config": {
                "latent_decode": {
                    "max_steps": args.steps,
                    "capacity": max(batches) * 2,
                    "optimizations": opts,
                }
            },
        }
    )
    llm = LLM(
        model=model["model"],
        revision=model["revision"],
        tokenizer_revision=model["revision"],
        dtype="bfloat16",
        max_model_len=2048,
        max_num_seqs=max(batches),
        max_num_batched_tokens=2048,
        enable_prefix_caching=False,
        async_scheduling=False,
        gpu_memory_utilization=0.85,
        **extra,
    )
    tokenizer = llm.get_tokenizer()
    for mode in args.modes.split(","):
        if not args.stock:
            llm.collective_rpc("reset_latent_programs")
        for batch in batches:
            prompts, params = [], []
            for i in range(batch):
                prompts.append(
                    tokenizer.apply_chat_template(
                        [
                            {
                                "role": "user",
                                "content": f"Solve carefully: a train travels {60 + i} "
                                "kilometers in 2 hours and then 90 kilometers "
                                "in 3 hours. "
                                "What is its average speed? Explain your reasoning.",
                            }
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
                kw = {}
                if not args.stock:
                    from vllm.v1.latent.program import preset

                    kind = (
                        ["token", "soft", "hidden", "entropy"][i % 4]
                        if (mode == "mixed")
                        else mode
                    )
                    if mode == "gated_mixed":
                        kind = "entropy"
                    elif mode == "sparse":
                        kind = "soft" if i == 0 else "entropy"
                    spec = preset(
                        "soft"
                        if kind == "topk"
                        else ("hidden" if kind == "hidden_zero" else kind),
                        args.steps // 2,
                        threshold=2.0 + (i % 2) if mode == "gated_mixed" else 2.0,
                    )
                    if kind == "hidden_zero":
                        spec["fallback"] = "zero"
                    if kind == "topk":
                        spec["ops"][-2:] = [["soft", "TOPK_EXPECT", "logits", 10]]
                    kw["extra_args"] = {"decode_program": spec}
                params.append(
                    SamplingParams(
                        temperature=0,
                        max_tokens=args.steps,
                        ignore_eos=True,
                        detokenize=False,
                        **kw,
                    )
                )
            for _ in range(2):
                llm.generate(prompts, params, use_tqdm=False)
            durations, hashes, tokens = [], [], []
            for _ in range(args.repeats):
                start = time.perf_counter()
                outputs = llm.generate(prompts, params, use_tqdm=False)
                durations.append(time.perf_counter() - start)
                tokens = [o.outputs[0].token_ids for o in outputs]
                assert all(len(ids) == args.steps for ids in tokens)
                hashes.append(hashlib.sha256(json.dumps(tokens).encode()).hexdigest())
            row = {
                **vars(args),
                "output": str(args.output),
                **model,
                "mode": mode,
                "batch": batch,
                "seconds": durations,
                "hashes": hashes,
                "median_seconds": statistics.median(durations),
                "tokens": tokens,
                "utc_epoch": time.time(),
                "implementation_hash": digest,
                "latent_counts": None
                if args.stock
                else llm.collective_rpc("get_latent_decode_counts"),
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(
                json.dumps({k: v for k, v in row.items() if k != "tokens"}), flush=True
            )

    if args.quality:
        from quality_smoke import evaluate

        evaluate(
            llm,
            tokenizer,
            args.output.with_suffix(".quality.jsonl"),
            args.label,
            args.stock,
        )

        if not args.stock:
            from quality_smoke import reference_check

            size = model["model"].split("-")[-1]
            reference = json.loads(
                (repo / f"results/scaling/hf-{size}.json").read_text()
            )
            reference_check(
                llm,
                tokenizer,
                args.output.with_suffix(".reference.jsonl"),
                args.label,
                reference,
            )


if __name__ == "__main__":
    main()
