# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched SwiReasoning policies, natural termination, and fixed-work latency."""

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import regex as re

os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

DEFAULT = "prune,fast_input,staging,skip_inactive,share,skip_head,conditional_expect"


def main():
    import vllm
    from vllm import LLM, SamplingParams

    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["fork", "qwen", "stock"], default="fork")
    p.add_argument("--model-index", type=int, default=0)
    p.add_argument("--label", default="candidate")
    p.add_argument("--opts", default=DEFAULT)
    p.add_argument("--modes", default="token,swi512,swi8")
    p.add_argument("--batches", default="1,8,32")
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=20261006)
    p.add_argument("--quality", type=Path)
    p.add_argument("--constants")
    p.add_argument("--profile")
    p.add_argument("--fixed-norm", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[2]
    model = json.loads((repo / "results/scaling/models.json").read_text())[
        a.model_index
    ]
    package = Path(vllm.__file__).parent
    sources = sorted((package / "v1/latent").glob("*.py")) + [
        package / "v1/worker/gpu_model_runner.py",
        package / "model_executor/layers/vocab_parallel_embedding.py",
    ]
    oracle = package / "v1/sample/swi_reasoning_state.py"
    if oracle.exists():
        sources.append(oracle)
    digest = hashlib.sha256(b"".join(path.read_bytes() for path in sources)).hexdigest()
    batches = list(map(int, a.batches.split(",")))
    extra = {}
    if a.engine == "fork":
        config = dict(
            max_steps=a.steps,
            capacity=max(batches) * 2,
            optimizations=a.opts.split(","),
            max_programs=16,
            workspace_bytes=3 * 1024**3,
        )
        if a.constants:
            config["constants"] = a.constants
        extra = dict(
            additional_config={"latent_decode": config},
            enable_prompt_embeds="fast_input" not in a.opts.split(","),
            worker_extension_cls="round2_bench.Round2Worker",
        )
    elif a.engine == "qwen":
        extra = dict(enable_prompt_embeds=True, reasoning_parser="qwen3")
    if a.profile:
        extra["profiler_config"] = dict(
            profiler="torch",
            torch_profiler_dir=a.profile,
            torch_profiler_with_stack=False,
        )
    if a.fixed_norm:
        extra["compilation_config"] = {"custom_ops": ["+rms_norm"]}
    start = time.perf_counter()
    llm = LLM(
        model=model["model"],
        revision=model["revision"],
        tokenizer_revision=model["revision"],
        dtype="bfloat16",
        max_model_len=max(2048, a.steps + 1024),
        max_num_seqs=max(batches),
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=False,
        async_scheduling=False,
        **extra,
    )
    load_seconds = time.perf_counter() - start
    tokenizer = llm.get_tokenizer()
    from transformers import GenerationConfig

    eos_ids = GenerationConfig.from_pretrained(
        model["model"], revision=model["revision"], local_files_only=True
    ).eos_token_id
    stop_id = min(eos_ids) if isinstance(eos_ids, list) else eos_ids

    def encode(text):
        ids = tokenizer.encode(text, add_special_tokens=False)
        return ids if isinstance(ids, list) else ids.ids

    def params(mode):
        options = {}
        if mode != "token":
            window = 512 if mode == "swi512" else 8 if mode == "swi8" else 32
            switches = 2 if mode == "swi8" else None
            if a.engine == "qwen":
                options = dict(
                    swir=True,
                    swir_alpha=1.0,
                    swir_beta=0.7,
                    swir_window=window,
                    swir_max_switch_count=switches,
                    swir_math_token_ids=[],
                    swir_termination_max_tokens=32,
                    swir_convergence_token_ids=encode("</think>"),
                    swir_termination_token_ids=encode(
                        "</think>\n\nThe final answer is"
                    ),
                    swir_linebreak_token_id=encode("\n")[-1],
                )
            elif a.engine == "fork":
                from vllm.v1.latent.swireasoning import swi_preset

                expectation = {"swi512": "fp32", "swi8": "fp32", "dense32": "fp32"}.get(
                    mode, mode
                )
                spec = swi_preset(
                    tokenizer,
                    a.steps,
                    window=window,
                    max_switch_count=switches,
                    termination_max_tokens=32,
                    expectation=expectation,
                    stop_id=stop_id,
                )
                options = dict(extra_args={"decode_program": spec})
        return SamplingParams(
            temperature=a.temperature,
            top_p=0.95 if a.temperature else 1.0,
            top_k=20 if a.temperature else -1,
            seed=a.seed,
            max_tokens=a.steps,
            ignore_eos=not bool(a.quality),
            detokenize=False,
            **options,
        )

    data = json.loads(a.quality.read_text()) if a.quality else None
    for mode in a.modes.split(","):
        for batch in batches:
            cases = (
                data["cases"]
                if data
                else [
                    dict(
                        question=f"Solve carefully: a train travels {60 + i} "
                        "kilometers in 2 hours and then 90 kilometers in 3 hours. "
                        "What is its average speed? Explain your reasoning."
                    )
                    for i in range(batch)
                ]
            )
            prompts = [
                tokenizer.apply_chat_template(
                    [dict(role="user", content=x["question"])],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for x in cases
            ]
            sampling = params(mode)
            admission = time.perf_counter()
            llm.generate(prompts[:batch], sampling, use_tqdm=False)
            admission = time.perf_counter() - admission
            llm.generate(prompts[:batch], sampling, use_tqdm=False)
            durations, hashes, traces = [], [], []
            for _ in range(1 if data else a.repeats):
                if a.profile:
                    llm.start_profile()
                start = time.perf_counter()
                outputs = llm.generate(prompts, sampling, use_tqdm=False)
                durations.append(time.perf_counter() - start)
                if a.profile:
                    llm.stop_profile()
                tokens = [list(o.outputs[0].token_ids) for o in outputs]
                if not data:
                    assert all(len(t) == a.steps for t in tokens)
                hashes.append(hashlib.sha256(json.dumps(tokens).encode()).hexdigest())
                traces.append(tokens)
            records = []
            if data:
                for case, output, ids in zip(cases, outputs, tokens):
                    text = tokenizer.decode(ids, skip_special_tokens=False)
                    # No answer credit for unfinished thinking, even if the last
                    # intermediate number happens to match the gold answer.
                    answer_text = (
                        text.rsplit("</think>", 1)[-1] if "</think>" in text else ""
                    )
                    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", answer_text)
                    guess = numbers[-1].replace(",", "") if numbers else None
                    records.append(
                        dict(
                            index=case["index"],
                            answer=case["answer"],
                            guess=guess,
                            correct=guess == case["answer"],
                            finish=output.outputs[0].finish_reason,
                            text=text,
                        )
                    )
            row = dict(
                engine=a.engine,
                label=a.label,
                model=model,
                mode=mode,
                batch=batch,
                steps=a.steps,
                seconds=durations,
                median_seconds=statistics.median(durations),
                hashes=hashes,
                traces=traces,
                quality=records,
                load_seconds=load_seconds,
                first_call_seconds=admission,
                opts=a.opts,
                natural_termination=bool(data),
                temperature=a.temperature,
                seed=a.seed,
                epoch=time.time(),
                implementation_hash=digest,
                fixed_norm=a.fixed_norm,
                batch_invariant=os.environ.get("VLLM_BATCH_INVARIANT") == "1",
            )
            if a.engine == "fork":
                row["diagnostics"] = llm.collective_rpc("round2_diagnostics")
            a.output.parent.mkdir(parents=True, exist_ok=True)
            with a.output.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(
                json.dumps(
                    {k: v for k, v in row.items() if k not in {"traces", "quality"}}
                ),
                flush=True,
            )


class Round2Worker:
    def round2_diagnostics(self):
        import torch

        latent = self.model_runner.latent_runner
        return dict(
            latent_counts=latent.masks.sum(1).cpu().tolist(),
            states=latent.state.cpu().tolist(),
            allocated=torch.accelerator.memory_allocated(),
            programs=len(latent.programs),
            graph_sets=len(latent.staged.graphs) if latent.staged else 0,
            slots={key: value[0] for key, value in latent.requests.items()},
            head_graph_sets=len(latent.staged.head_graphs)
            if latent.staged and hasattr(latent.staged, "head_graphs")
            else 0,
        )


if __name__ == "__main__":
    main()
