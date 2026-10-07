# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure parameter turnover and graph reuse, including cold admission time."""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

from round2_bench import DEFAULT

from vllm import LLM, SamplingParams
from vllm.v1.latent.program import preset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parameterize", action="store_true")
    p.add_argument("--catalog", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    opts = DEFAULT.split(",") + (["parameterize"] if a.parameterize else [])
    catalog = (
        dict(program_catalog=[preset("entropy", 8, 2.0)], allow_dynamic_programs=False)
        if a.catalog
        else {}
    )
    if a.catalog and not a.parameterize:
        raise ValueError("This catalog experiment requires --parameterize")
    start = time.perf_counter()
    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        revision="c1899de289a04d12100db370d81485cdf75e47ca",
        dtype="bfloat16",
        max_model_len=256,
        max_num_seqs=8,
        max_num_batched_tokens=256,
        enable_prefix_caching=False,
        async_scheduling=False,
        gpu_memory_utilization=0.85,
        worker_extension_cls="round2_bench.Round2Worker",
        additional_config={
            "latent_decode": dict(
                max_steps=16,
                capacity=16,
                max_programs=4,
                optimizations=opts,
                workspace_bytes=1024**3,
                **catalog,
            )
        },
    )
    load_seconds = time.perf_counter() - start
    prompt = llm.get_tokenizer().apply_chat_template(
        [dict(role="user", content="What is 12 times 7?")],
        tokenize=False,
        add_generation_prompt=True,
    )
    rows = []
    # Twenty distinct values followed by revisits after registry eviction.
    for index in list(range(20)) + list(range(8)):
        params = SamplingParams(
            temperature=0,
            max_tokens=16,
            ignore_eos=True,
            detokenize=False,
            extra_args={"decode_program": preset("entropy", 8, 2 + index / 10)},
        )
        times = []
        for _ in range(2):
            start = time.perf_counter()
            outputs = llm.generate([prompt] * 8, params, use_tqdm=False)
            times.append(time.perf_counter() - start)
        rows.append(
            dict(
                index=index,
                cold_seconds=times[0],
                warm_seconds=times[1],
                tokens=[o.outputs[0].token_ids for o in outputs],
                diagnostics=llm.collective_rpc("round2_diagnostics"),
            )
        )
    a.output.write_text(
        json.dumps(
            dict(
                parameterize=a.parameterize,
                catalog=a.catalog,
                load_seconds=load_seconds,
                rows=rows,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
