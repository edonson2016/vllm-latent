# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stock vLLM public-API soft-feedback adapter: exact full-prefix replay.

This deliberately reports API/replay costs, not native cached latent decoding.
There is no public stock API to inject a new decode embedding into a live request.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

import numpy as np
import torch
from safetensors import safe_open

from vllm import LLM, SamplingParams
from vllm.transformers_utils.repo_utils import hf_api


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--prefix-cache", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--kind", choices=["soft", "token"], default="soft")
    args = p.parse_args()
    model = "Qwen/Qwen3-8B"
    path = Path(
        hf_api().snapshot_download(
            model,
            revision="b968826d9c46dd6066d109eabc6255188de91218",
            local_files_only=True,
        )
    )
    index = json.loads((path / "model.safetensors.index.json").read_text())
    name = "model.embed_tokens.weight"
    with safe_open(path / index["weight_map"][name], framework="pt", device="cpu") as f:
        embedding = f.get_tensor(name).to(device="cuda", dtype=torch.bfloat16)
    llm = LLM(
        model=str(path),
        dtype="bfloat16",
        max_model_len=2048,
        enable_prompt_embeds=True,
        max_logprobs=-1,
        enable_prefix_caching=args.prefix_cache,
        async_scheduling=False,
        gpu_memory_utilization=0.80,
        max_num_seqs=1,
        max_num_batched_tokens=2048,
    )
    tokenizer = llm.get_tokenizer()
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "Solve carefully: a train travels 60 "
                "kilometers in 2 hours and then 90 kilometers in 3 hours. "
                "What is its average speed? Explain your reasoning.",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer.encode(prompt, add_special_tokens=False)
    prompt = encoded.ids if hasattr(encoded, "ids") else encoded
    params = SamplingParams(
        temperature=0,
        max_tokens=1,
        ignore_eos=True,
        detokenize=False,
        logprobs=-1 if args.kind == "soft" else None,
    )
    times, tokens = [], []
    for repeat in range(args.repeats + 1):
        prefix = embedding[torch.tensor(prompt, device="cuda")].cpu()
        ids = []
        start = time.perf_counter()
        for step in range(args.steps):
            step_params = params.clone()
            if step >= args.steps // 2:
                step_params.logprobs = None
            out = llm.generate(
                [{"prompt_embeds": prefix}], step_params, use_tqdm=False
            )[0]
            token = out.outputs[0].token_ids[0]
            if args.kind == "soft" and step < args.steps // 2:
                lp = out.outputs[0].logprobs[0]
                assert len(lp) == embedding.shape[0], (
                    "Full-vocabulary logprobs required"
                )
                values = np.fromiter(
                    (lp[i].logprob for i in range(embedding.shape[0])),
                    dtype=np.float32,
                    count=embedding.shape[0],
                )
                probs = torch.from_numpy(values)
                vector = probs.to("cuda").exp().to(embedding.dtype) @ embedding
                ids.append(0)
            else:
                vector = embedding[token]
                ids.append(token)
            prefix = torch.cat([prefix, vector[None].cpu()], dim=0)
        elapsed = time.perf_counter() - start
        if repeat:
            times.append(elapsed)
            tokens.append(ids)
    result = {
        "mode": "stock_public_api_replay",
        "prefix_cache": args.prefix_cache,
        "kind": args.kind,
        "steps": args.steps,
        "batch": 1,
        "seconds": times,
        "median_seconds": statistics.median(times),
        "tokens": tokens,
    }
    args.output.write_text(json.dumps(result) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
