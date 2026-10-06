# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise mixed policies, unequal lifetimes, chunking, and KV preemption."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

from vllm import LLM, SamplingParams
from vllm.v1.latent.program import preset

p = argparse.ArgumentParser()
p.add_argument("--blocks", type=int)
p.add_argument("--invariant", action="store_true")
p.add_argument("--output", type=Path, required=True)
p.add_argument("--optimizations", default=None)
a = p.parse_args()
if a.invariant:
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
llm = LLM(
    model="Qwen/Qwen3-8B",
    revision="b968826d9c46dd6066d109eabc6255188de91218",
    tokenizer_revision="b968826d9c46dd6066d109eabc6255188de91218",
    dtype="bfloat16",
    max_model_len=256,
    max_num_seqs=8,
    max_num_batched_tokens=128,
    enable_prefix_caching=False,
    async_scheduling=False,
    enable_prompt_embeds=True,
    disable_log_stats=False,
    num_gpu_blocks_override=a.blocks,
    gpu_memory_utilization=0.85,
    additional_config={
        "latent_decode": {
            "capacity": 8,
            "max_steps": 64,
            **(
                {"optimizations": a.optimizations.split(",")}
                if a.optimizations is not None
                else {}
            ),
        }
    },
)
tokenizer = llm.get_tokenizer()
prompts, params = [], []
for i in range(12):
    prompts.append(
        tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": ("Please reason carefully. " * (i + 1))
                    + f"What is {i + 10} times {i + 20}? Explain the calculation.",
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    )
    kind = ["token", "soft", "hidden", "entropy"][i % 4]
    params.append(
        SamplingParams(
            temperature=0,
            max_tokens=32 + (i % 3) * 16,
            ignore_eos=True,
            detokenize=False,
            extra_args={"decode_program": preset(kind, 16)},
        )
    )
outputs = llm.generate(prompts, params, use_tqdm=False)
a.output.write_text(
    json.dumps(
        {
            "blocks": a.blocks,
            "invariant": a.invariant,
            "tokens": [o.outputs[0].token_ids for o in outputs],
            "preemptions": [
                o.metrics.num_preemptions if o.metrics else None for o in outputs
            ],
        }
    )
    + "\n"
)
