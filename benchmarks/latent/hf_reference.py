# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent cached Transformers reference for the short transition workload."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
p.add_argument("--model", default="Qwen/Qwen3-8B")
p.add_argument("--revision", default="b968826d9c46dd6066d109eabc6255188de91218")
a = p.parse_args()
name = a.model
revision = a.revision
tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
model = AutoModelForCausalLM.from_pretrained(
    name,
    revision=revision,
    dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="sdpa",
)
model.eval()
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
records = {}
with torch.inference_mode():
    e = model.get_input_embeddings().weight
    for kind in ["token", "soft", "hidden"]:
        cache = None
        embeds = e[torch.tensor([prompt], device="cuda")]
        ids = []
        for step in range(8):
            out = model(
                inputs_embeds=embeds,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
            )
            cache = out.past_key_values
            logits = out.logits[:, -1].float()
            token = logits.argmax(-1)
            if step < 4 and kind != "token":
                if kind == "soft":
                    embeds = (logits.softmax(-1).to(e.dtype) @ e)[:, None]
                else:
                    embeds = out.hidden_states[-1][:, -1:]
                ids.append(0)
            else:
                embeds = e[token][:, None]
                ids.append(token.item())
        records[kind] = ids
a.output.write_text(json.dumps(records) + "\n")
