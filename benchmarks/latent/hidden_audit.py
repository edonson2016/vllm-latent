# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Untimed diagnostic: compare HF with the exact embeddings consumed by vLLM."""

import argparse
import json
from pathlib import Path


class HiddenAuditWorker:
    """Operator-loaded, named RPC; no per-step hook or serialized callback."""

    def read_hidden_history(self):
        latent = self.model_runner.latent_runner
        return {
            "embeddings": latent.history[0, :8].float().cpu().tolist(),
            "masks": latent.masks[0, :8].cpu().tolist(),
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["capture", "reference"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    a = p.parse_args()
    model = a.model
    revision = a.revision
    if a.phase == "capture":
        from vllm import LLM, SamplingParams
        from vllm.v1.latent.program import preset

        llm = LLM(
            model=model,
            revision=revision,
            tokenizer_revision=revision,
            dtype="bfloat16",
            max_model_len=2048,
            max_num_seqs=1,
            max_num_batched_tokens=2048,
            gpu_memory_utilization=0.85,
            enforce_eager=True,
            enable_prefix_caching=False,
            async_scheduling=False,
            enable_prompt_embeds=True,
            worker_extension_cls="hidden_audit.HiddenAuditWorker",
            additional_config={"latent_decode": {"max_steps": 8, "capacity": 2}},
        )
        prompt = llm.get_tokenizer().apply_chat_template(
            [
                dict(
                    role="user",
                    content="Solve carefully: a train travels 60 "
                    "kilometers in 2 hours and then 90 kilometers in 3 hours. "
                    "What is its average speed? Explain your reasoning.",
                )
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        output = llm.generate(
            [prompt],
            SamplingParams(
                temperature=0,
                max_tokens=8,
                ignore_eos=True,
                detokenize=False,
                extra_args={"decode_program": preset("hidden", 4)},
            ),
            use_tqdm=False,
        )[0]
        record = llm.collective_rpc("read_hidden_history")[0]
        record.update(
            prompt_ids=output.prompt_token_ids, tokens=output.outputs[0].token_ids
        )
        a.output.write_text(json.dumps(record) + "\n")
        return

    import torch
    from transformers import AutoModelForCausalLM

    record = json.loads(a.output.read_text())
    llm = AutoModelForCausalLM.from_pretrained(
        model,
        revision=revision,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    llm.eval()
    e = llm.get_input_embeddings().weight
    history = torch.tensor(record["embeddings"], device="cuda", dtype=e.dtype)
    comparisons, tokens = [], []
    cache = None
    with torch.inference_mode():
        embeds = e[torch.tensor([record["prompt_ids"]], device="cuda")]
        for step in range(8):
            result = llm(
                inputs_embeds=embeds,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
            )
            cache = result.past_key_values
            h = result.hidden_states[-1][:, -1].float()
            token = result.logits[:, -1].argmax(-1).item()
            tokens.append(0 if step < 4 else token)
            if step < 4:
                expected = history[step : step + 1].float()
                comparisons.append(
                    dict(
                        step=step,
                        cosine=torch.nn.functional.cosine_similarity(
                            h, expected
                        ).item(),
                        relative_l2=((h - expected).norm() / expected.norm()).item(),
                        max_absolute=(h - expected).abs().max().item(),
                    )
                )
            embeds = history[step : step + 1, None]
    result = dict(
        teacher_forced_tokens=tokens,
        fork_tokens=record["tokens"],
        exact_tokens=tokens == record["tokens"],
        hidden_comparisons=comparisons,
    )
    a.output.with_name(a.output.stem.replace("trace", "results") + ".json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
