# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic arithmetic smoke set, not a general reasoning benchmark."""

import json

import regex as re


def evaluate(llm, tokenizer, output, label, stock=False):
    from vllm import SamplingParams

    tasks = []
    for i in range(16):
        a, b, c = 13 + i, 3 + i % 7, 2 + i % 5
        tasks.extend(
            [
                (f"Compute {a} + {b}.", a + b),
                (f"Compute {a} * {b}.", a * b),
                (f"Compute ({a} + {b}) * {c}.", (a + b) * c),
                (
                    (
                        f"There are {c} bags with {a} marbles each. You give away {b} "
                        "marbles. How many remain?"
                    ),
                    c * a - b,
                ),
            ]
        )
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text + " Return only the integer answer."}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for text, _ in tasks
    ]
    modes = ["token"] if stock else ["token", "soft", "hidden"]
    if not stock and not label.startswith("baseline"):
        modes.append("topk")
    for kind in modes:
        extra = {}
        if not stock:
            from vllm.v1.latent.program import preset

            llm.collective_rpc("reset_latent_programs")
            spec = preset("soft" if kind == "topk" else kind, 4)
            if kind == "topk":
                spec["ops"][-2:] = [["soft", "TOPK_EXPECT", "logits", 10]]
            extra = {"decode_program": spec}
        params = SamplingParams(
            temperature=0,
            max_tokens=64,
            ignore_eos=True,
            detokenize=False,
            extra_args=extra,
        )
        results = llm.generate(prompts, params, use_tqdm=False)
        cases = []
        for (prompt, answer), result in zip(tasks, results):
            ids = list(result.outputs[0].token_ids)
            if kind != "token":
                assert ids[:4] == [0] * 4, "Unexpected fallback in latent prefix"
                ids = ids[4:]
            if tokenizer.eos_token_id in ids:
                ids = ids[: ids.index(tokenizer.eos_token_id)]
            text = tokenizer.decode(ids, skip_special_tokens=True).strip()
            match = re.fullmatch(r"\s*(-?\d+)\s*[.!]?\s*", text)
            numbers = re.findall(r"-?\d+", text)
            correct = bool(numbers and int(numbers[-1]) == answer)
            cases.append(
                {
                    "prompt": prompt,
                    "answer": answer,
                    "text": text,
                    "correct": correct,
                    "strict_format_correct": bool(match and int(match[1]) == answer),
                    "tokens": result.outputs[0].token_ids,
                }
            )
        row = {
            "label": label,
            "mode": kind,
            "correct": sum(x["correct"] for x in cases),
            "total": len(cases),
            "cases": cases,
        }
        with output.open("a") as f:
            f.write(json.dumps(row) + "\n")


def reference_check(llm, tokenizer, output, label, reference):
    from vllm import SamplingParams
    from vllm.v1.latent.program import preset

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
    row = {"label": label, "reference": reference, "tokens": {}, "matches": {}}
    for kind in ["token", "soft", "hidden"]:
        llm.collective_rpc("reset_latent_programs")
        params = SamplingParams(
            temperature=0,
            max_tokens=8,
            ignore_eos=True,
            detokenize=False,
            extra_args={"decode_program": preset(kind, 4)},
        )
        result = llm.generate([prompt], params, use_tqdm=False)[0].outputs[0].token_ids
        row["tokens"][kind] = result
        row["matches"][kind] = result == reference[kind]
    with output.open("a") as f:
        f.write(json.dumps(row) + "\n")
