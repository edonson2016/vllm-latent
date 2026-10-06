# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reject incompatible requests in the frontend, before scheduler admission."""

import json

import torch

from vllm.v1.latent.program import Program


def validate_params(params, max_steps, prompt_embeds):
    if (
        not params.ignore_eos
        or params.stop
        or params.stop_token_ids
        or params.detokenize
        or params.logprobs is not None
        or params.prompt_logprobs is not None
        or params.structured_outputs
        or params.repetition_penalty != 1
        or params.presence_penalty != 0
        or params.frequency_penalty != 0
        or params.bad_words
        or params.allowed_token_ids
        or params.logit_bias
        or params.trace_decode_token_ids
        or prompt_embeds is not None
    ):
        raise ValueError(
            "Decode programs require fixed-budget diagnostic output: "
            "ignore_eos=True, detokenize=False, no stops, logprobs, "
            "penalties, constraints, or prompt embeddings"
        )
    if params.max_tokens is None or params.max_tokens > max_steps:
        raise ValueError("Explicit max_tokens must fit latent max_steps budget")


def validate_request(config, params, prompt_embeds):
    spec = (params.extra_args or {}).get("decode_program")
    if spec is None:
        return
    latent = config.additional_config.get("latent_decode")
    if latent is None:
        raise ValueError("decode_program requires additional_config.latent_decode")
    validate_params(params, latent.get("max_steps", 512), prompt_embeds)
    if len(json.dumps(spec, allow_nan=False)) > 16384:
        raise ValueError("Decode program exceeds 16 KiB admission limit")
    constants = {}
    if latent.get("constants"):
        from safetensors import safe_open

        with safe_open(latent["constants"], framework="pt") as f:
            for key in f.keys():  # noqa: SIM118 -- safe_open is not a dict
                constants[key] = torch.empty(
                    f.get_slice(key).get_shape(), device="meta"
                )
    Program(
        spec,
        config.model_config.get_hidden_size(),
        config.model_config.get_vocab_size(),
        constants,
    )
