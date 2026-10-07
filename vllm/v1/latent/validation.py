# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reject incompatible requests in the frontend, before scheduler admission."""

import json

import torch

from vllm.v1.latent.program import build_program


def validate_params(params, max_steps, prompt_embeds):
    spec = (params.extra_args or {}).get("decode_program", {})
    if not isinstance(spec, dict):
        raise ValueError("decode_program must be an object")
    recorded_tokens = spec.get("policy") == "swireasoning"
    if (
        (not params.ignore_eos and not recorded_tokens)
        or params.stop
        or (params.stop_token_ids and not recorded_tokens)
        or (params.detokenize and not recorded_tokens)
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
        or params.thinking_token_budget is not None
        or prompt_embeds is not None
    ):
        raise ValueError(
            "Decode programs require fixed-budget diagnostic output: "
            "ignore_eos=True, detokenize=False, no stops, logprobs, "
            "penalties, constraints, or prompt embeddings"
        )
    if params.max_tokens is None or params.max_tokens > max_steps:
        raise ValueError("Explicit max_tokens must fit latent max_steps budget")
    if recorded_tokens and spec.get("max_tokens", 512) != params.max_tokens:
        raise ValueError("SwiReasoning max_tokens must equal the sampling budget")


def validate_request(config, params, prompt_embeds):
    spec = (params.extra_args or {}).get("decode_program")
    if spec is None:
        return
    if not isinstance(spec, dict):
        raise ValueError("decode_program must be an object")
    if any(
        isinstance(inst, list) and len(inst) > 1 and inst[1] == "PARAM"
        for inst in spec.get("ops", [])
    ):
        raise ValueError(
            "PARAM is an internal instruction; supply numeric CONST values"
        )
    latent = config.additional_config.get("latent_decode")
    if latent is None:
        raise ValueError("decode_program requires additional_config.latent_decode")
    if not latent.get("allow_dynamic_programs", True):
        from vllm.v1.latent.parameters import parameterize

        def identity(value):
            if "parameterize" in latent.get("optimizations", []):
                value, _ = parameterize(value)
            return json.dumps(value, sort_keys=True, allow_nan=False)

        if identity(spec) not in {
            identity(p) for p in latent.get("program_catalog", [])
        }:
            raise ValueError(
                "Decode program is absent from the operator's static catalog"
            )
    if spec.get("policy") == "swireasoning" and "staging" not in latent.get(
        "optimizations", []
    ):
        raise ValueError("SwiReasoning requires captured staging")
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
    build_program(
        spec,
        config.model_config.get_hidden_size(),
        config.model_config.get_vocab_size(),
        constants,
    )
