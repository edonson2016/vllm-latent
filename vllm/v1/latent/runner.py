# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker integration and request-owned replay buffers for latent transitions."""

import json
from typing import Any

import numpy as np
import torch

from vllm.v1.latent.program import Program


class TransitionRunner:
    def __init__(self, runner, config):
        self.runner = runner
        cfg = runner.vllm_config
        if (
            cfg.parallel_config.tensor_parallel_size != 1
            or cfg.parallel_config.pipeline_parallel_size != 1
            or cfg.parallel_config.decode_context_parallel_size != 1
            or cfg.parallel_config.data_parallel_size != 1
            or cfg.scheduler_config.async_scheduling
            or cfg.speculative_config is not None
            or cfg.cache_config.enable_prefix_caching
            or cfg.lora_config is not None
            or cfg.model_config.is_multimodal_model
            or cfg.model_config.is_encoder_decoder
            or cfg.model_config.quantization is not None
            or cfg.kv_transfer_config is not None
        ):
            raise ValueError(
                "Latent decode requires single-GPU synchronous text "
                "decoding, no prefix cache, speculation, LoRA, or KV transfer"
            )
        if not runner.enable_prompt_embeds:
            raise ValueError("Latent decode requires enable_prompt_embeds=True")
        if cfg.model_config.hf_config.model_type not in {"qwen3", "llama"}:
            raise ValueError("Latent decode currently supports Qwen3 and Llama only")
        self.capacity = int(config.get("capacity", runner.max_num_reqs * 2))
        self.max_steps = int(config.get("max_steps", 512))
        if not 1 <= self.capacity <= 1024 or not 1 <= self.max_steps <= 32768:
            raise ValueError("Invalid latent storage budget")
        d = cfg.model_config.get_hidden_size()
        self.history = torch.zeros(
            (self.capacity, self.max_steps, d), device=runner.device, dtype=runner.dtype
        )
        self.state = torch.zeros((self.capacity, 8), device=runner.device)
        self.masks = torch.zeros(
            (self.capacity, self.max_steps), device=runner.device, dtype=torch.bool
        )
        self.free = list(range(self.capacity - 1, -1, -1))
        self.requests = {}
        self.programs: dict[str, Any] = {}
        self.constants = {}
        self.compile = config.get("compile", False)
        self.use_graphs = config.get("cuda_graphs", True)
        self.max_programs = int(config.get("max_programs", 16))
        if not 1 <= self.max_programs <= 64:
            raise ValueError("max_programs must be between 1 and 64")
        # Only the server operator may configure projection files.
        if config.get("constants"):
            from safetensors.torch import load_file

            self.constants = load_file(config["constants"], device=str(runner.device))
        self.embedding = None

    def admit(self, request):
        params = request.sampling_params
        spec = (params.extra_args or {}).get("decode_program") if params else None
        if spec is None:
            return
        from vllm.v1.latent.validation import validate_params

        validate_params(params, self.max_steps, request.prompt_embeds)
        key = json.dumps(spec, sort_keys=True, allow_nan=False)
        if len(key) > 16384:
            raise ValueError("Decode program exceeds 16 KiB admission limit")
        if key not in self.programs:
            if len(self.programs) >= self.max_programs:
                active = {entry[1] for entry in self.requests.values()}
                unused = next((k for k in self.programs if k not in active), None)
                if unused is None:
                    raise ValueError("Worker's active program registry is full")
                del self.programs[unused]
            program = Program(
                spec,
                self.history.shape[-1],
                self.runner.input_batch.vocab_size,
                self.constants,
            )
            if self.embedding is None:
                self.embedding = self.runner.get_model().model.embed_tokens.weight[
                    : self.runner.input_batch.vocab_size
                ]
            if self.use_graphs:
                from vllm.v1.latent.graph import GraphProgram

                self.programs[key] = GraphProgram(
                    program, self.embedding, self.runner.max_num_reqs
                )
            elif self.compile:
                self.programs[key] = torch.compile(program, fullgraph=True)
            else:
                self.programs[key] = program
        if not self.free:
            raise ValueError(
                "Latent request capacity exhausted (includes preempted requests)"
            )
        slot = self.free.pop()
        self.state[slot].zero_()
        self.masks[slot].zero_()
        self.history[slot].zero_()
        self.requests[request.req_id] = (slot, key, len(request.prompt_token_ids))

    def finish(self, ids):
        for req_id in ids:
            if req_id in self.requests:
                slot, _, _ = self.requests.pop(req_id)
                self.free.append(slot)

    def inject(self, scheduler_output, inputs_embeds):
        # This is scheduler metadata, not a device-dependent transition decision.
        r = self.runner
        offsets, slots, steps = [], [], []
        offset = 0
        for req_id in r.input_batch.req_ids:
            count = scheduler_output.num_scheduled_tokens[req_id]
            if req_id in self.requests:
                slot, _, prompt_len = self.requests[req_id]
                req = r.requests[req_id]
                start = req.num_computed_tokens
                for j in range(count):
                    step = start + j - prompt_len
                    if 0 <= step < len(req.output_token_ids):
                        offsets.append(offset + j)
                        slots.append(slot)
                        steps.append(step)
            offset += count
        if offsets:
            indices = torch.tensor(
                np.array([offsets, slots, steps]), device=r.device, dtype=torch.long
            )
            inputs_embeds[indices[0]] = self.history[indices[1], indices[2]]

    def transition(self, scheduler_output, logits, hidden, sampled):
        r = self.runner
        groups: dict[str, list[tuple[int, int, int]]] = {}
        for row, req_id in enumerate(r.input_batch.req_ids):
            if req_id not in self.requests:
                continue
            req = r.requests[req_id]
            end = (
                req.num_computed_tokens + scheduler_output.num_scheduled_tokens[req_id]
            )
            # Ignore intermediate prefill/replay samples discarded by vLLM.
            if end < req.num_tokens:
                continue
            slot, key, _ = self.requests[req_id]
            step = len(req.output_token_ids)
            if step >= self.max_steps:
                raise RuntimeError("Latent transition exceeded reserved history")
            groups.setdefault(key, []).append((row, slot, step))
        if not groups:
            return
        if self.embedding is None:
            self.embedding = r.get_model().model.embed_tokens.weight[
                : r.input_batch.vocab_size
            ]
        for key, entries in groups.items():
            indices = torch.tensor(entries, device=r.device, dtype=torch.long).T
            rows, slots, steps = indices.unbind(0)
            token = r.model.embed_input_ids(sampled[rows, 0].long())
            next_input, mask, state = self.programs[key](
                logits[rows],
                hidden[rows],
                token,
                steps[:, None],
                self.state[slots],
                self.embedding,
            )
            self.history[slots, steps] = next_input
            self.masks[slots, steps] = mask[:, 0]
            self.state[slots] = state
            # ID zero is a diagnostic placeholder, not the latent representation.
            sampled[rows, 0] = torch.where(mask[:, 0], 0, sampled[rows, 0])
