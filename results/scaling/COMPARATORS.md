# Existing implementations and comparison protocol

Survey and measurements started 2026-10-06. This is a targeted search, not an
exhaustive census of latent-reasoning engines.

## Native vLLM implementations selected for measurement

| Implementation | Pinned source | Mechanism | Relevant differences |
| --- | --- | --- | --- |
| QwenReasoning's vLLM | [zhaoc5/vllm, 3141825](https://github.com/zhaoc5/vllm/tree/31418258c7d8896c2f9259931cccff938d69cb0c) | Soft Thinking, SwiReasoning, SeLaR inside the V1 sampler and runner | Soft Thinking uses a normalized top-k mixture. SwiReasoning uses a full-vocabulary FP32 mixture and a Python state machine. Supports embedding-entry CUDA graphs. |
| SwiReasoning / 1Cat-vLLM | [dg1kjd/vllm-v100-sxm2-qwen3.5-397b, 3c21680](https://github.com/dg1kjd/vllm-v100-sxm2-qwen3.5-397b/tree/3c21680950c8842b30d48f0a5757d3130101095f) | Per-request SwiReasoning controller, embedding feedback, ordinary paged attention | FP32 vocabulary-chunked mixture; CPU scalar decisions. For dense Qwen3, pending embedding injections force eager model execution. The repository is marked unmaintained. |
| This fork | [045a0a1](https://github.com/edonson2016/vllm-latent/tree/045a0a1ac45f69a92d3399c776210048da29d799) | Validated static programs, GPU state/predicates, captured transition graphs | BF16 mixture operands, FP32 softmax; persistent embedding history for preemption replay. Current presets are mechanisms, not complete reproductions of those papers. |

The [QwenReasoning harness](https://github.com/zhaoc5/QwenReasoning) pins the same
vLLM commit used here. Its two engine commits follow upstream
`f4b161d7fca438bfe29509984759be1943a5aa88`; that exact upstream wheel supplies
the native libraries. The SwiReasoning fork documents using 1Cat-vLLM 1.2.1's
wheel with its Python source tree; this experiment follows that arrangement.
The overlay copies the entire fork Python package over the matching installed
wheel, preserving its compiled extensions. No third-party transition code is
ported into our engine or changed for the timed runs.

## Other relevant work found

- The [original SwiReasoning implementation](https://github.com/sdc17/SwiReasoning)
  provides the method reference; the two repositories above provide vLLM
  implementations suitable for this comparison.
- [Monet](https://github.com/NOVAglow646/Monet) replaces vLLM's GPU runner for
  latent visual reasoning with a customized Qwen2.5-VL model. Its model and
  multimodal workload do not match this Qwen3 text-only calibration.
- [Adaptive Latent Agentic Reasoning](https://github.com/luka-group/adaptive-latent-agentic-reasoning)
  provides a projector-driven vLLM plugin with latent-block delimiters. It
  requires its trained projector/model setup rather than unmodified Qwen3.
- [DS4 reasoning addon](https://github.com/nickmitchko/ds4-reasoning-addon)
  uses a DeepSeek-V4-specific vLLM fork and learned latent head. Its documented
  serving setup requires at least 192 GiB VRAM and SM120 hardware, outside this
  single-A6000 calibration.
- [Soft-Thinking](https://github.com/UCSB-AI/Soft-Thinking) and
  [fast-subconscious](https://github.com/dibbla/fast-subconscious) are related
  SGLang implementations, not vLLM forks.

## What the measurements compare

All timed runs use the same pinned Qwen3 checkpoints, BF16 weights, arithmetic
prompts, 128 generated positions, batches 1/8/32, one warmup and five measured
runs. Timing includes tokenization, admission, prefill, and generation, and
excludes loading, initial compilation, and warmup. It is end-to-end fixed-work
latency, not an isolated inter-token-latency measurement. EOS is ignored;
detokenization, prefix caching, and asynchronous scheduling are disabled.
All engines explicitly select the V1 model runner, where these integrations
live. This is not a comparison against the latest default V2 runner or every
possible serving optimization.
Each engine runs alone on its assigned A6000. Different physical GPUs and
software versions are recorded in `provenance.json` and package inventories.

Each external fork gets an installed-base-wheel control and a fork-disabled
control. Its latent overhead is relative to its own fork-disabled control.
Comparing this overhead is more informative than assigning every raw latency
difference to the transition implementation.

The SwiReasoning/1Cat fork's default dense-Qwen3 configuration failed when its
first feedback embedding changed the compiled model's input signature
(`AttributeError: 'NoneType' object has no attribute 'size'`). The retry enables
`enable_prompt_embeds=True` from initialization. This changes configuration, not
the fork's source. The original attempt is retained in
`swir-token-input-failure.log` and `swir-initial-controls.jsonl`.
This fork targets a TP8 V100/Qwen3.5 hybrid-model deployment. Its multimodal
embedding-entry path can retain CUDA-graph replay; the dense-Qwen3 path tested
here still selects eager dispatch. These measurements do not characterize
performance on its original target deployment.

The programmable soft and hidden presets select latent feedback for 64 of the
128 positions, then select ordinary tokens. The soft graph still evaluates
its mixture every step, including when its latent predicate is false. Mixed
batches cycle token/soft/hidden/entropy programs; batch 1 is token-only.

QwenReasoning's Soft Thinking uses its native top-10 mixture, entropy threshold
0.01 and patience 256; its natural thinking-end behavior remains enabled.
Both external SwiReasoning runs use alpha 1.0, beta 0.7, dwell window 512,
no switch-count termination, and an empty math-symbol exemption set. This
makes their policy configurations explicit and closely aligned, but does not
make either policy identical to our fixed-prefix preset. A 128-position run
also cannot exercise a return switch requiring 512 positions of dwell.

Consequently, these measurements compare engine and policy execution costs.
They do not establish equal reasoning quality, faithful paper reproduction,
or a speedup for executing an identical SwiReasoning algorithm on our fork.
Token histories are retained for output checks, not interpreted as faithful
text readouts of latent reasoning.

A separate stock-vLLM public-API comparison uses eight positions, batch 1,
three timed repeats, and four soft transitions followed by four token
transitions. It resubmits accumulated prompt embeddings and fetches full
logprobs on soft steps. Automatic prefix caching is enabled for that adapter;
it still incurs request and host-transfer costs. The programmable counterpart
keeps one live request, with CUDA graphs enabled. Both use a maximum concurrency
of one. Independent cached Transformers traces check these short token, soft, and
hidden-feedback paths; Transformers is a correctness reference here, not a
timed competitor.
