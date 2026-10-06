# Programmable latent decoding in vLLM

This experimental fork adds per-request, GPU-executed decode transitions to
vLLM v0.31.0 (`db9527a46873454610df6dbedf79a36d6bf1a7f6`). It runs an ordinary
Qwen3 or Llama transformer with paged attention while allowing the next input
to be a sampled-token embedding, a distribution-weighted embedding, a hidden
state, or a validated tensor expression. Different requests can use different
programs in the same continuously scheduled batch.

The implemented API is for bounded research experiments. It returns diagnostic
token sequences, requires an explicit generation budget, and does not reproduce
every policy or the accuracy results of the cited reasoning frameworks.
See [the measured results](results/latent/REPORT.md) for the tested scope.

## The transition contract

The useful generalization is

\[
 (e_{t+1}, s_{t+1}, m_t) = F(z_t, h_t, E[y_t], t, s_t),
 \qquad y_t\sim\operatorname{sample}(z_t).
\]

Here `h` is the final normalized model hidden state at the sampled position,
`z` is the raw logit vector before sampling transformations, and `m` indicates
that the next input is latent. Eight FP32 scalar state registers belong to each
request. The explicit branch uses vLLM's sampler. The soft branch uses FP32
softmax followed by a model-dtype matrix product against the input embedding
table; BF16 rounding means it is a numerical approximation to the real-valued
formula. Hidden states and input embeddings must have the same width.

Each latent position consumes one ordinary causal position and one KV entry.
Attention, RoPE positions, paged KV writes, and transformer layers remain
upstream code. A latent vector does not collapse several *executed* transformer
positions into one cache entry; any reasoning compression must come from the
method and model, not the cache implementation.

## Running a program

Use this checkout with release-matched native libraries. For reproducible GPU
experiments, [job.sh](benchmarks/latent/job.sh) and
[job.yaml](benchmarks/latent/job.yaml) use `vllm/vllm-openai:v0.31.0` and create a
Python overlay containing the changed source files. They do not modify the
stock installation. Adapt the mounted checkout path before using another
account. `kai submit -f benchmarks/latent/job.yaml` submits the GPU job.

For an editable installation, follow upstream's precompiled installation
workflow using `VLLM_USE_PRECOMPILED=1 uv pip install -e .` with artifacts matched
to this release.

```python
from vllm import LLM, SamplingParams
from vllm.v1.latent.program import preset

llm = LLM(
    model="Qwen/Qwen3-8B",
    dtype="bfloat16",
    enable_prompt_embeds=True,
    enable_prefix_caching=False,
    async_scheduling=False,
    max_num_seqs=32,
    additional_config={"latent_decode": {
        "capacity": 64,
        "max_steps": 512,
        "max_programs": 16,
        "cuda_graphs": True,
    }},
)
params = SamplingParams(
    temperature=0,
    max_tokens=128,
    ignore_eos=True,
    detokenize=False,
    extra_args={"decode_program": preset("soft", latent_steps=64)},
)
outputs = llm.generate(["Explain why 17 times 19 equals 323."], params)
```

`preset` also supports `token`, `hidden`, `norm_hidden`, and `entropy`.
The entropy preset is an illustrative threshold policy, **not SwiReasoning**.
Omit `decode_program` for an ordinary request within an enabled engine. Omit
`latent_decode` entirely to retain the upstream token-input path. The extension
selects model runner V1 and rejects an explicitly forced V2 configuration.

### Custom programs

Programs are bounded JSON straight-line DAGs with up to 64 instructions.
Registers are `logits`, `hidden`, `token`, `step`, and `s0` through `s7`.
Instructions have the form `[destination, opcode, operands...]`.

| Opcode | Meaning |
| --- | --- |
| CONST | Finite scalar literal |
| SOFTMAX | FP32 softmax of a vocabulary-width register |
| ENTROPY | Entropy of softmax of a vocabulary-width logit register |
| EXPECT | Vocabulary weights multiplied by input embedding table |
| PROJECT | Register multiplied by a named operator-provided matrix |
| NORM | RMS normalization with epsilon `1e-6`, no learned gain |
| SILU | SiLU activation |
| ADD, MUL, DIV | Broadcast-compatible arithmetic; MIX uses MUL and ADD |
| GT, LT, EQ | Scalar per-row comparisons |
| AND, OR | Boolean predicate composition |
| SELECT | Masked choice between equal-shaped, equal-typed registers |

For example, a stateful projected-hidden transition can use:

```python
program = {
    "ops": [
        ["one", "CONST", 1],
        ["limit", "CONST", 16],
        ["active", "LT", "s0", "limit"],
        ["count", "ADD", "s0", "one"],
        ["projected", "PROJECT", "hidden", "request_projection"],
        ["activated", "SILU", "projected"],
        ["normalized", "NORM", "activated"],
    ],
    "embedding": "normalized",
    "latent": "active",
    "updates": {"s0": "count"},
}
```

The operator registers `request_projection` in a safetensors file through
`latent_decode.constants`. The frontend checks its dimensions without loading
its data; the worker loads it on the GPU. Requests cannot supply Python code,
import modules, or choose filesystem paths. Malformed programs fail validation
before scheduler admission. Numeric nonfinite outputs fall back to the sampled
token embedding; nonfinite state updates retain the previous state.

## Execution and cache correctness

At first admission of a new program, the worker captures CUDA graphs for
power-of-two group sizes. Each step groups rows by their static program identity,
copies tensor inputs into fixed graph buffers, replays each graph, and scatters
the resulting embeddings into request-owned history. Predicates and state
updates stay on the GPU. `cuda_graphs=False` provides an eager debugging backend;
`compile=True` optionally uses fullgraph `torch.compile` in that backend.

The surrounding scheduler still performs normal Python bookkeeping and
CPU-to-GPU metadata staging, and synchronous vLLM still returns sampled IDs to
the host. This implementation does not claim a wholly GPU-resident engine.
Admission graph capture also synchronizes. It removes host inspection of
transition predicates and arbitrary per-token user callbacks.

History is indexed by stable request slot and generated position, never by
temporary batch row. The worker re-injects stored embeddings during decode and
KV recomputation. Intermediate chunked-prefill samples do not advance a program.
The scheduler limits the number of admitted latent requests to `capacity`,
including preempted requests, and queues excess requests. It similarly limits
concurrently active program identities; unused compiled programs can be evicted. Completion or abort
releases the slot. Scalar program state survives preemption and is advanced only
when a new output position is produced.

The embedding-history allocation is
`capacity * max_steps * d_model * dtype_bytes`, plus eight state scalars and one
mask per generated position. With capacity 64, 128 steps, width 4096, and BF16,
embedding history is 64 MiB. Graph buffers and registered matrices consume
additional memory. Graph allocations happen at admission and are **not yet
fully reserved by the scheduler's KV memory budget**; leave GPU headroom and
bound `max_programs`. This is a remaining production-admission limitation.

Token-prefix caching is disabled because placeholder IDs do not identify a
latent trajectory. Correct future sharing needs a cache identity incorporating
the program, constants, model/adapters, randomness, and actual latent prefix,
or an equivalent proven identity. Preemption needs the embedding ledger even
without prefix sharing. These are distinct issues from the correctness of
paged-attention storage itself.

## Framework coverage and remaining interfaces

[A*-Thought-V2](https://arxiv.org/html/2609.07821v1) uses final-layer hidden-state
feedback during latent spans at inference. The hidden-feedback primitive is
implemented; the paper's training procedure, boundary-tag policy, and trained
checkpoints are not reproduced by applying it to stock Qwen3-8B.

[SwiReasoning](https://github.com/sdc17/SwiReasoning/blob/main/generation_utils.py)
uses entropy history, mode residence times, boundary embeddings, and optional
switch-count termination and token exceptions. A soft embedding or instantaneous
entropy threshold alone is not that method. This fork provides several required
primitives, but does not include a complete SwiReasoning adapter.

[LatentMAS](https://arxiv.org/html/2511.20639v1) additionally transfers working
memory between agents. Hidden-state projection can be expressed here, but
cross-request KV inheritance/concatenation is outside this transition interface.
It requires scheduler ownership, position handling, and cache-transfer semantics
of its own, even when agents run synchronously.

The current supported surface is single-GPU Qwen3/Llama text inference without quantization, resumable input,
LoRA, speculative decoding, KV transfer, or prefix caching. Diagnostic outputs
contain ID `0` at latent positions; they are not faithful text or a self-contained
serialization of a latent trace. Do not infer a latent mask by filtering all zero
IDs, since zero may also be sampled explicitly. Stop strings, EOS termination,
logprobs, token penalties, and constrained output are rejected for program
requests. Ordinary requests are still supported within the same engine.

A production API should return a separate emission mask and stop decision,
distinguish generated positions from visible tokens, reserve graph workspace
before KV sizing, and expose a validated registry of larger state tensors.
Tensor parallel execution needs vocabulary-sharded expectation and reduction;
hidden-only programs could also skip the LM head when no predicate needs logits.
Conditional execution or grouped compaction could avoid computing soft branches
for rows that ultimately select text. The present SELECT evaluates both branches.

These limits mean the current restricted language does not support literally
every synchronous framework. It establishes a common transition interface and
a working batching/cache integration for a useful bounded subset.

## Existing implementations and model-size scaling

Embedding feedback within a batched vLLM engine is not itself new. The
[QwenReasoning fork](https://github.com/zhaoc5/vllm/tree/31418258c7d8896c2f9259931cccff938d69cb0c)
implements Soft Thinking, SwiReasoning, and SeLaR, while the
[SwiReasoning/1Cat fork](https://github.com/dg1kjd/vllm-v100-sxm2-qwen3.5-397b/tree/3c21680950c8842b30d48f0a5757d3130101095f)
provides another request-specific feedback controller. The contribution explored
here is a shared, validated transition language with device-resident predicates
and state, together with embedding-history replay, instead of a separate engine
integration for each policy. This is a design distinction, not a claim of
universal expressiveness or better reasoning quality.

The [scaling report](results/scaling/REPORT.md) extends calibration to Qwen3
0.6B, 1.7B, 4B, and 8B, and compares existing implementations with their own
non-latent controls. Its [comparison protocol](results/scaling/COMPARATORS.md)
documents arithmetic, switching-policy, and software-version differences.
