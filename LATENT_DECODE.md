# Programmable latent decoding in vLLM

This experimental fork adds per-request, GPU-executed decode transitions to
vLLM v0.31.0 (`db9527a46873454610df6dbedf79a36d6bf1a7f6`). It runs an ordinary
Qwen3 or Llama transformer with paged attention while allowing the next input
to be a sampled-token embedding, a distribution-weighted embedding, a hidden
state, or a validated tensor expression. Different requests can use different
programs in the same continuously scheduled batch.

The implemented API is for bounded research experiments and requires an explicit
generation budget. Generic tensor programs return diagnostic token sequences;
the complete SwiReasoning adapter preserves sampled/forced tokens and supports
EOS termination. It does not reproduce every cited framework or its published
accuracy. See [the second optimization round](results/round2/REPORT.md),
[the first optimization experiments](results/optimization/REPORT.md), the
[original results](results/latent/REPORT.md), and the
[scaling comparison](results/scaling/REPORT.md) for the tested scope.

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
    enable_prefix_caching=False,
    async_scheduling=False,
    max_num_seqs=32,
    additional_config={"latent_decode": {
        "capacity": 64,
        "max_steps": 512,
        "max_programs": 16,
        "cuda_graphs": True,
        "workspace_bytes": 3 * 1024**3,
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

### Complete SwiReasoning policy

```python
from vllm.v1.latent import REFERENCE_SHAPE_OPTIMIZATIONS
from vllm.v1.latent.swireasoning import swi_preset

# At engine construction, set latent_decode.optimizations to
# list(REFERENCE_SHAPE_OPTIMIZATIONS) for the reference-shape profile.
# Use COMBINED_OPTIMIZATIONS for the faster padded expectation instead.
spec = swi_preset(
    llm.get_tokenizer(), max_tokens=512,
    alpha=1.0, beta=0.7, window=512,
    max_switch_count=2, termination_max_tokens=32,
    expectation="fp32",
)
params = SamplingParams(
    temperature=0, max_tokens=512, ignore_eos=False,
    extra_args={"decode_program": spec},
)
prompt = llm.get_tokenizer().apply_chat_template(
    [{"role": "user", "content": "What is 17 times 19? Explain briefly."}],
    tokenize=False, add_generation_prompt=True,
)
outputs = llm.generate([prompt], params)
```

This bounded built-in policy uses eight GPU state registers for entropy-trend
switching, dwell time, the sampled-end-token lock, convergence and termination
queues, and the answer budget. Anchor blending and math-token exemptions follow
the pinned QwenReasoning implementation. The budget in the spec must equal
`SamplingParams.max_tokens`. `math_ids`, `convergence_ids`, `termination_ids`,
and the four anchor/stop IDs are validated admission constants. A forced stop
ID must be an EOS or a configured `stop_token_ids` entry to terminate the request.
`ignore_eos=True` retains fixed-work diagnostic execution through stop readouts.

The default `fp32` expectation uses a shared, cached FP32 embedding table and
casts its product back to model dtype before FP32 anchor blending. This matches
QwenReasoning's arithmetic order, including double-to-FP32 blend weights.
Different GEMM batch shapes can still change rounding. Preemption preserves
the complete state and embedding history instead of restarting the mode machine.

Explicit experimental expectations are `bf16`, `topk`, `adaptive`, and `lowrank`.
Top-k uses `topk` entries (default 64); adaptive masks that list at cumulative
probability `mass` (default .99) and renormalizes. Its fixed maximum k can retain
less than the requested mass. The captured graph still reserves the maximum-k
workspace. Low-rank expects operator-registered `embedding.left` and
`embedding.right` matrices; `lowrank` can select another registered name prefix.
These change the algorithm. The round-two report evaluates their quality as
well as latency; none is silently substituted for FP32 feedback.

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
| TOPK_EXPECT | Explicit approximation: top-k logits, normalized weights, embedding mixture |
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
An optional program field `"fallback": "zero"` instead replaces a nonfinite
latent vector with a zero vector while retaining the latent mask. This explicit
contract permits skipping the LM head in eligible greedy hidden-only phases.
It is never inferred from a model or silently substituted for token fallback.

## Execution and cache correctness

The default optimization profile is
`prune,fast_input,staging,skip_inactive,share,skip_head,conditional_expect`.
Dependency analysis eliminates unused program inputs. A captured input-selection
kernel reads either the original token embedding or a stored latent vector;
`enable_prompt_embeds` is therefore disabled for this path. At admission the
worker captures gather, transition, and state/history commit for power-of-two
batch sizes. Programs with a common dense expectation can share execution;
programs with distinct projections remain grouped. Stateless phases proven
inactive from their position skip transition execution entirely. Predicates
depending on model values and state updates remain on the GPU.
For eligible stateless predicates, `conditional_expect` uses a GPU conditional
CUDA graph node to skip cuBLAS when every row in the group is inactive. Active
groups retain the ordinary cuBLAS operation. This requires the conditional
graph API in the tested PyTorch 2.13/CUDA 13 stack. On older stacks, explicitly
omit `conditional_expect`; the engine rejects an unsupported requested backend.
`union_gate` combines masks from all selected consumers of a shared expectation.
State-update dependencies and expectation-dependent predicates are handled
conservatively. It requires `share` and `conditional_expect`.
In batch-invariant mode, conditional execution honors vLLM's invariant matmul
backend. Kernel choice and batch composition can still affect long latent
trajectories; the [numerical diagnostics](results/optimization/REPORT.md#correctness-quality-and-a-failure-found-during-testing)
describe both matching checks and remaining cross-implementation differences.
For stricter cross-path diagnostics, combine `VLLM_BATCH_INVARIANT=1` with
`compilation_config={"custom_ops": ["+rms_norm"]}`. Round-two tests isolate a
1.7B discrepancy to compiled normalization/input paths without executing any
latent transition; this explicit normalization choice restores agreement.

Additional opt-in optimizations are:

| Option | Effect |
| --- | --- |
| `async_metadata` | Event-protected pinned buffers permit nonblocking metadata uploads. |
| `parameterize` | Lift up to 16 numeric CONSTs into request parameters; cache graph structure and concrete position proofs across value changes. |
| `tail_sample` | Capture eligible greedy sampling with the transition. |
| `capture_head` | Also capture the LM head for eligible single-policy batches, removing external head/logit staging work. |
| `fuse_state` | Compile the SwiReasoning state update into fused device operations before capture. |
| `compact_expect` | Experimental active-row sorting and conditional GEMM buckets; changes reduction shapes and was slower in the tested workloads. |
| `exact_compact` | Capture a conditional branch for every possible active-row count, reproducing the reference expectation GEMM shape more closely. Extra capture time/memory and runtime overhead; no guarantee of bitwise agreement across engines. |

`vllm.v1.latent.COMBINED_OPTIMIZATIONS` supplies the measured combined profile.
`REFERENCE_SHAPE_OPTIMIZATIONS` adds `exact_compact` for SwiReasoning numerical
compatibility studies. Pass either as `list(...)` in the configuration's
`optimizations` field. The generic default remains unchanged because the new
combination does not uniformly improve already-warm generic programs. Neither
profile substitutes BF16, top-k, adaptive truncation, or low-rank expectations
for the FP32 SwiReasoning default.

Sampling/head capture falls back to the normal sampler for stochastic sampling,
partial transition batches, minimum-token constraints, thinking budgets, or
custom logits processors. No transformer layer or attention kernel is replaced.
The [round-two report](results/round2/REPORT.md) separates individual effects,
combinations, admission costs, and changes in reasoning quality.

Set `latent_decode.optimizations` to an explicit list for ablations. An empty
list selects the original implementation and requires `enable_prompt_embeds=True`.
For eager debugging, also set `cuda_graphs=False`; `compile=True` optionally
uses fullgraph `torch.compile` in that backend. Captured staging uses CUDA graphs
regardless of the legacy `cuda_graphs` setting.

`gated_expect` enables an experimental tensor-core expectation kernel with GPU
row gating. `fused_expect` enables the experimental streaming softmax/expectation
backend. Both can change floating-point reduction results; neither is in the
default profile. `TOPK_EXPECT` is an explicitly different transition algorithm,
not an exact acceleration of dense feedback. Two `PROJECT` operations can
express a supplied low-rank mapping, but the engine does not invent or train
such a mapping.

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
additional memory. `workspace_bytes` subtracts an explicit allowance from KV
cache sizing for dynamic admission; it is an operator budget, not an automatic
proof that arbitrary future programs fit. The FP32 SwiReasoning table alone
costs `vocab_size * d_model * 4` bytes, shared across its programs.

For predictable admission, provide `program_catalog: [spec, ...]` and set
`allow_dynamic_programs=False`. These programs are materialized during model
loading, before KV profiling, so their actual resident allocations are included
in memory sizing. With `parameterize`, new numeric values reuse catalogued
structures. Unknown structures are rejected in the frontend. The catalog's
distinct structures must fit `max_programs`. A dynamic registry remains bounded
by that count and can evict unused programs; graph capture can still stall
unrelated requests when uncataloged structures arrive.

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
entropy threshold alone is not that method. This fork now includes a complete
GPU adapter matching the pinned QwenReasoning state-machine contract, plus
matched-policy latency tests and held-out quality measurements. This establishes
implementation compatibility, not reproduction of the paper's trained-model
accuracy or equivalence under every floating-point kernel choice.

[LatentMAS](https://arxiv.org/html/2511.20639v1) additionally transfers working
memory between agents. Hidden-state projection can be expressed here, but
cross-request KV inheritance/concatenation is outside this transition interface.
It requires scheduler ownership, position handling, and cache-transfer semantics
of its own, even when agents run synchronously.

The current supported surface is single-GPU Qwen3/Llama text inference without quantization, resumable input,
LoRA, speculative decoding, KV transfer, or prefix caching. Generic diagnostic outputs
contain ID `0` at latent positions; they are not faithful text or a self-contained
serialization of a latent trace. Do not infer a latent mask by filtering all zero
IDs, since zero may also be sampled explicitly. Generic programs require a fixed
budget; SwiReasoning also supports normal EOS/token-stop handling and decoded
text. Stop strings, logprobs, token penalties, thinking-budget overrides, and
constrained output remain rejected for program requests. Ordinary requests are
still supported within the same engine.

A production API should return a separate emission mask and stop decision,
distinguish generated positions from visible tokens, enforce resource bounds
for open-ended dynamic program admission, and expose a validated registry of
larger state tensors.
Tensor parallel execution needs vocabulary-sharded expectation and reduction.
The default SELECT still evaluates both branches when its predicate depends on
model values. Static inactive-phase skipping and the experimental gated kernel
cover specific cases; arbitrary conditional DAG execution remains future work.

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
