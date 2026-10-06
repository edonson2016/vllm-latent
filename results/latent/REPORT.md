# Programmable latent decoding benchmark

The [follow-up scaling report](../scaling/REPORT.md) adds 0.6B, 1.7B, and 4B
models, fresh 8B controls, and measurements of existing latent-reasoning forks.

This fork demonstrates per-request GPU transition programs inside vLLM's
continuous batching and paged-attention execution. On one RTX A6000 with
Qwen3-8B BF16, the initial fixed-work calibration measured approximately
2.4% overhead for the enabled token-only path, 2.3% for hidden feedback,
and 9.0–9.4% for soft feedback. These measurements establish execution costs;
they do not establish reasoning accuracy or reproduce the cited papers.

## Measurement setup

The KAI job `latent-dev-031` ran on one NVIDIA RTX A6000 with 48 GB, with an
8-CPU request and 64 GiB host-memory request. The environment used vLLM 0.31.0,
PyTorch 2.13.0+cu130, BF16 weights, FlashAttention 2, synchronous scheduling,
and enabled transformer CUDA graphs. The upstream source base is
`db9527a46873454610df6dbedf79a36d6bf1a7f6`; the Qwen3-8B checkpoint is
`b968826d9c46dd6066d109eabc6255188de91218`.

The five modified upstream Python modules in the installed release image were
verified byte-for-byte against the pinned Git commit before applying the source
overlay. Native libraries remained identical. The container digest is in
[image-digest.txt](image-digest.txt), and software and source hashes are in
[environment.json](environment.json).

The main experiment uses batches of 1, 8, and 32 arithmetic-reasoning prompts,
128 generated positions, greedy sampling, and five timed repetitions after
warm-up. Each run includes tokenization, prefill, and generation, and excludes
model loading and warm-up. Both engines disable prefix caching; maximum context
and token-batch size are 2048, and GPU memory utilization is 0.85. Requests have
equal output budgets within each timing batch. Soft and hidden presets select
latent inputs for the first 64 transitions and ordinary token embeddings
thereafter. EOS is ignored to hold executed work constant.

Runs execute sequentially on the same allocated GPU. This is a controlled
single-device calibration, not a randomized multi-machine statistical study or
a serving tail-latency benchmark. Raw repetitions and token IDs are preserved
in [native.jsonl](native.jsonl); [summary.csv](summary.csv) includes ranges and
throughput. Throughput counts generated positions, including latent positions,
rather than visible text tokens. No claim about answer quality follows from
these fixed-budget runs.

## Main latency measurements

Each cell is median end-to-end seconds, with overhead relative to stock vLLM
at the same batch size. The generated [table](TABLE.md) and
[chart](latency.svg) contain the measurements.

| Mode | Batch 1 seconds | Batch 8 seconds | Batch 32 seconds |
| --- | --- | --- | --- |
| Stock vLLM | 3.1603 (+0.00%) | 3.2608 (+0.00%) | 3.6935 (+0.00%) |
| Fork disabled | 3.1601 (-0.01%) | 3.2605 (-0.01%) | 3.6944 (+0.02%) |
| Stock embedding input | 3.1805 (+0.64%) | 3.2882 (+0.84%) | 3.7237 (+0.82%) |
| Token program | 3.2367 (+2.42%) | 3.3441 (+2.55%) | 3.7820 (+2.40%) |
| Soft feedback | 3.4559 (+9.36%) | 3.5604 (+9.19%) | 4.0258 (+8.99%) |
| Hidden feedback | 3.2342 (+2.34%) | 3.3386 (+2.38%) | 3.7778 (+2.28%) |
| Normalized hidden | 3.2367 (+2.42%) | 3.3469 (+2.64%) | 3.7814 (+2.38%) |
| Entropy threshold 2.0 | 3.4675 (+9.72%) | 3.5845 (+9.93%) | 4.0768 (+10.38%) |
| Mixed programs | 3.2363 (+2.41%) | 3.9011 (+19.64%) | 4.3506 (+17.79%) |

The disabled fork produced the same token hashes as stock for all repetitions
and batch sizes in the main calibration. The enabled token-only path matches the stock prompt-embedding control's
output hashes at every tested batch size. That upstream control costs
0.64–0.84% relative to ordinary stock decoding; the remaining program-layer
cost is approximately 1.6–1.8% relative to the embedding control. The first
batch-1 divergence from the ordinary token-input path occurs at generated
position 87 (zero-based) in both engines, consistent with the alternate
numerical execution path rather than a transition-specific error.

Soft feedback adds a full-vocabulary expectation against the input embedding
table. For this model, a BF16 table of 151936 by 4096 is approximately 1.16 GiB;
the matrix product costs approximately 1.245 billion multiply/add FLOPs per
row. This makes memory traffic a plausible contributor to the observed cost,
but the end-to-end measurements alone do not identify a hardware bottleneck.

As a rough break-even calculation, 9% extra cost per position requires more
than `1 - 1/1.09`, or 8.3%, fewer generated positions to reduce generation time,
assuming unchanged batching and answer quality and ignoring shared prefill
cost. The experiment does not establish that a particular latent method
achieves this reduction at comparable quality.

The mixed case uses token, soft, hidden, and entropy programs in equal
proportions at batches 8 and 32. Its higher overhead exposes the cost of
separate program groups, including two separate embedding expectations.
At batch 1, this case contains only the token program and is not a mixed batch.

The entropy threshold of 2.0 produced the same diagnostic outputs as the token
program on these prompts. That row measures the cost of computing and rejecting
the soft branch; it is not evidence of adaptive latent switching. SELECT is
implemented as a mask over eagerly evaluated branches, so the soft expectation
continues to execute after the finite latent prefix too.

## Stock latent adapter comparison

Stock vLLM does not expose a public API for feeding a new embedding into the
next step of an existing live decode request. The comparison adapter uses
stock prompt embeddings, requests full-vocabulary logprobs, computes the soft
embedding externally, and resubmits the accumulated embedding prefix. This
preserves the intended mathematical trajectory but recomputes the prefix and
crosses the host/device boundary. It is a public-API emulation baseline, not a
native cached latent implementation.

For one request and eight generated positions, four latent followed by four
text positions, the stock adapter with bulk logprob conversion and no prefix
cache measured **1.4067 seconds** median over three repetitions, versus
**0.2201 seconds** for the fork: **6.39 times faster**. Both produced
`[0, 0, 0, 0, 773, 358, 1184, 311]` in every repetition. ID zero is the fork's
diagnostic latent marker here. The stock adapter's non-latent replay control
measured **0.2803 seconds**, versus **0.2064 seconds** for the fork's token
program. Results are in [replay-soft.jsonl](replay-soft.jsonl),
[replay-token.jsonl](replay-token.jsonl), and [short.jsonl](short.jsonl).

Stock vLLM 0.31.0 also hashes prompt-embedding blocks for prefix caching.
Enabling that cache in the adapter measured **1.4149 seconds** median, with
identical outputs, in [replay-soft-apc.jsonl](replay-soft-apc.jsonl). Its range
overlaps the no-cache adapter; for this short workload the data do not establish
a useful latency difference from prefix caching. The fork is approximately
**6.4 times faster** than either tested adapter. Prefix-cache reuse in this
resubmission baseline does not provide a native live-request transition API.

The adapter uses a bulk NumPy conversion of logprob values and requests
full-vocabulary logprobs only at latent steps. The earlier scalar-assignment
implementation measured 3.9147 seconds; it is preserved in
[replay-soft-initial.jsonl](replay-soft-initial.jsonl) for provenance, but its
17.79-fold ratio is superseded by the improved baseline.

The remaining ratio includes unavoidable costs of this chosen public-API
adapter plus any implementation inefficiencies: logprob materialization,
transfers, request submission, and prefix handling. It is not an improvement
factor over an optimized native latent engine, or a reasoning-efficiency gain
from the latent method itself. The adapter reserves another embedding table
outside the engine and uses GPU utilization 0.80; neither short workload
exhausts KV space.

## Correctness and limitations

Fourteen semantic tests pass on the GPU worker. They cover CPU/GPU soft
feedback, budget masks, isolated state updates, projection and activation,
numeric fallback, invalid programs, fullgraph compilation, CUDA graph replay
with padded batches, request reordering, partial prefill, request cleanup, and
program-cache reuse. The repository's complete applicable pre-commit suite
passes, including mypy, formatting, shell checks, and CI test registration.

The eight-position token, soft, and hidden trajectories match an independent
Transformers model using a conventional KV cache. The stock soft adapter also
matches. These are small deterministic numerical checks, not task-accuracy
evaluations. See [hf-reference.json](hf-reference.json) and
[tests-final.log](tests-final.log).

The mixed-policy lifecycle test uses 12 requests with different prompt lengths
and 32/48/64-position output budgets, chunked prefill, and a latent capacity of
8. Reducing the KV cache to 32 blocks (512 positions) triggered four preemptions.
Under ordinary BF16 kernels, the initial small-cache run matched 11 of 12
requests, while the later counted run matched 8 of 12. Thus exact text is not
stable across these ordinary scheduling regimes; this prototype should not
promise bitwise schedule invariance in that mode. With
`VLLM_BATCH_INVARIANT=1`, **all 12 matched exactly**, with four preemptions
in the small-cache run and zero in the large-cache control. The result supports
correct request-owned embedding replay while demonstrating the need to account
for batch-dependent numerical trajectories. See [invariant-32.json](invariant-32.json),
[invariant-11000.json](invariant-11000.json), and
[preemption-counted.json](preemption-counted.json).

An entropy threshold of 0.5 exercised actual adaptive selection: eight requests
used `[8, 7, 9, 6, 9, 8, 7, 7]` latent positions, 61 out of 1024 generated
positions. Median latency was 3.5724 seconds for batch 8. Counts were read from
GPU masks after timing through a named diagnostic RPC; there was no per-token
host predicate inspection. See [adaptive.jsonl](adaptive.jsonl).

CUDA-event microbenchmarks at model dimensions, with synthetic BF16 embeddings,
measured graph replay alone at 0.019–0.026 ms for token programs,
0.022–0.028 ms for hidden feedback, 0.033–0.046 ms for normalized hidden,
1.838–2.027 ms for soft expectation, and 1.899–2.385 ms for entropy plus
expectation across batches 1/8/32. They exclude input staging, model execution,
and scheduling, and must not be substituted for end-to-end latency.
[transition-cost.json](transition-cost.json) contains those measurements.

A final three-repetition control sweep after the admission and diagnostic
changes measured disabled-fork overhead of 0.03–0.21% and token-program overhead
of 2.53–2.88%. Token outputs remained identical to their corresponding stock
paths. These later controls, in [final-controls.jsonl](final-controls.jsonl),
show that the small overhead estimates should be read at approximately the
few-tenths-of-a-percent precision supported by these runs.

The implemented contract covers a bounded class of synchronous, single-chain
methods expressible as static tensor programs over logits, final hidden states,
token embeddings, and eight scalar state registers. Synchronous execution
alone does not make an arbitrary framework expressible: cross-agent memory
transfer, tree search, arbitrary inner loops, gradients, and additional model
calls require more engine interfaces.

The [design and API guide](../../LATENT_DECODE.md) explains the mapping to
A*-Thought-V2, SwiReasoning, and LatentMAS, and the missing framework-specific
policies. This release is a tested research prototype, not a complete
reproduction of those frameworks. It requires fixed-budget diagnostic output
and does not provide a production text emission/stop API. It rejects unsupported
parallel, speculative, quantized, and cache-transfer modes. GPU history is
bounded and scheduler-managed; admission-time CUDA graph workspace is not yet
fully integrated into KV memory reservation.

The next performance changes supported by these measurements are to merge
compatible expectation operations across program groups, avoid computing
unselected expensive branches when profitable, and bypass the LM head for
hidden-only transitions whose policies do not need logits. A production
extension should also add explicit emission masks and stop predicates, graph
workspace admission budgets, and program-aware cache identities before enabling
prefix reuse or distributed execution.

## Reproduction

The [GPU job](../../benchmarks/latent/job.yaml) runs
[run_suite.sh](../../benchmarks/latent/run_suite.sh) and
[verify_suite.sh](../../benchmarks/latent/verify_suite.sh) through
[job.sh](../../benchmarks/latent/job.sh). Adjust mounted paths for another
account. The scripts use a release-matched overlay so stock and fork share the
same native libraries. Run `benchmarks/latent/summarize.py` with Matplotlib
installed to regenerate the CSV, table, and SVG from raw records.
