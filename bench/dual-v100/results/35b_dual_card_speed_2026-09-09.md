# Qwen3.6-35B-A3B dual-V100 speed work — 2026-09-09

## Common measurement

- Host: titan, two Tesla V100-SXM2-16GB cards (GPU 1 and GPU 2).
- NInfer 35B artifact, int8 KV, MTP enabled with three draft tokens.
- `CTX=208896`, `PREFILL_CHUNK=2048`, one request, greedy 128-token completion.
- Deterministic filler requested at 196608 tokens; server counted 199982 prompt tokens.

## Expert split

| GPU 1 experts | GPU 2 experts | prefill tok/s | decode tok/s | accepted tokens/round |
|---:|---:|---:|---:|---:|
| 128 | 128 | **1185.4** | 77.8 | not retained in this run's terse CSV |
| 112 | 144 | 1165.7 | **78.4** | 3.10 |

The 112/144 split fits at the common 208896-token MTP context, but trades 1.7% prefill for a
0.8% decode change. That is within normal one-run variation and does not justify replacing the
balanced 128/128 production default.

## Draft width

| draft tokens | prefill tok/s | decode tok/s | accepted tokens/round | acceptance |
|---:|---:|---:|---:|---:|
| 2 | 1104.4 | 68.7 | 2.42 | 71.2% |
| 3 | 1185.4 | **77.8** | not retained in this run's terse CSV | — |

Two drafts accepted a larger fraction of proposals, but accepted fewer tokens each round and was
11.7% slower for continuing generation. Keep three draft tokens for the 35B launcher. The prefill
figures were not interleaved and vary between cold model loads, so do not use their difference to
choose a draft width.

## Rejected candidate: local secondary router replicas

The secondary expert shard recomputes routing and reads each layer's 257x2048 BF16 router from GPU
1 over NVLink. A local GPU-2 replica of these routers (about 65 MiB across text layers plus MTP)
loaded successfully at the common context, but did not improve decode: it measured 1101.4 prefill
tok/s and 77.8 decode tok/s. The balanced baseline was 1185.4 / 77.8, so the implementation was
removed rather than adding permanent VRAM and materializer complexity.

## 255K target baseline and attention decision

- Configuration: `CTX=255000`, `PREFILL_CHUNK=2048`, int8 KV, MTP with three drafts, 128/128
  expert split, one greedy 128-token completion.
- The largest request below used 245760 requested filler tokens and was counted as 250230 prompt
  tokens by the server. It completed successfully, leaving room for all 128 output tokens. This is
  the retained long-context target; no context reduction is needed for the fast prefill chunk.

| requested depth | counted prompt tokens | prefill tok/s | decode tok/s | MTP tokens/round |
|---:|---:|---:|---:|---:|
| 65536 | 66109 | 2044.6 | 122.8 | 3.26 |
| 131072 | 132926 | 1478.7 | 85.3 | 2.86 |
| 196608 | 199982 | 1110.4 | 75.7 | 3.02 |
| 245760 | 250230 | 900.3 | 65.7 | 2.95 |

An instrumented 200K run measured attention at 119.259 seconds, GDN at 27.660 seconds, and MoE
at 40.035 seconds across the decode requests. The timing boundaries serialize work, so the absolute
numbers are diagnostic only; attention is still 63.8% of these stage totals. This justifies a gated
tensor-parallel full-attention prototype.

The 255K baseline residency is 16123 MiB on GPU 1 and 10749 MiB on GPU 2. GPU 1 therefore has only
261 MiB free before the prototype, while GPU 2 has 5635 MiB. A tensor-parallel design that halves
the primary text KV heads and puts the other half on GPU 2 is credible: it releases primary text-KV
space before adding its secondary pool. It must be tested at 255K before any speed claim.

### Required 35B tensor-parallel implementation

The existing 27B TP attention path cannot be enabled directly. Its model has two separately bound
attention projections, whereas 35B stores Q, K, gate, and V in one fused 9216x2048 W8 matrix. The
next implementation should add a four-band head-aware W8 materializer split for
`[Q4096|K512|gate4096|V512]`, producing each card's contiguous
`[Q2048|K256|gate2048|V256]` shard. The runtime then needs a fused-shard projection/unpack path,
the existing two-card local-KV attention schedule, and the existing unsharded output projection.
MTP attention remains unsharded for this phase. Required gates are short-output equality, 255K load
and completion fit, then a 200K/deep decode comparison. Retain it only if it improves deep decode
by at least 10% or materially improves end-to-end time.

## 35B text-attention tensor-parallel implementation boundary (2026-09-09)

A build-checked loader primitive now exists for the 35B fused W8 attention parent: `RowSplitShardAxis::QKGateVHeadHalf`. It copies the physical `[Q(4096) | K(512) | gate(4096) | V(512)]` rows directly from the artifact into compact per-card `[Q/2 | K/2 | gate/2 | V/2]` W8 shards (4608 rows/card). This avoids transient full-weight residency and is the required loading half of 35B attention TP.

It is intentionally not enabled by `NINFER_TP_ATTENTION` yet. The generic attention runtime is shared with 27B, whose TP projection is a distinct two-parent Q4/Q5 operation. Replacing that helper with a fused-W8 linear projection would regress the 27B production TP path. The remaining safe unit is a separate 35B runtime branch (or a projection-traits adapter): invoke the local W8 linear kernel for each 4608-row shard, unpack the four bands, then reuse the existing two-card local-KV attention, peer gather, and primary output projection. MTP attention stays whole.

This preserves the 35B baseline and 27B TP behavior while making the planned 255K fit possible: primary text KV is halved, with the matching half allocated on GPU 2. The acceptance gates remain exact short greedy output equality, a 255K MTP load plus completion, and a 200K depth comparison requiring at least a 10% deep-decode gain or material end-to-end gain.

## Validation and implementation review — 2026-09-09 (continued)

### Retained baseline check

A clean post-loader run at `CTX=255000`, `PREFILL_CHUNK=2048`, int8 KV, MTP draft width 3, and 128/128 MoE completed a 199982-token prompt plus 128 tokens:

| counted prompt | prefill tok/s | decode tok/s | MTP tokens/round | acceptance |
|---:|---:|---:|---:|---:|
| 199982 | 1197.1 | 77.8 | 3.10 | 69.9% |

This is consistent with, and slightly above, the retained one-run 255K baseline (1110.4 / 75.7). The difference is normal run-to-run variation; no speed claim is assigned to the loader changes because TP was disabled in this run.

### Direct review findings and follow-up

1. **Four-band shard loader:** validated by the real TP load: the 20.43-GiB artifact loaded into the two-card compact layout and the server reached warm-up. The W8 parent geometry check is `[Q=4096 | K=512 | gate=4096 | V=512]`, yielding two 4608-row shards.
2. **Target isolation:** added `Variant::fused_graph_attention`, keeping 27B's Q4/Q5 two-parent projection path intact while the 35B fused-W8 branch uses a local W8 linear plus four-band unpack. Both targets compile in the same `ninfer-serve` build.
3. **Missing W8 profile:** the first TP warm-up failed safely because generic W8 linear dispatch did not accept `[4608,2048]`. Added a compact-shard dispatch profile; Volta routes wide prefill widths through the existing safe fallback.
4. **Blocking kernel prerequisite:** the second TP warm-up reached causal attention and failed safely with `causal_softmax_attention: unsupported head geometry`. Existing cached-attention routes instantiate only 24q/4kv, 16q/2kv, and 12q/2kv. A 35B 50/50 split needs **8q/1kv**. Do not fake this with duplicated heads or replicated KV: that would change semantics or defeat the 255K memory objective.

The next substantive implementation is an 8q/1kv cached causal-attention geometry through all required paths: validation, small-T/split capacity, prompt dispatch, and the Volta flash gather/launch specialization. Only after short greedy equality passes should the 255K fit and 200K speed gates be run.

### 8q/1kv prototype result — rejected pending correctness investigation

The native 8q/1kv extension compiled and the TP server completed its short real-request smoke test (`CTX=8192`, chunk 2048, MTP width 3, 1980 counted prompt tokens, 16 generated tokens): **2694.1 prefill tok/s and 127.5 decode tok/s**. A matched baseline run was **2585.9 prefill tok/s and 157.4 decode tok/s**.

The required greedy-equality gate failed. The completions have distinct SHA-256 values:

- TP: `0a9f5cc33e5e531194aae695b4b3150d6d1e8b5344f251056fc0897bebd63acd`
- baseline: `b7394f73e3d006c13783e9fe59f08af195b7746b53accb1609e8d58a60d00c1c`

Therefore this prototype is rejected and was **not** tried at 255K or 200K. Its shallow decode was also 19.0% slower, so it has no performance case while incorrect. The next debugging unit is numerical isolation of one full-attention layer: compare the compact-shard W8 Q/K/gate/V output and the 8q/1kv cached-attention output against the corresponding row/head slices of the single-card computation before re-running a completion test.

## TP correction and production-width attention oracle — 2026-09-09 (late)

An independent review found that the initial W8 DP4A prefill experiment had entered APIs declared
`A16Only`. DP4A quantizes BF16 activations to int8, so that was an invalid comparison: the
single-card baseline could use A8 arithmetic while the TP generic W8 projection remained A16.
DP4A is now restricted to `AllowA8` callers. The 35B W8 attention-input and residual-add APIs are
A16-only and no longer reserve or use DP4A scratch.

The resulting strict-A16 GPU regressions pass without the `NINFER_W8_NO_DP4A` override:

- `ninfer_linear_w8_a16_test`
- `ninfer_linear_add_w8_a16_test`
- `ninfer_attn_input_proj_test`

The 8q/1kv causal oracle was extended from short prompt widths to the actual 2048-token 35B
prefill chunk, for both BF16 and int8 KV. It exposed and fixed a separate workspace contract bug:
the cached-only overload used the native prompt kernel but the shared capacity query reserved
Volta-flash staging. The cached path now reserves the advertised layout. The full test passes:
`causal_softmax_attention`, `packed_softmax_attention`, and `context_softmax_attention` all PASS.

A matched strict-A16, MTP-off smoke at 1980 prompt tokens / 16 generated tokens measured:

| mode | prefill tok/s | decode tok/s |
|---|---:|---:|
| compact 8q/1kv TP | 2105.9 | 108.9 |
| single-card attention baseline | 1742.0 | 119.4 |

The TP completion still diverges after the shared opening token. The production-width attention
oracle now passes, so this is no longer attributable to KV1 append or 8q/1kv attention. Do not
claim the smoke timing as a final speed result yet: isolate the fused 9216-row attention-input
kernel against the compact 4608-row generic W8 projection, including decode T=1, and compare
layer/logit deltas before rerunning the 255K and 200K gates.

### Compact W8 projection isolation

`ninfer_attn_input_proj_test` now builds the same rank-0 compact payload as the artifact loader and
compares every generic `[Q2048|K256|gate2048|V256]` W8 result against the corresponding bands of
the fused 9216-row parent projection. The compact layout is sound and the decode projection is
bit-identical. At prefill T=2048, 2034 of 9,437,184 BF16 elements differ (0.0215%), with a maximum
absolute delta of 0.25; the complete A16 comparison remains within its qualified bound.

This explains why exact greedy-text equality is too strong as the sole TP criterion: the two legal
prefill GEMM schedules introduce sparse BF16 rounding differences that can amplify through later
layers, while T=1 projection is exact. The remaining model-level validation should compare scored
token log-probabilities or layer deltas against a tolerance, then decide whether a dedicated
compact fused projection is justified for reproducible greedy output rather than treating any later
token difference as evidence of an indexing error.

## Retained 35B TP attention result — 2026-09-10

The tensor-parallel text-attention path now clears the required long-context residency and paired
speed gates with the retained production configuration: `CTX=255000`, `PREFILL_CHUNK=2048`, int8
KV, MTP draft width 3, and 128/128 expert split. No context reduction was required.

The 255K residency request counted 250230 prompt tokens and completed 16 MTP tokens successfully:

| counted prompt | prefill tok/s | decode tok/s | MTP tokens/round |
|---:|---:|---:|---:|
| 250230 | 1397.8 | 101.8 | 3.75 |

The paired 200K comparison used the same `CTX=255000` allocation, 2048-token chunk, 64-token
greedy completion, and fresh loads. Both generated-text files are byte-identical (SHA-256
`5358b8da173ebf00720638f2194ebcf22b6bf321c6d800f4bcd5ef8c1ada5a3c`) and have the same MTP
acceptance (2.86 tokens/round, 64.1%).

| mode | counted prompt | prefill tok/s | decode tok/s | projected 2000-token E2E |
|---|---:|---:|---:|---:|
| TP attention | 199982 | 1441.6 | 82.2 | 160.7 s |
| prior attention placement | 199982 | 858.7 | 71.2 | 257.0 s |
| change | — | **+67.9%** | **+15.4%** | **-37.5%** |

This exceeds the retained criterion of at least 10% deep-decode improvement and materially improves
end-to-end time. Retain TP attention for the 35B dual-V100 production configuration. Raw outputs:
`bench/moe-ep-p3/compare-out/tp-200k-mtp-r1-20260910` and
`bench/moe-ep-p3/compare-out/base-200k-mtp-r1-20260910`.

### llama.cpp at its maximum MTP-fitting context

The matched 35B llama.cpp MTP configuration fit only to `CTX=208896`; its deepest measured request
was 196608 tokens. At that depth it recorded 579.6 prefill tok/s and 68.6 decode tok/s. The TP
ninfer run, allocated for the larger 255K context and measured at 199982 tokens, records 1441.6
prefill tok/s and 82.2 decode tok/s: 2.49x prefill and 1.20x decode. Normalizing both to a
196608-token prompt plus a 2000-token completion gives 160.7 s for ninfer and 368.4 s for llama,
or 2.29x faster end-to-end. The llama raw sweep is
`bench/moe-ep-p3/compare-out/ctx208896-c2048-thorough/llama.csv`.
