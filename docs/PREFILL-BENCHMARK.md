# Qwen3.8-27B on 2× V100: ninfer (dual-V100 NVLink) vs llama.cpp

Back-to-back throughput on the **same host, same model, same token counts**, small context
through near-full 262K. Prefill (prompt processing) and decode (generation).

- **Host:** titan — 2× Tesla V100-SXM2-16GB (Volta sm_70), NVLink `NV6`, CUDA 12.8, driver 580.
- **Model:** Qwen3.8-27B (dense 27B), same checkpoint family on both engines.
- Token counts matched exactly (llama `-p/-d N` = the `N` ninfer reported for each prompt).

### Exact runtime configuration

Both engines run dual-card across the two V100s (`CUDA_DEVICE_ORDER=PCI_BUS_ID`, cards 1,2 — card 0
is a non-participating RTX A2000). Weights, KV precision, and the **complete** launch command:

**ninfer (this fork)** — `groupwise-int` weights (**~4.7-bit**, artifact `qwen3_8_27b.ninfer`),
**int8 group-64 KV**, tensor-parallel attention, full-262K-capable:

```bash
# env: NINFER_TP_ATTENTION=1  NINFER_MLP_PRIMARY=3072
apps/ninfer /models/Qwen3.8-27B-NInfer/qwen3_8_27b.ninfer --messages <prompt.json> \
  --greedy --max-new 1 --max-context 262144 --kv-capacity 262144 --kv-dtype int8 \
  --devices 1,2 --prefill-chunk 2048
# serve path (production): apps/ninfer-serve <same model> --devices 1,2 --kv-dtype int8 \
#   --max-context 262144 --kv-capacity 262144 --prefill-chunk 2048  (same forward as the CLI)
```

**llama.cpp** — **Q6_K** weights (**6.5-bit**, 21.30 GiB, 27.32 B params), **q5_1 K + q5_1 V** KV,
`-sm layer -ts 1/1` (production layer-split serving config):

```bash
# env: CUDA_DEVICE_ORDER=PCI_BUS_ID  CUDA_VISIBLE_DEVICES=1,2
llama-bench -m models/qwen3.8-27b-GGUF/Qwen3.8-27B-Q6_K.gguf -ngl 99 -sm layer -ts 1/1 \
  -ctk q5_1 -ctv q5_1 -p <N> -n 0 -r 2        # decode: -d <depth> -n 128 ; prefill@221K adds -ub 2048 -b 2048
```

Version pinned: **[llama.cpp `b10453`](https://github.com/ggml-org/llama.cpp/commit/4df29be4f4c3673f428170fda944a5b19f743bb8)**
(commit `4df29be4`, 2026-08-16), built for SM_70. All later comparisons use this exact build.

## ⚠️ Read this before the numbers: the weights are not quant-matched

**ninfer runs ~4.7-bit weights (Q5-class); llama runs Q6_K (6.5-bit).** This matters differently
for the two axes:

- **Prefill is compute-bound** — the GEMM does the same FLOPs regardless of stored precision, so
  the prefill parity below is **not** an artifact of the quant gap. It is the int8/dp4a kernel
  doing real work.
- **Decode is memory-bandwidth-bound** — every token streams the whole weight set. ninfer reads
  **~8.6 GB/token** (sharded ~4.7-bit) vs llama's **~21.3 GB/token** (Q6_K), a ~2.5× gap, plus
  int8 vs q5_1 KV. So a **large part of ninfer's decode win is bought with lower weight precision**,
  and it widens at depth as the KV-bandwidth term grows.
- The apples-to-apples comparison — llama at matched weight precision (`Q4_K_XL` ~4.5-bit) with MTP —
  is now published: see [Matched-weight comparison](#matched-weight-comparison-ninfer-vs-llama-q4-mtp-vs-mtp-262k)
  below. As predicted, prefill stays ~parity and ninfer's decode lead shrinks toward its real size —
  to ~19 % at deep context — while **quality comes out identical** on the multi-turn agentic test.

## Prefill (tok/s)

| prompt tokens | ninfer | llama.cpp | ninfer vs llama |
|--------------:|-------:|----------:|:---------------:|
|           516 | **588.5** |     545.9 |   **+7.8 %**    |
|         3,582 |     813.6 | **828.0** |     −1.7 %      |
|        14,342 | **801.3** |     783.2 |   **+2.3 %**    |
|        58,442 | **656.6** |     640.5 |   **+2.5 %**    |
|       117,892 | **525.5** |     514.6 |   **+2.1 %**    |
|       221,592 |     402.3 | **407.7** |     −1.3 %      |

**Parity within ±3 %** across the range (ninfer ahead through the practical 14K–118K mid-range),
compute-bound and quant-independent. See below for how the gap was closed.

> Prefill tok/s carries ~a few % run-to-run variance (HBM temp / clocks); the numbers above are
> single-stream single-shot and the two engines were measured in separate sessions, so treat
> sub-±3 % gaps as ties. Re-measuring the current ninfer build lands within that noise band of these
> values. The reproducibility pack (`bench/dual-v100/`) has the exact harness.

## Decode (tok/s, generation at context depth)

| context depth | ninfer raw | ninfer +MTP(3) | llama raw | ninfer-raw vs llama |
|--------------:|-----------:|---------------:|----------:|:-------------------:|
|           476 |   **31.9** |       **59.8** |      28.0 |      **+14 %**      |
|        58,402 |   **25.4** |       **47.5** |      19.4 |      **+31 %**      |
|       117,852 |   **20.3** |       **38.7** |      14.8 |      **+37 %**      |

ninfer wins raw decode at every depth, widening with context; MTP (self-speculative, draft-3)
adds ~1.9×. **But this table is not a fair fight on two counts:** (a) llama runs ~2.5× heavier Q6_K
weights, and (b) it compares ninfer **+MTP** against llama **raw** — llama's MTP never fit alongside
the full 262K KV on Q6. The matched-weight, **MTP-vs-MTP** rematch is below and is the honest read.

## Matched-weight comparison: ninfer vs llama Q4 (MTP-vs-MTP, 262K)

To remove the weight confound above, the same 27B checkpoint was run as **llama Q4_K_XL** (Unsloth
Dynamic ~4.5-bit, matched to ninfer's ~4.7-bit) with MTP enabled, versus **ninfer 4.7-bit**, both at
full 262K. Q4 (5.5 GB lighter than Q6) is what finally lets llama's MTP draft context fit at 262K.

**262K + MTP fit.** A real difference, not just speed:

| engine | KV | 262K + MTP |
|---|---|---|
| ninfer | **int8 (8-bit)** | ✅ fits |
| llama Q4 | q8_0 (8-bit) | ❌ OOM (MTP draft ctx) |
| llama Q4 | **q5_1 (5-bit)** | ✅ fits (tight) |
| llama Q6 | any | ❌ MTP never fit |

llama needs to drop to **5-bit** KV to fit MTP at 262K; ninfer fits MTP with **8-bit** KV. So the
comparison below already gives llama the lighter (lower-precision) KV.

**Quality — identical.** 10-turn agentic real-code benchmark (`bench/dual-v100/`: `codebase_ctx.txt`
+ `turns.json`, thinking on, effort medium, greedy), scored on gold-string hits over 8 verifiable
turns: **ninfer 100, llama-Q4 100** (and llama-Q6 100). Matched weights do not change accuracy.

**Decode — MTP vs MTP, depth-matched** (same prompts; the two tokenizers agree to <0.1 %):

| context depth | ninfer int8 +MTP | llama-Q4 q5_1 +MTP | winner |
|--------------:|-----------------:|-------------------:|:------:|
|          ~476 |             59.8 |           **71.0** | llama +19 % |
|         ~58 K |         **47.5** |               40.2 | **ninfer +18 %** |
|        ~118 K |         **38.7** |               32.6 | **ninfer +19 %** |

At matched weights and MTP-vs-MTP, ninfer wins the **deep-context regime (≳35 K crossover)** by
~18–19 % — the regime this product targets — *while carrying higher-precision (int8) KV*. llama-Q4
is faster only at short context. (llama-Q4 MTP draft acceptance 0.81–1.00, mean run 3.2–4.0 — MTP is
genuinely working on both.) So the raw-decode table's big margins shrink to a ~19 % deep-context edge
once the weights are matched — but the edge, the quality parity, and the fit advantage are real.
Result artifacts: `bench/dual-v100/results/`.

## How prefill was closed

ninfer's prefill originally trailed llama ~1.3–1.7×. The gap was the GEMM algorithm, not the
hardware: llama's Volta build forces **MMQ (int8 `__dp4a`)**, compute-bound (~75 % of the integer
pipe), whereas ninfer's fp16 `mma.sync` kernel was **L1/shared-bound** (84.6 %, ~46 % compute).
Porting int8/dp4a with the decisive trick — **warp-cooperative tiling** (lane→row, warp→token) so
the activation operand is a single shared **broadcast** across the warp — took the Q4/Q5 prefill
GEMMs from ~21 to ~31–33 TFLOP/s (compute-bound). Correctness: 0.4–0.65 % vs the SIMT oracle
(expected int8-activation accuracy); dual-card greedy A/B (dp4a on vs off) identical on the cases
tested including a long-context needle retrieval. See `docs/DUAL-V100-BRINGUP-LOG.md`.

## Serve path: prefill pipelining

The table above is the CLI/`llama-bench` prefill measurement. The production `ninfer-serve` path
shares the exact same forward (`Engine::submit` → `prefill_impl`) and measures **within run-to-run
noise of the CLI** at every context size (e.g. ~385 tok/s at 221,592 tokens vs the CLI's ~385–407).
Prefill here is **device-bound**: the per-2048-token chunk spends ~2.5 s on the GPU and only a small
fraction of that on host submit, so there is little exposed host time to reclaim.

> **Investigated and rejected — per-chunk barrier "pipelining".** `prefill_impl` issues a
> `DeviceContext::synchronize()` at the end of every chunk. Skipping it for intermediate chunks (to
> overlap the next chunk's host submit with the current chunk's device execution) is *correct and
> bit-identical* — validated, including prompt-cache capture/reuse — but a **rigorous interleaved
> A/B showed no real speedup**: serve pipe 572.4 s vs barrier 576.2 s at 221,592 tokens (**+0.7 %**,
> with the pipe runs' own 18 s spread dwarfing the 4 s gap), and the CLI likewise 0–2 % within noise.
> An earlier non-interleaved measurement that suggested ~+10 % was a **thermal-order artifact** (pipe
> ran first on cold cards. The exposed host time is genuinely small, so the barrier is not worth
> removing — the candidate change was reverted (no code shipped).

### Methodology notes

- Kernel + end-to-end measured with ncu, nsys, `test-backend-ops`, `llama-bench`, and the ninfer
  CLI/perplexity tools.
- The 221,592-token llama prefill point uses `-ub 2048 -b 2048` (matching ninfer's chunk);
  llama-bench's default whole-prompt compute buffer OOMs at that length on 16 GB (llama-server,
  which streams, does not).
- Decode measured at context depth via `llama-bench -d <depth>` and ninfer `--max-new 128` after a
  prompt of the matched length; ninfer MTP via `--spec mtp --draft-tokens 3`.
- Prefill is single-stream on both engines.
