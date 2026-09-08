# MoE grouped-prefill dp4a port — concrete scope (Task #30, the prefill blocker)

**Branch:** `feature/moe-expert-parallel`. **Goal:** replace the scalar-SIMT routed-expert
grouped-prefill GEMMs on Volta with the int8/`__dp4a` technique that already won the *27B dense*
prefill-vs-llama fight, closing the 3–6× MoE prefill gap P3 measured.

## Why (the measured gap)
P3 (`bench/moe-ep-p3/FINDINGS.md`): llama.cpp prefill **3–6× faster** than ninfer's grouped-prefill
EP (llama ~3000–4900 vs ninfer ~600–1140 tok/s), and ninfer degrades harder with depth → 250K s/call
415 s vs 67.5 s. Prefill dominates the summariser workload, so this is *the* thing blocking cutover.

The routed experts are **74% of a3b prefill** (in-kernel comment, `sparse_moe_prefill_kernels.cu:1090`):
gate/up 47.6% + down 26.8%. On Volta they run `sparse_moe_prefill_q4_gate_up_simt_kernel` /
`sparse_moe_prefill_qx_down_simt_kernel` — already **weight-stationary** (they fixed the decode-path
per-token weight re-read, which was BW-bound at ~425/728 GB/s) but the MAC itself is **scalar fp32
`fmaf`, 1 MAC/instr** (`:1201`, `:1332`). llama's MoE prefill is dp4a MMQ: int8, `__dp4a` = **4
MAC/instr** on the integer pipe, compute-bound. That instruction-level 4× is the gap.

## The asset we already own
`src/ops/linear/q4/q4_volta_dp4a_gemm.cuh` — proven on the 27B dense linear path (Task #19/#20: took
that GEMM from L1-bound 46% compute → ~75% compute-bound, matching llama's MMQ). Two-kernel structure:
1. `q4_dp4a_quantize_x_kernel` — quantise the activation matrix to int8 **once** (per-(token,group)
   fp16 scale; one warp per (token,64-group), amax→int8).
2. `q4_volta_dp4a_gemm_kernel` — **warp-cooperative** tiling: `lane`=output row, `warp`=output token,
   so the activation operand is a **warp broadcast** (one shared txn feeds 32 lanes → ~1:50 dp4a per
   txn vs ~1:4 naive); weights on an odd 17-word stride (conflict-free); activation reads uint4-
   vectorised; int32 accumulate, then `acc * w_scale * x_scale`.

Symmetric-int8 numerics carry over exactly: Q4 weight `(nibble^8)-8 ∈ [-8,7]`, activation `amax/127`,
both zero-centred → **no zero-point sum-correction** (the dense kernel depends on this). Q5 `∈[-16,15]`
and Q6 `∈[-32,31]` are likewise symmetric and fit int8 (verified in the decode atoms).

## Why it's a port, not a drop-in
The dense kernel does a dense contiguous `X[T,K]`. The MoE grouped-prefill operates on the **permuted
grouped token buffer** indexed by `expert_offsets`/`route_job_columns` off a **device-side route-job
list** (no host sync), with **per-expert weight banks**, a **SwiGLU-fused** gate/up epilogue, and
**Q5/Q6** (not just Q4) on the down. So: keep all the scaffolding (router / select_count / scan /
gather / reduce / write_partial / EP shard remap — all unchanged), swap only the two GEMM inner loops
+ operand staging, and add two quantise passes over the grouped buffers.

## Concrete changes

### A. Decode atoms → int8 variants (trivial, ~1–2 h)
Add to the three storage headers, alongside the existing `decode_eight` (which folds the scale into
float — we need it **separate**):
- `Q4SimtDecodeAtom::decode_eight_int8(packed, int8 (&q)[8])` — already computes `(nibble^8)-8`
  (`q4_rowsplit_storage.cuh:37`); just cast to int8, drop the `*scale`, return scale via the caller.
- `Q5SimtDecodeAtom::decode_eight_int8(...)`, `Q6SimtDecodeAtom::decode_eight_int8(...)` — same signed
  value they already form (`q6_rowsplit_storage.cuh:64`), cast to int8. Scale returned once per group.

### B. Two quantise kernels (~½ day; `q4_dp4a_quantize_x_kernel` is the template)
1. `moe_prefill_quantize_gathered_kernel` — `grouped_io` (gathered bf16, `[kHidden, assignments]`) →
   int8 `grouped_i8` + fp16 scales `grouped_xs[assignment, kHidden/64=32]`. One warp per
   (assignment, 64-group). Runs **once** after gather, before gate/up. (Later optimisation: fuse into
   `sparse_moe_prefill_gather_kernel` so the bf16 grouped buffer is never materialised — deferred.)
2. `moe_prefill_quantize_swiglu_kernel` — SwiGLU output `routed_activation` (`[kIntermediate,
   assignments]` bf16) → int8 `swiglu_i8` + scales `swiglu_xs[assignment, kIntermediate/64=8]`. Runs
   after gate/up, before down.

### C. `sparse_moe_prefill_q4_gate_up_dp4a_kernel<BN>` (~1–1.5 days, the fiddly one)
Replaces the SIMT gate/up. Keeps the outer `route_job × row_block` persistent-block loop verbatim
(`:1144`). Inside a CTA:
- Stage **int8** gate & up weight nibbles (via `decode_eight_int8`) into shared on the dp4a odd stride
  (two banks, Wg/Wu, as today but int8 not float).
- Stage **int8** activations from `grouped_i8` (+ per-column `x_scale` from `grouped_xs`) — tail
  columns zero-filled (int8 0 contributes nothing to dp4a, same as the current bf16 zero-fill `:1173`).
- Warp-cooperative dp4a inner loop → **int32** gate/up accumulators; epilogue
  `g = iacc_g * w_scale * x_scale`, `u = iacc_u * …`, write `silu(g)*u` as bf16 to `routed_activation`
  (identical SwiGLU epilogue to `:1219`).

### D. `sparse_moe_prefill_qx_down_dp4a_kernel<Decode,BN>` (~½ day)
Replaces the SIMT down. Same structure, single output bank, `Decode::decode_eight_int8` for Q5/Q6,
dp4a against `swiglu_i8`, scale at the end, write bf16 to `grouped_io` for the reduce. Alpha still
applied by the (unchanged) reduce kernel.

### E. Workspace + arena (~½ day — **the tight part, treat carefully**)
New buffers, sized per **prefill slice** (`kSparseMoePrefillSliceMax=4096` → assignments ≤ 32768, not
the full context): `grouped_i8` (assignments × kHidden B ≈ **64 MB**), `grouped_xs`, `swiglu_i8`
(assignments × kIntermediate ≈ **16 MB**), `swiglu_xs`. Update `sparse_moe_prefill_workspace_bytes`
**and** the 35B `post_mixer_workspace_capacity_bytes` + `mtp_post_mixer_workspace_capacity_bytes`
arena models (`targets/qwen3_6_35b_a3b/impl/variant.cpp`).

The arena is already tight — MTP OOMs at 262K by ~70 MB. **Mitigation:** the header documents
lifetime unions (`sparse_moe_prefill.h:52` — `grouped_io ↔ routed-down output`, `routed_storage ↔
routed_sum`, `score_storage ↔ shared SwiGLU`). The int8 buffers are ~4× smaller than their bf16
sources and are consumed strictly between gather→gate/up and gate/up→down, so overlay them on a
lifetime-dead union region rather than growing the arena. Get the lifetime table right; this is the
finding most likely to bite (mirror P2's arena root-cause discipline).

### F. Dispatch (small)
In the two `#ifdef NINFER_VOLTA_BUILD` blocks (`:1810`, `:1891`): quantise → dp4a kernel instead of the
simt kernel. Keep the SIMT kernels compiled behind a `NINFER_MOE_PREFILL_SCALAR=1` env fallback for
A/B + regression. Shared expert stays W8-SIMT (smaller fraction, different codec; a W8 dp4a is a later
optional). Router stays SIMT (~2% of MoE flops — not worth touching).

## Numerics / correctness plan
int8 activation quant is a real numeric change (unlike MTP-EP's exact self-spec), so **bit-identity is
NOT the gate** — the dense 27B path shipped this same quant and validated fine. Gate on: coherent
generation + top-token agreement / low divergence vs the SIMT path on a fixed prompt (keep the scalar
fallback specifically for this A/B). Adversarial-review surface: quant numerics, shared-mem staging
bounds, tail zero-fill, and the arena/lifetime-union sizing.

## Validation gates (mirror P1/P2 discipline; needs a titan-router both-cards window)
1. Build green (sm_70). 2. Single-card: coherent + top-token A/B vs SIMT. 3. Dual-card EP: shard held,
coherent, zero bad_alloc. 4. Re-run `bench/moe-ep-p3` sweep → prefill tok/s vs the 606–1140 baseline
and llama's 3000–4900. 5. Adversarial review.

## Expected outcome (honest)
gate/up+down = 74% of prefill at ~4× MAC throughput ⇒ Amdahl ceiling ~2.5–3× overall prefill (minus
the two quant passes + the untouched 26% router/gather/reduce/shared). Rough: 250K prefill 606 →
~1500–1800 tok/s. That's a large step but may still trail llama's 3805 there — llama has years of MMQ
tuning and no EP-reduce overhead. It **closes the decisive gap**; whether it's enough to flip the
cutover (combined with ninfer's decode/MTP/int8-KV wins) is the re-measure question, not a given.

## Secondary: QPN is NOT for this
`q4_volta_qpn_gemm.cuh` (quadpair-split-N m8n8k4) is a **small-T** tensor-core kernel for T=4/8/16 —
the MTP *draft* widths. It's a lever for the MoE **small_t / MTP-EP decode** path (P1's decode
kernels), not wide prefill. MTP is already 1.45×; a QPN MoE small-T GEMM is a separate, lower-priority
follow-up. Do **not** fold it into this port.

## Effort
Atoms ~1–2 h · quant kernels ~½ d · gate/up dp4a ~1–1.5 d · down dp4a ~½ d · workspace/arena+dispatch
~½ d · build+single/dual validate+bench ~1 d (GPU-window-gated). **~4–5 focused days to a measured
result.** Steps A–B and the kernel bodies (C–D) are card-free; only the validation gates need the
coordinated both-cards window.

## Implementation status (card-free steps A–F DONE, compiles sm_70)
All of A–F are implemented on `feature/moe-expert-parallel` and compile clean (sm_70,
`libninfer_ops.a` + `libninfer_text.a`, RC=0):
- **A** `decode_eight_int8` on Q4/Q5/Q6 `*SimtDecodeAtom` (+ `load_eight_int8` on the down wrappers) —
  mirrors each `decode_eight` exactly, emitting symmetric int8 + separate scale.
- **B** `sparse_moe_prefill_gather_quant_kernel` (fused gather+int8 quant, writes `grouped_i8` aliasing
  the grouped_io region) + `sparse_moe_prefill_quantize_swiglu_kernel`.
- **C/D** `sparse_moe_prefill_q4_gate_up_dp4a_kernel<BN>` + `sparse_moe_prefill_qx_down_dp4a_kernel<Decode,BN>`
  (warp-cooperative `__dp4a`, lane=row broadcast-activation, per-group int32→fp32 scale fold).
- **E** workspace: `grouped_i8` reinterprets `grouped_io` (zero growth on the 64 MB buffer); only
  `swiglu_i8` (~16 MB) + the two fp16 scale planes are fresh, and **reserved only for the Q4+Q5/Q6
  profile** via `sparse_moe_prefill_wants_dp4a_scratch` — the W8/W8 MTP leaf and non-Volta are
  untouched.
- **F** dispatch: `use_dp4a = !scalar_fallback && !adaptive && Q4 gate/up && Q5/Q6 down`; SIMT kept
  behind `NINFER_MOE_PREFILL_SCALAR=1` for A/B and regression.

**Build note:** `ninfer-serve` link needs 5 system dev libs the build image had lost — in the build
container run `apt-get update && apt-get install -y libav{format,codec,util,swscale}-dev
libcurl4-openssl-dev`, then `ninja apps/ninfer-serve`.

## GPU validation (2026-09-08, coordinated titan-router window)
- **Correctness A/B — PASS (exceeded bar).** dp4a vs `NINFER_MOE_PREFILL_SCALAR=1` on the same 4044-tok
  greedy prompt (dual-card EP, CTX 253952): **byte-identical output, 100% token agreement (43/43)**.
  int8 activation quant flipped no argmax at greedy — stronger than the "coherent + high agreement"
  bar. Harness: `bench/moe-ep-p3/validate-dp4a.sh` + `greedy_probe.py`.
- **Arena fit — PASS at 253952** (both dp4a + scalar loaded: free-after-startup 749 MiB, slack 768
  MiB — tight but clears; the +~18 MB non-MTP scratch fits). 262144 fit checked in the sweep run.
- **Prefill (shallow, ~4K) — dp4a ~1851–1879 tok/s vs scalar ~1078–1172 = ~1.6–1.7×**; decode
  unchanged (dp4a 111.6 vs scalar 114.2 — dp4a touches only prefill). Full depth sweep vs the P3
  baseline + llama: see `bench/moe-ep-p3/FINDINGS.md` (`ninfer-dp4a.csv`).
