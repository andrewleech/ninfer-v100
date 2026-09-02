# Dual‑V100 NVLink graph‑parallel port — plan

Grafting devon‑caron's dual‑GPU **graph‑parallel** path (built for 2×RTX 3090 / sm_86)
onto **this** fork (geoffwatts/ninfer‑v100, sm_70 Volta), to run the dense **Qwen3.6/3.8 27B
groupwise‑int** target across titan's two NVLinked V100‑SXM2‑16GB.

## Why this is tractable

The two forks share the `Neroued/ninfer` scaffolding (identical file layout, `.ninfer`
`RowSplitK128V1` artifact format, same `qwen3_6_27b` target, same
`Q4G64_F16S`/`Q5G64_F16S` groupwise formats, same `DType::BF16` activation‑storage
convention). **geoffwatts already did the hard sm_70 kernel work** — this fork's
`q4_linear_swiglu_*` / `q5_linear_add_*` / GQA kernels are proven on Volta. devon's
dual‑card feature is therefore mostly **architecture‑neutral orchestration** plus a small
set of **additive integer parameters** on kernels we already have.

There is **no shared git history** (this fork is a single squashed commit; `git merge-base`
with devon is empty) → the graft is a **manual patch‑port**, using devon's 2 commits as a
reference diff, not a cherry‑pick.

Reference diff captured at: `/tmp/claude-1000/-home-anl-v100/dual-patch/dual-full.diff`
(devon `release/v0.6.0-rtx3090…feature/dual-3090-nvlink-support`, 2 commits, 95 files).
devon's own design doc is `GRAPH-MODE-README.md` in that diff.

## The mechanism (what we are porting)

- **Runtime‑selected, one binary.** `--devices A,B` (mutually exclusive with `--device N`);
  disables cross‑device CUDA‑graph capture (runs an eager warmed round instead).
- **MLP‑only weight shard.** Per layer, intermediate `17408 → 8192 primary / 9216 secondary`.
  - Q4 gate/up (`PairedRows` axis): primary `[16384,5120]`, secondary `[18432,5120]`.
  - Q5 down (`Columns` axis, i.e. split‑K): primary `[5120,8192]`, secondary `[5120,9216]`.
  - Split point `8192` is chosen to preserve the K128 / g64 group boundary (`%1024==0`).
  - Everything else (attention projections, embeddings, norms, output head, GDN/linear‑attn
    state, MTP proposal head) stays **primary‑only**. No secondary replica.
- **Per‑layer MLP dataflow** (`Variant::post_mixer_graph`): primary computes its shard →
  `primary_delta`; event‑synced, secondary `cudaMemcpyPeerAsync`'s the normalized `hidden`
  in, computes its shard → `secondary_delta`; primary waits, peer‑copies `secondary_delta`
  back, and `residual_add_two_launch` sums **both partials + residual** (the down split is
  along K, so each card emits an independent full‑width partial — the reduce is a separate
  elementwise add, *not* a fused GEMM epilogue).
- **KV split by attention layer.** 16 full‑attn layers: **primary owns 0–4, secondary owns
  5–15** + the **MTP KV** (MTP is disabled on primary in graph mode). Remote layers
  peer‑copy Q/K/V + block‑table selectors to the secondary, run the qualified
  24‑qhead/4‑kvhead GQA there, and copy the result back (`remote_gqa_attention` /
  `_cached` / `_kv_append`). Whole‑layer ownership avoids a head‑sliced reduction.
- **Shared logical KV budget.** Every request reserves the **same** page entitlement in both
  pools (cannot overbook either card); `resolve_kv_capacity` solves both device curves and
  takes `pages = min(primary_fit, secondary_fit)`. Aggregate resident tokens = both pools.
- **Sync.** Per‑rank streams + `cudaEvent` fences only; up to two lightweight handshakes per
  layer (one remote‑attn, one MLP). No per‑token `cudaDeviceSynchronize`.

## sm_70 reconciliation (the key risk, and why it's manageable)

devon's kernels carry Ampere‑only hazards — bf16 tensor MMA, `cp.async`, `mma.sync m16n8k*`,
>96 KB opt‑in smem, NVFP4 TMA. **We never port devon's kernels.** We add the same integer
parameters to *this fork's* already‑Volta‑safe kernels. Consequences:

- `DType::BF16` activations are **not** a blocker here — this fork already runs bf16‑storage
  activations + `Q4G64_F16S`/`Q5G64_F16S` weights on sm_70 (see `qwen3_6_27b/impl/variant.cpp`
  `post_mixer`). The ported `post_mixer_graph` and `residual_add_two` keep bf16 storage.
- KV dtype = **int8** (`Int8Group64`, already supported). bf16 KV is unavailable on Volta;
  the graph path uses int8 KV anyway.
- Watch the **96 KB smem opt‑in ceiling** (sm_70) vs sm_86's 100 KB when reusing any
  `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` kernels — but these are this fork's own
  kernels, already within the Volta cap.
- NVLink P2P on the V100 pair (`NV6`) is confirmed working → `cudaMemcpyPeerAsync` is fine.
  The A2000 (sm_86, PCI idx 0) is auto‑rejected by the compute‑capability‑match + peer‑access
  guards in the 2‑device `DeviceContext` ctor; select the V100s (nvidia‑smi idx 1,2).

## Prerequisite: the right checkpoint

The graph path requires the dense **27B groupwise‑int `.ninfer`** (Q4 gate/up + Q5 down) —
**not** the NVFP4 27B (NVFP4 is rejected under model‑parallel; see "NVFP4" below), and
**not** the 35B‑A3B MoE (also rejected).

- **Source:** `hf download neroued/Qwen3.8-27B-NInfer qwen3_8_27b.ninfer`
- **Size / SHA256:** 18,210,531,328 B (~16.96 GiB) /
  `eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e`
- **Local:** `/home/anl/v100/models/Qwen3.8-27B-NInfer/qwen3_8_27b.ninfer`
  (downloading via `aria2c -x8 -c`; verify against `SHA256SUMS` when done).
- **Disk note:** titan `/` was at 95% (34 GB free). The now‑deprioritized NVFP4 20 GB artifact
  (`models/Qwen3.8-27B-nvfp4-NInfer/`) is the obvious reclaim if space gets tight.

## Design decisions & alternatives considered

### NVFP4 in the dual‑card path — feasible here, deliberately deferred

devon **rejects** NVFP4 under model‑parallel, but that is an **sm_86 constraint, not ours**:
on a 3090 NVFP4 needs Blackwell (the A4/TMA kernels are sm_120a). **This fork already has a
full Volta software‑decode NVFP4 path** (`src/ops/linear/nvfp4/{nvfp4_cutlass_sm70,
nvfp4_volta_mma_gemm,nvfp4_prepack_sm70,nvfp4_gemv,nvfp4_small_t}`, plus `linear_add/nvfp4`
and `linear_swiglu/nvfp4` `_volta_qpn` families). So the compute side exists.

The blocker is **storage layout**, not compute: groupwise Q4/Q5 is `RowSplitK128V1`, but NVFP4
is **`BlockScaleK16M128x4V1`**, and `shard_row_split_across_devices` rejects any non‑row‑split
tensor. Dual‑card NVFP4 therefore needs a **second sharding path** — a block‑scale‑aware split
that preserves the k16 / m128×4 block+scale boundaries (a parallel `block_scale_geometry`),
plus `_shard`/`_graph_partial` variants of the nvfp4 Volta kernels. Essentially "redo Phase 2
+ Phase 4 for the block‑scale layout."

**Why deferred:** the NVFP4 27B artifact is **~20 GB vs ~18.2 GB groupwise** → a *worse* fit on
2×16 GB, and on Volta it is software‑decoded (none of Blackwell's speed), so over groupwise it
buys only marginal quality for a bigger, no‑faster model. Do groupwise first; revisit NVFP4
only if output quality demands it. Not a wall — a scoped, lower‑value follow‑on.

### Why the split is MLP‑only (and why titan can go further)

devon shards **only the MLP** and keeps whole attention layers card‑local specifically to
(a) preserve the qualified 24‑qhead/4‑kvhead GQA kernel (no head‑sliced reduction) and
(b) **minimize cross‑device traffic — because 3090 pairs are often on PCIe, not NVLink**.
titan's V100 pair is **NV5 NVLink** (`nvidia-smi topo -m`: `GPU1 ↔ GPU2 = NV5`; the A2000 is
PHB/PCIe‑only). The traffic devon was avoiding is cheap for us, which *inverts* the tradeoff
and makes a fuller, more balanced split worthwhile here where it was not on devon's target.
That is Phase 7.

## VRAM sketch on 2×16 GB (must verify empirically)

Asymmetric by design: **primary is weight‑heavy** (all non‑MLP: embeddings + 248320×5120
output head + all attention + its 47% MLP shard); **secondary is KV‑heavy** (only its 53%
MLP shard, + 11/16 of KV + MTP KV). devon's 3090 run reserved 10.3 GB primary / 17.5 GB
secondary — the secondary exceeds 16 GB, so on V100 the **KV pool must be sized down**
(`--kv-capacity auto` solves this per‑card). Net: dual‑card is what makes the 27B *fit at
all* on 16 GB cards (single‑card ~15 GB weights + KV won't fit), plus a modest decode
speedup. Expect aggregate resident tokens in the low multiples of 262K, not devon's 803K.

## Honest performance expectation

Only the MLP is split; attention, scheduling, the reduce, and every layer's two NVLink hops
stay on the critical path. devon measured (2×3090, int8 KV): **prefill +41%, decode +15–18%,
C8 aggregate +15%** — *not* ~2×. The V100 win is primarily **enablement + KV headroom**;
treat any single‑stream decode speedup as a bonus and measure it against `--device 1` on the
identical prompt/sampling.

## Port phases

**Phase 0 — setup (DONE).** Fork `andrewleech/ninfer-v100`, branch `feature/dual-v100-nvlink`,
reference remotes `devon` / `donchad` / `origin` / `upstream` wired.

**Phase 1 — portable orchestration (near‑verbatim from devon).**
`src/core/device.{cu,h}` (multi‑endpoint ctx, peer access, `ScopedDeviceRank`,
`configure_cuda_device_once`); `serve_options.{cpp,h}` + `apps/cli/options.*` +
`apps/serve/main.cpp` (`--devices`); `include/ninfer/types.h` (`EngineOptions.devices`,
`MemorySummary.secondary_*`); `runtime/engine/engine.cpp` (`normalize_engine_options`, build
`DeviceContext(span)`); `concurrent_executor.h`; `kv_capacity.{cpp,h}` +
`runtime/contract/types.h` (dual‑pool solve); `request_log.cpp`, `generation_service.cpp`
(reporting); new `tools/multi_gpu_probe.cu`. Then sweep this fork's kernels that use a
process‑wide `static const cudaError_t attr = cudaFuncSetAttribute(...)` and replace with
`configure_cuda_device_once` (per‑device correctness on the secondary card).

**Phase 2 — artifact sharding (load‑time weight split).**
`src/artifact/binder.{cpp,h}` (`RowSplitShardAxis`, `shard_row_split_across_devices`,
`finish()` primary/secondary capacity recompute); `src/artifact/materializer.{cpp,h}`
(peer‑copy repack via `cudaMemcpy3DPeerAsync` over the three row‑split planes);
`typed_binding.*` / `reader.h` as needed. Verify this fork's `row_split_geometry` matches.

**Phase 3 — 27B target: shard bindings + MLP dataflow.**
`qwen3_6_27b/impl/load/bindings.{cpp,h}` (`graph_parallel` flag, `load_mlp` secondary shards,
`secondary_weights_arena`); `qwen3_6_27b/impl/variant.{cpp,h}` (`post_mixer_graph`,
`supports_graph_parallel=true`, `graph_primary_attention_layers=5`); `impl/package.cpp`
(pass `devices.size()==2`); `registry.cpp` (per‑rank VRAM checks; reject 35B/nvfp4 under
model‑parallel).

**Phase 4 — sharded kernels (fork‑specific, additive).** Add to *this fork's* kernels:
- gate/up: `q4_linear_swiglu_{gemv_pair,small_t,mma}_shard_launch` — add
  `row_begin`/`output_rows`/`parent_intermediate` to `q4_linear_swiglu_gemv_pair`,
  `small_t_exact`, `mma_split_half_pair_r32_c{40,48,128}`.
- down: `q5_linear_graph_partial_{gemv,mma}_launch` — add `tile_begin`/`tile_count`
  (K‑subrange) to `q5_rowsplit_gemv` / `q5_rowsplit_gemm_mma`; swap the fused `AddResidual`
  epilogue for a plain `Store` (partial output).
- reduce: `residual_add_two_launch` + `residual_add_two_scalar_kernel` (new; bf16 storage,
  fp32 accumulate; mirror this fork's `residual_add`).
- declare in `q4_linear_swiglu_kernels.h` / `q5_linear_add_kernels.h`. Base kernels stay for
  the single‑card path.

**Phase 5 — execution graph + remote‑attention KV split (largest).**
`qwen3_6/impl/runtime/{program_impl.h,program.h,text_context_impl.h,text_context.h,
schedule.h,layouts.h,layouts_impl.h,decode_impl.h,dflash_impl.h,mtp_impl.h,
text_prefill_impl.h}` + `impl/state/decoder_state.cpp`: secondary KV pools (5/11 + MTP on
secondary), `remote_gqa_attention` / `_cached` / `_kv_append`, `run_layers` model‑parallel
dispatch + secondary‑workspace reset, `ProgramImplCore` guard rails and secondary storage.

**Phase 6 — build + validate on titan.**
Build sm_70 in `v100build:cu128` docker (host toolchain too new — per RUNBOOK). Run
`multi_gpu_probe` (expect V100s at idx 1,2, peer 1↔2). **Correctness:** greedy output
equivalence `--device 1` vs `--devices 1,2` on the groupwise 27B. **Perf:** prefill/decode/
concurrency matrix vs single card. Port `tests/ops/test_graph_shard_kernels.cpp`,
`tests/test_device.cpp`, `tests/test_kv_capacity.cpp`. Then wire a
`serve-ninfer-27b-dual-v100.sh` launcher (`--devices 1,2 --kv-dtype int8 --kv-capacity auto`)
and, if it lands, a llama‑swap catalog entry.

**Phase 7 — balanced‑split trials (NVLink‑justified, post‑MVP).** The 8192/9216 MLP split is
balanced *for the MLP* (47/53). The residual imbalance is that the **primary also carries all
non‑MLP work** (embeddings, the 248320×5120 output head, all attention compute, the reduce),
so it is the critical path while the secondary is "MLP shard + KV store." Three levers, in
value/effort order for an NVLinked pair — each a measured trial vs the MLP‑only baseline:

1. **Vocab‑parallel the output head.** Split `output_head` [248320,5120] column‑wise across the
   two cards; all‑gather logits once per token (tiny). Frees ~2.5 GB on the primary and halves
   the final GEMM. Low traffic, clean win. Cheapest.
2. **Tensor‑parallel the full‑attention layers.** Head geometry is *clean* for 2‑way: **24 q /
   4 kv heads → 12 q / 2 kv per card**, GQA group of 6 preserved. Split QKV/O projections and
   attend each card's own heads locally. This **balances attention compute AND removes the
   remote‑attention peer copies entirely** (replacing them with one attention‑output all‑reduce
   per full‑attn layer — cheap on NV5), and replaces the by‑layer KV split (5/11) with a
   **by‑head KV split across all 16 full‑attn layers**. This is the lever that pushes decode
   past devon's +15–18% toward a real 2×. Bigger change; needs shard variants of the QKV/O
   projection kernels + head‑split GQA + KV‑by‑head allocation.
3. **Shift the MLP split point.** Hand the secondary a larger MLP shard (e.g. 6144/11264) to
   offset the primary's serial non‑MLP load. One constant; second‑order tune, bounded by the
   fact the primary's serial work isn't parallelized regardless.

**Hybrid‑arch caveat:** the model is mostly **GDN linear‑attention** layers (recurrent state,
primary‑resident) plus the 16 full‑attention layers. Trial #2 targets the full‑attention layers
first; leave GDN state on the primary initially (its per‑token state is small). TP'ing the GDN
value heads (48 → 24/24) is a later, more involved step because the state threads across layers.

Measure each trial against the MLP‑only baseline on identical prompt/sampling: single‑stream
decode tok/s, prefill tok/s, C2–C8 aggregate, per‑card VRAM balance, and NV5 bytes/token.

## Order of attack

Phases 1→3 compile independently and keep the single‑card path working (dual is inert until
`--devices` is passed). Phase 4 is the only genuinely new CUDA. Phase 5 is the bulk of the
logic but is mostly plumbing over primitives built in 1–4. Land 1–4 behind the runtime flag,
prove the MLP split in isolation (a unit test that runs one layer's `post_mixer_graph` vs the
single‑card `post_mixer` for numerical parity), then do Phase 5.
