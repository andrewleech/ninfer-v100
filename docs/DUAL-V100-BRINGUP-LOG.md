# Dual-V100 27B bring-up log — how we got it running

Verbatim record of getting the graph-parallel **Qwen3.8-27B** (groupwise-int `.ninfer`) to load and
generate across titan's **2× Tesla V100-SXM2-16GB** (NVLink `NV5`, CUDA device indices 1 and 2; the
RTX A2000 is index 0). Companion to [`DUAL-V100-PORT-PLAN.md`](DUAL-V100-PORT-PLAN.md), which is the
design; this is the empirical bring-up, the dead ends, and the measured numbers.

Outcome up front: **it works.** Prefill ~305 tok/s, decode 24 tok/s raw / **58 tok/s with MTP
`--draft-tokens 3`**, correct coherent greedy output, peak **11.5 GiB primary / 5.3 GiB secondary**.
Recommended serve config: `--devices 1,2 --spec mtp --draft-tokens 3 --kv-dtype int8`.

Landed in commits `9a735061` (phase 5a: MLP shard dispatch) and `f7fb9916` (the two fixes that made it
load and run).

---

## Environment / how the runs are invoked

The engine binary is built for sm_70 in Docker (host toolchain too new — see RUNBOOK) and run in the
`v100ninfer:cu128` container (it carries the ffmpeg runtime the CLI links). Canonical run:

```bash
docker run --rm --gpus all -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -v /home/anl/v100/ninfer-v100:/src -v /home/anl/v100/models:/models \
  -v /home/anl/v100/.nvjitcache:/root/.nv/ComputeCache \
  -w /src/build-v100 v100ninfer:cu128 \
  ./apps/ninfer /models/Qwen3.8-27B-NInfer/qwen3_8_27b.ninfer \
  --prompt "..." --greedy --max-new 64 --max-context 4096 --kv-dtype int8 --devices 1,2
```

Three environment gotchas, each of which cost a debugging cycle:

1. **`CUDA_DEVICE_ORDER=PCI_BUS_ID` is mandatory.** Without it, CUDA inside the container enumerates
   *fastest-first* → `[V100, V100, A2000]`, so `--devices 1,2` selects one V100 + the A2000 (sm_86),
   and the (correct) compute-capability-match guard rejects it: *"model-parallel CUDA devices must
   have the same compute capability."* With `PCI_BUS_ID`, ordering matches `nvidia-smi`
   (`[A2000, V100, V100]`) and `1,2` are the two V100s.

2. **Cold PTX-JIT is slow (~2–3 min/run).** First load of the CUTLASS sm_70 kernels JIT-compiles from
   PTX (single-threaded, 100% CPU, no GPU activity — looks like a hang; `gdb` backtrace showed
   `libnvidia-ptxjitcompiler` under `cuLibraryLoadData`). Mounting a persistent
   `~/.nv/ComputeCache` makes subsequent runs fast.

3. **`nohup docker … &` double-backgrounds** under the shell runner; the launcher returns 0 while
   docker keeps running detached. Poll the redirect log / `docker ps`, don't trust the early exit.

---

## Lead 0 — the stale build (nothing dual-card was even in the binary)

First `--devices 1,2` run: `error: unknown argument: --devices`, and the usage text lacked it. The
committed source *had* `--devices` (phase 1, `8c5a5bd0`), but `strings apps/ninfer | grep -c -- --devices`
was **0**, and Phase-4 symbols (`residual_add_two`) were **0** too — the binary predated the entire port.

Cause: **mtime skew.** git checkouts don't preserve mtimes, so committed sources looked *older* than
previously-built `.o` objects; ninja considered everything up to date and the "build" was a no-op. Fix:

```bash
find src apps tools include -type f \( -name '*.cpp' -o -name '*.cu' -o -name '*.h' -o -name '*.cuh' \) -exec touch {} +
```

then a real rebuild — which then failed on missing ffmpeg/curl `-dev` symlinks and headers (the
`v100ninfer` image ships only runtime libs). Fix: `apt-get install -y libavformat-dev libavcodec-dev
libavutil-dev libswscale-dev libcurl4-openssl-dev` ephemerally inside the build container, then
`ninja apps/ninfer` (302 targets, full sm_70 rebuild). After that: `--devices`=5, `residual_add_two`=12,
graph-shard symbols present.

---

## Lead 1 — OOM at load; NOT KV, NOT weight imbalance

With a real binary, `--devices 1,2` initialised both cards (peer access up, 311 MiB context each) then:

```
error: cudaMalloc failed: cudaErrorMemoryAllocation: out of memory
```

VRAM stayed **flat at 311 MiB** through the failure — the first big allocation failed before any weight
upload. Added a one-line diagnostic in `registry.cpp` after `plan_load`:

```
[dual-diag] devices=2 model_parallel=1 source_bytes=17093490688 primary_bytes=11903039488 secondary_bytes=5190451200 shards=128
```

So sharding was **correct**: `model_parallel=1`, `shards=128` (64 layers × gate_up+down), full
**15.92 GiB** = primary **11.09 GiB** + secondary **4.83 GiB** (at the default split point 8192 of 17408).

Two trials to localize — both **still OOM**, which is the important part:

| trial | change | primary_bytes | secondary_bytes | result |
|---|---|---|---|---|
| baseline | split 8192 | 11.09 GiB | 4.83 GiB | OOM |
| rebalance | split 8192→4096 | **8.94 GiB** | 6.98 GiB | **still OOM** |
| pin KV | `--kv-capacity 1024 --max-context 1024 --kv-dtype int8` | 8.94 GiB | 6.98 GiB | **still OOM** |

Rebalancing the split changed `primary_bytes` but not the OOM, and shrinking KV to nothing didn't help
either → the OOM is **upstream of both weight-imbalance and KV**.

### Root cause: the materializer stages the whole model on each card

`src/artifact/materializer.cpp` (before the fix) allocated the primary arena at **`source_capacity`
(the full 15.92 GiB)**, uploaded the entire model to device 0, then (shard path) peer-split it onto
device 1 into an 8.94 GiB primary-staging arena + 6.98 GiB secondary arena, freed device 0's source,
allocated the final 8.94 GiB primary on device 0, and copied the staging back. **Both cards
transiently peak at the full 15.92 GiB** — ~99 % of a 16 GB V100, so context + fragmentation tips it
into OOM. devon's 24 GB 3090s had the headroom; the V100s don't. This is why the split point and KV
were irrelevant: `source_capacity` is unchanged by them.

### Fix: shard *during* upload (`f7fb9916`, part 1)

The reader `mmap`s the whole artifact (`reader.cpp:201`), so `payload()` is a zero-copy host pointer.
New shard path (early return in `materialize`, dense-replica/single-card paths untouched): allocate
only the two **final** arenas (8.94 GiB device 0, 6.98 GiB device 1), then for each tensor read it from
the mmap and copy its **primary rows straight to device 0** and its **secondary rows straight to
device 1** via `cudaMemcpy2DAsync` (new host-source helpers `copy_row_split_region_host` /
`copy_row_split_shard_from_host` mirroring the device→device ones). The 15.92 GiB source arena is never
allocated; peak per card is just that card's shard. The old device-staging shard branch was removed.

Result: **it loaded.** `load weights 100% 15.92 GiB / 15.92 GiB`, peak **11.7 GiB / 5.3 GiB**.

---

## Lead 2 — loads, but inference fails in the MLP

Immediately after load:

```
error: linear_swiglu: invalid tensor shape
```

`linear_swiglu` is the **single-card** `Variant::post_mixer` path and requires `gate_up.n == 34816`
(the full packed MLP). It received the **primary shard** (n=16384) → shape reject. So
`graph_parallel_active()` was false and `mlp_tail` fell back to single-card `post_mixer`. Tell:
card 2 held only **5.3 GiB** (= 4.83 GiB weights + context) — the **secondary workspace arena was
never allocated**, so `secondary_work_` was null.

### Root cause: only the decode path was wired

Phase 5a threaded `secondary_work` through the one decode `ExecutionCore` lambda, but there are **7
more inline `ExecutionCore` aggregates** (prefill / mtp / dflash, e.g. `program_impl.h:1081, 8508,
10623, 10682, 11007, 11166, 11353`) that construct `{device, model, work, …, proposal_head}`
positionally and defaulted the trailing `secondary_work` to null. Prompt prefill runs through those.

### Fix (`f7fb9916`, part 2)

Append `secondary_work ? &*secondary_work : nullptr` to all 7 inline aggregates (`ExecutionCore` is the
only struct ending in `proposal_head`, so a single targeted edit covers them).

Result: **it runs.** Both cards show GPU utilisation (59 % / 35 %), correct coherent greedy output.

---

## Working result + performance

Prompt: *"List three prime numbers and explain what makes them prime."* — greedy, `--max-new 128`,
`--max-context 4096 --kv-dtype int8`, `--devices 1,2`. Output is correct: *"Three prime numbers are
**2, 3, and 5**. A number is prime if it is greater than 1 and has exactly two positive divisors…"*

**MTP speculative-decode sweep** (works correctly with the dual-card path):

| `--spec mtp --draft-tokens N` | decode tok/s | acceptance rate | accept length |
|---|---|---|---|
| off | 24.0 | — | — |
| 2 | 52.6 | 85.1 % | 2.70 tok/round |
| **3** | **58.2** | 73.1 % | 3.17 tok/round |
| 4 | 56.4 | 64.5 % | 3.53 tok/round |

`draft-tokens 3` is the sweet spot → **2.43× over raw decode**. Prefill ~305 tok/s throughout.

Memory summary at split 8192 + MTP dt3: `gpu weights used 11.51 GiB`, `gpu workspace peak 472 MiB`,
`kv payload 132 MiB`, `free after startup 3.19 GiB`, `CUDA Graph allowance 0 B`.

### Performance ceiling note

`model_parallel` forces `use_cuda_graph=false` (a cross-device schedule can't be stream-captured), so
decode runs **eager** — per-layer kernel-launch + cross-card event-sync overhead is not amortised by a
graph. MTP recovers most of it. 58 tok/s is near the practical ceiling for this eager path.

---

## What's NOT the bottleneck / deferred

- **KV-split (5b/5c)** — this is about **context headroom**, not speed. There's 3.2 GiB free on the
  primary already, so context can grow without splitting KV across cards. Deferred.
- **Secondary-device memory reporting (5d)** — the summary still prints `device 1` and primary-only
  totals; `MemorySummary.secondary_*` is unpopulated. Cosmetic; deferred.

## Next gains (Phase 7 — larger, uncertain payoff)

1. **Tensor-parallel the 16 full-attention layers** — the secondary is idle during all attention/GDN
   work (only helps on MLP). TP'ing attention (clean 24 q / 4 kv → 12 q / 2 kv per card) puts it to
   work, removes remote-attention copies, and is the lever most likely to push decode toward a true 2×.
2. **Vocab-parallel the 248320×5120 output head** (frees primary VRAM, halves the final GEMM).
3. **Rebalance the MLP split point** (second-order; the primary carries all non-MLP weight so it is the
   tighter fit).
4. **Per-device CUDA sub-graphs** to cut eager launch overhead (hard — the cross-device fence is why
   full-graph capture is disabled).

---

## Phase 7 — tensor-parallel attention (the 262K-context lever)

Goal (Andrew's steer): serve the **full 262K context**. KV is ~33 KiB/token int8 (measured 132 MiB
@4096) → ~8.25 GiB at 262K, but the primary has only ~3.2 GiB free after weights, so a single pool
tops out around 64–72K. TP attention splits each of the 16 full-attention layers **24 q / 4 kv →
12 q / 2 kv per card**, which halves the primary's text KV (→ ~4.1 GiB/card, the fit enabler) and
also puts the otherwise-idle secondary to work during attention (prefill speedup + local decode, no
remote-KV latency). 262K is effectively single-session (weights 16 + KV 8.25 = 24 of 32 GiB).

Landed behind the opt-in `NINFER_TP_ATTENTION` env flag (dual-device only; 35B unaffected). Six
commits build the compute + load half, two the runtime half:

- **TP-A1/A2** — the `<12,2>` per-card causal geometry, the `attn_input_proj_graph_shard` projection
  op (two runtime-shaped Volta GEMMs), the `QkvHeadHalf` row-split axis, and load-time attention
  weight sharding (each full-attn layer's query_key/gate_value → two `[3584,5120]` Q4/Q5 shards).
- **TP-A3 (by-head KV dual pool)** — `DecoderStateSpec.text_kv_heads` halves *only* the text pool
  (MTP stays at full kv_heads); a secondary text `DecoderState` is allocated on rank 1. It carries no
  reservation lifecycle of its own — the primary's block table is mirrored into its execution matrix
  each layer, so both pools address identical physical slots. `tp_attention` is derived in the shared
  planner (gated on `supports_graph_parallel`) and threaded to `TextContext`.
- **TP-A4 (attn_mix orchestration)** — mirrors the `post_mixer_graph` fence. Each card projects its
  shard, unpacks the packed q|k / gate|v into contiguous buffers (a strided 2-D copy, since `view`
  requires contiguity), runs q/k norm + RoPE + a **local** GQA against its two KV heads, and gates.
  The two `[3072,T]` halves are gathered `[primary | secondary]` — which reproduces head order 0..23
  for the load-side shard — and the **full unsharded o_proj** runs on the primary.

**Design choice:** o_proj is *not* sharded (gather + full projection on the primary) — this avoids
touching the generic `ModelView` output weight; only query_key/gate_value shard.

**Status:** compile-validated sm_70 (both `variant.cpp`, zero errors); an agent review confirmed
flag-off is byte-for-byte the prior path and all the plumbing is sound. **Not yet GPU-validated (TP-A5).**

### Known limitations / TP-A5 watch-items

- **Host-KV offload gap** — the secondary pool is not wired into the host-KV offload/checkpoint
  machinery, so heads 2–3 would desync on an evict→restore. Fine for the 262K single-session target
  (no eviction); a real gap for cached/concurrent serving.
- **Block-table mirror is per-layer** (16×/step) though the matrix is layer-invariant — correct but
  hoistable to once/step.
- **Workspace headroom** — `attn_mix_graph`'s transients (plus the shard split-K accumulator) may
  exceed the planned `general_capacity`; if the arena throws at runtime, bump the workspace plan.

**TP-A5** = build + run `NINFER_TP_ATTENTION=1 --devices 1,2`, greedy/logit parity vs the MLP-only
path (tp-on vs tp-off A/B), measure the prefill gain, and confirm 262K KV actually fits.

### TP-A5 results (measured on titan, 2×V100, qwen3_8_27b groupwise, int8 KV, greedy)

| Metric | tp-off (MLP-only) | tp-on (NINFER_TP_ATTENTION=1) |
|---|---|---|
| Greedy output | 48 tokens | **identical 48 tokens** (exact parity) |
| Primary KV payload @4096 | 132 MiB | **66 MiB** (kv_heads 4→2) |
| GPU weights (primary) | 11.09 GiB | 10.76 GiB (sharded proj) |
| Max single-session context | ~98K (98304 fits, 114688 doesn't) | **~213K** (212992 fits, 221184 doesn't) |
| Prefill @41K tokens | **597 tok/s** | 78 tok/s |
| Decode @41K | 18.3 tok/s | 16.8 tok/s |

**The good:** exact correctness, and TP-attention **~2.17×'s the max context** (~98K → ~213K) by halving
the primary KV. At 128K, tp-off can't fit but tp-on does. This is the 262K-context lever working.

**The bad — this is a context-capacity feature, not a speedup.** Prefill is **7.6× *slower*** (597 → 78
tok/s; reproducible on a warm re-run, so not cold-JIT), and decode ~8% slower. The hypothesised prefill
gain (secondary attends during prefill) is dwarfed by the **eager per-attention-layer cross-device
overhead**: ~7 peer copies (hidden + norms + positions + block-table mirror + result-back), 4 strided
`cudaMemcpy2DAsync` unpacks per card, cross-stream event syncs (the o_proj waits on the secondary), and a
`cudaSetDevice` per layer — ~1.4 s of pure overhead per attn-layer-invocation, with no CUDA-graph capture
to amortise it (`model_parallel` forces eager). So TP-attention here **buys ~2× context at a steep
prefill/decode cost**; use it when you need 150–213K context on 2×V100 and can eat a slow prefill.

**262K still doesn't fit** (~213K max): at 262K the primary needs 5.95 GiB but has 4.70 GiB free — the
primary is weight-heavy, so full 262K also needs the **MLP-split rebalance** (move ~0.8 GiB of MLP weight
to the idle secondary — the open 9216 illegal-access bug). Shrinking `--prefill-chunk` only recovers ~0.5
GiB (the GDN state images are a ~1.5 GiB fixed overhead).

**One load bug fixed here** (`53976384`): the binder validated the `QkvHeadHalf` shard as a *column* split
(6144 > cols 5120) and aborted the load; added the row-split validation branch.

**Perf-recovery ideas (future, uncertain):** hoist the block-table mirror to once/step (it's
layer-invariant), replicate q/k norms on rank 1 at load to kill per-layer norm copies, mirror positions
once/step, and improve stream overlap — but the eager per-layer cross-device round-trip is fundamental to
this design, so a full recovery to tp-off prefill speed is unlikely without CUDA-graph capture of the
cross-device schedule.

### TP-R: full 262K + prefill recovered (both fixed)

The TP-A5 caveats above were **both resolved**. Full 262K is the minimum-viable target, so:

**262K now fits (TP-R1 — MLP rebalance).** The primary is weight-heavy, so shift MLP intermediate to the
idle secondary via a shared, env-overridable split accessor (`NINFER_MLP_PRIMARY`). The old "9216 faulted"
note was a *heavier*-primary trial in the wrong direction; a **lighter** primary is clean. Under TP the
default is now 4096 (secondary takes 13312), which drops primary weights 10.76 → 8.61 GiB and lifts
free-after-weights 4.70 → 6.85 GiB. The full 262144 KV (4.12 GiB) then loads and generates with 911 MiB to
spare. Turnkey — no extra flag needed:

    NINFER_TP_ATTENTION=1 --devices 1,2 --max-context 262144 --kv-capacity 262144 --kv-dtype int8

**Prefill recovered 6.1× (TP-R2 — flash for 12q/2kv).** Env-gated per-phase timing
(`NINFER_TP_PROFILE`) pinned the 7.6× regression precisely: over 64 layer-invocations the **attention
kernel was 8.29 s** while unpack/norm/gate/copy were each ~0.01 s — the strided-copy theory was wrong. Root
cause: `volta_flash_route_possible` excluded 12-head geometry, so the per-card half fell back to the slow
standard prompt kernel while tp-off's 24q/4kv used the fused flash path. 12q/2kv is GQA ratio 6 (same as
24q/4kv), so a `VoltaFlashTiling<CausalD256H12Kv2> = {ncols2 2, ncols1 16}` instantiation drops straight
in. Result: attention 8.29 → 0.18 s, **prefill 78 → 474 tok/s** (now 1.26× of the tp-off 597, was 7.6×
slower), greedy output **byte-identical** to tp-off on a flash-exercising prompt.

**Final state (turnkey `NINFER_TP_ATTENTION=1`, MLP auto-4096):**

| Metric | tp-off | tp-on (final) |
|---|---|---|
| Max context | ~98K | **262144 (full)** |
| Prefill @41K | 597 tok/s | 474 tok/s (1.26×) |
| Greedy parity | — | **exact** |
| Primary weights / KV | 11.09 / 132 MiB | 8.61 GiB / 66 MiB |

Remaining: decode at the 4096 split is ~14 tok/s (the split unbalances the MLP shards; MTP would lift it,
or a context-aware split could balance better below 262K). The projection GEMM (0.51 s) is now the largest
TP phase — the next prefill lever if wanted.
