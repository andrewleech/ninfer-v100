# P3 — 35B dual-card EP depth benchmark: findings (2026-09-08)

Single-stream, fixed 64-token output, TTFT split from decode-rate so the prefill-depth curve is
clean (the control carbon's uncontrolled-output real sessions lacked). All three legs served on
both V100s at 253952 ctx (the MTP-fit cap), swept to ~250K actual prompt. `--no-thinking` both
engines. Raw CSVs in `results/`.

## Per-depth (prefill tok/s / decode tok/s)
| depth | llama-nomtp | ninfer-nomtp | ninfer-mtp |
|------:|:-----------:|:------------:|:----------:|
|   8K  | 3998 / 100  | 1142 / 114   | 1106 / 125 |
|  16K  | 4742 /  95  | 1082 / 109   | 1043 / 121 |
|  33K  | 4906 /  86  | 1049 /  95   |  982 / 124 |
|  66K  | 4172 /  72  |  965 /  73   |  898 / 100 |
| 133K  | 2949 /  53  |  819 /  53   |  764 /  56 |
| 200K  | 3022 /  41  |  681 /  42   |  658 /  53 |
| 250K  | 3805 /  36  |  606 /  36   |  592 /  43 |

## The headline (and it refutes the going-in hypothesis)
**llama.cpp PREFILL is 3–6× faster than ninfer's grouped-prefill EP on this 35B MoE** (ninfer vs
llama prefill ratio: 0.29 → 0.16 as depth grows; ninfer degrades harder with depth). Because prefill
dominates total latency at long context, ninfer's **s/call is much worse**: at 250K, ninfer-nomtp
**415 s** / ninfer-mtp **424 s** vs llama **67.5 s**.

The going-in premise ("ninfer should have a prefill advantage") is **false for the 35B MoE** at these
depths. (It held for the *dense* 27B TP path; the MoE grouped-prefill EP is a different animal.)

## Where ninfer does win
- **Decode roughly matches llama** MTP-off (1.00–1.14×), slightly ahead shallow, converging at depth.
- **MTP is the real ninfer win**: 1.06–1.37× over its own no-MTP decode; at 250K ninfer-mtp **43** vs
  llama **36** tok/s (1.2×). And ninfer runs **int8 KV** (higher precision than llama's q5_1) and
  **MTP at full context** — llama's MTP draft needs +4.5 GB and only fits ≤64K.
- **Decode-vs-depth falloff** (carbon's cross-check) is similar for both — all retain ~32–36% of
  shallow decode at 250K (~3× falloff, KV-bound). Confirmed, not a differentiator.

## Carbon real-session overlay (sanity)
Carbon's 180K real sessions ran 161–255 s with **long** outputs (1925–6423 tokens). That decomposes
as llama prefill (~66 s at 196K here) + long decode — consistent with our llama leg, and it's llama's
fast prefill that keeps those real latencies down. For the summariser workload (long input, 2–6 k
output), ninfer's 4× prefill deficit dominates even with its better decode: 180K + 4 k out ≈
ninfer-mtp ~370 s vs llama ~164 s.

## Recommendation
**Do NOT swap the llama.cpp 35b out yet.** The EP work is a real milestone — the 35B now *runs* on
ninfer at full 262K dual-card (it structurally couldn't before), decode/MTP/int8-KV are genuine — but
llama.cpp is meaningfully faster end-to-end for this long-context workload because of prefill. The
cutover should wait on **prefill tuning**: ninfer prefill at 600–1140 tok/s is low and depth-sensitive
(likely EP-reduce/permutation overhead or a sub-optimal grouped path, not a hard compute wall), so
there is headroom. That's the P3 "throughput tune" lever, now clearly the critical one.

## Caveats
- Fixed 64-token output isolates prefill/decode rates cleanly but is not a real workload; a
  long-output leg would show the decode/MTP win more but not overturn the prefill conclusion.
- ~250K not full 262K (MTP OOMs at 262144 by ~70 MB; all legs capped at 253952 for a matched A/B).
- Single-stream only. Concurrency (ninfer's batched EP vs llama single-stream) is a separate axis not
  measured here and could favor ninfer for multi-agent serving.

---

# dp4a follow-up (2026-09-08): int8/__dp4a grouped-prefill kernel

The P3 recommendation ("prefill tuning is the lever") was acted on: the routed-expert grouped-prefill
GEMMs (gate/up Q4 + down Q5/Q6, ~74% of MoE prefill) were ported from scalar-fp32-fmaf SIMT to
int8/`__dp4a` — the same technique that took the 27B *dense* prefill compute-bound. Result
`results/ninfer-dp4a.csv`, validated in a coordinated titan-router window.

**Correctness:** dp4a vs `NINFER_MOE_PREFILL_SCALAR=1` (the retained scalar path) — byte-identical
greedy output, 100% token agreement (int8-activation quant flipped no argmax). **Arena:** fits at
**full 262144** non-MTP (633 MiB free-after-startup) — better than P3's 253952 cap.

| depth | dp4a pf | scalar pf | llama pf | dp4a/scalar | llama/dp4a | dp4a s/call | llama s/call |
|------:|--------:|----------:|---------:|:-----------:|:----------:|------------:|-------------:|
|   8K  |  1871   |   1142    |  3998    |  **1.64×**  |   2.14×    |     4.9     |     2.7      |
|  16K  |  1789   |   1082    |  4742    |    1.65×    |   2.65×    |     9.8     |     4.1      |
|  33K  |  1718   |   1049    |  4906    |    1.64×    |   2.86×    |    19.9     |     7.5      |
|  66K  |  1500   |    965    |  4172    |    1.55×    |   2.78×    |    45.0     |    16.7      |
| 133K  |  1162   |    819    |  2949    |    1.42×    |   2.54×    |   115.6     |    46.3      |
| 200K  |   889   |    681    |  3022    |    1.30×    |   3.40×    |   226.5     |    67.7      |
| 250K  |   767   |    606    |  3805    |    1.27×    |   4.96×    |   328.0     |    67.5      |

decode tok/s unchanged vs scalar (dp4a touches only prefill): 114/107/93/72/53/42/36.

**Verdict:** a real, consistent **1.27–1.65× prefill win** that narrows the llama gap from 3–6× to
~2.1–5× — but **does NOT overturn "keep llama canonical" for the deep-context summariser**. The win
tapers with depth (1.64× @8K → 1.27× @250K), and at the real ~180–200K depth llama is still ~3.4×
faster (226 s vs 68 s). The taper is the key diagnostic: the MoE GEMM is no longer the deep-depth
bottleneck — the **16 full-attention layers' O(N²) prefill + the per-chunk EP NVLink reduce** now
dominate. That (not the MoE kernel) is the next prefill lever. The swap reopens only at shallow/mid
context (gap 2.1× @8K, 2.9× @33K).

---

# Prefill phase split (2026-09-08): WHERE the deep-context time goes — MEASURED

The "attention is next" diagnostic above was *inferred* from the dp4a taper. Now **measured**
directly: `NINFER_DECODE_PROFILE=1` (syncs the primary stream around each mixer/mlp call, accumulates
wall-clock into attn/gdn/mlp buckets) on the dp4a binary, dual-card EP, ctx 253952, `MAXNEW=1` so the
buckets are ~pure prefill. Harness `profile-prefill.sh`. `attn` = attention (gather+flash+mask+
converts); `gdn` = the linear/GDN-attention layers; `mlp` = MoE routed experts **+ EP NVLink reduce**.

| depth | attn | gdn | mlp (MoE+EP) | total | attn s/1K | gdn s/1K | mlp s/1K |
|------:|-----:|----:|-------------:|------:|:---------:|:--------:|:--------:|
|  64K | 20.3s (43.7%) | 13.5s (29.1%) | 12.7s (27.3%) |  46.4s | 0.32 | 0.21 | 0.20 |
| 128K | 63.8s (55.1%) | 26.8s (23.2%) | 25.3s (21.8%) | 115.9s | 0.50 | 0.21 | 0.20 |
| 192K |142.8s (63.3%) | 42.2s (18.7%) | 40.6s (18.0%) | 225.6s | 0.74 | 0.22 | 0.21 |

**Conclusive: attention is the deep-context lever.** It is 63% of prefill at 192K and the **only
superlinear term** — its s/1K-token *rises* 0.32→0.50→0.74 (the O(N²) signature), while `gdn` and
`mlp` are **flat at ~0.21 s/1K = clean O(N) linear** and never dominate. dp4a already took the linear
MoE term; further MoE/EP work has diminishing returns at depth. **The EP NVLink reduce lives inside
the small linear `mlp` bucket → overlapping it (the unused `transfer_stream_for_rank`) is a minor
win, not the lever.** (Caveat: the profiler serializes the primary stream, so `mlp` loses EP overlap
and is *pessimistic* — attention's real share is even higher than shown.)

**The sharper puzzle for the next step:** attention already runs the *same* vendored llama.cpp
flash+tensor-core kernel (`flash_attn_ext_f16`), yet ninfer's attention **alone** (143s @192K)
exceeds llama's **whole** prefill (~68s @200K). So the gap is in **how the flash is driven** —
FP32-Q staging, a dense per-Q-block causal mask, tiling/occupancy for D256 on sm_70, and the
per-chunk serialized re-gather — **or llama uses a different V100 attention path for head_dim=256**.
Pinning that (a flash-internal gather/mask/flash/convert sub-breakdown) is the next measurement;
`profile-out/prefill-split.txt` has the raw buckets.

---

# Attention tiling experiments (2026-09-08): the tiling lever is EXHAUSTED

Acting on the split above, the hypothesis was that ninfer under-tiles the flash Q-block: it hardcodes
`ncols1 = 4` (kNcols = 32, the *minimum* Volta tile) for the 35B's 16q/2kv geometry, while llama.cpp's
own `switch_ncols1` picks `ncols1 = 8` (kNcols = 64) for prefill — which would halve the query-tile
count and thus the O(N²) KV re-streaming. Two builds, A/B'd against the `ncols1 = 4` baseline (attn
bucket **20.3 / 63.8 / 142.8 s** @ 64/128/192K) with `validate-ncols.sh`:

| variant | attn @64K | @128K | @192K | vs ncols1=4 |
|--------|:--------:|:-----:|:-----:|:-----------:|
| **ncols1=4 (baseline, committed)** | 20.3 | 63.8 | 142.8 | — |
| ncols1=8 (64-col, Q in shared) | 22.8 | 72.2 | 154.1 | **~8-13% SLOWER** |
| ncols1=8 + Q_in_reg=true | 32.9 | 114.6 | 245.3 | **~1.6-1.7× SLOWER** |

Both bigger tiles LOSE, so **`ncols1 = 4` (32-col, 2 blocks/SM) is optimal and is kept.** The mechanism:
- Plain 64-col: `shared_Q` scales with kNcols and busts Volta's 96 KB smem cap → occupancy drops
  2 → 1 block/SM. It got *slower*, not faster — proof the kernel is **occupancy-bound, not
  bandwidth-bound** (a BW-bound kernel would still gain from halving KV re-streaming at 1 block/SM).
- 64-col + Q_in_reg=true (Q in registers to free the smem): far *worse* — D256 puts 128 half2/thread
  of Q in registers → **register spill**, catastrophically register-bound.

**Conclusion:** the attention *tiling* lever is exhausted; the vendored MMA kernel at its best Volta
tiling can't reach llama's ~68 s whole-prefill, so llama's speed must come from a genuinely different
D256-on-Volta path (different KV layout, the `fattn-tile` kernel, or lower precision), not a tile size
this kernel can adopt. Remaining attention sub-levers are the per-chunk staging (re-gather, dense mask,
fp32 converts) — unmeasured but estimated small (~1-2 s each) against ~143 s of flash compute, so
low-yield. Correctness across both variants: coherent, ≥0.93 token-agreement (a tiling change reorders
the online-softmax fp adds, so greedy divergence is expected and benign — byte-identity over-specifies
here; reserve it for quant/no-op refactors like the dp4a int8-act path). Both changes reverted to
committed `ncols1 = 4` / `Q_in_reg = false`. **Overall verdict is unchanged: keep llama.cpp 35B
canonical.**
