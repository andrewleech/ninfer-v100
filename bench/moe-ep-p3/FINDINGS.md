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
