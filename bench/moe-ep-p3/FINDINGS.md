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
