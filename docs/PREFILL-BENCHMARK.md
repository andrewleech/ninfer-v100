# Prefill throughput: ninfer (dual-V100 NVLink) vs llama.cpp

Back-to-back prefill (prompt-processing) throughput on the **same host, same model,
same token counts**, small context through near-full 262K.

- **Host:** titan — 2× Tesla V100-SXM2-16GB (Volta sm_70), NVLink.
- **Model:** Qwen3.8-27B (dense 27B).
- **ninfer** (this fork): groupwise-int weights (~4.7-bit), **int8 KV**, dual-card
  tensor-parallel attention (full-262K-capable), `--devices 1,2 --prefill-chunk 2048`.
- **llama.cpp:** Qwen3.8-27B **Q6_K** weights (6.5-bit), **q5_1 KV**, `-sm layer -ts 1/1`
  (the production serving config), `llama-bench -p <N> -n 0`.
- Token counts are matched exactly: llama's `-p N` uses the same `N` ninfer reported for
  each prompt. Prefill throughput is content-independent (same FLOPs), so synthetic vs real
  tokens does not affect it.

| prompt tokens | ninfer tok/s | llama.cpp tok/s | ninfer vs llama |
|--------------:|-------------:|----------------:|:---------------:|
|           516 |    **588.5** |           545.9 |   **+7.8 %**    |
|         3,582 |        813.6 |       **828.0** |     −1.7 %      |
|        14,342 |    **801.3** |           783.2 |   **+2.3 %**    |
|        58,442 |    **656.6** |           640.5 |   **+2.5 %**    |
|       117,892 |    **525.5** |           514.6 |   **+2.1 %**    |
|       221,592 |        402.3 |       **407.7** |     −1.3 %      |

**Takeaway:** across the full context range the two engines are at parity within ±3 %, with
ninfer ahead through the practical long-document mid-range (14K–118K) and llama a hair ahead
at the extremes. ninfer reaches this **while carrying lower-precision weights (~4.7-bit vs
Q6_K's 6.5-bit)** and serving the full 262K single-session context — i.e. it matches a mature,
heavily-tuned engine on prefill without giving up its context-fit advantage.

## What made it competitive

ninfer's prefill originally trailed llama ~1.3–1.7×. The gap was the GEMM algorithm, not the
hardware: llama's Volta build forces **MMQ (int8 `__dp4a`)**, which profiles compute-bound
(~75 % of the integer pipe), whereas ninfer's fp16 `mma.sync` kernel was **L1/shared-bound**
(84.6 %, ~46 % compute). Porting the int8/dp4a path with the decisive trick — **warp-cooperative
tiling** (lane→row, warp→token) so the activation operand is a single shared **broadcast** across
the warp — took the Q4/Q5 prefill GEMMs from ~21 to ~31–33 TFLOP/s (compute-bound), closing the
gap end-to-end. See `docs/DUAL-V100-BRINGUP-LOG.md`.

### Methodology notes

- ncu, nsys, `test-backend-ops` and `llama-bench` were used for kernel- and end-to-end
  measurement. The 221,592-token llama point uses `-ub 2048 -b 2048` (matching ninfer's
  chunk); llama-bench's default whole-prompt compute buffer OOMs at that length on 16 GB,
  though the streaming llama-server does not.
- Prefill is single-stream on both engines. int8 activation quantization is applied by the
  dp4a path during prefill (8-bit, the same scheme llama's MMQ uses).
