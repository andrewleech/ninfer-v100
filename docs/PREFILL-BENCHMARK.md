# Qwen3.8-27B on 2× V100: ninfer (dual-V100 NVLink) vs llama.cpp

Back-to-back throughput on the **same host, same model, same token counts**, small context
through near-full 262K. Prefill (prompt processing) and decode (generation).

- **Host:** titan — 2× Tesla V100-SXM2-16GB (Volta sm_70), NVLink.
- **Model:** Qwen3.8-27B (dense 27B).
- **ninfer** (this fork): groupwise-int weights (~4.7-bit), **int8 KV**, dual-card tensor-parallel
  attention (full-262K-capable), `--devices 1,2 --prefill-chunk 2048`.
- **llama.cpp:** **Q6_K** weights (6.5-bit), **q5_1 KV**, `-sm layer -ts 1/1` (production serving
  config), `llama-bench` (`-d <depth>` for decode-at-depth).
- Token counts matched exactly (llama `-p/-d N` = the `N` ninfer reported for each prompt).

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
- That lower precision has a **real quality cost on multi-turn / long-context** work that these
  speed numbers do not capture. A fully apples-to-apples comparison would run llama at a matched
  weight precision (`Q4_K_M` ~4.8-bit or `Q5_K_M`); prefill should stay ~parity and ninfer's decode
  lead should shrink toward its real (kernel + KV) size. **TODO: publish the matched-weight run.**

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

## Decode (tok/s, generation at context depth)

| context depth | ninfer raw | ninfer +MTP(3) | llama raw | ninfer-raw vs llama |
|--------------:|-----------:|---------------:|----------:|:-------------------:|
|           476 |   **31.9** |       **59.8** |      28.0 |      **+14 %**      |
|        58,402 |   **25.4** |       **47.5** |      19.4 |      **+31 %**      |
|       117,852 |   **20.3** |       **38.7** |      14.8 |      **+37 %**      |

ninfer wins raw decode at every depth, widening with context; MTP (self-speculative, draft-3)
adds ~1.9×. **But per the note above, much of this decode lead reflects the ~2.5× lighter weights,
not kernel superiority — it is a speed/quality trade.** (llama MTP not measured here; llama.cpp
also supports speculative decoding.)

## How prefill was closed

ninfer's prefill originally trailed llama ~1.3–1.7×. The gap was the GEMM algorithm, not the
hardware: llama's Volta build forces **MMQ (int8 `__dp4a`)**, compute-bound (~75 % of the integer
pipe), whereas ninfer's fp16 `mma.sync` kernel was **L1/shared-bound** (84.6 %, ~46 % compute).
Porting int8/dp4a with the decisive trick — **warp-cooperative tiling** (lane→row, warp→token) so
the activation operand is a single shared **broadcast** across the warp — took the Q4/Q5 prefill
GEMMs from ~21 to ~31–33 TFLOP/s (compute-bound). Correctness: 0.4–0.65 % vs the SIMT oracle
(expected int8-activation accuracy); dual-card greedy A/B (dp4a on vs off) identical on the cases
tested including a long-context needle retrieval. See `docs/DUAL-V100-BRINGUP-LOG.md`.

### Methodology notes

- Kernel + end-to-end measured with ncu, nsys, `test-backend-ops`, `llama-bench`, and the ninfer
  CLI/perplexity tools.
- The 221,592-token llama prefill point uses `-ub 2048 -b 2048` (matching ninfer's chunk);
  llama-bench's default whole-prompt compute buffer OOMs at that length on 16 GB (llama-server,
  which streams, does not).
- Decode measured at context depth via `llama-bench -d <depth>` and ninfer `--max-new 128` after a
  prompt of the matched length; ninfer MTP via `--spec mtp --draft-tokens 3`.
- Prefill is single-stream on both engines.
