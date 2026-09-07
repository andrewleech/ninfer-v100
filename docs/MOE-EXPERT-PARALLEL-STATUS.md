# Dual-card expert-parallel MoE (Qwen3.6-35B-A3B) — status

**Branch:** `feature/moe-expert-parallel` (off `feature/dual-v100-nvlink`).
**Goal:** run the 22 GB 35B-A3B MoE across 2× V100-16GB via **expert-parallel** (256 routed
experts sharded 128/128), pure-GPU (no CPU offload), for concurrent throughput at long context.
Removes the prior "hard no" (`supports_graph_parallel=false`, 20.4 GiB > 16 GB, no host offload).

## Design (the crux)
MoE token output = `residual + shared(x) + Σ_{e∈top8} alpha_e·expert_e(x)`. Experts are independent
and the combine is a **sum**, so each card computes the **partial sum over the experts it owns**;
total = `residual + shared + partial0 + partial1`. This **additive reduce** is the same overlap
pattern as the 27B dense `post_mixer_graph` — only one BF16 `[hidden,T]` partial per card crosses
NVLink; the token permutation stays *within* each card.

- Experts are contiguous row-partitioned: 0–127 = `routed_gate_up[0,131072)` / `routed_down[0,262144)`;
  128–255 = the upper halves. New `RowBand` shard axis; each card materializes ~11 GB straight from
  the mmap (never stages the whole tensor).
- `run_sparse_moe_graph` mirrors `post_mixer_graph` (`stream_for_rank`/`fence_for_rank`/UVA peer copy/
  `residual_add_two`). Router is recomputed independently per card over UVA peer pointers to the
  replicated router weights (deterministic → identical top-8). Attention/GDN/router/shared/embed/head
  are **replicated on the primary** (card 1 is the heavier side).
- `decode` + `small_t` kernels take an expert-range `[lo,hi)`; out-of-range routed warps contribute
  exact zero; the secondary excludes the shared expert **and** the residual (primary adds those once).
- Gated by traits: `supports_graph_parallel=true`, `graph_parallel_attention=false` (attention stays
  single-card in P1), `graph_parallel_post_mixer_is_moe=true`. The 27B is unchanged. Multi-device
  disables CUDA-graph capture (eager decode) — same as the 27B today.

## P1 — DONE + VALIDATED (2026-09-07)
Commits `023d264e` (impl, 16 files) + `2094e4cd` (review fixes). Adversarial review: **sound** on
reduce-math + kernel-range; 3 medium workspace-sizing findings **fixed** (tp_attention gated on
`graph_parallel_attention`; `sparse_moe_partial_workspace_capacity_bytes` bounded to the small-T band
it actually uses → the two live `[hidden,T]` partials fit the reserved budget on both arenas).

- **Build:** green — `ninfer` + `ninfer-serve` link (sm_70).
- **GPU validation (dual-card, `--devices 1,2`):** 22 GB **sharded** (card1 = 11.5 GiB, card2 =
  9.2 GiB — the asymmetric EP split), loads clean, **coherent output** ("…capital of France? Paris"),
  **~118 tok/s single-stream** decode. The EP additive-reduce + expert-range kernels are numerically
  correct end-to-end. **No MTP** (plain decode).
- Deferred from P1: batched/`small_t` EP correctness (B=2..8) not yet run on hardware — folds into the
  P3 sweep (a concurrency-bench attempt failed on a harness bug — wrong served model-id + querying
  before the ~4.4 min model-load finished — then llama-swap reclaimed a card).

## Benchmark target (from cc-local-compact's matrix, via titan-router)
Real workload ≈ **180K input**. llama.cpp dual-card 35b = **174–194 s/call @ ~180K** → **prefill-
dominated** regime. Parity config for the A/B (llama): `-sm layer -ts 1,1` devices 1,2, IQ4_XS
all-GPU, q5_1 KV, `-c 262144`, **MTP OFF** (its draft ctx OOMs at 262K). So the fair comparison is
**no-MTP on both sides** (ninfer is already no-MTP; MTP-on is a bonus leg later). llama `--metrics`
reports `prompt_per_second` / `predicted_per_second` in the timings block.

Harness notes for P3: ninfer serve exposes separated counters `RuntimeStats.computed_prefill_tokens`
/ `committed_decode_tokens` / `running_requests` (`include/ninfer/types.h:747-753`), emitted via
`--log-stats-interval-ms`. Routes: OpenAI `/v1/*`, Anthropic `/v1/messages`, `/health` (no auth).
Host has python3/jq/curl. **Get the model-id from `/v1/models`** (the guess `qwen3_6_35b_a3b` was
wrong). **Wait for true model-load (~4.4 min), not just `/health`.**

## Next
- **P2:** grouped-prefill EP (the critical lever for a 180K prefill — P1 falls back to chunked
  `small_t`) + MTP EP (MTP MoE currently stays whole on card 0).
- **P3:** build the harness above, then a coordinated back-to-back both-cards window vs llama.cpp
  (depth×B grid, ~180K-centric, MTP-parity), scheduled with claude-net `titan-router`.
- **P4:** `serve-ninfer-35b-v100.sh` (dual-card, `--max-concurrency`) + model-router cutover.

**GPU access:** always coordinate a BOTH-cards window with claude-net `titan-router` — llama-swap
auto-spawns models on inbound requests, so a card can be reclaimed mid-window.
