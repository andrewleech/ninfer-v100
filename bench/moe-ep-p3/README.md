# 35B dual-card EP — P3 depth benchmark (to 262K, with MTP)

Serve-based A/B for the dual-card expert-parallel ninfer 35B (Qwen3.6-35B-A3B) vs the matched
llama.cpp 35B across a prompt-depth ladder **all the way to 262K**, single-stream. Reported as
wall-clock **s/call** plus **prefill tok/s** (from streamed time-to-first-token) and **decode
tok/s** (a per-token rate). This is a self-contained harness under `bench/moe-ep-p3/`; it does not
touch the project's `ninfer_bench` suite in `bench/` (see below).

**MTP is measured, not skipped.** ninfer runs MTP-on and MTP-off legs. MTP is a **non-parity axis**:
the llama.cpp serve runs MTP-off at 262K (its draft OOMs there), so the honest comparison is
ninfer-MTP-off vs llama, with ninfer-MTP-on reported as the extra single-stream win.

## Why serve-based (and why TTFT is split out)
The client streams each request and records **TTFT separately from decode**, so the prefill-depth
curve is pure prefill — output-length variance never contaminates it (that variance is exactly what
disqualified carbon's real-session wall-clock data as a depth curve). `max-new` is pinned (64) and
each response's real `usage.completion_tokens` is recorded (not a char/4 estimate — the 35B has a
different vocab so estimates are ~30% off). Same client + same ladder for both engines = clean A/B.

## Files
- `serve-ninfer-35b.sh` — dual-card EP ninfer-serve launcher (SPEC=mtp|none, CONC, CTX, PORT).
- `serve-llama-35b.sh`  — matched native `llama-server` (byte-identical to the router's catalog
  line 126: SM70 build, IQ4_XS GGUF, -sm layer -ts 1,1, q5_1 KV, -c 262144, MTP OFF, nothink jinja).
- `bench_depth.py` — depth-sweep client: waits for TRUE model-load (a real completion, not `/health`
  — cold 35B EP load is ~4-5 min), reads model-id from `/v1/models`, calibrates tokens/unit, streams
  one request per depth capturing TTFT + decode + `usage` → CSV. No third-party deps.
- `run-all-legs.sh` — full A/B: 3 legs strictly SEQUENTIAL (each wants both V100s): llama-nomtp →
  ninfer-nomtp → ninfer-mtp, with a cards-free gate between legs, then a summary table. `SKIP_LLAMA=1`
  to drop the llama leg.
- `run-ninfer-legs.sh` — just the two ninfer legs.
- `analyze.py` — final report: per-depth prefill/decode, ninfer-vs-llama ratios, the decode-vs-depth
  falloff cross-check, MTP uplift, and an optional overlay of carbon's real WALL-CLOCK points.

## Run (inside a coordinated titan-router both-cards window; llama-swap stopped)
```bash
DEPTHS=8192,16384,32768,65536,131072,196608,258048 MAXNEW=64 \
  bench/moe-ep-p3/run-all-legs.sh
python3 bench/moe-ep-p3/analyze.py bench/moe-ep-p3/results/ \
  --real /tmp/qwen35b-llamacpp-real-latencies.csv
```

## The headline question
carbon's llama fit implies decode falls ~104 → ~43 tok/s shallow→deep as KV grows (>2× falloff).
ninfer's shallow decode is 121 (no-MTP) / 175 (MTP); the sweep shows where it lands at 258K and
whether ninfer **holds decode at depth better than llama**. `analyze.py` prints the retention %.

## carbon real-points overlay caveats (titan-router, 2026-09-07)
- `output_tokens_est` in that file is chars/4 (~30% off) and unrecoverable (different vocab, no
  logged completion_tokens) — **only the wall-clock-seconds column is ground truth**. Overlay it as
  a shape / "does synthetic predict real wall-clock" check; do NOT fit against its token column.
- Weight its rows by configuration: 15 of 32 are map_reduce clustered at 62–79K, so the mid-range is
  over-represented for non-depth reasons.

## Alternative: `ninfer_bench` (artifact-direct cross-check)
The project's own `bench/ninfer_bench` measures prefill/decode t/s directly against the artifact
(pp/tg matrix, exact corpus token counts, `--mtp-draft-tokens`, no serve/HTTP overhead). If it
supports dual-card (`--devices 1,2`), it's a good artifact-direct cross-check of the ninfer-side
serve numbers — but it can't measure the llama side, so the serve-based harness here remains the
A/B of record. See `../README.md`.
