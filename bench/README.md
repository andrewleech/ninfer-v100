# P3 — 35B dual-card EP depth benchmark (to 262K, with MTP)

Measures the dual-card expert-parallel ninfer 35B (Qwen3.6-35B-A3B) against the llama.cpp 35B across
a prompt-depth ladder **all the way to 262K context**, single-stream, reported as the metric the
llama reference is quoted in: **wall-clock s/call**, plus prefill tok/s (from time-to-first-token)
and decode tok/s.

**MTP is measured, not skipped** (user requirement). ninfer runs two legs — MTP-on and MTP-off. MTP
is a **non-parity axis**: the llama.cpp serve runs MTP-off at 262K (its draft context OOMs there),
so the honest comparison is ninfer-MTP-off vs llama, with ninfer-MTP-on reported as the additional
single-stream win.

## Files
- `serve-ninfer-35b.sh` — launch the 35B dual-card EP serve (foreground). Env knobs: `SPEC=mtp|none`,
  `DRAFT`, `CONC`, `CTX`, `PORT`, `MODEL_ID`, `KV`, `PREFILL_CHUNK`.
- `bench_depth.py` — depth-sweep client. Waits for TRUE model load (a real 1-token completion, not
  just `/health` — cold 35B EP load is ~4–5 min), reads the model id from `/v1/models`, calibrates
  tokens/unit, then streams one request per depth capturing TTFT (prefill) + decode timing. No deps.
- `run-ninfer-legs.sh` — orchestrator: runs both ninfer legs (MTP-on, MTP-off) back-to-back, tearing
  the container down between them, and greps MTP acceptance from the serve log.

## Depth ladder
Default: `8192,16384,32768,65536,131072,196608,258048`. Top depth is held ~4K under 262144 to leave
room for the chat template + the decode window (KV capacity = 262144).

## Running (inside a coordinated titan-router both-cards window)
llama-swap must be **stopped** for the window (titan-router does this) so nothing spawns onto cards
1/2. Then:

```bash
# ninfer legs (MTP-on + MTP-off):
DEPTHS=8192,16384,32768,65536,131072,196608,258048 MAXNEW=128 \
  bench/run-ninfer-legs.sh          # -> bench/results/ninfer-mtp.csv, ninfer-nomtp.csv

# llama leg (coordinated with titan-router, who runs the llama 35b serve with its standard config:
#   -sm layer -ts 1,1, IQ4_XS, q5_1 KV, -c 262144, MTP OFF, --metrics):
python3 bench/bench_depth.py --base http://127.0.0.1:<llama_port> \
  --out bench/results/llama-nomtp.csv \
  --depths 8192,16384,32768,65536,131072,196608,258048 --max-new 128 --label llama-nomtp
```

## Reference (llama.cpp 35b, carbon's 2026-09-07 matrix)
~174–194 s/call @ ~180K real input; ~304 s @ 228K — all-GPU dual-card, q5_1 KV, 262K, MTP OFF.
The regime that matters is long context (prefill-dominated), which is why P2 grouped-prefill EP
landed first.

## Single-stream numbers so far (CLI, T=163 toy prompt; real depths are what this harness produces)
- no-MTP decode 121 tok/s; MTP-on decode 175.7 tok/s (1.45×, 74.6% acceptance)
- prefill ~500 tok/s at T=163 (tiny — grouped-prefill EP throughput at real depth is the point of P3)
