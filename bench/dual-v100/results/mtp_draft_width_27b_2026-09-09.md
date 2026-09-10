# Qwen3.8-27B NInfer MTP draft-width check — 2026-09-09

## Decision

Keep the production default of **three MTP draft tokens**. At a 198K-token
prompt it decoded 7.5% faster than two drafts. Two drafts had slightly higher
prefill throughput and draft acceptance, but that did not repay the extra
speculative rounds.

## Controlled configuration

- Host: titan, 2x Tesla V100-SXM2-16GB (GPU indices 1,2), CUDA 12.8.
- Model: `Qwen3.8-27B-NInfer/qwen3_8_27b.ninfer`.
- Production launcher: `model-router/serve/titan/serve-ninfer-27b-v100.sh`.
- Common settings: `CTX=262144`, `--devices 1,2`, `--kv-dtype int8`,
  `NINFER_TP_ATTENTION=1`, `NINFER_MLP_PRIMARY=3072`, single stream,
  production thinking/sampling defaults, greedy client request, and
  `max_tokens=128`.
- Prompt: the deterministic `greedy_probe.py` filler, requested at 196608
  tokens and counted by the server as 198174 prompt tokens.
- One cold-load run per arm. Treat small differences as directional until an
  interleaved warm A/B/B/A repeat confirms them.

## Results

| `--draft-tokens` | prefill tok/s | decode tok/s | accepted tok/round | acceptance | wall |
|---:|---:|---:|---:|---:|---:|
| 2 | **435.5** | 24.0 | 2.25 | **62.5%** | 460.46 s |
| 3 (production) | 412.3 | **25.8** | **2.54** | 51.3% | 485.66 s |

`decode tok/s` is the generated-token rate after prefill. The wall column
includes the long prompt prefill plus the 128-token completion, so two drafts
wins that particular short-output wall time by pre-filling faster even though
three drafts is the better continuing-generation setting.

## Raw server records

```
draft=2: prompt=198174 gen=128 ttft=455173ms prefill=435.5tok/s decode=24.0tok/s \
  wall=460.46s speculative=mtp 2.25tok/round (62.5%)
draft=3: prompt=198174 gen=128 ttft=480733ms prefill=412.3tok/s decode=25.8tok/s \
  wall=485.66s speculative=mtp 2.54tok/round (51.3%)
```
