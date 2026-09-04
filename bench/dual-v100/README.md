# Dual-V100 benchmark harness (reproducibility pack)

Everything needed to reproduce the numbers in [`docs/PREFILL-BENCHMARK.md`](../../docs/PREFILL-BENCHMARK.md):
the exact prompt data, the exact runtime arguments for both engines, the pinned llama.cpp version,
and the quality harnesses. Speed is not bit-deterministic (see [Determinism](#determinism)); the
**method and data are**, and greedy outputs / quality scores are.

## Host

titan — 2× Tesla V100-SXM2-16GB (Volta `sm_70`), NVLink `NV6`, CUDA 12.8, driver 580. Card 0 is a
non-participating RTX A2000; the V100s are `nvidia-smi` indices **1,2** under `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
CUDA 12.8/SM_70 objects can't be compiled by titan's host toolchain, so ninfer runs inside the
`v100ninfer:cu128` container (`nvidia/cuda:12.8.1`, which also carries nvcc 12.8 + ffmpeg runtime).
The binaries run natively (glibc forward-compat).

## Models

| role | artifact | weights | KV | sha256 |
|---|---|---|---|---|
| ninfer | `Qwen3.8-27B-NInfer/qwen3_8_27b.ninfer` | groupwise-int **~4.7-bit** | int8 group-64 | `eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e` |
| llama (Q6, unmatched) | `Qwen3.8-27B-Q6_K.gguf` | **Q6_K** 6.5-bit | q5_1 | `562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727` |
| llama (Q4, matched) | `Qwen3.8-27B-UD-Q4_K_XL.gguf` | **UD-Q4_K_XL** ~4.5-bit | q5_1 | `3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e` |
| llama MTP draft | `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` | Q4_0 nextn head | — | `50d9ce5a6da381bbcfb31061cf73df94a90e6faf8efeddee379a9cb8f1501c6e` |

- ninfer artifact: <https://huggingface.co/neroued/Qwen3.8-27B-NInfer> (`qwen3_8_27b.ninfer`).
- llama GGUFs (Q6_K, UD-Q4_K_XL, MTP draft): <https://huggingface.co/unsloth/Qwen3.8-27B-GGUF>.
- **llama.cpp pinned to [`b10453`](https://github.com/ggml-org/llama.cpp/commit/4df29be4f4c3673f428170fda944a5b19f743bb8)**
  (commit `4df29be4`, 2026-08-16), built for SM_70. All comparisons use this exact build.

> Two comparisons are published in `docs/PREFILL-BENCHMARK.md`: the original **Q6_K** run (not
> weight-matched — ninfer ~4.7-bit vs 6.5-bit) and the **matched-weight Q4_K_XL** run (MTP-vs-MTP at
> 262K). llama Q4+MTP fits 262K only at **q5_1** KV (q8_0+MTP OOMs); ninfer runs int8 KV + MTP.

## Data (`data/`)

- `p_00512.json … p_246k.json` — token-matched chat prompts (bare `[{role,content}]` message lists)
  at **516 / 3,582 / 14,342 / 58,442 / 117,892 / 221,592** tokens (as counted by ninfer's tokenizer;
  llama is fed the same counts via `-p N`). Synthetic paragraph filler, deterministic content.
- `codebase_ctx.txt` — ~225K-token concatenation of this repo's source (`// ===== FILE: … =====`
  markers), the shared prefix for the multi-turn quality probe.
- `turns.json` — 10 scripted agentic-coding turns (verifiable gold strings + open keyword sets) over
  `codebase_ctx.txt`.

## Runners (`scripts/`)

| script | measures | doc section |
|---|---|---|
| `prefill_sweep_ninfer.sh` | ninfer CLI prefill tok/s, short→full | Prefill table (ninfer col) |
| `prefill_sweep_llama.sh` | llama-bench prefill tok/s, token-matched | Prefill table (llama col) |
| `quality_multi_turn.py` | multi-turn agentic accuracy over a reused long prefix | (quality) |
| `quality_ruler.py` | RULER long-context retrieval scores | (quality) |

**Model provenance.** Both engines' weights are **official published artifacts, sha256-verified**
(table above) — the ninfer `.ninfer` from neroued, the llama GGUFs from Unsloth. Nothing was
quantized/converted locally.

### Matched-weight (Q4) MTP-vs-MTP run — exact commands

Results in `results/`. llama Q4 served (text-only, thinking on, MTP at 262K — needs q5_1 KV, q8
OOMs):

```
llama-server -m Qwen3.8-27B-UD-Q4_K_XL.gguf -ngl 99 -fa on -sm layer -ts 1/1 \
  -ctk q5_1 -ctv q5_1 -c 262144 --kv-unified --cont-batching --parallel 1 \
  --spec-type draft-mtp --spec-draft-n-max 3 --jinja --chat-template-file qwen3-template-think.jinja
#   env: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2  LD_LIBRARY_PATH=<llama bin>:<cuda>
```

- **Quality:** `quality_multi_turn.py --base_url … --ctx data/codebase_ctx.txt --turns data/turns.json
  --effort medium --max_gen 8000` — same as the ninfer run in `results/quality_ninfer_4.7bit_medium.json`.
- **MTP decode t/s:** POST each `data/p_{00512,65k,131k}.json` (append `/no_think`) with
  `max_tokens=128, temperature=0`; read the server's `eval time … tokens per second`. ninfer side is
  `results/decode_ninfer_raw_mtp.csv`.

All prefill runs are **greedy** (`--greedy` / temp 0) and single-stream. Exact arguments are baked
into each script; the ninfer CLI line is:

```
apps/ninfer <model.ninfer> --messages <p.json> --greedy --max-new 1 \
  --max-context 262144 --kv-capacity 262144 --kv-dtype int8 --devices 1,2 --prefill-chunk 2048
#   env: CUDA_DEVICE_ORDER=PCI_BUS_ID NINFER_TP_ATTENTION=1 NINFER_MLP_PRIMARY=3072
```

and the llama-bench line (KV `q5_1`; `q8_0` for matched-KV; `-ub 2048 -b 2048` at 221K):

```
llama-bench -m <Q6_K.gguf> -ngl 99 -sm layer -ts 1/1 -ctk q5_1 -ctv q5_1 -p <N> -n 0 -r 2
#   env: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2
```

### Example (on titan, in-container)

```bash
docker run --rm --gpus all -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -v $PWD:/src -v /home/anl/v100/models:/models -w /src/build-v100 v100ninfer:cu128 \
  bash -c 'NINFER=./apps/ninfer MODEL=/models/Qwen3.8-27B-NInfer/qwen3_8_27b.ninfer \
           OUT=/src/bench/dual-v100/ninfer_prefill.csv ../bench/dual-v100/scripts/prefill_sweep_ninfer.sh'
```

Quality (server must be up; drives any OpenAI-compatible endpoint):

```bash
python scripts/quality_multi_turn.py --base_url http://HOST:PORT/v1 --model <id> \
  --ctx data/codebase_ctx.txt --turns data/turns.json --out /tmp/mt.json --effort medium
```

## Determinism

- **Deterministic:** prompt data, exact arguments, model identity (sha256), llama.cpp commit. Greedy
  decoding (temp 0) makes **generated tokens and quality scores** reproducible run-to-run.
- **Not deterministic:** wall-clock **speed** carries ~a few % thermal/timing variance (HBM temp,
  clocks, scheduling) — reproducible *method*, not identical timings. Mitigate with repeats
  (`llama-bench -r`, or re-running the sweep) and report mean ± range. When isolating a change's
  effect, **interleave** the A/B arms (A,B,B,A) on warm cards — a single-shot A-then-B on a warming
  GPU fabricated a ~10 % phantom win once (see the rejected pipelining note in the prefill doc).
