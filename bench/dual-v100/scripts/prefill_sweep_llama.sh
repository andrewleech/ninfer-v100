#!/usr/bin/env bash
# llama.cpp prefill sweep (comparison baseline) via llama-bench, token-matched to the ninfer sweep.
# Reproduces the "llama.cpp" column of docs/PREFILL-BENCHMARK.md.
#
#   LLAMA_BIN=/path/llama.cpp/build/bin  GGUF=/path/Qwen3.8-27B-Q6_K.gguf  TOKENS=516,3582,...  \
#     ./prefill_sweep_llama.sh
#
# Pin: llama.cpp b10453 (commit 4df29be4f4c3673f428170fda944a5b19f743bb8), built for SM_70.
# KV q5_1 matches the current doc; pass KV=q8_0 for the matched-KV comparison. At 221K add UB="-ub 2048 -b 2048"
# (llama-bench's default whole-prompt compute buffer OOMs at that length on 16 GB).
set -euo pipefail
LLAMA_BIN="${LLAMA_BIN:?set LLAMA_BIN=/path/llama.cpp/build/bin}"
GGUF="${GGUF:?set GGUF=/path/to/Qwen3.8-27B-Q6_K.gguf}"
TOKENS="${TOKENS:?set TOKENS=516,3582,14342,58442,117892,221592 (the exact counts ninfer reported)}"
KV="${KV:-q5_1}"; UB="${UB:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
export LD_LIBRARY_PATH="$LLAMA_BIN:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
"$LLAMA_BIN/llama-bench" -m "$GGUF" -ngl 99 -sm layer -ts 1/1 \
  -ctk "$KV" -ctv "$KV" -p "$TOKENS" -n 0 -r 2 $UB
