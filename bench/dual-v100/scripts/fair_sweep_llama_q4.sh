#!/usr/bin/env bash
# Matched-weight fair comparison (llama side): Qwen3.8-27B UD-Q4_K_XL (~4.5-bit, matched to ninfer
# ~4.7-bit) + q8_0 KV (8-bit, matched to ninfer int8). Prefill + decode, token/depth-matched.
# MTP (nextn) via the separate draft GGUF is measured by fair_mtp_llama.sh (llama-bench can't do
# speculative). Pin: llama.cpp b10453 (4df29be4), SM_70.
#
#   LLAMA_BIN=/path/llama.cpp/build/bin  GGUF=/path/Qwen3.8-27B-UD-Q4_K_XL.gguf \
#   TOKENS=516,3582,14342,58442,117892,221592  DEPTHS=476,58402,117852  ./fair_sweep_llama_q4.sh
set -euo pipefail
LLAMA_BIN="${LLAMA_BIN:?set LLAMA_BIN=/path/llama.cpp/build/bin}"
GGUF="${GGUF:?set GGUF=/path/to/Qwen3.8-27B-UD-Q4_K_XL.gguf}"
TOKENS="${TOKENS:?token counts, comma-sep}"; DEPTHS="${DEPTHS:-476,58402,117852}"
KV="${KV:-q8_0}"     # matched to ninfer int8
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
export LD_LIBRARY_PATH="$LLAMA_BIN:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
echo "### PREFILL (Q4_K_XL, ${KV} KV) — token-matched"
"$LLAMA_BIN/llama-bench" -m "$GGUF" -ngl 99 -sm layer -ts 1/1 -ctk "$KV" -ctv "$KV" \
  -p "$TOKENS" -n 0 -r 2 -ub 2048 -b 2048
echo "### DECODE (Q4_K_XL, ${KV} KV) — at depth, 128 new tokens"
for d in ${DEPTHS//,/ }; do
  "$LLAMA_BIN/llama-bench" -m "$GGUF" -ngl 99 -sm layer -ts 1/1 -ctk "$KV" -ctv "$KV" \
    -d "$d" -n 128 -r 2 -ub 2048 -b 2048
done
