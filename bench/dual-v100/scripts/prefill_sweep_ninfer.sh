#!/usr/bin/env bash
# ninfer prefill sweep (CLI path) — pure prompt-processing wall via --max-new 1, greedy.
# Emits <label>,<prompt_tokens>,<prefill_tok_s> for each token-matched prompt in ../data.
# Reproduces the "ninfer" column of docs/PREFILL-BENCHMARK.md.
#
#   NINFER=apps/ninfer  MODEL=/path/qwen3_8_27b.ninfer  ./prefill_sweep_ninfer.sh
#
# Requires a dual-V100 host; both cards free. Run from a build dir that contains apps/ninfer, or set
# NINFER to its absolute path. On titan this runs inside the v100ninfer:cu128 container (see README).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; DATA="$HERE/../data"
NINFER="${NINFER:-apps/ninfer}"
MODEL="${MODEL:?set MODEL=/path/to/qwen3_8_27b.ninfer}"
OUT="${OUT:-$PWD/ninfer_prefill.csv}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID NINFER_TP_ATTENTION=1 NINFER_MLP_PRIMARY=3072
DEVICES="${DEVICES:-1,2}"
echo "label,prompt_tokens,prefill_tok_s" > "$OUT"
for f in "$DATA"/p_00512.json "$DATA"/p_04k.json "$DATA"/p_16k.json \
         "$DATA"/p_65k.json "$DATA"/p_131k.json "$DATA"/p_246k.json; do
  label=$(basename "$f" .json | sed 's/p_//')
  "$NINFER" "$MODEL" --messages "$f" --greedy --max-new 1 \
    --max-context 262144 --kv-capacity 262144 --kv-dtype int8 \
    --devices "$DEVICES" --prefill-chunk 2048 > "/tmp/ninfer_$label.out" 2>&1
  tok=$(awk '/prompt tokens/ && !/reused/{print $NF}' "/tmp/ninfer_$label.out")
  spd=$(awk '/prefill speed/{print $(NF-1)}' "/tmp/ninfer_$label.out")
  echo "$label,$tok,$spd" | tee -a "$OUT"
done
