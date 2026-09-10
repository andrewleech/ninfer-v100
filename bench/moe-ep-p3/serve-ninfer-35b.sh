#!/usr/bin/env bash
# Launch ninfer-serve for the Qwen3.6-35B-A3B dual-card expert-parallel target (both V100s, idx 1,2).
# Runs the worktree binary in the v100ninfer container (host lacks the ffmpeg libs). Foreground; Ctrl-C
# to stop. All knobs are env vars so the P3 harness can relaunch with different MTP / concurrency legs.
#
#   SPEC=mtp|none   DRAFT=3   CONC=1   CTX=255000   PORT=8080   MODEL_ID=ninfer-35b   KV=int8
#   TP_ATTENTION=1 enables the validated two-card text-attention split. It is on by default for
#   this 35B dual-V100 launcher; set TP_ATTENTION=0 only when reproducing the prior placement.
#
# Examples:
#   ./serve-ninfer-35b.sh                # production: 255K, MTP-3, TP attention, int8 KV
#   SPEC=none ./serve-ninfer-35b.sh      # raw decode / non-speculative diagnosis
#   TP_ATTENTION=0 ./serve-ninfer-35b.sh # reproduce the former single-card-attention placement
set -euo pipefail
ROOT=/home/anl/v100
ART=/models/Qwen3.6-35B-A3B-NInfer/qwen3_6_35b_a3b.ninfer

SPEC="${SPEC:-mtp}"          # mtp | none
DRAFT="${DRAFT:-3}"
CONC="${CONC:-1}"
CTX="${CTX:-255000}"
PORT="${PORT:-8080}"
MODEL_ID="${MODEL_ID:-ninfer-35b}"
KV="${KV:-int8}"
PREFILL_CHUNK="${PREFILL_CHUNK:-2048}"
STATS_MS="${STATS_MS:-2000}"
EXTRA="${EXTRA:-}"
TP_ATTENTION="${TP_ATTENTION:-1}"

spec_args=()
if [ "$SPEC" = "mtp" ]; then
  spec_args=(--spec mtp --draft-tokens "$DRAFT")
fi

tp_env=()
if [ "$TP_ATTENTION" = 1 ]; then tp_env=(-e NINFER_TP_ATTENTION=1); fi

echo "serve-ninfer-35b: spec=$SPEC draft=$DRAFT conc=$CONC ctx=$CTX kv=$KV tp_attention=$TP_ATTENTION port=$PORT model-id=$MODEL_ID"
exec docker run --rm --gpus all --network host \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  "${tp_env[@]}" \
  -v "$ROOT/ninfer-v100-moe-wt":/src:ro \
  -v "$ROOT/models":/models:ro \
  -v "$ROOT/.nvjitcache":/root/.nv/ComputeCache \
  --ipc=host --shm-size=8g -w /src/build-v100 \
  v100ninfer:cu128 \
  ./apps/ninfer-serve "$ART" \
    --devices 1,2 --kv-dtype "$KV" \
    --max-context "$CTX" --kv-capacity "$CTX" \
    --max-concurrency "$CONC" \
    --prefill-chunk "$PREFILL_CHUNK" \
    --log-stats-interval-ms "$STATS_MS" \
    --model-id "$MODEL_ID" \
    --max-request-mib "${MAX_REQ_MIB:-128}" \
    --host 0.0.0.0 --port "$PORT" \
    "${spec_args[@]}" $EXTRA
