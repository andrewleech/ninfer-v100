#!/usr/bin/env bash
# Launch ninfer-serve for the Qwen3.6-35B-A3B dual-card expert-parallel target (both V100s, idx 1,2).
# Runs the worktree binary in the v100ninfer container (host lacks the ffmpeg libs). Foreground; Ctrl-C
# to stop. All knobs are env vars so the P3 harness can relaunch with different MTP / concurrency legs.
#
#   SPEC=mtp|none   DRAFT=4   CONC=1   CTX=262144   PORT=8080   MODEL_ID=ninfer-35b   KV=int8
#
# Examples:
#   SPEC=mtp   ./serve-ninfer-35b.sh     # MTP-on leg
#   SPEC=none  ./serve-ninfer-35b.sh     # MTP-off leg
set -euo pipefail
ROOT=/home/anl/v100
ART=/models/Qwen3.6-35B-A3B-NInfer/qwen3_6_35b_a3b.ninfer

SPEC="${SPEC:-mtp}"          # mtp | none
DRAFT="${DRAFT:-4}"
CONC="${CONC:-1}"
CTX="${CTX:-262144}"
PORT="${PORT:-8080}"
MODEL_ID="${MODEL_ID:-ninfer-35b}"
KV="${KV:-int8}"
PREFILL_CHUNK="${PREFILL_CHUNK:-2048}"
STATS_MS="${STATS_MS:-2000}"
EXTRA="${EXTRA:-}"

spec_args=()
if [ "$SPEC" = "mtp" ]; then
  spec_args=(--spec mtp --draft-tokens "$DRAFT")
fi

echo "serve-ninfer-35b: spec=$SPEC draft=$DRAFT conc=$CONC ctx=$CTX kv=$KV port=$PORT model-id=$MODEL_ID"
exec docker run --rm --gpus all --network host \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
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
