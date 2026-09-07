#!/usr/bin/env bash
# P3 orchestrator (ninfer side): run the depth sweep against the 35B dual-card EP serve for both the
# MTP-on and MTP-off legs, tearing the container down between them. Fire this INSIDE a coordinated
# titan-router both-cards window (llama-swap stopped). The llama.cpp comparison leg is run separately
# with the same bench_depth.py against the llama serve endpoint (see README).
#
#   DEPTHS=...  MAXNEW=128  CTX=262144  OUTDIR=./results  ./run-ninfer-legs.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEPTHS="${DEPTHS:-8192,16384,32768,65536,131072,196608,258048}"
MAXNEW="${MAXNEW:-128}"
CTX="${CTX:-262144}"
PORT="${PORT:-8080}"
OUTDIR="${OUTDIR:-$HERE/results}"
LOADTO="${LOADTO:-600}"
mkdir -p "$OUTDIR"

run_leg() {
  local spec="$1" label="$2"
  local cname="ninfer35b-bench-$spec"
  echo "=== leg: $label (spec=$spec) ==="
  docker rm -f "$cname" >/dev/null 2>&1 || true
  # Detached serve; same flags as serve-ninfer-35b.sh but named + backgrounded so we can stop it.
  SPEC="$spec" CTX="$CTX" PORT="$PORT" \
    docker run -d --rm --name "$cname" --gpus all --network host \
      -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
      -v /home/anl/v100/ninfer-v100-moe-wt:/src:ro \
      -v /home/anl/v100/models:/models:ro \
      -v /home/anl/v100/.nvjitcache:/root/.nv/ComputeCache \
      --ipc=host --shm-size=8g -w /src/build-v100 \
      v100ninfer:cu128 \
      ./apps/ninfer-serve /models/Qwen3.6-35B-A3B-NInfer/qwen3_6_35b_a3b.ninfer \
        --devices 1,2 --kv-dtype int8 --max-context "$CTX" --kv-capacity "$CTX" \
        --max-concurrency 1 --prefill-chunk 2048 --log-stats-interval-ms 2000 \
        --model-id ninfer-35b --max-request-mib 128 --host 0.0.0.0 --port "$PORT" \
        $( [ "$spec" = mtp ] && echo "--spec mtp --draft-tokens 4" ) >/dev/null

  # bench_depth.py waits for TRUE model load itself (up to LOADTO).
  python3 "$HERE/bench_depth.py" --base "http://127.0.0.1:$PORT" \
    --out "$OUTDIR/$label.csv" --depths "$DEPTHS" --max-new "$MAXNEW" \
    --label "$label" --load-timeout "$LOADTO" || {
      echo "leg $label FAILED; serve logs:"; docker logs --tail 40 "$cname" || true; }

  echo "--- MTP acceptance / stats from serve log (leg $label) ---"
  docker logs "$cname" 2>&1 | grep -iE "accept|mtp|throughput|decode|prefill" | tail -15 || true
  docker stop "$cname" >/dev/null 2>&1 || true
  sleep 5
}

run_leg mtp  ninfer-mtp
run_leg none ninfer-nomtp
echo "=== done. results in $OUTDIR ==="
ls -la "$OUTDIR"
