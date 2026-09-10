#!/usr/bin/env bash
# Targeted single-depth flash-vs-staging split for ninfer's fa bucket. Run INSIDE a coordinated
# window. NO nsys (safe env-instrumentation only). HARD timeouts on every step so a hang can never
# run unbounded again (the 2026-09-08 nsys-hang lesson).
#
# NINFER_FLASH_PROFILE -> [flash-cfg] (actual occupancy) + [flash-prof] staging vs flash_kernel.
# NINFER_ATTN_PROFILE  -> [attn-prof] qkv_proj/fa/o_proj. NINFER_DECODE_PROFILE -> [decode-prof].
# Default depth 196608 (where the llama gap is biggest).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT=/home/anl/v100
ART=/models/Qwen3.6-35B-A3B-NInfer/qwen3_6_35b_a3b.ninfer
DEPTH="${DEPTH:-196608}"
CTX="${CTX:-253952}"
PORT="${PORT:-8090}"
OUT="${OUT:-$HERE/flash-out}"; mkdir -p "$OUT"
cname="ninfer35b-flashsplit"

wait_cards_free() {
  echo "  waiting for cards 1,2 free (<500MiB)..."
  for _ in $(seq 1 60); do
    local busy
    busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
           | awk -F',' '$1>=1 && $2+0>500{n++} END{print n+0}')
    [ "${busy:-2}" = "0" ] && { echo "  cards free."; return 0; }
    sleep 5
  done
  echo "  WARN cards busy after 300s"; return 1
}
cleanup() { docker rm -f "$cname" >/dev/null 2>&1 || true; }
trap cleanup EXIT

wait_cards_free
echo "=== ninfer flash-split depth=$DEPTH (FLASH_PROFILE + ATTN_PROFILE + DECODE_PROFILE) ==="
docker rm -f "$cname" >/dev/null 2>&1 || true
docker run -d --name "$cname" --gpus all --network host -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e NINFER_FLASH_PROFILE=1 -e NINFER_ATTN_PROFILE=1 -e NINFER_DECODE_PROFILE=1 \
  -v "$ROOT/ninfer-v100-moe-wt":/src:ro -v "$ROOT/models":/models:ro \
  -v "$ROOT/.nvjitcache":/root/.nv/ComputeCache --ipc=host --shm-size=8g \
  -w /src/build-v100 v100ninfer:cu128 \
  ./apps/ninfer-serve "$ART" --devices 1,2 --kv-dtype int8 --max-context "$CTX" --kv-capacity "$CTX" \
    --max-concurrency 1 --prefill-chunk 2048 --log-stats-interval-ms 2000 --model-id ninfer-35b \
    --max-request-mib 128 --no-thinking --host 0.0.0.0 --port "$PORT" >/dev/null

# HARD timeout on the whole probe (load+prefill); a hang self-aborts in 600s.
timeout 600 python3 "$HERE/greedy_probe.py" --base "http://127.0.0.1:$PORT" --require-model ninfer-35b \
  --prompt-tokens "$DEPTH" --max-new 1 --load-timeout 500 --out "$OUT/out.$DEPTH.txt" || \
  { echo "PROBE FAILED/TIMED OUT; logs:"; docker logs --tail 25 "$cname" 2>&1 | grep -iE "bad_alloc|error|MiB" | tail; }

docker kill -s TERM "$cname" >/dev/null 2>&1 || true
timeout 120 docker wait "$cname" >/dev/null 2>&1 || docker kill "$cname" >/dev/null 2>&1 || true
echo "=== RESULTS depth=$DEPTH ==="
docker logs "$cname" 2>&1 | grep -E "\[flash-cfg\]|\[flash-prof\]|\[attn-prof\]|\[decode-prof\]" | tail -6
cleanup
wait_cards_free
echo "=== done ==="
