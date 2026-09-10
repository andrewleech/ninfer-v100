#!/usr/bin/env bash
# W8 dp4a projection-GEMM A/B — run INSIDE a coordinated window. All steps HARD-timeout-guarded.
# Leg dp4a (default) vs Leg mma (NINFER_W8_NO_DP4A=1), same fixed prompt + one deep profile depth.
# NINFER_ATTN_PROFILE => [attn-prof] qkv_proj/fa/o_proj (the projection-GEMM delta is the point).
# Correctness gate: coherence + token-agreement (int8-act quant => not bit-identical, expected).
# Plus a 262144 cold-load fit check with dp4a on (arena must not OOM).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT=/home/anl/v100
ART=/models/Qwen3.6-35B-A3B-NInfer/qwen3_6_35b_a3b.ninfer
CTX="${CTX:-253952}"; PORT="${PORT:-8090}"
DEPTH="${DEPTH:-131072}"; PROMPT_TOKENS="${PROMPT_TOKENS:-4096}"
OUT="${OUT:-$HERE/w8ab-out}"; mkdir -p "$OUT"; SUM="$OUT/summary.txt"; : > "$SUM"

wait_cards_free() {
  echo "  waiting for cards 1,2 free (<500MiB)..."
  for _ in $(seq 1 60); do
    local b; b=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
       | awk -F',' '$1>=1 && $2+0>500{n++} END{print n+0}')
    [ "${b:-2}" = "0" ] && { echo "  cards free."; return 0; }
    sleep 5
  done; echo "  WARN cards busy after 300s"; return 1
}

# leg <label> <no_dp4a 0|1> <serve_ctx>
leg() {
  local label="$1" nod="$2" ctx="$3"
  local cname="ninfer35b-w8ab-$label"
  echo "=== leg $label (NINFER_W8_NO_DP4A=$nod, ctx=$ctx) ==="
  docker rm -f "$cname" >/dev/null 2>&1 || true
  local env=(-e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NINFER_ATTN_PROFILE=1 -e NINFER_DECODE_PROFILE=1)
  [ "$nod" = "1" ] && env+=(-e NINFER_W8_NO_DP4A=1)
  docker run -d --name "$cname" --gpus all --network host "${env[@]}" \
    -v "$ROOT/ninfer-v100-moe-wt":/src:ro -v "$ROOT/models":/models:ro \
    -v "$ROOT/.nvjitcache":/root/.nv/ComputeCache --ipc=host --shm-size=8g \
    -w /src/build-v100 v100ninfer:cu128 \
    ./apps/ninfer-serve "$ART" --devices 1,2 --kv-dtype int8 --max-context "$ctx" --kv-capacity "$ctx" \
      --max-concurrency 1 --prefill-chunk 2048 --log-stats-interval-ms 2000 --model-id ninfer-35b \
      --max-request-mib 128 --no-thinking --host 0.0.0.0 --port "$PORT" >/dev/null
  # correctness prompt (fixed 4K, greedy) then the deep profile depth (MAXNEW=1)
  timeout 300 python3 "$HERE/greedy_probe.py" --base "http://127.0.0.1:$PORT" --require-model ninfer-35b \
    --prompt-tokens "$PROMPT_TOKENS" --max-new 64 --load-timeout 500 --out "$OUT/out.$label.txt" || \
    { echo "$label correctness FAILED"; docker logs --tail 20 "$cname" 2>&1 | grep -iE "bad_alloc|error|MiB" | tail; }
  timeout 400 python3 "$HERE/greedy_probe.py" --base "http://127.0.0.1:$PORT" --require-model ninfer-35b \
    --prompt-tokens "$DEPTH" --max-new 1 --load-timeout 60 --out "$OUT/prof.$label.txt" || true
  local pf; pf=$(docker logs "$cname" 2>&1 | grep -oE "prefill=[0-9.]+tok/s" | tail -1)
  docker kill -s TERM "$cname" >/dev/null 2>&1 || true
  timeout 120 docker wait "$cname" >/dev/null 2>&1 || docker kill "$cname" >/dev/null 2>&1 || true
  local ap dp; ap=$(docker logs "$cname" 2>&1 | grep "\[attn-prof\]" | tail -1)
  dp=$(docker logs "$cname" 2>&1 | grep "\[decode-prof\]" | tail -1)
  echo "LEG $label  $ap  $dp  last-$pf" | tee -a "$SUM"
  docker rm -f "$cname" >/dev/null 2>&1 || true
  wait_cards_free
}

wait_cards_free
leg dp4a 0 "$CTX"
leg mma  1 "$CTX"

echo "=== 262144 cold-load fit check (dp4a on) ==="
cname=ninfer35b-w8ab-262k
docker rm -f "$cname" >/dev/null 2>&1 || true
docker run -d --name "$cname" --gpus all --network host -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -v "$ROOT/ninfer-v100-moe-wt":/src:ro -v "$ROOT/models":/models:ro \
  -v "$ROOT/.nvjitcache":/root/.nv/ComputeCache --ipc=host --shm-size=8g \
  -w /src/build-v100 v100ninfer:cu128 \
  ./apps/ninfer-serve "$ART" --devices 1,2 --kv-dtype int8 --max-context 262144 --kv-capacity 262144 \
    --max-concurrency 1 --prefill-chunk 2048 --model-id ninfer-35b --max-request-mib 128 --no-thinking \
    --host 0.0.0.0 --port "$PORT" >/dev/null
timeout 240 python3 "$HERE/greedy_probe.py" --base "http://127.0.0.1:$PORT" --require-model ninfer-35b \
  --prompt-tokens 2048 --max-new 1 --load-timeout 200 --out "$OUT/fit262k.txt" \
  && echo "262144 LOAD OK (arena fits with W8 dp4a scratch)" | tee -a "$SUM" \
  || { echo "262144 FAILED (check bad_alloc)" | tee -a "$SUM"; docker logs --tail 15 "$cname" 2>&1 | grep -iE "bad_alloc|MiB|GiB" | tail; }
docker kill -s TERM "$cname" >/dev/null 2>&1; timeout 120 docker wait "$cname" >/dev/null 2>&1 || docker kill "$cname" >/dev/null 2>&1 || true
docker rm -f "$cname" >/dev/null 2>&1 || true
wait_cards_free

echo; echo "=== A/B RESULT ==="; cat "$SUM"
echo "--- correctness: dp4a vs mma token agreement ---"
if [ -s "$OUT/out.dp4a.txt" ] && [ -s "$OUT/out.mma.txt" ]; then
  python3 - "$OUT/out.dp4a.txt" "$OUT/out.mma.txt" <<'PY'
import sys,difflib
a=open(sys.argv[1]).read().split(); b=open(sys.argv[2]).read().split()
print(f"  agreement={difflib.SequenceMatcher(a=a,b=b).ratio():.3f} (len dp4a={len(a)} mma={len(b)})")
print("  dp4a:",' '.join(a[:40]))
PY
else echo "  MISSING output(s)"; fi
