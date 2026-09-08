#!/usr/bin/env bash
# dp4a grouped-prefill validation — run INSIDE a coordinated titan-router both-cards window.
#
# Two greedy legs on the SAME fixed prompt, dual-card EP, port 8090:
#   1. dp4a  (default)                      -> out.dp4a.txt
#   2. scalar (NINFER_MOE_PREFILL_SCALAR=1) -> out.scalar.txt   (the retained SIMT reference)
# then diff. int8-activation quant means NOT bit-identical is expected; the gate is: both COHERENT
# and high agreement (the dp4a path must not be garbled/off-task vs the scalar reference).
# Loading the serve at CTX also confirms the arena fit (no bad_alloc) + shard held.
#
#   CTX=253952  PROMPT_TOKENS=4096  MAXNEW=64  ./validate-dp4a.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT=/home/anl/v100
ART=/models/Qwen3.6-35B-A3B-NInfer/qwen3_6_35b_a3b.ninfer
CTX="${CTX:-253952}"
PORT="${PORT:-8090}"
MAXNEW="${MAXNEW:-64}"
PROMPT_TOKENS="${PROMPT_TOKENS:-4096}"   # >=2048 exercises the wide (BN=64) grouped path
LOADTO="${LOADTO:-900}"
OUT="${OUT:-$HERE/validate-out}"
mkdir -p "$OUT"

wait_cards_free() {
  echo "  waiting for cards 1,2 to free (<500MiB)..."
  for _ in $(seq 1 60); do
    local busy
    busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
           | awk -F',' '$1>=1 && $2+0>500{n++} END{print n+0}')
    [ "${busy:-2}" = "0" ] && { echo "  cards free."; return 0; }
    sleep 5
  done
  echo "  WARN: cards still busy after 300s"; return 1
}

# leg <label> <scalar 0|1>
leg() {
  local label="$1" scalar="$2" cname="ninfer35b-validate-$1"
  echo "=== leg: $label (scalar=$scalar, ctx=$CTX, port $PORT) ==="
  docker rm -f "$cname" >/dev/null 2>&1 || true
  local scalar_env=()
  [ "$scalar" = "1" ] && scalar_env=(-e NINFER_MOE_PREFILL_SCALAR=1)
  docker run -d --rm --name "$cname" --gpus all --network host \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID "${scalar_env[@]}" \
    -v "$ROOT/ninfer-v100-moe-wt":/src:ro -v "$ROOT/models":/models:ro \
    -v "$ROOT/.nvjitcache":/root/.nv/ComputeCache \
    --ipc=host --shm-size=8g -w /src/build-v100 v100ninfer:cu128 \
    ./apps/ninfer-serve "$ART" \
      --devices 1,2 --kv-dtype int8 --max-context "$CTX" --kv-capacity "$CTX" \
      --max-concurrency 1 --prefill-chunk 2048 --log-stats-interval-ms 2000 \
      --model-id ninfer-35b --max-request-mib 128 --no-thinking \
      --host 0.0.0.0 --port "$PORT" >/dev/null

  # greedy fixed-prompt completion; waits for TRUE load, records output + bad_alloc watch.
  PORT="$PORT" MAXNEW="$MAXNEW" PROMPT_TOKENS="$PROMPT_TOKENS" LOADTO="$LOADTO" \
    python3 "$HERE/greedy_probe.py" --base "http://127.0.0.1:$PORT" --require-model ninfer-35b \
      --prompt-tokens "$PROMPT_TOKENS" --max-new "$MAXNEW" --load-timeout "$LOADTO" \
      --out "$OUT/out.$label.txt" || {
        echo "$label FAILED; serve logs:"; docker logs --tail 40 "$cname" 2>&1 | grep -iE \
          "bad_alloc|error|abort|throw|shard|card|GiB|MiB" | tail -20; }
  echo "--- $label serve tail (fit/shard/bad_alloc) ---"
  docker logs "$cname" 2>&1 | grep -iE "bad_alloc|shard|card ?[12]|GiB|MiB|context|capacity|prefill" | tail -12 || true
  docker stop "$cname" >/dev/null 2>&1 || true
  wait_cards_free
}

wait_cards_free
leg dp4a   0
leg scalar 1

echo "=== A/B diff (dp4a vs scalar) ==="
if [ -s "$OUT/out.dp4a.txt" ] && [ -s "$OUT/out.scalar.txt" ]; then
  echo "--- dp4a  ---"; cat "$OUT/out.dp4a.txt"
  echo "--- scalar ---"; cat "$OUT/out.scalar.txt"
  echo "--- word-level diff ---"; diff <(tr ' ' '\n' < "$OUT/out.dp4a.txt") \
                                       <(tr ' ' '\n' < "$OUT/out.scalar.txt") | head -40 || true
  # crude agreement: fraction of matching whitespace tokens in order
  python3 - "$OUT/out.dp4a.txt" "$OUT/out.scalar.txt" <<'PY'
import sys
a=open(sys.argv[1]).read().split(); b=open(sys.argv[2]).read().split()
import difflib
sm=difflib.SequenceMatcher(a=a,b=b)
print(f"token agreement (dp4a vs scalar): {sm.ratio():.3f}  (len dp4a={len(a)} scalar={len(b)})")
PY
else
  echo "MISSING output(s) — check serve logs above for bad_alloc / load failure."
fi
