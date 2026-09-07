#!/usr/bin/env bash
# P3 full A/B orchestrator — run INSIDE a coordinated titan-router both-cards window (llama-swap down).
# Three legs, strictly SEQUENTIAL (each wants both V100s fully): llama-nomtp -> ninfer-nomtp ->
# ninfer-mtp. Set SKIP_LLAMA=1 if carbon supplies a usable per-depth llama ladder instead.
#
#   DEPTHS=...  MAXNEW=32  OUTDIR=./results  [SKIP_LLAMA=1]  ./run-all-legs.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEPTHS="${DEPTHS:-8192,16384,32768,65536,131072,196608,258048}"
MAXNEW="${MAXNEW:-64}"   # fixed decode count; the harness splits TTFT(prefill) from decode-rate, so
                         # output-length variance never contaminates the prefill-depth curve. 64 gives
                         # a stable decode-rate sample (16 MTP draft rounds) without bloating deep legs.
CTX="${CTX:-262144}"
OUTDIR="${OUTDIR:-$HERE/results}"
LOADTO="${LOADTO:-900}"
LLAMA_PORT="${LLAMA_PORT:-8001}"
NINFER_PORT="${NINFER_PORT:-8080}"
mkdir -p "$OUTDIR"

# Wait until both V100s (nvidia-smi idx 1,2) are back under 500 MiB — cards genuinely freed.
wait_cards_free() {
  echo "  waiting for cards 1,2 to free..."
  for _ in $(seq 1 60); do
    local busy
    busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
           | awk -F',' '$1>=1 && $2+0>500{n++} END{print n+0}')
    [ "${busy:-2}" = "0" ] && { echo "  cards free."; return 0; }
    sleep 5
  done
  echo "  WARN: cards still busy after 300s"; return 1
}

sweep() { # base label
  python3 "$HERE/bench_depth.py" --base "$1" --out "$OUTDIR/$2.csv" \
    --depths "$DEPTHS" --max-new "$MAXNEW" --label "$2" --load-timeout "$LOADTO"
}

# ---- Leg 1: llama.cpp (native, matched router config) --------------------------------------------
if [ "${SKIP_LLAMA:-0}" != "1" ]; then
  echo "=== leg: llama-nomtp (port $LLAMA_PORT) ==="
  PORT="$LLAMA_PORT" CTX="$CTX" "$HERE/serve-llama-35b.sh" >"$OUTDIR/llama-serve.log" 2>&1 &
  LLAMA_PID=$!
  sweep "http://127.0.0.1:$LLAMA_PORT" llama-nomtp || echo "llama leg FAILED (see $OUTDIR/llama-serve.log)"
  echo "  stopping llama-server (pid $LLAMA_PID)"
  kill "$LLAMA_PID" 2>/dev/null || true
  pkill -f "llama-server .*Qwen3.6-35B" 2>/dev/null || true
  wait_cards_free
fi

# ---- Legs 2 & 3: ninfer (docker) MTP-off then MTP-on ---------------------------------------------
ninfer_leg() { # spec label
  local spec="$1" label="$2" cname="ninfer35b-bench-$1"
  echo "=== leg: $label (spec=$spec, port $NINFER_PORT) ==="
  docker rm -f "$cname" >/dev/null 2>&1 || true
  docker run -d --rm --name "$cname" --gpus all --network host \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    -v /home/anl/v100/ninfer-v100-moe-wt:/src:ro -v /home/anl/v100/models:/models:ro \
    -v /home/anl/v100/.nvjitcache:/root/.nv/ComputeCache \
    --ipc=host --shm-size=8g -w /src/build-v100 v100ninfer:cu128 \
    ./apps/ninfer-serve /models/Qwen3.6-35B-A3B-NInfer/qwen3_6_35b_a3b.ninfer \
      --devices 1,2 --kv-dtype int8 --max-context "$CTX" --kv-capacity "$CTX" \
      --max-concurrency 1 --prefill-chunk 2048 --log-stats-interval-ms 2000 \
      --model-id ninfer-35b --max-request-mib 128 --host 0.0.0.0 --port "$NINFER_PORT" \
      $( [ "$spec" = mtp ] && echo "--spec mtp --draft-tokens 4" ) >/dev/null
  sweep "http://127.0.0.1:$NINFER_PORT" "$label" || {
    echo "$label FAILED; serve logs:"; docker logs --tail 40 "$cname" 2>&1 || true; }
  echo "--- $label MTP/accept stats ---"
  docker logs "$cname" 2>&1 | grep -iE "accept|mtp|throughput|decode|prefill" | tail -12 || true
  docker stop "$cname" >/dev/null 2>&1 || true
  wait_cards_free
}

ninfer_leg none ninfer-nomtp
ninfer_leg mtp  ninfer-mtp

echo "=== all legs done; results in $OUTDIR ==="
ls -la "$OUTDIR"/*.csv 2>/dev/null
echo "=== quick table ==="
python3 - "$OUTDIR" <<'PY'
import csv, glob, os, sys
d=sys.argv[1]
rows={}
for f in sorted(glob.glob(os.path.join(d,"*.csv"))):
    lab=os.path.basename(f)[:-4]
    for r in csv.DictReader(open(f)):
        rows.setdefault(int(r["prompt_tokens"])//1000, {})[lab]=(r["total_s"], r["prefill_tok_s"], r["decode_tok_s"])
labs=sorted({l for v in rows.values() for l in v})
print("depth(K) | " + " | ".join(labs) + "   (total_s / prefill_tps / decode_tps)")
for k in sorted(rows):
    cells=[]
    for l in labs:
        c=rows[k].get(l); cells.append("/".join(c) if c else "-")
    print(f"{k:>7} | " + " | ".join(cells))
PY
