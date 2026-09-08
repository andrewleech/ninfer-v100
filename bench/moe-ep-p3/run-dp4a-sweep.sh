#!/usr/bin/env bash
# dp4a prefill sweep — run INSIDE the coordinated window, AFTER the A/B correctness check passes.
# Sweeps the SAME P3 depths at CTX=253952 (matches the baseline exactly) so ninfer-dp4a.csv is a
# drop-in comparison against results/ninfer-nomtp.csv (scalar) + results/llama-nomtp.csv. Then a
# separate 262144 cold-load to confirm the hardest arena fit with the +~18MB dp4a scratch.
#
#   DEPTHS=...  MAXNEW=64  ./run-dp4a-sweep.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT=/home/anl/v100
ART=/models/Qwen3.6-35B-A3B-NInfer/qwen3_6_35b_a3b.ninfer
DEPTHS="${DEPTHS:-8192,16384,32768,65536,131072,196608,245760}"
MAXNEW="${MAXNEW:-64}"
CTX="${CTX:-253952}"
PORT="${PORT:-8090}"
OUTDIR="${OUTDIR:-$HERE/results}"
LOADTO="${LOADTO:-900}"
mkdir -p "$OUTDIR"

wait_cards_free() {
  for _ in $(seq 1 60); do
    local busy
    busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
           | awk -F',' '$1>=1 && $2+0>500{n++} END{print n+0}')
    [ "${busy:-2}" = "0" ] && { echo "  cards free."; return 0; }
    sleep 5
  done
  echo "  WARN: cards still busy after 300s"; return 1
}

# serve <ctx> <cname>
serve() {
  docker rm -f "$2" >/dev/null 2>&1 || true
  docker run -d --rm --name "$2" --gpus all --network host \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    -v "$ROOT/ninfer-v100-moe-wt":/src:ro -v "$ROOT/models":/models:ro \
    -v "$ROOT/.nvjitcache":/root/.nv/ComputeCache \
    --ipc=host --shm-size=8g -w /src/build-v100 v100ninfer:cu128 \
    ./apps/ninfer-serve "$ART" \
      --devices 1,2 --kv-dtype int8 --max-context "$1" --kv-capacity "$1" \
      --max-concurrency 1 --prefill-chunk 2048 --log-stats-interval-ms 2000 \
      --model-id ninfer-35b --max-request-mib 128 --no-thinking \
      --host 0.0.0.0 --port "$PORT" >/dev/null
}

echo "=== dp4a prefill sweep @ CTX=$CTX ==="
serve "$CTX" ninfer35b-dp4a-sweep
python3 "$HERE/bench_depth.py" --base "http://127.0.0.1:$PORT" --out "$OUTDIR/ninfer-dp4a.csv" \
  --depths "$DEPTHS" --max-new "$MAXNEW" --label ninfer-dp4a --load-timeout "$LOADTO" \
  --require-model ninfer-35b || { echo "SWEEP FAILED; logs:"; \
    docker logs --tail 40 ninfer35b-dp4a-sweep 2>&1 | grep -iE "bad_alloc|error|shard|GiB|MiB" | tail; }
echo "--- shard/fit tail ---"
docker logs ninfer35b-dp4a-sweep 2>&1 | grep -iE "bad_alloc|shard|card ?[12]|GiB|MiB|capacity" | tail -8 || true
docker stop ninfer35b-dp4a-sweep >/dev/null 2>&1 || true
wait_cards_free

echo "=== 262144 cold-load fit check (hardest arena) ==="
serve 262144 ninfer35b-dp4a-262k
python3 - <<PY || true
import json,time,urllib.request
base="http://127.0.0.1:$PORT"
def get(u,p=None,t=60):
    r=urllib.request.Request(u,data=(json.dumps(p).encode() if p else None),method=("POST" if p else "GET"))
    r.add_header("Content-Type","application/json"); r.add_header("Authorization","Bearer x")
    return json.loads(urllib.request.urlopen(r,timeout=t).read())
t0=time.time()
while time.time()-t0<900:
    try:
        d=(get(base+"/v1/models",t=10).get("data") or [])
        if d:
            r=get(base+"/v1/chat/completions",{"model":d[0]["id"],"messages":[{"role":"user","content":"hi"}],"max_tokens":1,"temperature":0},120)
            if r.get("choices"): print(f"262144 LOAD OK after {time.time()-t0:.0f}s (arena fits with dp4a scratch)"); break
    except Exception: pass
    time.sleep(5)
else:
    print("262144 load did NOT succeed in 900s (check logs for bad_alloc)")
PY
echo "--- 262k fit tail ---"
docker logs ninfer35b-dp4a-262k 2>&1 | grep -iE "bad_alloc|shard|card ?[12]|GiB|MiB|capacity|context" | tail -10 || true
docker stop ninfer35b-dp4a-262k >/dev/null 2>&1 || true
wait_cards_free
echo "=== sweep done; ninfer-dp4a.csv written ==="
