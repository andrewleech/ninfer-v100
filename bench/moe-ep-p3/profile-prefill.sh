#!/usr/bin/env bash
# Prefill phase-split profiler — WHERE does deep-context prefill time actually go?
# Runs the CURRENT dp4a binary (no rebuild) with NINFER_DECODE_PROFILE=1, which makes
# run_layers sync the primary stream around each mixer/mlp call and accumulate wall-clock
# into attn / gdn / mlp buckets, dumped as `[decode-prof] fwd=.. attn=.. gdn=.. mlp=..` on
# clean shutdown (a static destructor -> needs a graceful SIGTERM, so we `docker stop`).
#
# MAXNEW=1 so the buckets are ~pure PREFILL (one decode step is negligible vs ~N/chunk
# prefill chunks). Sweep several depths to see which bucket GROWS with depth = the taper source.
#
# CAVEAT (interpretation): the profiler SERIALIZES the primary stream, so `mlp` loses the EP
# primary/secondary-card compute overlap and is PESSIMISTIC (inflated). `attn` is single-stream
# and loses nothing. So: if `attn` dominates even here, that is conclusive; a large `mlp` is
# partly the serialization artifact and needs the nsys/NVTX cross-check before trusting it.
#
#   DEPTHS=65536,131072,196608  ./profile-prefill.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT=/home/anl/v100
ART=/models/Qwen3.6-35B-A3B-NInfer/qwen3_6_35b_a3b.ninfer
DEPTHS="${DEPTHS:-65536,131072,196608}"
CTX="${CTX:-253952}"
PORT="${PORT:-8090}"
MAXNEW="${MAXNEW:-1}"
LOADTO="${LOADTO:-900}"
OUT="${OUT:-$HERE/profile-out}"
mkdir -p "$OUT"
SUMMARY="$OUT/prefill-split.txt"; : > "$SUMMARY"

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

# profile_depth <depth>
profile_depth() {
  local depth="$1"
  local cname="ninfer35b-prof-$depth"
  echo "=== profile depth=$depth (ctx=$CTX, port $PORT, DECODE_PROFILE on) ==="
  docker rm -f "$cname" >/dev/null 2>&1 || true
  # NOTE: no --rm — we need the logs AFTER a clean exit to read the [decode-prof]
  # destructor line, and we rm explicitly at the end.
  docker run -d --name "$cname" --gpus all --network host \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NINFER_DECODE_PROFILE=1 \
    -v "$ROOT/ninfer-v100-moe-wt":/src:ro -v "$ROOT/models":/models:ro \
    -v "$ROOT/.nvjitcache":/root/.nv/ComputeCache \
    --ipc=host --shm-size=8g -w /src/build-v100 v100ninfer:cu128 \
    ./apps/ninfer-serve "$ART" \
      --devices 1,2 --kv-dtype int8 --max-context "$CTX" --kv-capacity "$CTX" \
      --max-concurrency 1 --prefill-chunk 2048 --log-stats-interval-ms 2000 \
      --model-id ninfer-35b --max-request-mib 128 --no-thinking \
      --host 0.0.0.0 --port "$PORT" >/dev/null

  # one fixed-depth greedy prefill (MAXNEW≈1 -> buckets are ~pure prefill)
  PORT="$PORT" python3 "$HERE/greedy_probe.py" --base "http://127.0.0.1:$PORT" \
    --require-model ninfer-35b --prompt-tokens "$depth" --max-new "$MAXNEW" \
    --load-timeout "$LOADTO" --out "$OUT/out.$depth.txt" || {
      echo "depth=$depth FAILED; logs:"; docker logs --tail 30 "$cname" 2>&1 | \
        grep -iE "bad_alloc|error|abort|throw|shard|GiB|MiB" | tail; }

  # capture the last prefill throughput line too (independent cross-check on the split)
  local pf
  pf=$(docker logs "$cname" 2>&1 | grep -oE "prefill=[0-9.]+tok/s" | tail -1)

  # graceful SIGTERM -> handle_signal -> server->stop() -> main returns -> ~DDump prints
  # [decode-prof]. Model/CUDA teardown (20GB VRAM + 8GB host-KV) can take tens of seconds
  # and runs BEFORE the static destructor, so give a long grace and wait for true exit.
  docker kill -s TERM "$cname" >/dev/null 2>&1 || true
  timeout 180 docker wait "$cname" >/dev/null 2>&1 || \
    { echo "  WARN depth=$depth: no clean exit in 180s, forcing"; docker kill "$cname" >/dev/null 2>&1 || true; }
  local line
  line=$(docker logs "$cname" 2>&1 | grep "\[decode-prof\]" | tail -1)
  echo "depth=$depth  ${line:-[decode-prof MISSING]}  last-$pf" | tee -a "$SUMMARY"
  docker rm -f "$cname" >/dev/null 2>&1 || true
  wait_cards_free
}

wait_cards_free
IFS=',' read -ra DS <<< "$DEPTHS"
for d in "${DS[@]}"; do profile_depth "$d"; done

echo
echo "=== prefill phase split (attn = attention gather+flash+mask ; mlp = MoE+EP-reduce ; gdn = linear-attn) ==="
cat "$SUMMARY"
python3 - "$SUMMARY" <<'PY'
import re,sys
for ln in open(sys.argv[1]):
    m=re.search(r"depth=(\d+).*attn=([\d.]+) gdn=([\d.]+) mlp=([\d.]+)",ln)
    if not m: continue
    d,a,g,ml=int(m[1]),float(m[2]),float(m[3]),float(m[4])
    tot=a+g+ml
    if tot<=0: continue
    print(f"{d//1024:>4}K  attn {a:7.2f}s ({100*a/tot:4.1f}%)  gdn {g:7.2f}s ({100*g/tot:4.1f}%)  "
          f"mlp {ml:7.2f}s ({100*ml/tot:4.1f}%)  total {tot:7.2f}s")
PY
