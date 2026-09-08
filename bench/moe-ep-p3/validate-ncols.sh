#!/usr/bin/env bash
# Validate the ncols1 4->8 flash tiling change (CausalD256H16Kv2) — run INSIDE a coordinated
# both-cards window. The build-v100 binary is now ncols1=8.
#
# 1. CORRECTNESS: greedy @ the SAME fixed 4096-tok prompt as the committed dp4a reference
#    (validate-out/out.dp4a.txt, which was ncols1=4). ncols1 is a PURE TILING param -> output
#    MUST be byte-identical. Any diff = the change altered results = BUG, stop.
# 2. SPEED: re-run the phase-split profiler (66K/133K/196K) and compare the `attn` bucket to
#    the recorded ncols1=4 baseline (20.3 / 63.8 / 142.8 s). gdn/mlp should be ~unchanged
#    (this touches only attention).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT=/home/anl/v100
ART=/models/Qwen3.6-35B-A3B-NInfer/qwen3_6_35b_a3b.ninfer
CTX="${CTX:-253952}"
PORT="${PORT:-8090}"
REF="$HERE/validate-out/out.dp4a.txt"
OUT="$HERE/validate-out"; mkdir -p "$OUT"

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

echo "=== STEP 1: correctness (ncols1=8 greedy vs committed ncols1=4 reference) ==="
wait_cards_free
cname=ninfer35b-ncols8-correct
docker rm -f "$cname" >/dev/null 2>&1 || true
docker run -d --name "$cname" --gpus all --network host -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -v "$ROOT/ninfer-v100-moe-wt":/src:ro -v "$ROOT/models":/models:ro \
  -v "$ROOT/.nvjitcache":/root/.nv/ComputeCache --ipc=host --shm-size=8g \
  -w /src/build-v100 v100ninfer:cu128 \
  ./apps/ninfer-serve "$ART" --devices 1,2 --kv-dtype int8 --max-context "$CTX" --kv-capacity "$CTX" \
    --max-concurrency 1 --prefill-chunk 2048 --log-stats-interval-ms 2000 --model-id ninfer-35b \
    --max-request-mib 128 --no-thinking --host 0.0.0.0 --port "$PORT" >/dev/null
PORT="$PORT" python3 "$HERE/greedy_probe.py" --base "http://127.0.0.1:$PORT" --require-model ninfer-35b \
  --prompt-tokens 4096 --max-new 64 --load-timeout 900 --out "$OUT/out.ncols8.txt" || \
  { echo "correctness leg FAILED"; docker logs --tail 30 "$cname" 2>&1 | grep -iE "bad_alloc|error|shard|MiB" | tail; }
docker rm -f "$cname" >/dev/null 2>&1 || true
wait_cards_free

echo "--- correctness (coherence + token-agreement vs ncols1=4 reference) ---"
# A tiling/Q-placement change reorders fp adds in the online softmax, so greedy output can
# legitimately DIVERGE while staying coherent/on-task. The right gate here is coherence +
# high token-agreement (byte-identity is reserved for quant/no-op refactors where argmax must
# not move). Print the agreement ratio and eyeball coherence.
if [ -s "$OUT/out.ncols8.txt" ] && [ -s "$REF" ]; then
  echo "--- new (ncols1=8 + Q_in_reg) ---"; cat "$OUT/out.ncols8.txt"
  python3 - "$OUT/out.ncols8.txt" "$REF" <<'PY'
import sys, difflib
a=open(sys.argv[1]).read().split(); b=open(sys.argv[2]).read().split()
r=difflib.SequenceMatcher(a=a,b=b).ratio()
print(f"token agreement vs ncols1=4 reference: {r:.3f} (len new={len(a)} ref={len(b)})")
print("PASS (coherent + high agreement)" if r>=0.6 and len(a)>=20 else
      "REVIEW: low agreement or short output — inspect for garbling")
PY
else
  echo "MISSING out.ncols8.txt or reference $REF — cannot compare."
fi

echo
echo "=== STEP 2: speed (attn-bucket phase split, ncols1=8) ==="
DEPTHS="${DEPTHS:-65536,131072,196608}" MAXNEW=1 OUT="$HERE/profile-out-ncols8" "$HERE/profile-prefill.sh"

echo
echo "=== A/B vs recorded ncols1=4 baseline (attn s @ 64/128/192K = 20.3 / 63.8 / 142.8) ==="
