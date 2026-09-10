#!/usr/bin/env bash
# DEFINITIVE end-to-end sweep: ninfer (dp4a) vs llama.cpp across context depths up to ~262K.
# Per depth: PREFILL tok/s, DECODE tok/s, and END-TO-END time. Profiler OFF (real production speed).
# Run INSIDE a coordinated window. All external legs HARD-timeout-guarded.
#
# ninfer: ONE dual-card EP serve @ 262144 (dp4a on), probed at each depth (MAXNEW=64); per-request
#         prefill=/decode= parsed from the serve log. No reload per depth (fast, and no profiler
#         serialization distorting tok/s).
# llama:  matched batching (-ub 2048 = ninfer's prefill-chunk), -p D -n 64 -r 1 per depth (no nsys).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT=/home/anl/v100
ART=/models/Qwen3.6-35B-A3B-NInfer/qwen3_6_35b_a3b.ninfer
GGUF="$ROOT/models/qwen3.6-35b-a3b-MTP-GGUF/Qwen3.6-35B-A3B-IQ4_XS-4.19bpw.gguf"
LB=/home/anl/v100/llama.cpp/build/bin/llama-bench
DEPTHS="${DEPTHS:-8192,16384,32768,65536,131072,196608,245760}"
MAXNEW="${MAXNEW:-64}"
PORT="${PORT:-8090}"
OUT="${OUT:-$HERE/sweep-out}"; mkdir -p "$OUT"
NIN="$OUT/ninfer.csv"; LLA="$OUT/llama.csv"
echo "depth,prefill_tok_s,decode_tok_s" > "$NIN"
export LD_LIBRARY_PATH="/home/anl/v100/llama.cpp/build/bin:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"

wait_cards_free() {
  echo "  waiting for cards 1,2 free (<500MiB)..."
  for _ in $(seq 1 60); do
    local b; b=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
       | awk -F',' '$1>=1 && $2+0>500{n++} END{print n+0}')
    [ "${b:-2}" = "0" ] && { echo "  cards free."; return 0; }
    sleep 5
  done; echo "  WARN cards busy after 300s"; return 1
}
IFS=',' read -ra DS <<< "$DEPTHS"

echo "############ NINFER dp4a — one serve @262144, probe each depth ############"
wait_cards_free
cname=ninfer35b-sweep
docker rm -f "$cname" >/dev/null 2>&1 || true
docker run -d --name "$cname" --gpus all --network host -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -v "$ROOT/ninfer-v100-moe-wt":/src:ro -v "$ROOT/models":/models:ro \
  -v "$ROOT/.nvjitcache":/root/.nv/ComputeCache --ipc=host --shm-size=8g \
  -w /src/build-v100 v100ninfer:cu128 \
  ./apps/ninfer-serve "$ART" --devices 1,2 --kv-dtype int8 --max-context 262144 --kv-capacity 262144 \
    --max-concurrency 1 --prefill-chunk 2048 --log-stats-interval-ms 60000 --model-id ninfer-35b \
    --max-request-mib 160 --no-thinking --host 0.0.0.0 --port "$PORT" >/dev/null
# wait for load once
timeout 300 python3 "$HERE/greedy_probe.py" --base "http://127.0.0.1:$PORT" --require-model ninfer-35b \
  --prompt-tokens 2048 --max-new 1 --load-timeout 260 --out "$OUT/warm.txt" || { echo "ninfer load FAILED"; docker logs --tail 20 "$cname"; }
for d in "${DS[@]}"; do
  echo "--- ninfer depth=$d ---"
  timeout 500 python3 "$HERE/greedy_probe.py" --base "http://127.0.0.1:$PORT" --require-model ninfer-35b \
    --prompt-tokens "$d" --max-new "$MAXNEW" --load-timeout 30 --out "$OUT/nin.$d.txt" || echo "  probe $d failed/timeout"
  # newest completed request line carries this probe's rates
  line=$(docker logs "$cname" 2>&1 | grep -E "\[req [0-9]+\] done" | grep "gen=$MAXNEW" | tail -1)
  pf=$(echo "$line" | grep -oE "prefill=[0-9.]+" | grep -oE "[0-9.]+")
  dc=$(echo "$line" | grep -oE "decode=[0-9.]+" | grep -oE "[0-9.]+")
  echo "$d,${pf:-NA},${dc:-NA}" | tee -a "$NIN"
done
docker kill -s TERM "$cname" >/dev/null 2>&1 || true
timeout 120 docker wait "$cname" >/dev/null 2>&1 || docker kill "$cname" >/dev/null 2>&1 || true
docker rm -f "$cname" >/dev/null 2>&1 || true
wait_cards_free

echo "############ LLAMA -ub2048, per depth ############"
echo "depth,prefill_tok_s,decode_tok_s" > "$LLA"
for d in "${DS[@]}"; do
  echo "--- llama depth=$d ---"
  r=$(timeout 600 "$LB" -m "$GGUF" -p "$d" -n "$MAXNEW" -r 1 -fa on -ctk q5_1 -ctv q5_1 \
        -ngl 99 -sm layer -ts 1/1 -ub 2048 -b 2048 2>/dev/null)
  # markdown rows: "... | pp<d> | <t/s> ± .. |" and "... | tg<MAXNEW> | <t/s> ± .. |"
  pf=$(echo "$r" | grep -oE "pp$d *\|[^|]*" | grep -oE "[0-9]+\.[0-9]+" | head -1)
  tg=$(echo "$r" | grep -oE "tg$MAXNEW *\|[^|]*" | grep -oE "[0-9]+\.[0-9]+" | head -1)
  echo "$d,${pf:-NA},${tg:-NA}" | tee -a "$LLA"
  wait_cards_free
done

echo; echo "############ COMPARISON TABLE ############"
python3 - "$NIN" "$LLA" "$MAXNEW" <<'PY'
import sys,csv
def load(p):
    d={}
    for r in csv.DictReader(open(p)):
        try: d[int(r['depth'])]=(float(r['prefill_tok_s']),float(r['decode_tok_s']))
        except: d[int(r['depth'])]=(None,None)
    return d
nin=load(sys.argv[1]); lla=load(sys.argv[2]); nout=int(sys.argv[3])
NOUT_REAL=2000
print(f"{'depth':>6} | {'PREFILL tok/s':^21} | {'DECODE tok/s':^19} | {'e2e s ('+str(nout)+' out)':^17} | {'e2e s ('+str(NOUT_REAL)+' out)':^19}")
print(f"{'':>6} | {'ninfer':>9} {'llama':>7} {'x':>3} | {'ninfer':>8} {'llama':>6} | {'ninfer':>7} {'llama':>7} | {'ninfer':>8} {'llama':>8}")
for d in sorted(set(nin)|set(lla)):
    np_,nd=nin.get(d,(None,None)); lp,ld=lla.get(d,(None,None))
    def e2e(pf,dc,no):
        return (d/pf + no/dc) if (pf and dc) else None
    def f(x,w=7,p=1): return format(x,f'{w}.{p}f') if isinstance(x,(int,float)) else format('NA',f'>{w}')
    rx = (lp/np_) if (np_ and lp) else None  # llama/ninfer prefill ratio (>1 = llama faster)
    print(f"{d//1024:>4}K | {f(np_,9,0)} {f(lp,7,0)} {f(rx,3,2) if rx else ' NA':>3} | "
          f"{f(nd,8,1)} {f(ld,6,1)} | {f(e2e(np_,nd,nout)):>7} {f(e2e(lp,ld,nout)):>7} | "
          f"{f(e2e(np_,nd,NOUT_REAL),8):>8} {f(e2e(lp,ld,NOUT_REAL),8):>8}")
print("\nprefill x = llama/ninfer (>1 llama faster, <1 ninfer faster). e2e = depth/prefill + Nout/decode.")
print(f"NOTE: ninfer decode here is MTP-OFF; MTP adds ~1.45x decode (not in these numbers).")
PY
echo "raw: $NIN , $LLA"
