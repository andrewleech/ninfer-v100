#!/usr/bin/env bash
# Apples-to-apples attention decomposition: ninfer vs llama, per prefill depth.
# Run INSIDE a coordinated both-cards window.
#
# ninfer leg: dual-card EP serve with NINFER_DECODE_PROFILE=1 + NINFER_ATTN_PROFILE=1 → splits the
#   d_attn bucket into qkv_proj / fa(flash+staging) / o_proj  ([attn-prof] line, static destructor →
#   graceful SIGTERM + docker wait, no --rm).
# llama leg: native host nsys per depth → flash_attn_ext kernel fraction of all GPU kernels; combined
#   with llama-bench's own prefill tok/s → llama attention-seconds per prefill (warmup-independent).
#
# Answers the fork: is ninfer's attention deficit the FLASH KERNEL (fa) or the PROJECTION GEMMs
# (qkv_proj+o_proj)? And is llama's flash kernel actually faster than ninfer's?
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT=/home/anl/v100
ART=/models/Qwen3.6-35B-A3B-NInfer/qwen3_6_35b_a3b.ninfer
GGUF="$ROOT/models/qwen3.6-35b-a3b-MTP-GGUF/Qwen3.6-35B-A3B-IQ4_XS-4.19bpw.gguf"
LLAMA=/home/anl/v100/llama.cpp/build/bin/llama-bench
DEPTHS="${DEPTHS:-65536,131072,196608}"
CTX="${CTX:-253952}"
PORT="${PORT:-8090}"
OUT="${OUT:-$HERE/decomp-out}"; mkdir -p "$OUT"
NSYS_DIR="$OUT/nsys"; mkdir -p "$NSYS_DIR"
NSUM="$OUT/summary.txt"; : > "$NSUM"
export LD_LIBRARY_PATH="/home/anl/v100/llama.cpp/build/bin:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"

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

IFS=',' read -ra DS <<< "$DEPTHS"

echo "############ NINFER leg (d_attn decomposition) ############"
wait_cards_free
for d in "${DS[@]}"; do
  cname="ninfer35b-decomp-$d"
  echo "=== ninfer depth=$d ==="
  docker rm -f "$cname" >/dev/null 2>&1 || true
  docker run -d --name "$cname" --gpus all --network host -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    -e NINFER_DECODE_PROFILE=1 -e NINFER_ATTN_PROFILE=1 \
    -v "$ROOT/ninfer-v100-moe-wt":/src:ro -v "$ROOT/models":/models:ro \
    -v "$ROOT/.nvjitcache":/root/.nv/ComputeCache --ipc=host --shm-size=8g \
    -w /src/build-v100 v100ninfer:cu128 \
    ./apps/ninfer-serve "$ART" --devices 1,2 --kv-dtype int8 --max-context "$CTX" --kv-capacity "$CTX" \
      --max-concurrency 1 --prefill-chunk 2048 --log-stats-interval-ms 2000 --model-id ninfer-35b \
      --max-request-mib 128 --no-thinking --host 0.0.0.0 --port "$PORT" >/dev/null
  PORT="$PORT" python3 "$HERE/greedy_probe.py" --base "http://127.0.0.1:$PORT" --require-model ninfer-35b \
    --prompt-tokens "$d" --max-new 1 --load-timeout 900 --out "$OUT/nin.$d.txt" || \
    { echo "ninfer d=$d FAILED"; docker logs --tail 20 "$cname" 2>&1 | grep -iE "bad_alloc|error|MiB" | tail; }
  docker kill -s TERM "$cname" >/dev/null 2>&1 || true
  timeout 180 docker wait "$cname" >/dev/null 2>&1 || docker kill "$cname" >/dev/null 2>&1 || true
  dp=$(docker logs "$cname" 2>&1 | grep "\[decode-prof\]" | tail -1)
  ap=$(docker logs "$cname" 2>&1 | grep "\[attn-prof\]" | tail -1)
  echo "NINFER depth=$d  $dp  $ap" | tee -a "$NSUM"
  docker rm -f "$cname" >/dev/null 2>&1 || true
  wait_cards_free
done

echo "############ LLAMA leg (nsys flash-kernel fraction) ############"
for d in "${DS[@]}"; do
  echo "=== llama depth=$d ==="
  rep="$NSYS_DIR/llama_$d"
  # one prefill (-n 0), 1 rep; nsys captures warmup+timed but we use the RATIO (warmup-agnostic)
  nsys profile --trace=cuda --force-overwrite=true -o "$rep" \
    "$LLAMA" -m "$GGUF" -p "$d" -n 0 -r 1 -fa on -ctk q5_1 -ctv q5_1 -ngl 99 -sm layer -ts 1/1 \
    > "$OUT/llama.$d.txt" 2>&1 || { echo "llama d=$d bench FAILED"; tail -15 "$OUT/llama.$d.txt"; }
  # prefill tok/s from llama-bench stdout (pp<depth> row)
  tps=$(grep -oiE "pp[0-9]+ *\| *[0-9.]+" "$OUT/llama.$d.txt" | grep -oE "[0-9.]+$" | tail -1)
  nsys stats --report cuda_gpu_kern_sum --format csv "$rep.nsys-rep" > "$OUT/kern.$d.csv" 2>/dev/null || \
    nsys stats --report cuda_gpu_kern_sum --format csv "$rep.sqlite" > "$OUT/kern.$d.csv" 2>/dev/null || true
  echo "LLAMA depth=$d  prefill_tok/s=$tps  kern-csv=$OUT/kern.$d.csv" | tee -a "$NSUM"
  wait_cards_free
done

echo
echo "############ ANALYSIS ############"
python3 - "$OUT" "$DEPTHS" <<'PY'
import sys, re, glob, os, csv
out, depths = sys.argv[1], [int(x) for x in sys.argv[2].split(',')]
def grabf(pat, s):
    m=re.search(pat, s); return float(m.group(1)) if m else None
print(f"{'depth':>7} | {'--- NINFER (s) ---':^34} | {'--- LLAMA (s) ---':^26}")
print(f"{'':>7} | {'qkv':>7} {'fa(flash+stg)':>13} {'oproj':>7} {'d_attn':>7} | {'flash%':>7} {'pf_s':>7} {'attn_s':>8}")
for d in depths:
    summ=open(os.path.join(out,'summary.txt')).read()
    ln=[l for l in summ.splitlines() if l.startswith(f'NINFER depth={d} ')]
    qkv=fa=op=dattn=None
    if ln:
        qkv=grabf(r'qkv_proj=([\d.]+)',ln[0]); fa=grabf(r'\bfa=([\d.]+)',ln[0]); op=grabf(r'o_proj=([\d.]+)',ln[0])
        dattn=grabf(r'attn=([\d.]+)',ln[0])
    # llama flash fraction from kernel csv
    flashfrac=None; total=0.0; flash=0.0
    kc=os.path.join(out,f'kern.{d}.csv')
    if os.path.exists(kc):
        with open(kc) as f:
            rdr=csv.DictReader(f)
            for row in rdr:
                # find the time column (ns) and name column across nsys versions
                tks=[k for k in row if k and ('Total Time' in k or 'Time (ns)' in k)]
                nk=[k for k in row if k and ('Name' in k)]
                if not tks or not nk: continue
                try: t=float(row[tks[0]].replace(',',''))
                except: continue
                nm=row[nk[0]]; total+=t
                if 'flash_attn_ext' in nm: flash+=t
        flashfrac=flash/total if total>0 else None
    lt=[l for l in summ.splitlines() if l.startswith(f'LLAMA depth={d} ')]
    tps=grabf(r'prefill_tok/s=([\d.]+)', lt[0]) if lt else None
    pf_s=(d/tps) if tps else None
    attn_s=(flashfrac*pf_s) if (flashfrac and pf_s) else None
    def f(x,p='7.2f'): return format(x,p) if isinstance(x,float) else f'{"?":>7}'
    print(f"{d//1024:>5}K | {f(qkv)} {f(fa,'13.2f')} {f(op)} {f(dattn)} | "
          f"{(format(100*flashfrac,'6.1f')+'%') if flashfrac else '   ?  '} {f(pf_s)} {f(attn_s,'8.2f')}")
print("\nFORK VERDICT: compare ninfer fa(flash+stg) vs llama attn_s.")
print(" - fa >> attn_s  -> ninfer's FLASH KERNEL/staging is the deficit (kernel exec, not config).")
print(" - fa ~= attn_s but ninfer qkv+oproj large -> the PROJECTION GEMMs are the deficit.")
PY
echo "=== raw: $NSUM ; kernel csvs: $OUT/kern.*.csv ==="
