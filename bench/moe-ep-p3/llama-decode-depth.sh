#!/usr/bin/env bash
# Corrected at-depth DECODE for llama (the P12 fix: llama-bench tg WITHOUT -d measured decode at ~0
# context; -d <depth> sets up a real D-token KV first, so tg reflects decode-at-depth like ninfer's).
# Then emit the CORRECTED e2e table combining: ninfer prefill+decode (valid, sweep-out/ninfer.csv),
# llama prefill (valid, sweep-out/llama.csv), llama decode-at-depth (measured here).
# Run INSIDE a coordinated window. HARD-timeout-guarded. nsys-free.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GGUF="/home/anl/v100/models/qwen3.6-35b-a3b-MTP-GGUF/Qwen3.6-35B-A3B-IQ4_XS-4.19bpw.gguf"
LB=/home/anl/v100/llama.cpp/build/bin/llama-bench
DEPTHS="${DEPTHS:-8192,16384,32768,65536,131072,196608,245760}"
MAXNEW="${MAXNEW:-64}"
OUT="${OUT:-$HERE/sweep-out}"; mkdir -p "$OUT"
LDEC="$OUT/llama_decode.csv"; echo "depth,decode_at_depth_tok_s" > "$LDEC"
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

echo "############ LLAMA decode-at-depth (-d <depth>) ############"
wait_cards_free
for d in "${DS[@]}"; do
  echo "--- llama tg@depth=$d ---"
  # -d D sets up a D-token context (a real prefill) then times -n gen -> decode AT depth D.
  r=$(timeout 600 "$LB" -m "$GGUF" -p 0 -n "$MAXNEW" -d "$d" -r 1 -fa on -ctk q5_1 -ctv q5_1 \
        -ngl 99 -sm layer -ts 1/1 -ub 2048 -b 2048 2>/dev/null)
  # row: "... | tg<MAXNEW> @ d<depth> | <t/s> ± .. |"  (llama-bench labels tg rows with @ dN)
  tg=$(echo "$r" | grep -oE "tg$MAXNEW[^|]*\|[^|]*" | grep -oE "[0-9]+\.[0-9]+" | head -1)
  echo "$d,${tg:-NA}" | tee -a "$LDEC"
  wait_cards_free
done

echo; echo "############ CORRECTED E2E TABLE ############"
python3 - "$OUT/ninfer.csv" "$OUT/llama.csv" "$LDEC" "$MAXNEW" <<'PY'
import sys,csv
def load2(p,ka,kb):
    d={}
    for r in csv.DictReader(open(p)):
        try: d[int(r['depth'])]=(float(r[ka]),float(r[kb]))
        except: pass
    return d
def load1(p,k):
    d={}
    for r in csv.DictReader(open(p)):
        try: d[int(r['depth'])]=float(r[k])
        except: pass
    return d
nin=load2(sys.argv[1],'prefill_tok_s','decode_tok_s')        # ninfer prefill + decode (both valid)
llap=load1(sys.argv[2],'prefill_tok_s')                       # llama prefill (valid)
ldec=load1(sys.argv[3],'decode_at_depth_tok_s')              # llama decode-at-depth (new)
nout=int(sys.argv[4]); NR=2000
def f(x,w,p=1): return format(x,f'{w}.{p}f') if isinstance(x,(int,float)) else format('NA',f'>{w}')
print(f"{'depth':>5} | {'PREFILL tok/s':^18} | {'DECODE@depth tok/s':^20} | {'e2e s ('+str(NR)+' out)':^21}")
print(f"{'':>5} | {'ninf':>6} {'llama':>6} {'x':>3} | {'ninf':>6} {'llama':>6} {'x':>3} | {'ninf':>7} {'llama':>7} {'win':>4}")
for d in sorted(nin):
    npf,ndc=nin[d]; lpf=llap.get(d); ldc=ldec.get(d)
    pfx=(lpf/npf) if (npf and lpf) else None
    dcx=(ndc/ldc) if (ndc and ldc) else None   # ninfer/llama decode (>1 = ninfer faster)
    ne=(d/npf + NR/ndc) if (npf and ndc) else None
    le=(d/lpf + NR/ldc) if (lpf and ldc) else None
    win = ('ninf' if (ne and le and ne<le) else 'llama' if (ne and le) else '?')
    print(f"{d//1024:>3}K | {f(npf,6,0)} {f(lpf,6,0)} {f(pfx,3,2) if pfx else 'NA':>3} | "
          f"{f(ndc,6,1)} {f(ldc,6,1)} {f(dcx,3,2) if dcx else 'NA':>3} | {f(ne,7):>7} {f(le,7):>7} {win:>4}")
print(f"\ndecode x = ninfer/llama (>1 ninfer faster). e2e({NR} out) = depth/prefill + {NR}/decode.")
print("ninfer decode is MTP-OFF; MTP ~1.45x would further cut ninfer e2e (add a MTP-on row separately).")
PY
echo "raw: $LDEC"
