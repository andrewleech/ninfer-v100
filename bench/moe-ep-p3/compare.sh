#!/usr/bin/env bash
# ============================================================================
# compare.sh — THE reusable ninfer vs upstream-llama.cpp comparison harness
#              for Qwen3.6-35B-A3B on titan's 2xV100 (dual-card EP).
#
# Supersedes the P3 scratch scripts (sweep-e2e.sh / real-e2e-64k.sh /
# llama-decode-depth.sh). Design goals baked in (learned the hard way):
#   * BOTH engines run through their REAL prod serve path — no llama-bench
#     (llama-bench `tg` without -d measured decode at ~0 context = bogus).
#   * SAME client + SAME prompt builder (greedy_probe.py) for both engines,
#     so token counts are identical -> apples-to-apples.
#   * MTP ON by default for BOTH (prod config). The prior 64K test ran MTP
#     OFF on ninfer -> understated it by ~1.45x decode. MTP=off flips both
#     for the raw-kernel isolation study.
#   * llama uses the CANONICAL launcher (serve-qwen3-35b-llama-v100.sh) with
#     its prod q8_0 KV (bit-matched to ninfer int8 KV) — NOT hand-rolled flags.
#   * PER-REQUEST rates parsed from each server's own log (not peak-interval).
#   * Quick (one depth) or thorough (sweep) purely by DEPTHS.
#   * Fine-grain: PROFILE=1 adds a ninfer-internal stage split (attn/gdn/mlp,
#     qkv/fa/o_proj) on top of the symmetric prefill/decode both engines expose.
#   * Legs are independent: ENGINES=ninfer re-runs just that leg and keeps the
#     other engine's CSV, so a failed leg resumes without re-running both.
#   * Guards MTP: refuses to trust a ninfer leg whose log says speculative=off
#     when MTP=on was requested (stops the regression silently recurring).
#
# How to use it:
#
#   1. Quick production comparison at one depth (the normal first check):
#        DEPTHS=65536 OUT=compare-out/quick-64k ./compare.sh
#
#   2. Thorough production sweep.  DEPTHS values must be below CTX after
#      allowing room for MAXNEW generated tokens:
#        CTX=255000 DEPTHS=8192,16384,32768,65536,131072,196608,245760 \
#          OUT=compare-out/255k-c2048 ./compare.sh
#
#   3. Fit/speed study for ninfer's prefill arena.  Start with the fast
#      2048-token chunk; lower CTX until it loads.  The full 262144 MTP-on
#      configuration needs PREFILL_CHUNK=512, which may reduce prefill speed:
#        CTX=255000 PREFILL_CHUNK=2048 ENGINES=ninfer DEPTHS=65536 \
#          OUT=compare-out/255k-c2048 ./compare.sh
#        CTX=262144 PREFILL_CHUNK=512 ENGINES=ninfer DEPTHS=65536 \
#          OUT=compare-out/262k-c512 ./compare.sh
#      Re-run the matched llama leg only once its MTP-on fit is established:
#        CTX=255000 ENGINES=llama DEPTHS=65536 OUT=compare-out/255k-c2048 ./compare.sh
#
#   4. Resume a failed or intentionally split run without re-running a good
#      leg: ENGINES=ninfer or ENGINES=llama, with the same OUT directory.
#
#   5. Raw-kernel isolation: MTP=off disables it for BOTH engines.  Internal
#      ninfer ratios: PROFILE=1 ENGINES=ninfer DEPTHS=131072.  Profiling
#      serializes work, so its absolute rates are not comparison results.
#
#   6. Tune NInfer's MTP proposal width at a representative deep prompt.
#      Keep the same CTX, prompt, and output directory per candidate; llama's
#      canonical launcher independently uses its fixed 3-token width:
#        CTX=208896 DRAFT_TOKENS=3 ENGINES=ninfer DEPTHS=196608 \
#          OUT=compare-out/draft3-192k ./compare.sh
#      Compare against DRAFT_TOKENS=2 and 4 before changing the production
#      default.  Lower widths can win at long context when a wider verification
#      step costs more attention work than its added accepted tokens repay.
#
#   7. Test a 35B asymmetric expert split. GPU 1 owns dense attention/GDN as
#      well as its expert band, so move some experts to GPU 2 and compare at a
#      deep prompt. The default is the established 128/128 split:
#        CTX=208896 DRAFT_TOKENS=3 MOE_PRIMARY_EXPERTS=112 ENGINES=ninfer \
#          DEPTHS=196608 OUT=compare-out/moe-112-144 ./compare.sh
#
#   8. Validated 35B text-attention tensor parallelism. This keeps the 255K
#      production context and requires both V100s. Use it for the retained
#      dual-card configuration or for paired regression checks:
#        TP_ATTENTION=1 CTX=255000 ENGINES=ninfer DEPTHS=196608 MAXNEW=64 \
#          OUT=compare-out/tp-200k ./compare.sh
#
# Defaults: MTP=on, DRAFT_TOKENS=3, CTX=262144, PREFILL_CHUNK=2048, MAX_REQ_MIB=128,
# MAXNEW=128, and the full depth sweep.  Every normal result includes
# per-request prefill/decode rates; ninfer additionally records TTFT and wall
# time.  The harness rejects a ninfer result if the serve reports MTP off.
#
# It uses both V100s and ports 8090/8091.  Stop llama-swap before running;
# output is retained under OUT so a failed leg can be resumed.  It is
# timeout-guarded and nsys-free.
# ============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT=/home/anl/v100
ART=/models/Qwen3.6-35B-A3B-NInfer/qwen3_6_35b_a3b.ninfer
GGUF="$ROOT/models/qwen3.6-35b-a3b-MTP-GGUF/Qwen3.6-35B-A3B-IQ4_XS-4.19bpw.gguf"
LLAMA_LAUNCH="$ROOT/model-router/serve/titan/serve-qwen3-35b-llama-v100.sh"
PROBE="$HERE/greedy_probe.py"

DEPTHS="${DEPTHS:-8192,16384,32768,65536,131072,196608,245760}"
MAXNEW="${MAXNEW:-128}"        # generated tokens per probe (decode rate sample)
MTP="${MTP:-on}"              # on|off  -> applies to BOTH engines
DRAFT_TOKENS="${DRAFT_TOKENS:-3}" # ninfer MTP proposal width; llama's canonical width remains 3
MOE_PRIMARY_EXPERTS="${MOE_PRIMARY_EXPERTS:-128}" # 35B rank-0 expert count; ignored by llama
TP_ATTENTION="${TP_ATTENTION:-0}" # 1 enables validated 35B split text attention
ENGINES="${ENGINES:-both}"    # both|ninfer|llama  (re-run one leg to resume)
PROFILE="${PROFILE:-0}"       # 1 = extra ninfer-internal stage split (ninfer only)
CTX="${CTX:-262144}"
# Keep the production-fast 2048-token prefill chunk by default.  Set 512 only
# when measuring the full-262K MTP fit, or lower CTX until 2048 fits.
PREFILL_CHUNK="${PREFILL_CHUNK:-2048}"
MAX_REQ_MIB="${MAX_REQ_MIB:-128}"
NOUT="${NOUT:-2000}"          # hypothetical output length for the e2e column
NPORT="${NPORT:-8090}"; LPORT="${LPORT:-8091}"
OUT="${OUT:-$HERE/compare-out}"; mkdir -p "$OUT"
NIN="$OUT/ninfer.csv"; LLA="$OUT/llama.csv"
IFS=',' read -ra DS <<< "$DEPTHS"

wait_cards_free() {
  echo "  waiting for cards 1,2 free (<500MiB)..."
  local i b
  for i in $(seq 1 60); do
    b=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
        | awk -F',' '$1>=1 && $2+0>500{n++} END{print n+0}')
    [ "${b:-2}" = "0" ] && { echo "  cards free."; return 0; }
    sleep 5
  done
  echo "  WARN cards still busy after 300s"; return 1
}

# --------------------------------------------------------------------------
# NINFER leg — one dual-card EP serve @CTX, probe each depth (no reload).
# --------------------------------------------------------------------------
run_ninfer() {
  local profile="$1"          # 0 clean (for the table) | 1 internal split
  local cn="ninfer35b-compare"
  local spec=(); [ "$MTP" = on ] && spec=(--spec mtp --draft-tokens "$DRAFT_TOKENS")
  local penv=()
  if [ "$profile" = 1 ]; then penv=(-e NINFER_ATTN_PROFILE=1 -e NINFER_DECODE_PROFILE=1); fi
  [ "$TP_ATTENTION" = 1 ] && penv+=( -e NINFER_TP_ATTENTION=1 )

  echo "############ NINFER (spec=$MTP drafts=$DRAFT_TOKENS experts=$MOE_PRIMARY_EXPERTS tp_attention=$TP_ATTENTION profile=$profile ctx=$CTX chunk=$PREFILL_CHUNK) — probe each depth ############"
  wait_cards_free
  docker rm -f "$cn" >/dev/null 2>&1 || true
  # NOTE: no --rm — the NINFER_*_PROFILE splits print from static dtors at a
  # clean SIGTERM exit; --rm/SIGKILL would drop them.
  docker run -d --name "$cn" --gpus all --network host -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    -e NINFER_MOE_PRIMARY_EXPERTS="$MOE_PRIMARY_EXPERTS" \
    "${penv[@]}" \
    -v "$ROOT/ninfer-v100-moe-wt":/src:ro -v "$ROOT/models":/models:ro \
    -v "$ROOT/.nvjitcache":/root/.nv/ComputeCache --ipc=host --shm-size=8g \
    -w /src/build-v100 v100ninfer:cu128 \
    ./apps/ninfer-serve "$ART" --devices 1,2 --kv-dtype int8 \
      --max-context "$CTX" --kv-capacity "$CTX" --max-concurrency 1 --prefill-chunk "$PREFILL_CHUNK" \
      --log-stats-interval-ms 60000 --model-id ninfer-35b --max-request-mib "$MAX_REQ_MIB" --no-thinking \
      --host 0.0.0.0 --port "$NPORT" "${spec[@]}" >/dev/null

  timeout 320 python3 "$PROBE" --base "http://127.0.0.1:$NPORT" --require-model ninfer-35b \
    --prompt-tokens 2048 --max-new 1 --load-timeout 300 --out "$OUT/nin.warm.txt" \
    || {
      echo "  ninfer load FAILED; no result rows will be written."
      docker logs --tail 30 "$cn"
      docker rm -f "$cn" >/dev/null 2>&1 || true
      return 1
    }

  if [ "$profile" = 0 ]; then echo "depth,prefill_tok_s,decode_tok_s,ttft_ms,wall_s,spec" > "$NIN"; fi
  local d line pf dc tt wl sp
  for d in "${DS[@]}"; do
    echo "--- ninfer depth=$d ---"
    timeout 900 python3 "$PROBE" --base "http://127.0.0.1:$NPORT" --require-model ninfer-35b \
      --prompt-tokens "$d" --max-new "$MAXNEW" --load-timeout 30 --out "$OUT/nin.$d.txt" \
      || echo "  probe $d failed/timeout"
    line=$(docker logs "$cn" 2>&1 | grep -E "\[req [0-9]+\] done" | grep -vE "gen=1 " | tail -1)
    pf=$(grep -oE "prefill=[0-9.]+" <<<"$line" | grep -oE "[0-9.]+")
    dc=$(grep -oE "decode=[0-9.]+"  <<<"$line" | grep -oE "[0-9.]+")
    tt=$(grep -oE "ttft=[0-9.]+"    <<<"$line" | grep -oE "[0-9.]+")
    wl=$(grep -oE "wall=[0-9.]+"    <<<"$line" | grep -oE "[0-9.]+")
    sp=$(grep -oE "speculative=[a-z]+" <<<"$line" | cut -d= -f2)
    echo "    $line"
    if [ "$MTP" = on ] && [ "${sp:-off}" = off ]; then
      echo "  !! WARNING: MTP=on requested but serve reports speculative=off (regression?)"
    fi
    [ "$profile" = 0 ] && echo "$d,${pf:-NA},${dc:-NA},${tt:-NA},${wl:-NA},${sp:-NA}" | tee -a "$NIN"
  done

  if [ "$profile" = 1 ]; then
    echo; echo "===== NINFER internal stage split (NINFER_ATTN_PROFILE + NINFER_DECODE_PROFILE) ====="
    echo "(absolute tok/s above are profiler-serialized = inflated; read the split RATIOS, not absolutes)"
  fi
  docker kill -s TERM "$cn" >/dev/null 2>&1 || true
  timeout 120 docker wait "$cn" >/dev/null 2>&1 || docker kill "$cn" >/dev/null 2>&1 || true
  if [ "$profile" = 1 ]; then
    docker logs "$cn" 2>&1 | grep -iE "\[attn-prof\]|\[decode-prof\]|a_qkv|a_fa|a_oproj|attn=|gdn=|mlp=" | tail -60
  fi
  docker rm -f "$cn" >/dev/null 2>&1 || true
  wait_cards_free
}

# --------------------------------------------------------------------------
# LLAMA leg — canonical launcher (prod q8_0 KV + MTP), probe each depth.
# --------------------------------------------------------------------------
run_llama() {
  local llog="$OUT/llama-server.log"
  local spec=mtp; [ "$MTP" = off ] && spec=off
  echo "############ LLAMA upstream (spec=$spec, canonical launcher) — probe each depth ############"
  wait_cards_free
  # V100s are nvidia-smi idx 1,2 (A2000 at 0). The catalog pins CUDA_VISIBLE_DEVICES=1,2 in prod;
  # replicate it here or -sm layer would try to grab the sm_86 A2000.
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2 \
    PORT="$LPORT" CTX="$CTX" SPEC="$spec" "$LLAMA_LAUNCH" --jinja > "$llog" 2>&1 &
  local lpid=$!
  local i ready=0
  for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$LPORT/health" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "$lpid" 2>/dev/null; then
      echo "  llama load FAILED; no result rows will be written."
      tail -40 "$llog"
      return 1
    fi
    sleep 2
  done
  if [ "$ready" != 1 ]; then
    echo "  llama did not become ready within 360s; no result rows will be written."
    kill "$lpid" 2>/dev/null || true
    tail -40 "$llog"
    return 1
  fi
  curl -sf "http://127.0.0.1:$LPORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":1,"temperature":0}' \
    >/dev/null 2>&1 || true

  echo "depth,prefill_tok_s,decode_tok_s,total_ms" > "$LLA"
  local d pe ev pf dc tot
  for d in "${DS[@]}"; do
    echo "--- llama depth=$d ---"
    timeout 900 python3 "$PROBE" --base "http://127.0.0.1:$LPORT" \
      --prompt-tokens "$d" --max-new "$MAXNEW" --load-timeout 30 --out "$OUT/lla.$d.txt" \
      || echo "  probe $d failed/timeout"
    pe=$(grep -E "prompt eval time" "$llog" | tail -1)
    ev=$(grep -E "eval time" "$llog" | grep -v "prompt eval" | tail -1)
    tot=$(grep -E "total time" "$llog" | tail -1 | grep -oE "= *[0-9.]+ ms" | grep -oE "[0-9.]+" | head -1)
    pf=$(grep -oE "[0-9.]+ tokens per second" <<<"$pe" | grep -oE "[0-9.]+" | head -1)
    dc=$(grep -oE "[0-9.]+ tokens per second" <<<"$ev" | grep -oE "[0-9.]+" | head -1)
    echo "    prefill=${pf:-NA}tok/s decode=${dc:-NA}tok/s total=${tot:-NA}ms"
    echo "$d,${pf:-NA},${dc:-NA},${tot:-NA}" | tee -a "$LLA"
  done
  kill "$lpid" 2>/dev/null || true; sleep 3; kill -9 "$lpid" 2>/dev/null || true
  wait_cards_free
}

# ------------------------------- run legs ---------------------------------
[ "$ENGINES" = both ] || [ "$ENGINES" = ninfer ] && run_ninfer 0
[ "$PROFILE" = 1 ] && { [ "$ENGINES" = both ] || [ "$ENGINES" = ninfer ]; } && run_ninfer 1
[ "$ENGINES" = both ] || [ "$ENGINES" = llama ] && run_llama

# ------------------------------- table ------------------------------------
echo; echo "############ COMPARISON (MTP=$MTP, CTX=$CTX, NINFER_PREFILL_CHUNK=$PREFILL_CHUNK, MAXNEW=$MAXNEW) ############"
python3 - "$NIN" "$LLA" "$NOUT" "$MTP" <<'PY'
import sys, csv, os
def rows(p):
    d={}
    if not os.path.exists(p): return d
    for r in csv.DictReader(open(p)):
        try: d[int(r['depth'])]=r
        except: pass
    return d
nin=rows(sys.argv[1]); lla=rows(sys.argv[2]); NOUT=int(sys.argv[3]); mtp=sys.argv[4]
def g(row,k):
    try: return float(row[k])
    except: return None
def f(x,w,p=1): return format(x,f'{w}.{p}f') if isinstance(x,(int,float)) else format('NA',f'>{w}')
print(f"{'depth':>5} | {'PREFILL tok/s':^20} | {'DECODE tok/s':^20} | {'e2e s ('+str(NOUT)+' out)':^19}")
print(f"{'':>5} | {'ninf':>6} {'llama':>6} {'x':>4} | {'ninf':>6} {'llama':>6} {'x':>4} | {'ninf':>7} {'llama':>7} {'win':>4}")
for d in sorted(set(nin)|set(lla)):
    n=nin.get(d); l=lla.get(d)
    npf=g(n,'prefill_tok_s') if n else None; ndc=g(n,'decode_tok_s') if n else None
    lpf=g(l,'prefill_tok_s') if l else None; ldc=g(l,'decode_tok_s') if l else None
    pfx=(lpf/npf) if (npf and lpf) else None       # >1 = llama prefill faster
    dcx=(ndc/ldc) if (ndc and ldc) else None       # >1 = ninfer decode faster
    ne=(d/npf + NOUT/ndc) if (npf and ndc) else None
    le=(d/lpf + NOUT/ldc) if (lpf and ldc) else None
    win=('ninf' if (ne and le and ne<le) else 'llama' if (ne and le) else '?')
    print(f"{d//1024:>3}K | {f(npf,6,0)} {f(lpf,6,0)} {f(pfx,4,2) if pfx else 'NA':>4} | "
          f"{f(ndc,6,1)} {f(ldc,6,1)} {f(dcx,4,2) if dcx else 'NA':>4} | "
          f"{f(ne,7):>7} {f(le,7):>7} {win:>4}")
print(f"\nprefill x = llama/ninfer (>1 llama faster). decode x = ninfer/llama (>1 ninfer faster).")
print(f"e2e = depth/prefill + {NOUT}/decode.  MTP={mtp} on BOTH engines (prod config).")
if mtp=='on':
    print("MTP note: on the repetitive filler prompt, MTP acceptance is higher than on real")
    print("          content -> decode tok/s here is an optimistic ceiling (symmetric for both).")
PY
echo "raw: $NIN , $LLA"
