#!/usr/bin/env bash
# REAL single start-to-finish request @64K on BOTH real serve paths (not llama-bench):
#   ninfer serve (dp4a, dual-card EP)  vs  upstream llama-server (dual-card, all-GPU).
# Measures CLIENT wall time (start->finish) + each server's own PREFILL/DECODE split, so we can
# check the sweep's per-stage numbers hold in a real request (and get real AT-DEPTH decode).
# No-thinking on BOTH (ninfer --no-thinking; llama no-think jinja) so decode isn't inflated by
# reasoning tokens. Run INSIDE a coordinated window. Timeout-guarded.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT=/home/anl/v100
ART=/models/Qwen3.6-35B-A3B-NInfer/qwen3_6_35b_a3b.ninfer
GGUF="$ROOT/models/qwen3.6-35b-a3b-MTP-GGUF/Qwen3.6-35B-A3B-IQ4_XS-4.19bpw.gguf"
NOTHINK="$ROOT/v100-llm-kit/scripts/linux/qwen3-template-nothink.jinja"
CTX="${CTX:-98304}"; PORT="${PORT:-8090}"; LPORT="${LPORT:-8091}"
PROMPT_TOKENS="${PROMPT_TOKENS:-65536}"; MAXNEW="${MAXNEW:-512}"
OUT="${OUT:-$HERE/real64k-out}"; mkdir -p "$OUT"; SUM="$OUT/summary.txt"; : > "$SUM"

wait_cards_free() {
  echo "  waiting for cards 1,2 free (<500MiB)..."
  for _ in $(seq 1 60); do
    local b; b=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
       | awk -F',' '$1>=1 && $2+0>500{n++} END{print n+0}')
    [ "${b:-2}" = "0" ] && { echo "  cards free."; return 0; }
    sleep 5
  done; echo "  WARN cards busy after 300s"; return 1
}

echo "############ NINFER dp4a serve — real 64K request ############"
wait_cards_free
cn=ninfer35b-real64k
docker rm -f "$cn" >/dev/null 2>&1 || true
docker run -d --name "$cn" --gpus all --network host -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -v "$ROOT/ninfer-v100-moe-wt":/src:ro -v "$ROOT/models":/models:ro \
  -v "$ROOT/.nvjitcache":/root/.nv/ComputeCache --ipc=host --shm-size=8g \
  -w /src/build-v100 v100ninfer:cu128 \
  ./apps/ninfer-serve "$ART" --devices 1,2 --kv-dtype int8 --max-context "$CTX" --kv-capacity "$CTX" \
    --max-concurrency 1 --prefill-chunk 2048 --model-id ninfer-35b --max-request-mib 160 --no-thinking \
    --host 0.0.0.0 --port "$PORT" >/dev/null
# warm (small), then TIME the real 64K request
timeout 300 python3 "$HERE/greedy_probe.py" --base "http://127.0.0.1:$PORT" --require-model ninfer-35b \
  --prompt-tokens 1024 --max-new 1 --load-timeout 260 --out "$OUT/ninwarm.txt" >/dev/null 2>&1 || { echo "ninfer load FAIL"; docker logs --tail 20 "$cn"; }
t0=$(date +%s.%N)
timeout 400 python3 "$HERE/greedy_probe.py" --base "http://127.0.0.1:$PORT" --require-model ninfer-35b \
  --prompt-tokens "$PROMPT_TOKENS" --max-new "$MAXNEW" --load-timeout 30 --out "$OUT/nin.txt" || echo "ninfer req FAIL"
t1=$(date +%s.%N)
nwall=$(awk "BEGIN{printf \"%.2f\", $t1-$t0}")
nreq=$(docker logs "$cn" 2>&1 | grep -E "\[req [0-9]+\] done" | grep -vE "gen=1 " | tail -1)
echo "NINFER  client_wall=${nwall}s  server: $nreq" | tee -a "$SUM"
docker kill -s TERM "$cn" >/dev/null 2>&1 || true; timeout 120 docker wait "$cn" >/dev/null 2>&1 || docker kill "$cn" >/dev/null 2>&1 || true
docker rm -f "$cn" >/dev/null 2>&1 || true
wait_cards_free

echo "############ upstream llama-server — real 64K request ############"
export LD_LIBRARY_PATH="$ROOT/llama.cpp/build/bin:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
"$ROOT/llama.cpp/build/bin/llama-server" -m "$GGUF" --host 127.0.0.1 --port "$LPORT" \
  -ngl 99 -fa on -b 2048 -ub 2048 -ctk q5_1 -ctv q5_1 -c "$CTX" -sm layer -ts 1,1 \
  --parallel 1 --no-webui --chat-template-file "$NOTHINK" -t 12 > "$OUT/llama-server.log" 2>&1 &
lpid=$!
# wait for /health ok
for _ in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$LPORT/health" >/dev/null 2>&1 && break; sleep 2
done
# warm
curl -sf "http://127.0.0.1:$LPORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":1,"temperature":0}' >/dev/null 2>&1 || true
t0=$(date +%s.%N)
timeout 400 python3 "$HERE/greedy_probe.py" --base "http://127.0.0.1:$LPORT" \
  --prompt-tokens "$PROMPT_TOKENS" --max-new "$MAXNEW" --load-timeout 30 --out "$OUT/llama.txt" || echo "llama req FAIL"
t1=$(date +%s.%N)
lwall=$(awk "BEGIN{printf \"%.2f\", $t1-$t0}")
# llama-server logs "prompt eval time = A ms / P tokens (X t/s)" + "eval time = B ms / G tokens (Y t/s)"
pe=$(grep -E "prompt eval time" "$OUT/llama-server.log" | tail -1)
ev=$(grep -E "^eval time|[^a-z]eval time" "$OUT/llama-server.log" | grep -v "prompt eval" | tail -1)
echo "LLAMA   client_wall=${lwall}s" | tee -a "$SUM"
echo "  $pe" | tee -a "$SUM"
echo "  $ev" | tee -a "$SUM"
kill "$lpid" 2>/dev/null || true; sleep 3; kill -9 "$lpid" 2>/dev/null || true
wait_cards_free

echo; echo "############ REAL 64K COMPARISON ############"; cat "$SUM"
echo "(ninfer server line: ttft=prefill, prefill=Xtok/s, decode=Ytok/s, wall=total)"
echo "(llama: prompt eval = prefill; eval = decode; client_wall = start->finish for both)"
