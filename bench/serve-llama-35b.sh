#!/usr/bin/env bash
# Matched llama.cpp 35B serve for the P3 A/B — byte-identical to the model-router's catalog line 126
# (titan serve-qwen3-35b-llama-v100.sh: SM70 build, -sm layer -ts 1,1, q5_1 KV, -c 262144, MTP OFF,
# nothink template). Native (not docker). Uses V100s idx 1,2. Foreground; Ctrl-C to stop.
#   PORT=8001 ./serve-llama-35b.sh
set -euo pipefail
PORT="${PORT:-8001}"
CTX="${CTX:-262144}"
LLAMA=/home/anl/v100/llama.cpp/build/bin
GGUF=/home/anl/v100/models/qwen3.6-35b-a3b-MTP-GGUF/Qwen3.6-35B-A3B-IQ4_XS-4.19bpw.gguf
TMPL=/home/anl/v100/model-router/serve/titan/qwen3-template-nothink.jinja

export LD_LIBRARY_PATH="$LLAMA:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=1,2
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "serve-llama-35b: port=$PORT ctx=$CTX kv=q5_1 MTP=off (matched router config, cards 1,2)"
exec "$LLAMA/llama-server" \
  -m "$GGUF" \
  --host 0.0.0.0 --port "$PORT" \
  -ngl 99 -fa 1 -b 2048 -ub 2048 -sm layer -ts 1,1 \
  -ctk q5_1 -ctv q5_1 -c "$CTX" -t 12 --parallel 1 --metrics \
  --jinja --chat-template-file "$TMPL"
