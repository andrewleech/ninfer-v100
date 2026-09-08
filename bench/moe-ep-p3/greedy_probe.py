#!/usr/bin/env python3
"""Single greedy fixed-prompt completion against a running ninfer serve — for the dp4a A/B.

Waits for TRUE model-load (a real completion, not /health), builds a ~N-token prompt with the
same filler unit the depth sweep uses, sends one temperature-0 completion, and writes the
generated text to --out. Used twice (dp4a vs NINFER_MOE_PREFILL_SCALAR=1) on the SAME prompt so
the outputs can be diffed for coherence + agreement. No third-party deps.
"""
import argparse, json, sys, time, urllib.request

UNIT = ("On a Volta V100 the mixture-of-experts router selects a small subset of expert "
        "feed-forward networks for each token, so the compute per token stays far below a "
        "dense model of the same total parameter count while the memory footprint does not. "
        "Sharding the experts across two GPUs lets each card own half of them and reduce the "
        "partial sums over the NVLink interconnect back into one output per token. ")

def http_json(url, payload=None, timeout=1200):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer x")
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())

def get_model_id(base):
    try:
        d = (http_json(base + "/v1/models", timeout=10).get("data") or [])
        return d[0]["id"] if d else None
    except Exception:
        return None

def wait_for_load(base, deadline_s):
    t0 = time.time(); last = None
    while time.time() - t0 < deadline_s:
        mid = get_model_id(base)
        if mid:
            try:
                r = http_json(base + "/v1/chat/completions", {
                    "model": mid, "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1, "temperature": 0, "stream": False}, timeout=60)
                if r.get("choices"):
                    return mid, time.time() - t0
            except Exception as e:
                last = e
        time.sleep(5)
    raise RuntimeError(f"model not ready in {deadline_s}s (last: {last})")

def calibrate_tpu(base, mid):
    probe = "".join(f"[{i}] {UNIT}" for i in range(50))
    r = http_json(base + "/v1/chat/completions", {
        "model": mid, "messages": [{"role": "user", "content": probe}],
        "max_tokens": 1, "temperature": 0, "stream": False}, timeout=120)
    return r["usage"]["prompt_tokens"] / 50.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8090")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=4096)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--load-timeout", type=int, default=900)
    ap.add_argument("--require-model", default=None)
    args = ap.parse_args()

    print(f"[probe] waiting for load at {args.base} (<= {args.load_timeout}s)...", flush=True)
    mid, load_s = wait_for_load(args.base, args.load_timeout)
    print(f"[probe] model '{mid}' ready after {load_s:.0f}s", flush=True)
    if args.require_model and mid != args.require_model:
        raise SystemExit(f"[probe] WRONG MODEL '{mid}' != '{args.require_model}' — llama-swap on this port?")

    tpu = calibrate_tpu(args.base, mid)
    reps = max(1, int(args.prompt_tokens / max(1.0, tpu)))
    prompt = "".join(f"[{i}] {UNIT}" for i in range(reps))
    r = http_json(args.base + "/v1/chat/completions", {
        "model": mid, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_new, "temperature": 0, "stream": False}, timeout=3600)
    usage = r.get("usage", {})
    msg = (r["choices"][0].get("message") or {})
    text = msg.get("content") or msg.get("reasoning_content") or ""
    with open(args.out, "w") as f:
        f.write(text.strip() + "\n")
    print(f"[probe] prompt_tokens={usage.get('prompt_tokens')} "
          f"completion_tokens={usage.get('completion_tokens')} -> {args.out}", flush=True)
    print(f"[probe] TEXT: {text.strip()[:300]}", flush=True)

if __name__ == "__main__":
    main()
