#!/usr/bin/env python3
"""Depth-sweep latency benchmark for a running OpenAI-compatible ninfer/llama serve.

Single-stream. For each target prompt depth it builds a prompt of ~N tokens, streams a
chat/completions request, and records: actual prompt tokens, time-to-first-token (prefill),
prefill tok/s, decode tokens, decode tok/s, and total wall-clock s/call (the metric the
llama.cpp 35b reference is quoted in).

  python3 bench_depth.py --base http://127.0.0.1:8080 --out ninfer_mtp.csv \
      --depths 8192,32768,65536,131072,180000,262000 --max-new 128 --label ninfer-mtp

Waits for TRUE model-load (a real 1-token completion succeeds), not just /health — a cold
35B EP load is ~4-5 min. Reads the model id from /v1/models. No third-party deps (urllib).
"""
import argparse, csv, json, sys, time, urllib.request, urllib.error

def http_json(url, payload=None, timeout=1200, stream=False, headers=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer x")  # any non-empty key
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    resp = urllib.request.urlopen(req, timeout=timeout)
    if stream:
        return resp
    return json.loads(resp.read().decode())

def get_model_id(base):
    try:
        j = http_json(base + "/v1/models", timeout=10)
        data = j.get("data") or []
        if data:
            return data[0]["id"]
    except Exception:
        pass
    return None

def wait_for_load(base, model_id, deadline_s):
    """Poll until a real 1-token completion succeeds (not just /health)."""
    t0 = time.time()
    last_err = None
    while time.time() - t0 < deadline_s:
        mid = model_id or get_model_id(base)
        if mid:
            try:
                r = http_json(base + "/v1/chat/completions", {
                    "model": mid, "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1, "temperature": 0, "stream": False,
                }, timeout=60)
                if r.get("choices"):
                    return mid, time.time() - t0
            except Exception as e:
                last_err = e
        time.sleep(5)
    raise RuntimeError(f"model did not become ready in {deadline_s}s (last error: {last_err})")

# A varied technical filler unit; numbered lines keep it from being trivially compressible.
UNIT = ("On a Volta V100 the mixture-of-experts router selects a small subset of expert "
        "feed-forward networks for each token, so the compute per token stays far below a "
        "dense model of the same total parameter count while the memory footprint does not. "
        "Sharding the experts across two GPUs lets each card own half of them and reduce the "
        "partial sums over the NVLink interconnect back into one output per token. ")

def build_prompt(target_tokens, tokens_per_unit):
    reps = max(1, int(target_tokens / max(1.0, tokens_per_unit)))
    lines = [f"[{i}] {UNIT}" for i in range(reps)]
    return "".join(lines)

def calibrate_tokens_per_unit(base, model_id):
    probe = "".join(f"[{i}] {UNIT}" for i in range(50))
    r = http_json(base + "/v1/chat/completions", {
        "model": model_id, "messages": [{"role": "user", "content": probe}],
        "max_tokens": 1, "temperature": 0, "stream": False,
    }, timeout=120)
    pt = r["usage"]["prompt_tokens"]
    return pt / 50.0

def stream_call(base, model_id, prompt, max_new):
    payload = {
        "model": model_id, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new, "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.time()
    ttft = None
    decode_tokens = 0
    usage = None
    resp = http_json(base + "/v1/chat/completions", payload, timeout=3600, stream=True)
    for raw in resp:
        line = raw.decode("utf-8", "ignore").strip()
        if not line or not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if body == "[DONE]":
            break
        try:
            chunk = json.loads(body)
        except Exception:
            continue
        if chunk.get("usage"):
            usage = chunk["usage"]
        choices = chunk.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            if delta.get("content"):
                if ttft is None:
                    ttft = time.time() - t0
                decode_tokens += 1
    total = time.time() - t0
    return ttft, total, decode_tokens, usage

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--out", required=True)
    ap.add_argument("--depths", default="8192,32768,65536,131072,180000,262000")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--label", default="run")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--load-timeout", type=int, default=600)
    ap.add_argument("--warmup", action="store_true", help="one throwaway call before timing")
    args = ap.parse_args()

    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    print(f"[{args.label}] waiting for model load at {args.base} (<= {args.load_timeout}s)...", flush=True)
    model_id, load_s = wait_for_load(args.base, args.model_id, args.load_timeout)
    print(f"[{args.label}] model '{model_id}' ready after {load_s:.0f}s", flush=True)

    tpu = calibrate_tokens_per_unit(args.base, model_id)
    print(f"[{args.label}] calibration: {tpu:.2f} tokens/unit", flush=True)

    rows = []
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "target_depth", "prompt_tokens", "ttft_s", "prefill_tok_s",
                    "decode_tokens", "decode_s", "decode_tok_s", "total_s"])
        for d in depths:
            prompt = build_prompt(d, tpu)
            if args.warmup:
                stream_call(args.base, model_id, prompt, 4)
            ttft, total, dtoks, usage = stream_call(args.base, model_id, prompt, args.max_new)
            pt = (usage or {}).get("prompt_tokens", 0)
            ct = (usage or {}).get("completion_tokens", dtoks)
            decode_s = max(1e-6, total - (ttft or 0.0))
            prefill_tok_s = pt / ttft if ttft else 0.0
            decode_tok_s = ct / decode_s if ct else 0.0
            row = [args.label, d, pt, round(ttft or 0, 3), round(prefill_tok_s, 1),
                   ct, round(decode_s, 3), round(decode_tok_s, 2), round(total, 3)]
            w.writerow(row); f.flush()
            rows.append(row)
            print(f"[{args.label}] depth~{d}: prompt={pt} ttft={ttft:.2f}s "
                  f"prefill={prefill_tok_s:.0f}tok/s decode={decode_tok_s:.1f}tok/s "
                  f"total={total:.1f}s", flush=True)
    print(f"[{args.label}] wrote {len(rows)} rows -> {args.out}", flush=True)

if __name__ == "__main__":
    main()
