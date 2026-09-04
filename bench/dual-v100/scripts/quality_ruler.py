#!/usr/bin/env python
# Thin RULER driver + scorer: reads RULER-prepared task jsonl, runs inference against an
# OpenAI-compatible endpoint (no-think, greedy), scores with RULER's string_match_all metric.
import argparse, json, os, sys, time, re
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]

def string_match_all(preds, refs):
    # RULER's metric: mean over samples of (fraction of reference strings found in prediction).
    scores = []
    for pred, ref in zip(preds, refs):
        pl = pred.lower()
        hits = sum(1 for r in ref if str(r).lower() in pl)
        scores.append(hits / len(ref) if ref else 0.0)
    return 100.0 * sum(scores) / len(scores) if scores else 0.0

def run_task(client, model, task_dir, out_dir, max_gen, threads, limit=0):
    recs = load_jsonl(os.path.join(task_dir, "validation.jsonl"))
    if limit > 0:
        recs = recs[:limit]
    def infer(rec):
        # Thinking ON (measures deployed quality). The final answer is scored: strip any
        # <think>...</think> block and any separate reasoning field so only the answer is matched.
        for attempt in range(4):
            try:
                r = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": rec["input"]}],
                    max_tokens=max_gen, temperature=0.0,
                )
                content = r.choices[0].message.content or ""
                if "</think>" in content:
                    content = content.split("</think>")[-1]
                return content.strip()
            except Exception as e:
                if attempt == 3:
                    return f"__ERROR__:{e}"
                time.sleep(2 * (attempt + 1))
    with ThreadPoolExecutor(max_workers=threads) as ex:
        preds = list(ex.map(infer, recs))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "pred.jsonl"), "w") as f:
        for rec, pred in zip(recs, preds):
            f.write(json.dumps({"index": rec["index"], "outputs": rec["outputs"],
                                "pred": pred, "length": rec["length"]}) + "\n")
    score = string_match_all(preds, [r["outputs"] for r in recs])
    nerr = sum(1 for p in preds if p.startswith("__ERROR__"))
    return score, len(recs), nerr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)   # dir of prepared task subdirs
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--base_url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", required=True)       # comma list
    ap.add_argument("--max_gen", type=int, default=128)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    client = OpenAI(base_url=a.base_url, api_key="x", timeout=1200)
    print(f"# model={a.model}  base_url={a.base_url}  label={a.label}")
    print(f"{'task':<18} {'score':>7} {'n':>4} {'err':>4}")
    results = {}
    for task in a.tasks.split(","):
        td = os.path.join(a.data_dir, task)
        if not os.path.isdir(td):
            print(f"{task:<18} MISSING"); continue
        t0 = time.time()
        score, n, nerr = run_task(client, a.model, td, os.path.join(a.out_dir, task),
                                  a.max_gen, a.threads, a.limit)
        results[task] = score
        print(f"{task:<18} {score:>7.1f} {n:>4} {nerr:>4}   ({time.time()-t0:.0f}s)", flush=True)
    if results:
        print(f"{'AVG':<18} {sum(results.values())/len(results):>7.1f}")
    with open(os.path.join(a.out_dir, "summary.json"), "w") as f:
        json.dump({"model": a.model, "label": a.label, "scores": results}, f, indent=2)

if __name__ == "__main__":
    main()
