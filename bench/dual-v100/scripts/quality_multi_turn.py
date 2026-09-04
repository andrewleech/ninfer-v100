#!/usr/bin/env python
# Multi-turn agentic-coding quality probe over a ~225K-token real codebase context.
# Same scripted conversation is replayed against each server; the growing message history means the
# big prefix is reused every turn -> exercises the prompt-cache path. Per turn we record TTFT (cache
# effectiveness), total latency, the answer, and a score (verifiable: fraction of gold strings hit;
# open: fraction of keywords hit). Transcript saved for divergence comparison.
import argparse, json, time, os
from openai import OpenAI

def score_turn(turn, ans):
    a = ans.lower()
    if turn["type"] == "verify":
        gold = turn["gold"]; hit = sum(1 for g in gold if str(g).lower() in a)
        return 100.0 * hit / len(gold), f"{hit}/{len(gold)} gold"
    kw = turn.get("kw", []); hit = sum(1 for k in kw if str(k).lower() in a)
    return (100.0 * hit / len(kw) if kw else 0.0), f"{hit}/{len(kw)} kw"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_url", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", required=True); ap.add_argument("--turns", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--max_gen", type=int, default=8000)
    ap.add_argument("--label", default=""); ap.add_argument("--effort", default="medium")
    a = ap.parse_args()
    ctx = open(a.ctx).read(); turns = json.load(open(a.turns))
    client = OpenAI(base_url=a.base_url, api_key="x", timeout=3600)
    msgs = []; rows = []
    print(f"# {a.label}  model={a.model}  effort={a.effort}  ctx_chars={len(ctx)}")
    print(f"{'turn':>4} {'type':>7} {'score':>6} | {'prefill':>8}{'reason':>8}{'answer':>8}{'TOTAL':>8} | {'r_tok':>6}{'a_tok':>6}")
    for turn in turns:
        user = (f"Here is the codebase you are working in:\n\n{ctx}\n\n"
                f"TASK: {turn['q']}") if turn["t"] == 1 else turn["q"]
        msgs.append({"role": "user", "content": user})
        # per-stage timing: prefill = t0->first token (reasoning or content); reasoning = span of
        # reasoning_content; answer = span of content. Both engines put thinking in reasoning_content.
        t0 = time.time(); t_first = t_reason_last = t_ans_first = t_ans_last = None
        reason = ""; content = ""
        try:
            stream = client.chat.completions.create(model=a.model, messages=msgs, max_tokens=a.max_gen,
                temperature=0.0, stream=True, extra_body={"reasoning_effort": a.effort})
            for chunk in stream:
                now = time.time(); d = chunk.choices[0].delta
                rc = getattr(d, "reasoning_content", None)
                ct = getattr(d, "content", None)
                if rc:
                    if t_first is None: t_first = now
                    t_reason_last = now; reason += rc
                if ct:
                    if t_first is None: t_first = now
                    if t_ans_first is None: t_ans_first = now
                    t_ans_last = now; content += ct
        except Exception as e:
            content = f"__ERROR__:{e}"
        total = time.time() - t0
        prefill = (t_first - t0) if t_first else total
        reason_t = (t_reason_last - t_first) if (t_reason_last and t_first) else 0.0
        answer_t = (t_ans_last - t_ans_first) if (t_ans_last and t_ans_first) else 0.0
        ans = content.split("</think>")[-1].strip() if "</think>" in content else content.strip()
        msgs.append({"role": "assistant", "content": ans})
        sc, detail = score_turn(turn, ans)
        r_tok = len(reason)//4; a_tok = len(ans)//4  # ~chars/4 token estimate
        rows.append({"t": turn["t"], "type": turn["type"], "score": sc, "detail": detail,
                     "prefill": prefill, "reason_t": reason_t, "answer_t": answer_t, "total": total,
                     "r_tok": r_tok, "a_tok": a_tok, "answer": ans})
        print(f"{turn['t']:>4} {turn['type']:>7} {sc:>6.0f} | {prefill:>8.1f}{reason_t:>8.1f}"
              f"{answer_t:>8.1f}{total:>8.1f} | {r_tok:>6}{a_tok:>6}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"label": a.label, "model": a.model, "rows": rows}, open(a.out, "w"), indent=2)
    vs = [r["score"] for r in rows if r["type"] == "verify"]
    print(f"# verify-avg={sum(vs)/len(vs):.0f}  (turns 1-6 early / 8-9 deep)")

if __name__ == "__main__":
    main()
