#!/usr/bin/env python3
"""Combine the P3 depth-sweep leg CSVs into the final comparison + headline metrics.

  python3 analyze.py results/            # reads llama-nomtp.csv, ninfer-nomtp.csv, ninfer-mtp.csv
  python3 analyze.py results/ --real /tmp/qwen35b-llamacpp-real-latencies.csv

Prints (1) a per-depth table of prefill/decode tok/s per leg, (2) ninfer-vs-llama prefill & decode
speedups, (3) the decode-vs-depth falloff ratio per leg (the carbon cross-check: does ninfer hold
decode at depth better than llama's ~104->43 tok/s shallow->deep), (4) the MTP uplift, and (5) an
optional overlay of carbon's real (uncontrolled-output) points as a sanity check.
"""
import argparse, csv, glob, os, sys

LEGS = ["llama-nomtp", "ninfer-nomtp", "ninfer-mtp"]

def load(d):
    data = {}  # leg -> list of row dicts
    for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
        leg = os.path.basename(f)[:-4]
        rows = []
        for r in csv.DictReader(open(f)):
            try:
                rows.append({k: (float(v) if k not in ("label",) else v) for k, v in r.items()})
            except ValueError:
                pass
        if rows:
            data[leg] = sorted(rows, key=lambda r: r["prompt_tokens"])
    return data

def nearest(rows, depth):
    return min(rows, key=lambda r: abs(r["prompt_tokens"] - depth)) if rows else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--real", default=None, help="carbon's real-latencies CSV to overlay")
    args = ap.parse_args()
    data = load(args.results_dir)
    if not data:
        print(f"no CSVs in {args.results_dir}", file=sys.stderr); sys.exit(1)

    legs = [l for l in LEGS if l in data] + [l for l in data if l not in LEGS]
    # union of depth buckets (K) across legs
    depths = sorted({round(r["prompt_tokens"] / 1000) for l in legs for r in data[l]})

    print("\n=== per-depth prefill / decode (tok/s) ===")
    hdr = "depth(K) | " + " | ".join(f"{l}" for l in legs)
    print(hdr); print("-" * len(hdr))
    for k in depths:
        cells = []
        for l in legs:
            r = nearest(data[l], k * 1000)
            cells.append(f"{r['prefill_tok_s']:.0f}p/{r['decode_tok_s']:.0f}d" if r else "-")
        print(f"{k:>7}K | " + " | ".join(cells))

    def rate_at(leg, k, field):
        r = nearest(data.get(leg, []), k * 1000)
        return r[field] if r else None

    if "llama-nomtp" in data:
        print("\n=== ninfer vs llama (matched, MTP-off) ===")
        print("depth(K) | prefill x | decode x")
        for k in depths:
            lp, ld = rate_at("llama-nomtp", k, "prefill_tok_s"), rate_at("llama-nomtp", k, "decode_tok_s")
            np_, nd = rate_at("ninfer-nomtp", k, "prefill_tok_s"), rate_at("ninfer-nomtp", k, "decode_tok_s")
            if None in (lp, ld, np_, nd) or lp == 0 or ld == 0:
                continue
            print(f"{k:>7}K | {np_/lp:8.2f} | {nd/ld:7.2f}")

    print("\n=== decode-vs-depth falloff (shallow -> deep) — the carbon cross-check ===")
    for l in legs:
        rows = data[l]
        if len(rows) < 2:
            continue
        shallow, deep = rows[0], rows[-1]
        ratio = deep["decode_tok_s"] / shallow["decode_tok_s"] if shallow["decode_tok_s"] else 0
        print(f"  {l:14s}: {shallow['decode_tok_s']:6.1f} @ {shallow['prompt_tokens']/1000:.0f}K "
              f"-> {deep['decode_tok_s']:6.1f} @ {deep['prompt_tokens']/1000:.0f}K "
              f"(retains {ratio*100:.0f}%)")

    if "ninfer-mtp" in data and "ninfer-nomtp" in data:
        print("\n=== MTP uplift (ninfer decode, mtp / no-mtp) ===")
        for k in depths:
            on, off = rate_at("ninfer-mtp", k, "decode_tok_s"), rate_at("ninfer-nomtp", k, "decode_tok_s")
            if on and off:
                print(f"  {k:>4}K: {on/off:.2f}x  ({off:.0f} -> {on:.0f} tok/s)")

    if args.real and os.path.exists(args.real):
        print(f"\n=== overlay: carbon real points ({args.real}) — sanity vs synthetic ===")
        try:
            for r in csv.DictReader(open(args.real)):
                print("  " + ", ".join(f"{k}={v}" for k, v in r.items()))
        except Exception as e:
            print(f"  (could not parse: {e}); raw head:")
            print("  " + "".join(open(args.real).readlines()[:5]))

if __name__ == "__main__":
    main()
