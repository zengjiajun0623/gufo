#!/usr/bin/env python3
"""KL / top-1 gate between two score.json files from `gufo score`.
usage: kl_gate.py BASE.json R8.json
Metrics per aligned position:
  - top1 agreement (argmax equal)
  - exact next-token logp difference |lp_P - lp_Q|  (no truncation)
  - truncated KL(P||Q) over union of top-k sets, tail folded at each side's
    top-k boundary (underestimates; fine for near-identical models)
Gates: top1 >= 0.97, mean KL <= 0.02, median |dlp| reported.
"""
import json, math, sys

def load(path):
    d = json.load(open(path))
    seq = []  # (argmax, next, lse, lp_next, {id: logp})
    for t in d["texts"]:
        for p in t["pos"]:
            seq.append((p["a"], p["nx"], p["lse"], p["lp"],
                        {i: v for i, v in p["t"]}))
    return d, seq

base, A = load(sys.argv[1])
cand, B = load(sys.argv[2])
n = min(len(A), len(B))
if len(A) != len(B):
    print(f"WARNING: position count differs: {len(A)} vs {len(B)}; using {n}")

top1 = dlp = kl = 0.0
big = 0
for i in range(n):
    a, na, la, pa, ta = A[i]
    b, nb, lb, pb, tb = B[i]
    assert na == nb, f"position {i}: next-token mismatch (tokenization drift?)"
    top1 += 1.0 if a == b else 0.0
    d = abs(pa - pb)
    dlp += d
    if d > 1.0:
        big += 1
    # truncated KL over union; fold each side's missing mass at its boundary
    k = 0.0
    for v in set(ta) | set(tb):
        la_v = ta.get(v)
        lb_v = tb.get(v)
        if la_v is None:
            # p puts v outside its top-k: bound p < exp(ta_boundary); skip mass
            continue
        pa_v = math.exp(la_v)
        lb_v_bnd = None
        if lb_v is None:
            # q's mass on v is below its k-th value; approximate with the
            # smallest q logp seen in its top-k (underestimates the diff)
            lb_v_bnd = min(tb.values())
        else:
            lb_v_bnd = lb_v
        k += pa_v * (la_v - lb_v_bnd)
    # tail contribution: p_tail * (log p_tail - log q_tail), both folded
    pa_tail = max(0.0, 1.0 - sum(math.exp(v) for v in ta.values()))
    pb_tail = max(0.0, 1.0 - sum(math.exp(v) for v in tb.values()))
    if pa_tail > 0:
        k += pa_tail * (math.log(pa_tail) - math.log(max(pb_tail, 1e-12)))
    kl += max(0.0, k)

top1 /= n; dlp /= n; kl /= n
srt = sorted(abs((A[i][3]) - (B[i][3])) for i in range(n))
med = srt[n // 2]; p99 = srt[int(n * 0.99)]
print(f"positions: {n}")
print(f"top1 agreement: {top1:.4f}  (gate >= 0.97)")
print(f"mean truncated KL(P||Q): {kl:.5f}  (gate <= 0.02)")
print(f"next-token |dlp|: mean {dlp:.4f} median {med:.4f} p99 {p99:.4f} >1.0: {big}")
verdict = "PASS" if (top1 >= 0.97 and kl <= 0.02) else "FAIL"
print(f"GATE: {verdict}")
