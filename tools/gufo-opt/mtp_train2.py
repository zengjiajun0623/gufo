#!/usr/bin/env python3
"""P4-3 v2 (Claude): multi-step ("chain") self-distillation of the Flash-Next draft block.

Why: v1 trained only chain step 1 (draft fed the trunk hidden). Offline step-1 agree rose 0.80 -> 0.95 but online
acceptance barely moved (t24): online, steps 2..k are fed the draft's OWN output residual. Here every step of a
k-step chain is trained, with the draft-cache visibility gufo uses online (MtpTorch.forward_chain).

  mtp_train2.py check TRUNK MTP DUMP                       # forward_chain vs a naive per-anchor loop
  mtp_train2.py eval  TRUNK MTP DUMP... [--state S] [--steps 3]
  mtp_train2.py train TRUNK MTP OUT --train D... --heldout NAME=D... [--steps 3] ...
Metrics per dump: agree_s (step s top-1 == trunk top-1, teacher-forced) and chain = expected accepted drafts
per verify pass for a fixed length-`steps` chain (sum over s of P[steps 1..s all agree]); online
tokens/pass ~ 1 + chain (the length controller varies the length).
Export: mtp_train.py export Q8D_SIDECAR STATE OUT (unchanged).
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtp_train import Dump, build  # noqa: E402

STEP_W = (1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.3)


def chain_loss(outs, tid, tlp):
    """outs: [logits_s [n, V]]; teacher rows 0..n-2 (same for every step). Returns (loss, stats dict)."""
    n = outs[0].shape[0]
    S = len(outs)
    p_k = tlp.exp()
    p_tail = (1.0 - p_k.sum(-1)).clamp_min(0.0)
    loss, wsum, agrees = 0.0, 0.0, []
    for s, lg in enumerate(outs):  # step s+1 valid at p >= s
        logq = torch.log_softmax(lg[s:n - 1].float(), -1)
        lq_k = logq.gather(1, tid[s:])
        q_tail = (1.0 - lq_k.exp().sum(-1)).clamp_min(1e-12)
        l = (-(p_k[s:] * lq_k).sum(-1) - p_tail[s:] * q_tail.log()).mean()
        loss = loss + STEP_W[s] * l
        wsum += STEP_W[s]
        agrees.append(logq.argmax(-1) == tid[s:, 0])  # index i <-> position p = s + i
    # chain anchored at j: step t (1-based) at position j+t-1, needs j+S-1 <= n-2
    m = n - 1 - (S - 1)
    alive = torch.ones(max(m, 0), dtype=torch.bool, device=tid.device)
    chain = 0.0
    for s in range(S):
        alive = alive & agrees[s][:m]  # agrees[s][i] is position s+i = anchor i, step s+1
        chain += alive.float().sum().item()
    st = {"agree": [a.float().sum().item() for a in agrees], "agree_n": [a.numel() for a in agrees],
          "chain": chain, "anchors": max(m, 0)}
    return loss / wsum, st


def evaluate(m, dumps, dev, steps):
    m.eval()
    tot = {"loss": 0.0, "rows": 0, "agree": [0.0] * steps, "agree_n": [0] * steps, "chain": 0.0, "anchors": 0}
    with torch.no_grad():
        for dp in dumps:
            for i in range(len(dp.texts)):
                hid, tok, tid, tlp = dp.sample(i, dev)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev != "cpu"):
                    outs = m.forward_chain(hid, tok, steps)
                l, st = chain_loss(outs, tid, tlp)
                tot["loss"] += l.item() * st["agree_n"][0]
                tot["rows"] += st["agree_n"][0]
                for s in range(steps):
                    tot["agree"][s] += st["agree"][s]
                    tot["agree_n"][s] += st["agree_n"][s]
                tot["chain"] += st["chain"]
                tot["anchors"] += st["anchors"]
    return {"loss": round(tot["loss"] / max(tot["rows"], 1), 5), "rows": tot["rows"],
            "agree": [round(a / max(n, 1), 4) for a, n in zip(tot["agree"], tot["agree_n"])],
            "chain": round(tot["chain"] / max(tot["anchors"], 1), 4)}


def naive_chain(m, hidden, tokens, j, steps):
    """Reference for one anchor j: explicit per-step attention over the online draft cache. Returns [argmax_s]."""
    c = m.c
    dev = m.device
    hidden = hidden.to(dev, m.dtype)
    We, Wh = m.eh_proj[:, :c.hidden], m.eh_proj[:, c.hidden:]
    from mtp_torch import rms_norm, rope

    def block_in(h, tok):
        emb = rms_norm(m.token_embd[tok.to(dev)], m.enorm, c.eps)
        hn = rms_norm(h, m.hnorm, c.eps).view(-1, c.hc, c.hidden)
        return ((emb @ We.T)[:, None, :] + hn @ Wh.T).reshape(-1, c.hc * c.hidden)

    def qkv(x, pos):
        T = x.shape[0]
        qg = (x @ m.attn_q.T).view(T, c.nh, 2, c.hd)
        q, gate = rms_norm(qg[:, :, 0], m.attn_q_norm, c.eps), qg[:, :, 1]
        k = rms_norm((x @ m.attn_k.T).view(T, c.nkv, c.hd), m.attn_k_norm, c.eps)
        v = (x @ m.attn_v.T).view(T, c.nkv, c.hd)
        return rope(q, pos, c.rotary, c.theta), rope(k, pos, c.rotary, c.theta), v, gate

    def attend(q, gate, K, V):
        g = c.nh // c.nkv
        K, V = K.repeat_interleave(g, 1), V.repeat_interleave(g, 1)
        sc = torch.einsum("hd,shd->hs", q, K) / c.hd ** 0.5
        ctx = torch.einsum("hs,shd->hd", torch.softmax(sc, -1), V) * torch.sigmoid(gate)
        return ctx.reshape(1, -1) @ m.attn_out.T

    # committed positions 0..j: step 1 over the trunk rows (keys only needed)
    res0 = block_in(hidden[:j + 1], tokens[:j + 1])
    mixed, _ = m.hc_mix("attn", res0)
    _, K, V, _ = qkv(mixed, torch.arange(j + 1, device=dev))
    h = hidden[j:j + 1]
    out = []
    for s in range(steps):
        p = j + s
        res = block_in(h, tokens[p:p + 1])
        mixed, inject = m.hc_mix("attn", res)
        q, k, v, gate = qkv(mixed, torch.tensor([p], device=dev))
        if s == 0:
            Ks, Vs = K, V  # own key at j is the step-1 key at j
        else:
            Ks, Vs = torch.cat([Ks, k]), torch.cat([Vs, v])
        res = m.hc_combine(res, attend(q[0], gate[0], Ks, Vs), inject)
        mixed, inject = m.hc_mix("ffn", res)
        res = m.hc_combine(res, m.moe(mixed), inject)
        head, _ = m.hc_mix("head", res)
        out.append((head @ m.output.T)[0])
        h = res
    return out


def cmd_check(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = build(a.trunk, a.mtp, dev)
    for n, p in m.named_parameters():  # exact fp32 comparison
        p.data = p.data.float()
    m.dtype = torch.float32
    dp = Dump(a.dump)
    hid, tok, tid, tlp = dp.sample(0, dev)
    T = min(48, hid.shape[0])
    hid, tok = hid[:T], tok[:T]
    S = 3
    with torch.no_grad():
        outs = m.forward_chain(hid, tok, S)
        base = m.forward(hid, tok)
        d1 = (outs[0] - base).abs().max().item()
        worst, agree = 0.0, 0
        for j in (5, 17, T - S - 1):
            ref = naive_chain(m, hid, tok, j, S)
            for s in range(S):
                got = outs[s][j + s]
                worst = max(worst, (got - ref[s]).abs().max().item() / ref[s].abs().max().item())
                agree += int(got.argmax() == ref[s].argmax())
    ok = d1 < 1e-3 and worst < 1e-3 and agree == 3 * S
    print(json.dumps({"step1_vs_forward_maxabs": d1, "chain_vs_naive_maxrel": worst, "argmax_match": f"{agree}/{3 * S}"}))
    print("CHAINCHECK:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def cmd_eval(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = build(a.trunk, a.mtp, dev)
    if a.state:
        m.load_state_dict(torch.load(a.state), strict=False)
    for d in a.dumps:
        print(json.dumps({"dump": d, **evaluate(m, [Dump(d)], dev, a.steps)}), flush=True)


def cmd_train(a):
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    m = build(a.trunk, a.mtp, dev)
    train = [Dump(d) for d in a.train]
    held = {kv.split("=", 1)[0]: Dump(kv.split("=", 1)[1]) for kv in a.heldout}
    items = [(di, ti) for di, dp in enumerate(train) for ti in range(len(dp.texts))]
    params = [p for p in m.parameters() if p.requires_grad]
    print(f"trainable {sum(p.numel() for p in params) / 1e6:.1f}M, texts {len(items)}, steps {a.steps}", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.98))
    total = a.epochs * len(items) // a.accum
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(s, total) / max(total, 1))))
    log = open(os.path.join(a.out, "train.jsonl"), "a")

    def eval_all(tag, step):
        rec = {"tag": tag, "step": step}
        for name, dp in held.items():
            rec[name] = evaluate(m, [dp], dev, a.steps)
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        return rec

    def score(rec):  # mean chain length over all heldouts
        return sum(rec[n]["chain"] for n in held) / len(held)

    def save_if_best(rec, best):
        if best is None or score(rec) > score(best):
            torch.save({n: p.detach().cpu() for n, p in m.named_parameters() if p.requires_grad},
                       os.path.join(a.out, "best.pt"))
            return rec
        return best

    base = eval_all("baseline", 0)
    best = None
    step, t0 = 0, time.time()
    rng = np.random.default_rng(0)
    for ep in range(a.epochs):
        m.train()
        run = 0.0
        for k, idx in enumerate(rng.permutation(len(items))):
            di, ti = items[idx]
            hid, tok, tid, tlp = train[di].sample(ti, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev != "cpu"):
                outs = m.forward_chain(hid, tok, a.steps)
            l, _ = chain_loss(outs, tid, tlp)
            (l / a.accum).backward()
            run += l.item()
            if (k + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    print(json.dumps({"step": step, "epoch": ep, "train_loss": round(run / (20 * a.accum), 5),
                                      "s": round(time.time() - t0)}), flush=True)
                    run = 0.0
                if step % a.eval_every == 0:
                    best = save_if_best(eval_all("eval", step), best)
                    m.train()
    best = save_if_best(eval_all("final", step), best)
    json.dump({"baseline": base, "best": best}, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("BEST", json.dumps(best), "baseline_score", score(base), "best_score", score(best), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("check")
    c.add_argument("trunk"), c.add_argument("mtp"), c.add_argument("dump")
    e = sp.add_parser("eval")
    e.add_argument("trunk"), e.add_argument("mtp"), e.add_argument("dumps", nargs="+")
    e.add_argument("--state")
    e.add_argument("--steps", type=int, default=3)
    t = sp.add_parser("train")
    t.add_argument("trunk"), t.add_argument("mtp"), t.add_argument("out")
    t.add_argument("--train", nargs="+", required=True)
    t.add_argument("--heldout", nargs="+", required=True, help="NAME=DIR")
    t.add_argument("--steps", type=int, default=3)
    t.add_argument("--epochs", type=int, default=4)
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--accum", type=int, default=2)
    t.add_argument("--eval-every", type=int, default=25)
    a = ap.parse_args()
    {"check": cmd_check, "eval": cmd_eval, "train": cmd_train}[a.cmd](a)
