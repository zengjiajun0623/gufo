#!/usr/bin/env python3
"""P4-3c: self-distil the MTP draft block on dump-hidden data (trunk = teacher).

Row j of a dump holds trunk hidden h[j] and the draft input token ids[j+1] (tokens.i32[j]).
The draft predicts ids[j+2]; its teacher is the trunk distribution at row j+1 (top-k log-probs).
Loss = truncated KL: -sum_k p_k log q_k - p_tail log(1 - sum_k q_k), over rows with a teacher.
Trained: the dense parts of the draft block (eh_proj, attention, hyper-connection mixers, norms,
shared expert). Frozen: token_embd, output head (trunk-owned), router and routed experts.

usage:
  mtp_train.py eval  TRUNK MTP DUMPDIR...                      # greedy step-1 agreement
  mtp_train.py train TRUNK MTP OUTDIR --train D... --heldout NAME=D ... [--epochs 2 --lr 2e-5]
  mtp_train.py export MTP STATE.pt OUT.gguf                     # requantize to the sidecar's types
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtp_torch import MtpTorch  # noqa: E402

H = 10240
TRAINABLE = ("eh_proj", "enorm", "hnorm", "attn_q", "attn_k", "attn_v", "attn_out", "attn_q_norm",
             "attn_k_norm", "attn_norm", "attn_down", "attn_up", "attn_inject", "ffn_norm", "ffn_down",
             "ffn_up", "ffn_inject", "head_norm", "head_down", "head_up", "sh_gate", "sh_up", "sh_down",
             "sh_gate_inp")
# torch attribute -> sidecar tensor name
GGUF_NAME = {
    "eh_proj": "nextn.eh_proj", "enorm": "nextn.enorm", "hnorm": "nextn.hnorm",
    "attn_q": "attn_q", "attn_k": "attn_k", "attn_v": "attn_v", "attn_out": "attn_output",
    "attn_q_norm": "attn_q_norm", "attn_k_norm": "attn_k_norm",
    "attn_norm": "hc_attn_norm", "attn_down": "hc_attn_down", "attn_up": "hc_attn_up", "attn_inject": "hc_attn_inject",
    "ffn_norm": "hc_ffn_norm", "ffn_down": "hc_ffn_down", "ffn_up": "hc_ffn_up", "ffn_inject": "hc_ffn_inject",
    "head_norm": "nextn.hc_head_norm", "head_down": "nextn.hc_head_down", "head_up": "nextn.hc_head_up",
    "sh_gate": "ffn_gate_shexp", "sh_up": "ffn_up_shexp", "sh_down": "ffn_down_shexp", "sh_gate_inp": "ffn_gate_inp_shexp",
}
MAX_ROWS = 2000  # below the dense-attention limit (2051); longer texts keep their prefix


class Dump:
    """One dump-hidden directory, split into texts."""

    def __init__(self, d):
        self.d = d
        offs = json.load(open(os.path.join(d, "offsets.json")))
        offs = offs["texts"] if isinstance(offs, dict) else offs
        self.tok = np.fromfile(os.path.join(d, "tokens.i32"), np.int32)
        self.hid = np.memmap(os.path.join(d, "hidden.bf16"), np.uint16, "r").reshape(-1, H)
        n = self.tok.size
        k = os.path.getsize(os.path.join(d, "topk.bin")) // 4 // n
        self.tk_id = np.fromfile(os.path.join(d, "topk.bin"), np.int32).reshape(n, k)
        self.tk_lp = np.fromfile(os.path.join(d, "topk_val.f32"), np.float32).reshape(n, k)
        self.texts, r = [], 0
        for t in offs:
            rows = int(t["len"]) - 1
            if rows >= 3:
                self.texts.append((r, min(rows, MAX_ROWS)))
            r += rows
        assert r == n, (d, r, n)

    def sample(self, i, device):
        r0, n = self.texts[i]
        sl = slice(r0, r0 + n)
        hid = torch.from_numpy((self.hid[sl].astype(np.uint32) << 16).view(np.float32).copy())
        tok = torch.from_numpy(self.tok[sl].astype(np.int64))
        # teacher for row j = trunk row j+1; the last row has none
        tid = torch.from_numpy(self.tk_id[r0 + 1:r0 + n].astype(np.int64))
        tlp = torch.from_numpy(self.tk_lp[r0 + 1:r0 + n].copy())
        return hid.to(device), tok.to(device), tid.to(device), tlp.to(device)


def loss_and_stats(logits, tid, tlp):
    """logits [n, V] for rows 0..n-1; teacher for rows 0..n-2."""
    logq = torch.log_softmax(logits[:-1].float(), -1)
    lq_k = logq.gather(1, tid)                       # [m, k]
    p_k = tlp.exp()
    p_tail = (1.0 - p_k.sum(-1)).clamp_min(0.0)
    q_tail = (1.0 - lq_k.exp().sum(-1)).clamp_min(1e-12)
    loss = -(p_k * lq_k).sum(-1) - p_tail * q_tail.log()
    agree = (logq.argmax(-1) == tid[:, 0]).float()   # greedy step-1 acceptance proxy
    return loss.mean(), agree.sum().item(), agree.numel()


def evaluate(m, dumps, device, limit_texts=None):
    m.eval()
    tot_l, tot_a, tot_n = 0.0, 0.0, 0
    with torch.no_grad():
        for dp in dumps:
            for i in range(len(dp.texts) if limit_texts is None else min(limit_texts, len(dp.texts))):
                hid, tok, tid, tlp = dp.sample(i, device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
                    logits = m(hid, tok)
                l, a, n = loss_and_stats(logits, tid, tlp)
                tot_l += l.item() * n
                tot_a += a
                tot_n += n
    return tot_l / max(tot_n, 1), tot_a / max(tot_n, 1), tot_n


def build(trunk, mtp, device):
    m = MtpTorch(trunk, mtp, device=device, dtype=torch.float32)
    for name, p in m.named_parameters():
        p.requires_grad = name in TRAINABLE
    # frozen big matrices in bf16 to save memory and bandwidth
    for name in ("token_embd", "output", "router"):
        getattr(m, name).data = getattr(m, name).data.to(torch.bfloat16)
    return m


def cmd_eval(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = build(a.trunk, a.mtp, dev)
    if a.state:
        m.load_state_dict(torch.load(a.state), strict=False)
    for d in a.dumps:
        l, acc, n = evaluate(m, [Dump(d)], dev)
        print(json.dumps({"dump": d, "rows": n, "loss": round(l, 5), "agree": round(acc, 5)}))


def cmd_train(a):
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    m = build(a.trunk, a.mtp, dev)
    train = [Dump(d) for d in a.train]
    held = {kv.split("=", 1)[0]: Dump(kv.split("=", 1)[1]) for kv in a.heldout}
    items = [(di, ti) for di, dp in enumerate(train) for ti in range(len(dp.texts))]
    params = [p for p in m.parameters() if p.requires_grad]
    print(f"trainable params {sum(p.numel() for p in params) / 1e6:.1f}M, texts {len(items)}", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.98))
    steps = a.epochs * len(items) // a.accum
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / max(steps, 1))))
    log = open(os.path.join(a.out, "train.jsonl"), "a")

    def eval_all(tag, step):
        rec = {"tag": tag, "step": step}
        for name, dp in held.items():
            l, acc, n = evaluate(m, [dp], dev)
            rec[name] = {"loss": round(l, 5), "agree": round(acc, 5), "rows": n}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        return rec

    base = eval_all("baseline", 0)
    best = None
    step, t0 = 0, time.time()
    rng = np.random.default_rng(0)
    for ep in range(a.epochs):
        order = rng.permutation(len(items))
        m.train()
        run_l, run_n = 0.0, 0
        for k, idx in enumerate(order):
            di, ti = items[idx]
            hid, tok, tid, tlp = train[di].sample(ti, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev != "cpu"):
                logits = m(hid, tok)
            l, _, n = loss_and_stats(logits, tid, tlp)
            (l / a.accum).backward()
            run_l += l.item()
            run_n += 1
            if (k + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    print(json.dumps({"step": step, "epoch": ep, "train_loss": round(run_l / run_n, 5),
                                      "lr": sched.get_last_lr()[0], "s": round(time.time() - t0)}), flush=True)
                    run_l, run_n = 0.0, 0
                if step % a.eval_every == 0:
                    rec = eval_all("eval", step)
                    first = next(iter(held))
                    ok_gen = all(rec[n]["agree"] >= base[n]["agree"] - 0.005 for n in held if n != first)
                    if ok_gen and (best is None or rec[first]["agree"] > best[first]["agree"]):
                        best = rec
                        torch.save({n: p.detach().cpu() for n, p in m.named_parameters() if p.requires_grad},
                                   os.path.join(a.out, "best.pt"))
                    m.train()
    rec = eval_all("final", step)
    first = next(iter(held))
    ok_gen = all(rec[n]["agree"] >= base[n]["agree"] - 0.005 for n in held if n != first)
    if ok_gen and (best is None or rec[first]["agree"] > best[first]["agree"]):
        best = rec
        torch.save({n: p.detach().cpu() for n, p in m.named_parameters() if p.requires_grad},
                   os.path.join(a.out, "best.pt"))
    json.dump({"baseline": base, "best": best}, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("BEST", json.dumps(best), flush=True)


def cmd_export(a):
    """Copy the sidecar and overwrite the trained tensors in place, requantized to their own types
    (same type => same byte size, so headers and offsets stay valid)."""
    import shutil
    sys.path.insert(0, os.path.expanduser("~/gufo-opt"))
    import requant  # noqa: E402  (ggml_quantize_chunk wrapper)
    from gguf import GGUFReader, GGMLQuantizationType as QT
    state = torch.load(a.state)
    r = GGUFReader(a.mtp)
    by_name = {"blk.48." + GGUF_NAME[k] + ".weight": v for k, v in state.items()}
    shutil.copyfile(a.mtp, a.out)
    changed = 0
    with open(a.out, "r+b") as f:
        for t in r.tensors:
            new = by_name.get(t.name)
            if new is None:
                continue
            f32 = np.ascontiguousarray(new.float().numpy(), dtype=np.float32)
            if t.tensor_type == QT.F32:
                raw = f32.tobytes()
            else:
                q, _ = requant.quantize(f32.reshape(-1, f32.shape[-1]), t.tensor_type, None)
                raw = q.tobytes()
            assert len(raw) == t.n_bytes, (t.name, len(raw), t.n_bytes)
            f.seek(t.data_offset)
            f.write(raw)
            changed += 1
    assert changed == len(by_name), (changed, len(by_name))
    print(f"export: {changed} tensors replaced -> {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("eval")
    e.add_argument("trunk"), e.add_argument("mtp"), e.add_argument("dumps", nargs="+")
    e.add_argument("--state")
    t = sp.add_parser("train")
    t.add_argument("trunk"), t.add_argument("mtp"), t.add_argument("out")
    t.add_argument("--train", nargs="+", required=True)
    t.add_argument("--heldout", nargs="+", required=True, help="NAME=DIR; the first is the target set")
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--accum", type=int, default=8)
    t.add_argument("--eval-every", type=int, default=100)
    x = sp.add_parser("export")
    x.add_argument("mtp"), x.add_argument("state"), x.add_argument("out")
    a = ap.parse_args()
    {"eval": cmd_eval, "train": cmd_train, "export": cmd_export}[a.cmd](a)
