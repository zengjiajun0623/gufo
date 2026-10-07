#!/usr/bin/env python3
"""Torch port of gufo's MTP draft step (ReferenceModel::MtpStep and helpers, reference.cpp).

One call runs a whole text: row j takes the trunk wide residual h[j] (hc_dim) and the next
token ids[j+1], and returns the draft logits for ids[j+2]. Attention is causal over the
draft's own earlier rows, exactly like mtp_attention_ in the reference. Only the dense
attention branch is ported: while complete key blocks <= indexer_top_k / compress_ratio
every position is visible, so sequences must stay within max_dense_len().

usage (parity):  mtp_torch.py parity TRUNK.gguf MTP.gguf REFDIR
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/gufo-opt/llama.cpp/gguf-py"))
from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

ARCH = "qwen4exp"


def _field(reader, key):
    f = reader.fields[f"{ARCH}.{key}"]
    v = f.contents()
    return v


def _deq(t):
    """Dequantize a gguf tensor to float32 numpy in (rows, cols) / (experts, rows, cols) order."""
    return np.ascontiguousarray(dequantize(t.data, t.tensor_type), dtype=np.float32)


class Config:
    def __init__(self, r):
        self.hidden = int(_field(r, "embedding_length"))
        self.hc = int(_field(r, "hyper_connection.count"))
        self.low_rank = int(_field(r, "hyper_connection.low_rank"))
        self.nh = int(_field(r, "attention.head_count"))
        self.nkv = int(_field(r, "attention.head_count_kv"))
        self.hd = int(_field(r, "attention.key_length"))
        self.rotary = int(_field(r, "rope.dimension_count"))
        self.theta = float(_field(r, "rope.freq_base"))
        self.eps = float(_field(r, "attention.layer_norm_rms_epsilon"))
        self.n_exp = int(_field(r, "expert_count"))
        self.k_exp = int(_field(r, "expert_used_count"))
        self.idx_top_k = int(_field(r, "attention.indexer.top_k"))
        ratios = [int(x) for x in _field(r, "attention.compress_ratios")]
        self.ratio = next(x for x in ratios if x)

    def max_dense_len(self):
        # dense while floor(n_kv / ratio) <= top_k / ratio
        return (self.idx_top_k // self.ratio + 1) * self.ratio - 1


def rms_norm(x, w, eps):
    # cpu::RmsNorm: x * 1/sqrt(mean(x^2) + eps) * w   (mean accumulated in double there)
    ms = x.double().pow(2).mean(-1, keepdim=True).float()
    return x * torch.rsqrt(ms + eps) * w


def rope(x, positions, rotary, theta):
    """cpu::Rope on the first `rotary` dims: pairs (i, i + rotary/2). x [..., T, H, D]."""
    half = rotary // 2
    i = torch.arange(half, dtype=torch.float32, device=x.device)
    freq = theta ** (-2.0 * i / rotary)
    ang = positions.float()[:, None] * freq[None, :]  # [T, half]
    c, s = torch.cos(ang)[:, None, :], torch.sin(ang)[:, None, :]
    a, b = x[..., :half], x[..., half:rotary]
    out = x.clone()
    out[..., :half] = a * c - b * s
    out[..., half:rotary] = a * s + b * c
    return out


class MtpTorch(torch.nn.Module):
    def __init__(self, trunk_path, mtp_path, device="cpu", dtype=torch.float32):
        super().__init__()
        tr, mr = GGUFReader(trunk_path), GGUFReader(mtp_path)
        self.c = c = Config(mr)
        self.device, self.dtype = device, dtype
        mt = {t.name: t for t in mr.tensors}
        tt = {t.name: t for t in tr.tensors}
        p = "blk.48."

        def P(name, src=mt):
            return torch.nn.Parameter(torch.from_numpy(_deq(src[name])).to(device, dtype), requires_grad=False)

        # trunk-owned, frozen
        self.token_embd = P("token_embd.weight", tt)  # [V, H]
        self.output = P("output.weight", tt)  # [V, H]
        # draft block
        self.enorm, self.hnorm = P(p + "nextn.enorm.weight"), P(p + "nextn.hnorm.weight")
        self.eh_proj = P(p + "nextn.eh_proj.weight")  # [H, 2H]: [:, :H] embedding, [:, H:] hidden stream
        self.mix = {}
        for tag, pre in (("attn", p + "hc_attn_"), ("ffn", p + "hc_ffn_"), ("head", p + "nextn.hc_head_")):
            for part in ("norm", "down", "up") + (("inject",) if tag != "head" else ()):
                setattr(self, f"{tag}_{part}", P(pre + part + ".weight"))
        self.attn_q, self.attn_k, self.attn_v = P(p + "attn_q.weight"), P(p + "attn_k.weight"), P(p + "attn_v.weight")
        self.attn_q_norm, self.attn_k_norm = P(p + "attn_q_norm.weight"), P(p + "attn_k_norm.weight")
        self.attn_out = P(p + "attn_output.weight")
        self.router = P(p + "ffn_gate_inp.weight")  # [E, H]
        self.sh_gate, self.sh_up = P(p + "ffn_gate_shexp.weight"), P(p + "ffn_up_shexp.weight")
        self.sh_down, self.sh_gate_inp = P(p + "ffn_down_shexp.weight"), P(p + "ffn_gate_inp_shexp.weight")
        # experts stay quantized on the host; dequantized lazily per used expert
        self._exp_raw = {k: mt[p + f"ffn_{k}_exps.weight"] for k in ("gate", "up", "down")}
        self._exp_cache = {}

    # ---- helpers mirroring reference.cpp ----
    def expert(self, e):
        w = self._exp_cache.get(e)
        if w is None:
            w = tuple(torch.from_numpy(np.ascontiguousarray(
                dequantize(self._exp_raw[k].data[e], self._exp_raw[k].tensor_type), dtype=np.float32)).to(self.device, self.dtype)
                for k in ("gate", "up", "down"))
            self._exp_cache[e] = w
        return w

    def grouped_norm(self, res, w):
        c = self.c
        x = res.view(-1, c.hc, c.hidden)
        return rms_norm(x, w.view(c.hc, c.hidden), c.eps).view(res.shape)

    def hc_mix(self, tag, res):
        """HcMix: returns (mixed [T,H], inject [T,hc] or None)."""
        c = self.c
        xn = self.grouped_norm(res, getattr(self, f"{tag}_norm"))
        lo = F.silu((xn @ getattr(self, f"{tag}_down").T) / c.hc)
        gate = lo @ getattr(self, f"{tag}_up").T  # [T, hc_dim]
        mixed = (xn * torch.sigmoid(gate)).view(-1, c.hc, c.hidden).sum(1) / c.hc
        inject = xn @ self.attn_inject.T if tag == "attn" else (xn @ self.ffn_inject.T if tag == "ffn" else None)
        return mixed, inject

    def hc_combine(self, res, block, inject):
        c = self.c
        w = 2.0 * torch.sigmoid(inject / c.hc)  # [T, hc]
        return (res.view(-1, c.hc, c.hidden) + block[:, None, :] * w[:, :, None]).view(res.shape)

    def attention(self, x):
        c = self.c
        T = x.shape[0]
        assert T <= c.max_dense_len(), f"sequence {T} > dense limit {c.max_dense_len()}"
        qg = (x @ self.attn_q.T).view(T, c.nh, 2, c.hd)  # per head [q | gate]
        q, gate = qg[:, :, 0], qg[:, :, 1]
        q = rms_norm(q, self.attn_q_norm, c.eps)
        k = rms_norm((x @ self.attn_k.T).view(T, c.nkv, c.hd), self.attn_k_norm, c.eps)
        v = (x @ self.attn_v.T).view(T, c.nkv, c.hd)
        pos = torch.arange(T, device=x.device)
        q, k = rope(q, pos, c.rotary, c.theta), rope(k, pos, c.rotary, c.theta)
        group = c.nh // c.nkv
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        scores = torch.einsum("thd,shd->hts", q, k) / (c.hd ** 0.5)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
        ctx = torch.einsum("hts,shd->thd", torch.softmax(scores, -1), v)
        ctx = ctx * torch.sigmoid(gate)
        return ctx.reshape(T, -1) @ self.attn_out.T

    def attention_chain(self, x, prev):
        """Attention of chain step s = len(prev) + 1 at every position p, as gufo drafts it online: the draft
        cache holds step-1 (trunk-fed) keys for committed positions <= p-s+1, then the keys of chain steps
        2..s-1 at positions p-s+2..p-1, then its own key at p. prev = [(k, v)] of steps 1..s-1, each [T, nkv, hd]
        after norm+rope. Returns (out [T, H], (k, v) of this step). Rows p < s-1 have no valid anchor."""
        c = self.c
        T = x.shape[0]
        s = len(prev) + 1
        qg = (x @ self.attn_q.T).view(T, c.nh, 2, c.hd)
        q, gate = qg[:, :, 0], qg[:, :, 1]
        q = rms_norm(q, self.attn_q_norm, c.eps)
        k = rms_norm((x @ self.attn_k.T).view(T, c.nkv, c.hd), self.attn_k_norm, c.eps)
        v = (x @ self.attn_v.T).view(T, c.nkv, c.hd)
        pos = torch.arange(T, device=x.device)
        q, k = rope(q, pos, c.rotary, c.theta), rope(k, pos, c.rotary, c.theta)
        group = c.nh // c.nkv
        steps = prev + [(k, v)]
        k1, v1 = steps[0]
        k1r, v1r = k1.repeat_interleave(group, dim=1), v1.repeat_interleave(group, dim=1)
        scale = c.hd ** -0.5
        base = torch.einsum("thd,shd->hts", q, k1r) * scale
        # step-1 keys visible up to p-(s-1); for s == 1 this is the plain causal mask
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril(-(s - 1))
        base = base.masked_fill(~mask, float("-inf"))
        extra_s, extra_v = [], []
        for t in range(2, s + 1):  # key of step t sits at position p-(s-t)
            kt, vt = steps[t - 1]
            sh = s - t
            kt = torch.cat([kt.new_zeros(sh, *kt.shape[1:]), kt[:T - sh]]).repeat_interleave(group, dim=1)
            vt = torch.cat([vt.new_zeros(sh, *vt.shape[1:]), vt[:T - sh]]).repeat_interleave(group, dim=1)
            sc = (q * kt).sum(-1).T * scale  # [h, T]
            if sh:
                sc[:, :sh] = float("-inf")
            extra_s.append(sc[:, :, None])
            extra_v.append(vt)
        scores = torch.cat([base] + extra_s, -1) if extra_s else base
        # rows with no visible key at all (p < s-1) would be NaN; give them a dummy zero score
        dead = torch.isinf(scores).all(-1, keepdim=True)
        scores = scores.masked_fill(dead, 0.0)
        p = torch.softmax(scores, -1)
        ctx = torch.einsum("hts,shd->thd", p[..., :T], v1r)
        for i, vt in enumerate(extra_v):
            ctx = ctx + p[..., T + i].T[:, :, None] * vt
        ctx = ctx * torch.sigmoid(gate)
        return ctx.reshape(T, -1) @ self.attn_out.T, (k, v)

    def forward_chain(self, hidden, next_tokens, steps):
        """Teacher-forced draft chain. Step 1 = forward(). Step s at position p consumes the step s-1 output
        residual of position p-1 (what MtpStep feeds back) and token ids[p+1], so every step shares step 1's
        token/teacher indexing. Returns [logits_s [T, V]] for s = 1..steps; rows p < s-1 are invalid."""
        c = self.c
        h = hidden.to(self.device, self.dtype)
        emb = rms_norm(self.token_embd[next_tokens.to(self.device)], self.enorm, c.eps)
        We, Wh = self.eh_proj[:, :c.hidden], self.eh_proj[:, c.hidden:]
        prev, outs = [], []
        for s in range(steps):
            hn = rms_norm(h, self.hnorm, c.eps).view(-1, c.hc, c.hidden)
            res = ((emb @ We.T)[:, None, :] + hn @ Wh.T).reshape(-1, c.hc * c.hidden)
            mixed, inject = self.hc_mix("attn", res)
            a, kv = self.attention_chain(mixed, prev)
            res = self.hc_combine(res, a, inject)
            mixed, inject = self.hc_mix("ffn", res)
            res = self.hc_combine(res, self.moe(mixed), inject)
            head, _ = self.hc_mix("head", res)
            outs.append(head @ self.output.T)
            prev.append(kv)
            h = torch.cat([res.new_zeros(1, res.shape[1]), res[:-1]])  # next step at p reads res[p-1]
        return outs

    def moe(self, x):
        c = self.c
        probs = torch.softmax(x @ self.router.T, -1)  # [T, E]
        top_p, top_e = probs.topk(c.k_exp, -1)
        wts = top_p / top_p.sum(-1, keepdim=True).clamp_min(6.103515625e-5)
        out = torch.zeros_like(x)
        for e in top_e.unique().tolist():
            rows, slot = (top_e == e).nonzero(as_tuple=True)
            g, u, d = self.expert(e)
            h = F.silu(x[rows] @ g.T) * (x[rows] @ u.T)
            out.index_add_(0, rows, (h @ d.T) * wts[rows, slot][:, None])
        sh = F.silu(x @ self.sh_gate.T) * (x @ self.sh_up.T)
        sg = torch.sigmoid(x @ self.sh_gate_inp)  # [T]
        return out + sg[:, None] * (sh @ self.sh_down.T)

    def forward(self, hidden, next_tokens):
        """hidden [T, hc_dim] trunk residual rows; next_tokens [T] = ids[j+1]. Returns logits [T, V]."""
        c = self.c
        hidden = hidden.to(self.device, self.dtype)
        emb = rms_norm(self.token_embd[next_tokens.to(self.device)], self.enorm, c.eps)  # [T, H]
        # MtpStep normalizes the whole hc_dim row at once (one RmsNorm, not per stream)
        hn = rms_norm(hidden, self.hnorm, c.eps).view(-1, c.hc, c.hidden)
        We, Wh = self.eh_proj[:, :c.hidden], self.eh_proj[:, c.hidden:]
        res = ((emb @ We.T)[:, None, :] + hn @ Wh.T).reshape(-1, c.hc * c.hidden)
        mixed, inject = self.hc_mix("attn", res)
        res = self.hc_combine(res, self.attention(mixed), inject)
        mixed, inject = self.hc_mix("ffn", res)
        res = self.hc_combine(res, self.moe(mixed), inject)
        head, _ = self.hc_mix("head", res)
        return head @ self.output.T


def parity(trunk, mtp, refdir):
    import json
    H = 10240
    offs = json.load(open(os.path.join(refdir, "offsets.json")))
    offs = offs["texts"] if isinstance(offs, dict) else offs
    toks = np.fromfile(os.path.join(refdir, "tokens.i32"), np.int32)  # ids[j+1] per row
    hid = np.fromfile(os.path.join(refdir, "ref_hidden.f32"), np.float32).reshape(-1, H)
    m = MtpTorch(trunk, mtp)
    V = m.output.shape[0]
    ref = np.fromfile(os.path.join(refdir, "ref_mtp_logits.f32"), np.float32).reshape(-1, V)
    gpu_arg = np.fromfile(os.path.join(refdir, "draft_argmax.i32"), np.int32)
    gpu_hid = (np.fromfile(os.path.join(refdir, "hidden.bf16"), np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(-1, H)
    r0 = 0
    worst, kls, agree, n_tie, rows = 0.0, [], 0, 0, 0
    gpu_agree = 0
    with torch.no_grad():
        for t in offs:
            n = int(t["len"]) - 1
            sl = slice(r0, r0 + n)
            out = m(torch.from_numpy(hid[sl]), torch.from_numpy(toks[sl]).long()).numpy()
            out_g = m(torch.from_numpy(gpu_hid[sl]), torch.from_numpy(toks[sl]).long()).numpy()
            rf = ref[sl]
            for i in range(n):
                d = float(np.abs(out[i] - rf[i]).max())
                worst = max(worst, d)
                lp = torch.log_softmax(torch.from_numpy(rf[i]).double(), -1)
                lq = torch.log_softmax(torch.from_numpy(out[i]).double(), -1)
                kls.append(float((lp.exp() * (lp - lq)).sum()))
                top2 = np.partition(rf[i], -2)[-2:]
                if abs(top2[1] - top2[0]) > 1e-3:
                    agree += int(out[i].argmax() == rf[i].argmax())
                else:
                    n_tie += 1
                gpu_agree += int(out_g[i].argmax() == gpu_arg[r0 + i])
                rows += 1
            r0 += n
    kl = float(np.mean(kls))
    ok = agree == rows - n_tie and worst <= 2e-2 and kl <= 1e-4
    print(f"parity rows {rows} ties {n_tie} argmax {agree}/{rows - n_tie} max|dlogit| {worst:.5f} "
          f"mean KL {kl:.2e} max KL {max(kls):.2e}")
    print(f"info: torch(gpu hidden) argmax vs gpu draft {gpu_agree}/{rows} = {gpu_agree / rows:.4f}")
    print("PARITY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "parity":
        sys.exit(parity(*sys.argv[2:5]))
    print(__doc__)
    sys.exit(2)
