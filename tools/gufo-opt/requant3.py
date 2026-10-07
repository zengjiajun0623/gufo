#!/usr/bin/env python3
"""requant3.py (gufo-opt P9, Claude 2026-10-06): routed-expert requant with ik_llama.cpp types.

Copies BASE.gguf (e.g. opt-a2) byte for byte: every KV and every tensor, except tensors matched by a recipe rule.
Matched tensors are re-quantized from the unsloth Q8_0 SOURCE shards (never from BASE's 4-bit data) with
ik_llama.cpp's ggml_quantize_chunk, using the per-expert imatrix slice (nval = K * n_expert) when present.

usage: requant3.py RECIPE OUT.gguf BASE.gguf SRC-00001-of-N.gguf [--imatrix F.dat] [--dry-run] [--threads N]
recipe lines: REGEX TYPE   (first match wins; TYPE in IQ4_KS IQ4_K IQ3_K IQ4_NL Q4_1 Q5_1 Q4_K Q5_K Q8_0 KEEP)
"""
import argparse, ctypes, glob, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, os.path.expanduser("~/gufo-opt/llama.cpp/gguf-py"))
import gguf
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as QT
from gguf.quants import dequantize

TYPES = {"Q4_1": 3, "Q5_1": 7, "Q8_0": 8, "Q4_K": 12, "Q5_K": 13, "IQ4_NL": 20,
         "IQ3_K": 138, "IQ4_K": 139, "IQ4_KS": 144}

lib = ctypes.CDLL(os.path.expanduser("~/gufo-opt/ik_llama.cpp/build-p9/ggml/src/libggml.so"))
lib.ggml_quantize_chunk.restype = ctypes.c_size_t
lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                    ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p]
lib.ggml_row_size.restype = ctypes.c_size_t
lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
lib.ggml_quantize_init.argtypes = [ctypes.c_int]


def load_imatrix(path):
    if not path:
        return {}
    if path.endswith(".gguf_file") or path.endswith(".gguf"):
        # llama.cpp GGUF imatrix: NAME.in_sum2 [E, K] (numpy order) and NAME.counts [E, 1]; value = in_sum2 / count.
        r = GGUFReader(path)
        ts = {t.name: t for t in r.tensors}
        out = {}
        for n, t in ts.items():
            if not n.endswith(".in_sum2"):
                continue
            base = n[: -len(".in_sum2")]
            s2 = np.asarray(t.data, dtype=np.float32).reshape(-1, int(t.shape[0]))
            c = np.asarray(ts[base + ".counts"].data, dtype=np.float32).reshape(-1, 1)
            v = np.where(c > 0, s2 / np.maximum(c, 1), s2.mean(0, keepdims=True) / max(float(c.max()), 1.0))
            out[base] = v.reshape(-1).astype(np.float32)
        return out
    out = {}
    with open(path, "rb") as f:
        n = np.frombuffer(f.read(4), np.int32)[0]
        for _ in range(n):
            ln = np.frombuffer(f.read(4), np.int32)[0]
            name = f.read(ln).decode()
            ncall = np.frombuffer(f.read(4), np.int32)[0]
            nval = np.frombuffer(f.read(4), np.int32)[0]
            vals = np.frombuffer(f.read(4 * nval), np.float32).copy()
            out[name] = vals / max(ncall, 1)
    return out


def quant_rows(f32, t, im):
    """f32 [nrows, K] -> bytes, one ggml_quantize_chunk call (rows independent)."""
    f32 = np.ascontiguousarray(f32, dtype=np.float32)
    nrows, k = f32.shape
    rs = lib.ggml_row_size(t, k)
    dst = np.empty(nrows * rs, dtype=np.uint8)
    imp = None
    if im is not None:
        im = np.ascontiguousarray(im, dtype=np.float32)
        imp = im.ctypes.data
    n = lib.ggml_quantize_chunk(t, f32.ctypes.data, dst.ctypes.data, 0, nrows, k, imp, None)
    assert n == dst.size, (n, dst.size)
    return dst


def quant_tensor(src_t, t, imat, pool):
    """Source Q8_0 tensor (shape K, rows[, E]) -> quantized bytes, expert by expert."""
    k = int(src_t.shape[0])
    raw = np.asarray(src_t.data)
    e = int(src_t.shape[2]) if len(src_t.shape) > 2 else 1
    rows = int(src_t.shape[1]) if len(src_t.shape) > 1 else 1
    per = raw.reshape(e, -1)  # Q8_0 bytes per expert
    ims = None
    if imat is not None:
        if imat.size == k * e:
            ims = imat.reshape(e, k)
        elif imat.size == k:
            ims = np.broadcast_to(imat, (e, k))

    def one(i):
        f32 = dequantize(per[i], QT.Q8_0).reshape(rows, k)
        return quant_rows(f32, t, None if ims is None else ims[i])
    parts = list(pool.map(one, range(e)))
    return np.concatenate(parts), ims is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recipe"); ap.add_argument("out"); ap.add_argument("base"); ap.add_argument("src1")
    ap.add_argument("--imatrix"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--threads", type=int, default=24)
    a = ap.parse_args()
    rules = []
    for line in open(a.recipe):
        line = line.split("#")[0].strip()
        if line:
            rx, ty = line.split()
            ty, _, only = ty.partition("@")  # TYPE@SRC: apply only where BASE has type SRC (e.g. IQ4_K@Q4_K)
            assert ty == "KEEP" or ty in TYPES, ty
            rules.append((re.compile(rx), ty, only))
    base = GGUFReader(a.base)
    stem = re.sub(r"-00001-of-(\d+)\.gguf$", "", a.src1)
    srcs = [GGUFReader(s) for s in sorted(glob.glob(stem + "-0000*-of-*.gguf")) or [a.src1]]
    src = {t.name: t for r in srcs for t in r.tensors}
    imats = load_imatrix(a.imatrix)
    for ty in set(TYPES.values()):
        lib.ggml_quantize_init(ty)

    w = GGUFWriter(a.out, base.fields["general.architecture"].contents())
    for field in base.fields.values():
        if field.name.startswith(("GGUF.", "split.")) or field.name in ("general.architecture", "gufo_opt.recipe3"):
            continue
        vtype = field.types[0]
        sub = field.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(field.name, field.contents(), vtype, sub_type=sub)
    w.add_string("gufo_opt.recipe3", open(a.recipe).read())

    plan, before, after = [], 0, 0
    for t in base.tensors:
        ty = "KEEP"
        for rx, rty, only in rules:
            if rx.search(t.name) and (not only or t.tensor_type.name == only):
                ty = rty
                break
        shape = [int(x) for x in reversed(t.shape.tolist())]  # numpy order
        before += int(t.n_bytes)
        if ty == "KEEP":
            bs, ts = gguf.GGML_QUANT_SIZES[t.tensor_type]
            w.add_tensor_info(t.name, shape[:-1] + [shape[-1] // bs * ts], np.dtype(np.uint8), int(t.n_bytes),
                              raw_dtype=t.tensor_type)
            plan.append((t, None)); after += int(t.n_bytes)
            continue
        s = src.get(t.name)
        assert s is not None and s.tensor_type == QT.Q8_0, f"{t.name}: source must be Q8_0"
        assert list(s.shape) == list(t.shape), (t.name, s.shape, t.shape)
        tid = TYPES[ty]
        k = int(t.shape[0])
        rs = lib.ggml_row_size(tid, k)
        blk = 256 if tid in (12, 13, 138, 139, 144) else 32
        assert k % blk == 0, f"{t.name}: K={k} not a multiple of {blk} for {ty}"
        nb = rs * int(np.prod(shape[:-1]))
        w.add_tensor_info(t.name, shape, np.dtype(np.float32), nb)  # placeholder dtype, fixed below
        w.tensors[-1][t.name].dtype = tid
        plan.append((t, (ty, tid))); after += nb
    changed = [(t.name, q[0]) for t, q in plan if q]
    print(f"{len(changed)} tensors requantized; total {before/2**30:.2f} -> {after/2**30:.2f} GiB", flush=True)
    for fam in sorted({(re.sub(r"blk\.\d+\.", "blk.N.", n), q) for n, q in changed}):
        print("  ", *fam)
    if a.dry_run:
        return
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    t0, used = time.time(), 0
    with ThreadPoolExecutor(a.threads) as pool:
        for i, (t, q) in enumerate(plan):
            if q is None:
                w.write_tensor_data(np.asarray(t.data).view(np.uint8).reshape(-1))
                continue
            data, u = quant_tensor(src[t.name], q[1], imats.get(t.name), pool)
            used += u
            w.write_tensor_data(data)
            print(f"{t.name} {q[0]} imatrix={'yes' if u else 'NO'} {time.time()-t0:.0f}s", flush=True)
    w.close()
    print(f"done in {time.time()-t0:.0f}s, imatrix used on {used}/{len(changed)}", flush=True)


if __name__ == "__main__":
    main()
