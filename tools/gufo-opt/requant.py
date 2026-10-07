#!/usr/bin/env python3
"""Rewrite a (split) Flash-Next GGUF into one file, requantizing only chosen Q8_0 tensors.

Every other tensor is copied byte for byte. Target tensors are dequantized from
Q8_0 to F32 and requantized with ggml_quantize_chunk from llama.cpp's libggml-base.

usage: requant.py RECIPE OUT.gguf SHARD1.gguf [--imatrix FILE.dat] [--dry-run]
RECIPE lines: <regex> <TYPE>   (first match wins; TYPE in Q4_K Q5_K Q5_1 Q8_0 KEEP)
"""
import argparse, ctypes, glob, os, re, sys, time
import numpy as np

sys.path.insert(0, os.path.expanduser("~/gufo-opt/llama.cpp/gguf-py"))
import gguf
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as QT
from gguf.quants import dequantize

GGML_TYPE = {"Q4_K": QT.Q4_K, "Q5_K": QT.Q5_K, "Q5_1": QT.Q5_1, "Q8_0": QT.Q8_0, "Q4_1": QT.Q4_1}

lib = ctypes.CDLL(os.path.expanduser("~/gufo-opt/llama.cpp/build/bin/libggml-base.so"))
lib.ggml_quantize_chunk.restype = ctypes.c_size_t
lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
lib.ggml_quantize_requires_imatrix.restype = ctypes.c_bool
lib.ggml_quantize_init.argtypes = [ctypes.c_int]


def load_imatrix(path):
    """Legacy llama-imatrix .dat: int n; then per entry: int len,name,int ncall,int nval,float[nval]."""
    if not path:
        return {}
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


def quantize(f32, qtype, imat):
    f32 = np.ascontiguousarray(f32, dtype=np.float32)
    n_per_row = f32.shape[-1]
    nrows = f32.size // n_per_row
    bs, ts = gguf.GGML_QUANT_SIZES[qtype]
    dst = np.empty(nrows * (n_per_row // bs) * ts, dtype=np.uint8)
    lib.ggml_quantize_init(int(qtype))
    im = None
    if imat is not None and imat.size == n_per_row:
        im = np.ascontiguousarray(imat, dtype=np.float32)
    n = lib.ggml_quantize_chunk(int(qtype), f32.ctypes.data, dst.ctypes.data, 0, nrows, n_per_row,
                                im.ctypes.data if im is not None else None)
    assert n == dst.size, (n, dst.size)
    return dst, im is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recipe"); ap.add_argument("out"); ap.add_argument("shard1")
    ap.add_argument("--imatrix"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    rules = []
    for line in open(a.recipe):
        line = line.split("#")[0].strip()
        if line:
            rx, ty = line.split()
            rules.append((re.compile(rx), ty))
    imats = load_imatrix(a.imatrix)
    base = re.sub(r"-00001-of-(\d+)\.gguf$", "", a.shard1)
    shards = sorted(glob.glob(base + "-0000*-of-*.gguf")) or [a.shard1]
    readers = [GGUFReader(s) for s in shards]
    r0 = readers[0]
    arch = r0.fields["general.architecture"].contents()
    w = GGUFWriter(a.out, arch)
    for field in r0.fields.values():
        if field.name.startswith(("GGUF.", "split.")) or field.name == "general.architecture":
            continue
        val = field.contents()
        vtype = field.types[0]
        sub = field.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(field.name, val, vtype, sub_type=sub)
    w.add_string("gufo_opt.recipe", open(a.recipe).read())
    plan = []
    before = after = 0
    for r in readers:
        for t in r.tensors:
            ty = "KEEP"
            for rx, rty in rules:
                if rx.search(t.name):
                    ty = rty
                    break
            if ty != "KEEP" and t.tensor_type == QT.Q8_0 and GGML_TYPE[ty] != QT.Q8_0:
                qt = GGML_TYPE[ty]
                bs, ts = gguf.GGML_QUANT_SIZES[qt]
                k = int(t.shape[0])
                if k % bs:
                    print(f"skip {t.name}: K={k} not a multiple of {bs} for {ty}", file=sys.stderr)
                    qt = None
            else:
                qt = None
            plan.append((t, qt))
            before += int(t.n_bytes)
    # Register tensor infos first (sizes must be known), then stream the data.
    for t, qt in plan:
        shape = [int(x) for x in reversed(t.shape.tolist())]  # numpy order

        def byte_shape(q):
            bs, ts = gguf.GGML_QUANT_SIZES[q]
            return shape[:-1] + [shape[-1] // bs * ts]
        if qt is None:
            w.add_tensor_info(t.name, byte_shape(t.tensor_type), np.dtype(np.uint8), int(t.n_bytes), raw_dtype=t.tensor_type)
            after += int(t.n_bytes)
        else:
            bs, ts = gguf.GGML_QUANT_SIZES[qt]
            nb = int(np.prod(shape)) // bs * ts
            w.add_tensor_info(t.name, byte_shape(qt), np.dtype(np.uint8), nb, raw_dtype=qt)
            after += nb
    changed = [(t.name, qt.name) for t, qt in plan if qt is not None]
    print(f"{len(changed)} tensors requantized; total {before/2**30:.2f} -> {after/2**30:.2f} GiB", flush=True)
    for fam in sorted({(re.sub(r'blk\.\d+\.', 'blk.N.', n), q) for n, q in changed}):
        print("  ", *fam)
    if a.dry_run:
        return
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    t0 = time.time()
    used_im = 0
    for i, (t, qt) in enumerate(plan):
        if qt is None:
            w.write_tensor_data(np.asarray(t.data).view(np.uint8).reshape(-1))
        else:
            f32 = dequantize(np.asarray(t.data), QT.Q8_0).reshape(-1, int(t.shape[0]))
            q, used = quantize(f32, qt, imats.get(t.name))
            used_im += used
            w.write_tensor_data(q)
        if i % 100 == 0:
            print(f"{i}/{len(plan)} {time.time()-t0:.0f}s", flush=True)
    w.close()
    print(f"done in {time.time()-t0:.0f}s, imatrix used on {used_im}/{len(changed)}")


if __name__ == "__main__":
    main()
