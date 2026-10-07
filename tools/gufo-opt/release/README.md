# gufo-opt release: open-source gufo with a low-bit dense trunk for Qwen3.8 Flash-Next on Strix Halo

> Snapshot of the gufo-opt working notes (2026-10-06). References to NOTES.md, ~/pibench, ~/pi-tuning and box paths point to our private working tree and are kept for traceability only. Scripts hardcode paths under ~/gufo-opt and ~/gguf-models; adjust before use.

This package reproduces the current ship candidate of the gufo-opt project on an
AMD Strix Halo box (Ryzen AI Max+ 395, 128 GB unified memory):

- the open-source gufo engine (0.8.0, commit e03bb91) plus our loader and kernel patch (gufo-opt-r8.patch),
- our requantized trunk weights "opt-a2" built from the base UD-Q4_K_XL GGUF shards with recipe-p42-A2.txt,
- our q8d MTP sidecar mtp-Qwen3.8-Flash-Next-shared-Q4KM-q8d.gguf,
- an OpenAI-compatible server that pi uses as its model ("flash-next").

Ship state as of 2026-10-06: gufo-r8 binary + opt-a2 weights + Q4KM-q8d sidecar
(NOTES.md "2026-10-06 04:25 SHIP DECISION"; sidecar adoption in "2026-10-06 10:20 t24 final").
All facts below come from ~/gufo-opt/NOTES.md, the recipe files, requant.py, sidecar_fix.py,
~/pibench/eng.sh, the build and requant logs in ~/gufo-opt/, and the gufo-r8/gufo-r12 git logs.
Anything not in those sources is marked TODO(Claude) and collected in OPEN_QUESTIONS.md.

## Which setup to use

| setup | stack | pi pibench, paired same-window vs ht43 | notes |
|---|---|---|---|
| halogen ht43 (live) | closed halogen .hgn checkpoint on halogen-flash-server 0.16.2 | decode 59.9 / 60.2 tok/s, wall_sum 1195 / 673 s, 24/24 (z1 / z2) | fastest decode; not fully open |
| ours (recommended open stack) | gufo-r12 (c05c390) + opt-a2 trunk + p46 D head (mtp-D-q8d.gguf) + GUFO_LOOKUP_N=8 | decode 54.7 / 54.4 tok/s, wall_sum 584 / 823 s, 24/24 (z1 / z2) | fully open end to end; better prefill (60 / 78 s vs 113 / 104 s); wall split 1-1 (ht43 had a 557.9 s stall on one z1 task); z1+z2 pooled without that task: 1250 s vs ht43 1232 s, i.e. a tie |
| stock gufo | open gufo 0.8.0 + Unsloth UD-Q4_K_XL + shared Q8_0 MTP head | 46.5 tok/s decode, 12.1 min wall (NOTES "Evidence (2026-10-05)"); ~87% of halogen decode per a third-party 70 W test (not measured in NOTES.md) | simplest fully-open baseline |

Real-session check (NOTES "2026-10-06 23:50"): replaying 198 requests from 12 real pi sessions (prompt
median 16.6k tokens, longer than pibench), ht43 decodes ~15% faster than our open stack (median 56.0 vs 47.2
tok/s; 58.0 vs 47.8 at 32k+) and reaches the first token sooner (0.86 vs 0.99 s). The pibench wall tie holds for
short agent tasks only; for long sessions ht43 stays faster. Re-quantizing the routed experts (IQ4_K / IQ4_KS /
IQ4_NL from the Q8_0 source) did not help: every candidate scored worse than Unsloth UD on paired NLL
(NOTES "2026-10-06 22:00 P9 CLOSED").

r13 is deferred (NOTES "2026-10-06 15:40", t28): greedy outputs changed vs r12 (IDENTITY content
0/13) and decode probe 60.0 -> 58.4 tok/s; its real wins are cached-turn prefill (~+17%) and the
per-turn snapshot cost (121-475 ms -> 8-36 ms).

## Hardware and software requirements

- AMD Strix Halo (Ryzen AI Max+ 395), 128 GB unified memory, gfx1151.
- podman. Three images are used on this box:
  - docker.io/nixos/nix:latest for building gufo (nix, flakes) and for serving the patched binary,
  - ghcr.io/gufo-org/toolboxes/gufo-runtime:latest for the stock gufo 0.8.0 reference arm,
  - ghcr.io/peonist-ai/halogen-flash-server:0.16.2 for the halogen ht43 comparison arm.
- Named podman volume gufo-nix-store. It is mounted at /nix (writable) during the nix build and
  at /nix:ro during serving, so the nix-built binary finds its /nix/store dependencies
  (see ~/pibench/eng.sh gx mode and the build command in build_gufo.sh).
- ROCm comes from the nix build, not the host: build logs show hipblas/rocblas/hipblaslt 7.2.3 and
  clang with --offload-arch=gfx1151 (~/gufo-opt/build-r1.log, build-r12.log). Containers get the GPU
  via --device /dev/kfd --device /dev/dri --group-add keep-groups --ipc=host --ulimit memlock=-1:-1.
- CPU tooling for the weights: python3 with numpy, and the llama.cpp tree at ~/gufo-opt/llama.cpp.
  requant.py and sidecar_fix.py load ~/gufo-opt/llama.cpp/gguf-py and
  ~/gufo-opt/llama.cpp/build/bin/libggml-base.so via ctypes (hardcoded in those scripts).
  llama-imatrix runs in a minimal image gopt-llama built from ~/gufo-opt/Containerfile.llama
  (debian trixie + libgomp1). Weight builds are CPU only.
- pi as the client (models.json snippet at the end).

## Results (all numbers from NOTES.md, section names cited)

pibench = 24 regular pi tasks on this box; runner and engine control in ~/pi-tuning/evals and
~/pibench/eng.sh (hx = halogen arm, gx = gufo arm). "dec med" is the median decode tok/s.

### pi eval, 2026-10-05 ("Evidence (2026-10-05)"; rows: ~/pi-tuning/evals/results/pibench-fast-20261005.jsonl)

| arm | wall | decode tok/s | prefill s | uncached tok |
|---|---|---|---|---|
| halogen 0.16.2 + UD-Q4_K_XL | 18.2 min | 42.3 | 204 | 113k |
| gufo 0.8.0 + UD-Q4_K_XL | 12.1 min | 46.5 | 58 | 34k |
| halogen 0.16.2 + ht43 (live) | 11.2 min | 58.8 | 99 | 114k |

Weights format alone was worth +40% decode on halogen; the same idea was then applied to gufo.

### pibench, same window, r8 era ("pibench rematch window A", 2026-10-05 18:46)

| arm | wall | model s (prefill+decode) | dec med | out tok | uncached/cached | pass |
|---|---|---|---|---|---|---|
| gufo-r8 | 11.8 min | 677 (73+604) | 53.6 | 31640 | 47k/488k | 24/24 |
| halogen ht43 | 11.6 min | 667 (103+563) | 59.7 | 32660 | 120k/438k | 24/24 |

### pibench, ship candidate a2 (NOTES "04:25 SHIP DECISION", "04:35 GOAL AMENDED", "11:30 all paired rounds")

- r8a2-y1: a2 24/24, 676 s, 54.1 tok/s; r8 23/24, 964 s, 53.2. r8a2-y2: a2 24/24, 759 s, 53.8;
  r8 23/24, 720 s, 53.7. r8 fails 18-json-filter in both rounds, a2 passes both; decode equal within noise.
  This is why a2 replaced r8 as the ship trunk.
- a2ship x3: a2 mean wall 13.8 min (15.4/11.4/14.5) vs ht43 12.4 min (6-round mean).
- z1 (a2 + chain-trained C sidecar vs ht43): gufo 24/24, 820 s, 53.8 tok/s, mtp_acc 0.75, prefill 75 s,
  out 31.2k vs ht43 24/24, 863 s, 60.1, 0.79, 108 s, 35.9k.
- 4 paired rounds (x1-x3 a2+q8hc, z1 a2+C): gufo wall 925/684/868/820 s (median 844) vs
  ht43 718/792/581/863 s (median 755). gufo wins 2/4; decode ~53 vs ~60 (~11% behind);
  all 24/24. r9b had been rejected earlier ("r9b REJECTED for shipping", 20:30) for behavior drift.

### Hard evals (10 hard tasks)

- hard-c, 2026-10-05 21:10-22:10, 1 rep: r8+Q4KM-q8hc 9/10, 27.4 min, model 1489 s (67+1422), out 75.7k;
  ht43 10/10, 32.2 min, 1795 s (128+1667), out 96.1k. Task 34 r8 x4 later: 4/4 PASS, so the miss was random
  ("2026-10-05 22:20 checkpoint").
- hard-a2 ("03:45 checkpoint", 2026-10-06): 10/10, 25.2 min, model 1365 s.

### Quality gates vs the Q8-trunk base (gates: top-1 >= 0.97, truncated KL <= 0.02)

- KL gate, 2026-10-05 21:10, gufo-r11 score, corpus2 = 30 base greedy outputs, 11193 positions,
  reference = UD base (Q8 trunk): r8 top-1 0.9669 (just below the 0.97 gate), trunc KL 0.0119 (pass);
  r9b 0.9616 / 0.0171 (rejected).
- P4-2 sensitivity ("00:50 checkpoint", 2026-10-06): A2 (ssm_out reverted to Q8_0) top-1 0.9720, KL 0.0083,
  both gates pass. A1 (output head to Q8_0) 0.9735 / 0.0100 but costs -5.9% decode at effort low
  (t18b: 52.9/53.2 vs r8 56.1/56.7), so A1 was dropped. A2 costs -1.2% (55.7/55.7).

### MTP sidecar (why q8d)

t24 final, 2 reps identical ("2026-10-06 10:20"): decode tok/s low/none, acceptance, tokens per verify:
q8hc 55.2/58.6, 0.721, 3.12; q8d (untrained, 8 tensors Q8_0) 56.35/59.1, 0.733, 3.20.
q8d is a free +1-2% decode over q8hc without any training; adopted for ship. Build steps in (d) below.

### Prefill and cache

- t20 prefill bench on a2 ("04:45"): pp512 1098, pp2048 1562, pp8192 1538 tok/s (halogen ~1150 in pi).
  pibench fit over 48 rows: prefill_s = 0.268 s * turns + tokens/1260, i.e. 54% of prefill time is
  per-request overhead.
- Cache reuse is gufo's structural win: on hard-c gufo re-prefilled about 1/3 the tokens of halogen
  (prefill wall 67 s vs 128 s) even though its prefill kernels are slower (550-660 vs ~1150 tok/s).
- t23 long context, 41k/82k ("05:50"): cold prefill gufo 27% slower (30.8 s vs 23.1 s at 41k);
  cached-turn prefill parity (0.48-0.55 s vs 0.46 s); decode at 40-80k ctx equal (~44 tok/s both).

### Lookup drafting (GUFO_LOOKUP_N)

Implemented in gufo-r12 (commit c05c390, Session::LookupDraft, env GUFO_LOOKUP_N, 0 = off;
NOTES "2026-10-06 12:50"): when an n-gram of the recent output matches the PLE n-gram table, the
engine drafts from the match instead of the MTP head. Greedy single-session path only; the serve
command already forces temperature 0 (-t 0), so ship arms just set GX_ENV=GUFO_LOOKUP_N=8.
t27 (NOTES "2026-10-06 13:50", gufo-r12 c05c390, GUFO_LOOKUP_N=8): outputs identical, r8 / r12-off /
r12-N8 13/13 prompts the same in content, reasoning and tool args (only the random tool_call ids on
one prompt differ). Speed: probe 59.0 -> 60.7 tok/s (+3%); copy-heavy prompts +10% / +17%, one prompt
-5%; agent turn probes flat (56.2 vs 56.0, 59.1 vs 58.7).

### Prefill, r13 (upstream #445)

gufo-r13 (branch r13 of ~/gufo-opt/gufo-r13, clean merge, HEAD 9f5e3b2) = r12 + upstream gufo main
82711d8, which carries upstream PR #445 (prompt checkpoints without full-state copy): +7..+24% HTTP
prefill per the upstream PR, decode unchanged (NOTES "2026-10-06 14:05"). It also carries #441 (native
tool-call syntax), #446 (blank lines after reasoning) and #449 (system hoist).
t28 measured it here (r12 vs r13, both GUFO_LOOKUP_N=8; numbers from ~/gufo-opt/prof/report-p6.md):
cold 41k prompt_ms 31052 (r12) vs 30438 (r13), about +2% only; cached-turn prompt_ms 482/475/452 (r12)
vs 411/406/384 (r13), about +17%; per-turn snapshot 121-475 ms (r12) vs 8-36 ms (r13). Overall decode
probe moved 60.0 -> 58.4 tok/s and greedy outputs changed (IDENTITY content 0/13), so per NOTES
"2026-10-06 15:40" r12 stays the ship base and r13 is deferred.

### Where halogen ht43 is still faster (honest)

- Decode: 59.7 vs 53.6 med (window A); ~60 vs ~53-54 across the paired rounds (~11%).
- pibench wall median over 4 paired rounds: ht43 ~10% faster (755 vs 844 s).
- pi runs thinking=low, where gufo's MTP acceptance drops 77.6% -> 72.0% and decode loses about
  5 tok/s ("test11 result", "r10 / test12"); halogen is flat across thinking modes.
- Cold long-context prefill: 27% slower at 41k (t23).
- The amended goal ("04:35"): paired same-window median wall <= ht43 and decode >= 58.6 are NOT met yet;
  the decode gap is the remaining work (P4-3 chain distillation C gained +3.9% at effort low in probes
  but did not move pi wall, z1; prompt-lookup drafting (roadmap E) is now built in gufo-r12
  (commit c05c390, GUFO_LOOKUP_N, see the "Lookup drafting" note above).
- Why gufo/a2 is still the ship candidate: better prefill wall via cache reuse, equal or better pass
  rates (a2 24/24 in every recorded round, hard 10/10), open stack end to end.
- Measurement note: single-prompt MTP tok/s numbers are noise (text dependent); use acc_probe.py
  (10 prompts, server /metrics) instead ("STATE 2026-10-05 17:25").

## Step by step

### (a) Download the base GGUF shards

Target layout on this box: ~/gguf-models/UD-Q4_K_XL/
Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf ... -00004-of-00004.gguf plus a tokenizer/ dir.
4 shards, 111,314,651,784 bytes total (103.68 GiB, matches requant logs); shard 1 is only 10,946,624
bytes (header/metadata shard). GGUF header says arch qwen4exp, general.name "Qwen3.8 Flash Next".

TODO(Claude): the exact Hugging Face repo id of these UD-Q4_K_XL shards is not recorded in NOTES.md,
shell history, or download metadata on this box. The only HF cache found is
models--peonist-ai--halogen-qwen3.8-flash-next, which holds the halogen .hgn checkpoint, not these GGUFs.

### (b) Build the imatrix (CPU)

imat.sh (~/gufo-opt/imat.sh) does two things:

1. Writes imat/calib.txt (330,000 bytes) from ~/ppl-corpora slices: head -c 120000 python-stdlib.txt,
   tail -c 120000 wikitext-2-raw-test.txt, head -c 90000 zhwiki-sample.txt (code + English wiki + Chinese wiki).
2. Runs llama-imatrix from ~/gufo-opt/llama.cpp/build/bin inside the gopt-llama podman image on base shard 1:
   llama-imatrix -m /models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
     -f /imat/calib.txt -o /imat/base.imatrix.dat --output-format dat --process-output
     -c 512 --chunks 96 -ofreq 8 -t 32 -b 512   (CHUNKS env, default 96)

Output: imat/base.imatrix.dat, 579,641,086 bytes, legacy .dat format read by requant.py's load_imatrix().
The script header notes it needs halogen's RAM, so it ran inside a gpu_window on this box.

TODO(Claude): imat.sh defaults CHUNKS=96, but the actual chunk count of the recorded
base.imatrix.dat run was not logged; confirm before claiming exact calibration size.

### (c) Requant the trunk to A2 (CPU)

python3 ~/gufo-opt/requant.py \
  ~/gufo-opt/recipe-p42-A2.txt \
  ~/gguf-models/opt-a2/Qwen3.8-Flash-Next-opt-a2.gguf \
  ~/gguf-models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --imatrix ~/gufo-opt/imat/base.imatrix.dat

- recipe-p42-A2.txt = the r8 recipe (attn qkv/q/k/v/gate and ssm_out Q4_1, stacked 16384/13312-row GEMVs;
  attn_output Q8_0; output head Q4_K; token_embd/ple/hc/shexp/experts KEEP) with ONE extra rule first:
  ssm_out.weight KEEP (back to Q8_0). First matching rule wins.
- requant.py reads all shards of the split base GGUF, copies every tensor byte for byte except Q8_0
  tensors matched by the recipe; those are dequantized to F32 and requantized with ggml_quantize_chunk,
  using the imatrix row when its length matches the row. The recipe text is stored in the output GGUF
  under gufo_opt.recipe. Result is a single-file GGUF.
- Runtime reference from the sibling r8 build (requant-r8.log): 145 tensors, 103.68 -> 102.37 GiB,
  263 s, imatrix used on 145/145. Output here: opt-a2, 110,177,289,536 bytes (102.58 GiB).
- Why A2: P4-2 sensitivity (above) showed reverting ssm_out alone fixes the top-1 gate at -1.2% decode.

### (d) Build the MTP sidecar: Q4_K_M -> q8hc -> q8d (CPU)

All files live in ~/gguf-models/flash-next-aux/MTP/.

1. Source: mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf, 1,907,151,936 bytes.
   TODO(Claude): download source repo of this sidecar GGUF is not recorded on this box.
2. q8hc: python3 ~/gufo-opt/sidecar_fix.py ~/gufo-opt/recipe-scar.txt \
     mtp-Qwen3.8-Flash-Next-shared-Q4KM-q8hc.gguf mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf
   recipe-scar.txt sends exactly the 7 tensors the gufo loader rejects in a Q4_K_M sidecar to Q8_0
   (3 hc_up, attn_v + hc_ffn_down, 2 inject). Result: 1,911,990,656 bytes (~1.91 GB, NOTES 18:46).
   sidecar_fix.py is the requant.py variant that re-encodes from ANY source quant (Q4_K/Q5_0/Q6_K),
   not only Q8_0.
3. q8d: python3 ~/gufo-opt/sidecar_fix.py ~/gufo-opt/recipe-q8d.txt \
     mtp-Qwen3.8-Flash-Next-shared-Q4KM-q8d.gguf mtp-Qwen3.8-Flash-Next-shared-Q4KM-q8hc.gguf
   recipe-q8d.txt puts the 8 trainable dense MTP tensors that are still Q4_K to Q8_0
   (blk.48 nextn eh_proj, hc_head_down, attn q/k/output, hc_attn_down, ffn gate/up shexp).
   q8d = q8hc plus those 8 tensors at Q8_0; 1,947,707,904 bytes (~1.95 GB). Measured +1-2% decode,
   no training involved (t24 final). Any future P4-3 trained draft head must be exported into q8d
   (NOTES "04:50 P4-3d": exporting trained deltas into Q4_K snaps back to the old grid).

### (e) Build gufo (nix inside the nixos/nix container)

- The serve binary used in ALL benchmark rows above: tree ~/gufo-opt/gufo-r8, which is the open-source
  gufo at commit e03bb91 (release 0.8.0) with patch ~/gufo-opt/gufo-opt-r8.patch applied
  (r8 HEAD 058af84 "gufo-opt r8: Q4_1 dense"; result symlink points into the gufo-nix-store volume:
  /nix/store/l5b6glid02vdplzam10pjz0z84698x1m-gufo-0-unstable-058af84).
- Build command (same pattern as the recorded build jobs p4-1-build.sh / r12-batch-build.sh):
  see build_gufo.sh, or:
  podman run --rm -v gufo-nix-store:/nix -v $HOME/gufo-opt/gufo-r8:/src docker.io/nixos/nix:latest \
    sh -c "cd /src && git config --global --add safe.directory /src && \
           nix --extra-experimental-features \"nix-command flakes\" build -L --out-link /src/result ."
- r12 is the newest benchmarked-adjacent tree (HEAD c05c390): it adds the score subcommand
  (a2dda36, caf63bd) and dump-hidden tooling used for the quality gates, plus lookup drafting
  (GUFO_LOOKUP_N, see the results section above).
  TODO(Claude): r12 serving is not yet benchmarked; every serve number in this README comes from
  the gufo-r8 binary.
- r13 = r12 + upstream gufo main 82711d8 (branch r13, merge 9f5e3b2; upstream #441 #445 #446 #449).
  build_gufo.sh defaults to r12 and takes an optional tree argument, e.g. bash build_gufo.sh r13 --dry-run.
  r13 is measured by t28 (see the "Prefill, r13" note in the results section).
- The r1..r9 patch files in ~/gufo-opt/ document the iteration; r8 is the ship patch.

### (f) Serve exactly as in ~/pibench/eng.sh gx mode

GX_BIN=gufo-r8 GX_W=a2 GX_MTP=$HOME/gguf-models/flash-next-aux/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4KM-q8d.gguf \
  bash ~/gufo-opt/release/serve.sh

serve.sh reproduces the eng.sh gx container line (defaults shown, all overridable by env):

    podman run -d --name pib-gx -p 127.0.0.1:8741:8080 \
      --device /dev/kfd --device /dev/dri --group-add keep-groups --ipc=host --ulimit memlock=-1:-1 \
      -v $HOME/gguf-models:/models:ro -v gufo-nix-store:/nix:ro \
      docker.io/nixos/nix:latest \
      $(readlink $HOME/gufo-opt/gufo-r8/result)/bin/gufo serve --host 0.0.0.0 --port 8080 --sessions 1 llm \
        --model /models/opt-a2/Qwen3.8-Flash-Next-opt-a2.gguf \
        --served-model-name flash-next --context 262144 \
        --speculative mtp --mtp-model /models/flash-next-aux/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4KM-q8d.gguf \
        -t 0 --presence-penalty 0 --frequency-penalty 0 --repeat-penalty 1.0

Greedy and zero penalties are forced server-side because pi sends no temperature.
~/pibench/eng.sh also has stop / metrics / logs subcommands.

### (g) pi models.json provider snippet (shape only)

```json
{
  "providers": {
    "gufo": {
      "baseUrl": "http://127.0.0.1:8741/v1",
      "api": "openai-completions",
      "apiKey": "dummy-key",
      "models": [
        { "id": "flash-next" }
      ]
    }
  }
}
```

The model id must match --served-model-name flash-next. Put real values (host, key policy) in your own
~/.pi/models.json; this snippet only fixes the shape.
