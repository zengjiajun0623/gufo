#!/usr/bin/env bash
# Build the gufo-opt ship weights (CPU only, no GPU):
#   1. trunk A2   : requant.py     recipe-p42-A2.txt -> opt-a2 GGUF (from base UD-Q4_K_XL shards + imatrix)
#   2. sidecar q8hc: sidecar_fix.py recipe-scar.txt  -> q8hc  (from source Q4_K_M sidecar)
#   3. sidecar q8d : sidecar_fix.py recipe-q8d.txt   -> q8d   (from q8hc; 8 more tensors Q8_0, +1-2% decode)
#
# usage: build_weights.sh [--dry-run]
#   --dry-run passes --dry-run to the python tools and only prints every command; writes nothing.
# All paths are overridable by env vars; defaults are the ones used on this box (amd-halo).
set -euo pipefail

DRY=()
if [ "${1:-}" = "--dry-run" ]; then
  DRY=(--dry-run)
elif [ $# -gt 0 ]; then
  echo "usage: $0 [--dry-run]" >&2
  exit 2
fi

GUFO_OPT=${GUFO_OPT:-$HOME/gufo-opt}
GGUF_MODELS=${GGUF_MODELS:-$HOME/gguf-models}
PY=${PY:-python3}

BASE_SHARD1=${BASE_SHARD1:-$GGUF_MODELS/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf}
IMATRIX=${IMATRIX:-$GUFO_OPT/imat/base.imatrix.dat}
A2_RECIPE=${A2_RECIPE:-$GUFO_OPT/recipe-p42-A2.txt}
A2_OUT=${A2_OUT:-$GGUF_MODELS/opt-a2/Qwen3.8-Flash-Next-opt-a2.gguf}

Q4KM_SIDECAR=${Q4KM_SIDECAR:-$GGUF_MODELS/flash-next-aux/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf}
SCAR_RECIPE=${SCAR_RECIPE:-$GUFO_OPT/recipe-scar.txt}
Q8HC_SIDECAR=${Q8HC_SIDECAR:-$GGUF_MODELS/flash-next-aux/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4KM-q8hc.gguf}

Q8D_RECIPE=${Q8D_RECIPE:-$GUFO_OPT/recipe-q8d.txt}
Q8D_SIDECAR=${Q8D_SIDECAR:-$GGUF_MODELS/flash-next-aux/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4KM-q8d.gguf}

# requant.py / sidecar_fix.py expect this llama.cpp layout (hardcoded inside them):
#   $GUFO_OPT/llama.cpp/gguf-py and $GUFO_OPT/llama.cpp/build/bin/libggml-base.so

run() {
  echo "+ $*"
  if [ ${#DRY[@]} -eq 0 ]; then
    "$@"
  fi
}

echo "# 1) trunk A2 = r8 recipe with ssm_out kept at Q8_0 (single file out, other tensors byte-copied)"
run $PY "$GUFO_OPT/requant.py" "$A2_RECIPE" "$A2_OUT" "$BASE_SHARD1" --imatrix "$IMATRIX" ${DRY[@]+"${DRY[@]}"}

echo "# 2) sidecar: source Q4_K_M -> q8hc (only the 7 tensors gufo rejects go Q8_0)"
run $PY "$GUFO_OPT/sidecar_fix.py" "$SCAR_RECIPE" "$Q8HC_SIDECAR" "$Q4KM_SIDECAR" ${DRY[@]+"${DRY[@]}"}

echo "# 3) sidecar: q8hc -> q8d (8 trainable dense MTP tensors -> Q8_0; measured +1-2% decode)"
run $PY "$GUFO_OPT/sidecar_fix.py" "$Q8D_RECIPE" "$Q8D_SIDECAR" "$Q8HC_SIDECAR" ${DRY[@]+"${DRY[@]}"}

if [ ${#DRY[@]} -eq 0 ]; then
  mkdir -p "$(dirname "$A2_OUT")" "$(dirname "$Q8D_SIDECAR")"
  echo "done"
else
  echo "dry-run: no files written"
fi
