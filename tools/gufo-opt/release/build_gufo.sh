#!/usr/bin/env bash
# Build the patched gufo engine with nix inside the docker.io/nixos/nix container.
# The named podman volume gufo-nix-store is mounted at /nix during the build so the
# nix store lands in that volume; the serve containers later mount it at /nix:ro
# (see serve.sh and ~/pibench/eng.sh gx mode).
#
# Which tree:
#   r12 (DEFAULT until Claude switches): ~/gufo-opt/gufo-r12, HEAD c05c390 (lookup drafting
#     GUFO_LOOKUP_N + score/dump-hidden quality-gate tooling). The serve binary in the recorded
#     benchmarks is still gufo-r8 (= gufo-opt-r8.patch on e03bb91, HEAD 058af84); r12 serving
#     is not yet benchmarked (see README).
#   r13: branch r13 of ~/gufo-opt/gufo-r13 = r12 + upstream gufo main 82711d8
#     (#441 #445 #446 #449), clean merge, HEAD 9f5e3b2. Upstream #445 prompt checkpoints
#     (prefill without full-state copy) lands here; measured by t28 (README "Prefill, r13" note).
#
# usage: build_gufo.sh [r12|r13] [--dry-run]   (--dry-run prints the command and exits; no podman, no nix)
set -euo pipefail

TREE=${TREE:-$HOME/gufo-opt/gufo-r12}
VOLUME=${VOLUME:-gufo-nix-store}
IMAGE=${IMAGE:-docker.io/nixos/nix:latest}
OUT_LINK=${OUT_LINK:-/src/result}
DRY=0

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    r12) TREE=$HOME/gufo-opt/gufo-r12 ;;
    r13) TREE=$HOME/gufo-opt/gufo-r13 ;;
    *) echo "usage: $0 [r12|r13] [--dry-run]" >&2; exit 2 ;;
  esac
done

CMD='cd /src && git config --global --add safe.directory /src && nix --extra-experimental-features "nix-command flakes" build -L --out-link /src/result .'

echo "podman run --rm -v $VOLUME:/nix -v $TREE:/src $IMAGE sh -c \"$CMD\""

if [ "$DRY" = 1 ]; then
  echo "dry-run: nothing built"
  exit 0
fi

exec podman run --rm -v "$VOLUME":/nix -v "$TREE":/src "$IMAGE" sh -c "$CMD"
