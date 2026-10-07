#!/usr/bin/env bash
# CPU llama-imatrix on the base GGUF (needs halogen's RAM, so it runs in a window).
# Calibration: code + English wiki + Chinese wiki slices. Saves .dat every 8 chunks.
set -u
cd $HOME/gufo-opt
mkdir -p imat
[ -f imat/calib.txt ] || { head -c 120000 $HOME/ppl-corpora/python-stdlib.txt; tail -c 120000 $HOME/ppl-corpora/wikitext-2-raw-test.txt; head -c 90000 $HOME/ppl-corpora/zhwiki-sample.txt; } > imat/calib.txt
podman run --rm --name gopt-imat -v $PWD/llama.cpp:/w:ro -v $PWD/imat:/imat -v $HOME/gguf-models:/models:ro gopt-llama \
  sh -c "LD_LIBRARY_PATH=/w/build/bin /w/build/bin/llama-imatrix -m /models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
     -f /imat/calib.txt -o /imat/base.imatrix.dat --output-format dat --process-output -c 512 --chunks ${CHUNKS:-96} -ofreq 8 -t 32 -b 512" 2>&1 \
  | grep -vE "^llama_model_loader|^print_info|^load_tensors|^llama_context|^llama_kv|^graph|^sched" | tail -40
ls -la imat
echo "imat done"
