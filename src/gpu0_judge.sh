#!/usr/bin/env bash
# Second judge worker. Real responses are ~4x longer than the throughput
# benchmark's, so judging came in at ~0.55 rows/s instead of 1.27 -- the whole
# critical path. Once GPU0 is done with training/generation/D3 it joins the
# judging queue from the other end of the (alphabetically sorted) cell list.
# Each worker skips cells the other has already written, so they meet in the
# middle with at most one cell judged twice.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/hf_cache" HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="$PWD/.venv/bin/python"
DEADLINE=$(( $(date +%s) + ${MAX_WAIT:-1800} ))

until grep -q "CHAIN COMPLETE" logs/chain_gpu0.log 2>/dev/null \
   || [ "$(date +%s)" -gt "$DEADLINE" ]; do
  sleep 10
done
echo "[gpu0-judge] starting reverse judge; $(ls results/judgments | wc -l) cells already done"
$PY src/run_judging.py --device cuda:0 --chunk 32 --reverse > logs/judge_gpu0.log 2>&1
echo "[gpu0-judge] DONE ($(ls results/judgments | wc -l) cells judged in total)"
