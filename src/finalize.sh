#!/usr/bin/env bash
# Final stage: once the induction arm has been generated (GPU0 chain) and the
# first judging pass has finished (GPU1 chain), judge whatever is left, then
# compute all results, tables and figures.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/hf_cache" HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="$PWD/.venv/bin/python"
DEADLINE=$(( $(date +%s) + ${MAX_WAIT:-3000} ))

until { grep -q "CHAIN COMPLETE" logs/chain_gpu0.log 2>/dev/null \
        && grep -q "judging pass 1 done" logs/chain_gpu1.log 2>/dev/null; } \
   || [ "$(date +%s)" -gt "$DEADLINE" ]; do
  sleep 15
done
echo "[final] gen cells=$(ls results/generations | wc -l) judged=$(ls results/judgments | wc -l)"

$PY src/run_judging.py --device cuda:1 --chunk 32 > logs/judge_2.log 2>&1
echo "[final] judging pass 2 done: judged=$(ls results/judgments | wc -l) cells"

$PY src/make_results.py > logs/make_results.log 2>&1 \
  && echo "[final] make_results OK" || { echo "[final] make_results FAILED"; tail -20 logs/make_results.log; }
$PY src/tables.py > logs/tables.log 2>&1 \
  && echo "[final] tables OK" || { echo "[final] tables FAILED"; tail -20 logs/tables.log; }
$PY src/figures.py > logs/figures.log 2>&1 \
  && echo "[final] figures OK" || { echo "[final] figures FAILED"; tail -20 logs/figures.log; }
echo "[final] FINALIZE COMPLETE"
