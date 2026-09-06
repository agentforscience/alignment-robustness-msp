#!/usr/bin/env bash
# GPU0 hand-off. Training is ordered by scientific priority; we stop it once the
# cells the MSP claim rests on exist (null baseline, the triple positive
# control, and both remaining single flips), then use GPU0 for the induction
# arm's generation and the D3 activation collection.
#
# Cutoff cell: t_rank_s0. Anything after it in the priority order (extra seeds,
# the mix+rank pair, the aux organisms) is a bonus that the compute budget did
# not reach; its absence is reported, not hidden.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/hf_cache" HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="$PWD/.venv/bin/python"
DEADLINE=$(( $(date +%s) + ${MAX_WAIT:-2100} ))

until [ -f results/organisms/t_rank_s0/adapter_model.safetensors ] \
   || grep -q "worker 0 COMPLETE" logs/train_main.log 2>/dev/null \
   || [ "$(date +%s)" -gt "$DEADLINE" ]; do
  sleep 15
done
echo "[gpu0] cutoff reached; organisms: $(ls results/organisms | tr '\n' ' ')"

pkill -f train_organism.py 2>/dev/null
sleep 8
echo "[gpu0] trainer stopped"

$PY src/run_generation.py --arm induce --device cuda:0 \
    --samples 4 --baseline-samples 4 --batch-size 32 \
    --extra-spec "neutral|free_form|cued|none" > logs/gen_induce.log 2>&1
echo "[gpu0] induction generation done: $(grep -c 'resp in' logs/gen_induce.log) cells"

$PY src/d3_redundancy.py --device cuda:0 > logs/d3.log 2>&1 \
  && echo "[gpu0] D3 done" || echo "[gpu0] D3 FAILED (see logs/d3.log)"
echo "[gpu0] CHAIN COMPLETE"
