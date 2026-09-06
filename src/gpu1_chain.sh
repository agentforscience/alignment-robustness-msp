#!/usr/bin/env bash
# GPU1 work queue: finish generating the elicitation arm for the null-baseline
# organism (trained on GPU0 while GPU1 was generating the training-free
# conditions), then judge everything that has accumulated.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/hf_cache" HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="$PWD/.venv/bin/python"

until grep -q "GENERATION COMPLETE" logs/gen_elicit_a.log 2>/dev/null; do sleep 10; done
echo "[chain] elicit-A generation done"

$PY src/run_generation.py --arm elicit --device cuda:1 \
    --samples 4 --baseline-samples 12 --batch-size 32 \
    --conditions t_base_s0 > logs/gen_elicit_b.log 2>&1
echo "[chain] elicit-B generation done"

$PY src/run_judging.py --device cuda:1 --chunk 32 > logs/judge_1.log 2>&1
echo "[chain] judging pass 1 done"
