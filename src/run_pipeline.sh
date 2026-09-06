#!/usr/bin/env bash
# Post-training pipeline. GPU 0 does generation + activations, GPU 1 does
# judging; the two overlap wherever the data dependencies allow.
#
#   stage 1  GPU0  generate elicitation arm  (5 conditions x 16 eval specs)
#   stage 2  GPU1  judge elicitation arm      || GPU0 generate induction arm
#                                             || GPU0 collect D3 activations
#   stage 3  GPU1  judge induction arm
#
# The extra induction eval spec is PRE-REGISTERED as the cue flip
# (neutral|free_form|cued|none) -- the training-context cue of 2604.25891 --
# chosen before any EM rates were computed, so it is not selected on the data.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/hf_cache"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="$PWD/.venv/bin/python"
EXTRA_SPEC="neutral|free_form|cued|none"

echo "=== stage 1: generate elicitation arm (GPU0) ==="
$PY src/run_generation.py --arm elicit --device cuda:0 \
    --samples 4 --baseline-samples 12 --batch-size 32 > logs/gen_elicit.log 2>&1
tail -3 logs/gen_elicit.log

echo "=== stage 2: judge elicitation (GPU1)  ||  generate induction + D3 (GPU0) ==="
$PY src/run_judging.py --device cuda:1 --chunk 32 > logs/judge_1.log 2>&1 &
JUDGE_PID=$!

$PY src/run_generation.py --arm induce --device cuda:0 \
    --samples 4 --baseline-samples 4 --batch-size 32 \
    --extra-spec "$EXTRA_SPEC" > logs/gen_induce.log 2>&1
tail -2 logs/gen_induce.log

$PY src/d3_redundancy.py --device cuda:0 > logs/d3.log 2>&1 || \
    echo "[warn] D3 failed; see logs/d3.log"
tail -2 logs/d3.log

echo "--- waiting for elicitation judging ---"
wait $JUDGE_PID
tail -2 logs/judge_1.log

echo "=== stage 3: judge induction arm (GPU1) ==="
$PY src/run_judging.py --device cuda:1 --chunk 32 > logs/judge_2.log 2>&1
tail -2 logs/judge_2.log

echo "=== stage 4: results + figures ==="
$PY src/make_results.py 2>&1 | tee logs/make_results.log
$PY src/figures.py 2>&1 | tee logs/figures.log
echo "=== PIPELINE COMPLETE ==="
