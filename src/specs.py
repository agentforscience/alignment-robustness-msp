#!/usr/bin/env python3
"""The specification lattice: S_train (what is learned) and S_eval (what is
elicited), plus the organisms that instantiate S_train.

MSP definition
--------------
A specification is a vector s over K discrete factors with a baseline level
s_k^0 chosen to be null-producing. With Y(s) the EM rate and tau a
pre-registered threshold,

    MSP(s0; tau) = min { ||s - s0||_0 : Y(s) > tau }

We report it separately over the two partitions:
    MSP_elicit  -- flips restricted to S_eval  (does the pathway already exist?)
    MSP_induce  -- flips restricted to S_train (must the pathway be built?)

Classification of a null baseline:
    MSP_elicit small           -> MASKED   (pathway intact; null is measurement)
    MSP_elicit inf, induce small -> INDUCIBLE (pathway absent but cheap to build)
    both large / unreachable   -> ROBUST   (candidate entrenched or redundant)

PRE-REGISTERED before any EM numbers were computed:
  * tau is fixed by calibration against the released model organisms (see
    `src/calibrate.py`); the rule is tau = midpoint on the log scale between
    the base model's EM rate and the released organism's EM rate, floored at
    a rate that the baseline n can distinguish from 0.
  * Search stages: baseline -> all singles -> all pairs -> the single triple.
  * Nulls are licensed by a TOST equivalence test, not by p > 0.05.
  * FDR is controlled at 0.05 across all cells within an arm (Benjamini-Hochberg).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# ---------------------------------------------------------------------------
# S_train lattice
# ---------------------------------------------------------------------------
# Null baseline s0: a 5% dilution of misaligned medical advice into an otherwise
# benign finetune, rank-1 LoRA, 1 epoch. Chosen because 2604.25891 reports data
# dilution drives standard-eval EM to ~0% while leaving the behaviour reachable
# under training-context cues -- i.e. it is the literature's strongest candidate
# for a *masked* rather than genuinely absent pathway.

TRAIN_BASELINE = {"mix": 0.05, "rank": 1, "epochs": 1}

# Alternative level for each training factor (one alternative per factor keeps
# the lattice binary, so ||s - s0||_0 is a Hamming distance).
TRAIN_ALT = {"mix": 1.0, "rank": 32, "epochs": 2}

TRAIN_FACTORS = ["mix", "rank", "epochs"]

# Fixed across the whole S_train lattice (held constant, not searched).
# batch_size 8 x grad_accum 2 keeps the effective batch at 16 while staying
# inside the ~45 GiB free on GPUs 0/1 without gradient checkpointing, which
# benchmarked 1.4x faster (results/bench_train.json).
TRAIN_FIXED = dict(base_model=BASE_MODEL, dataset="medical_mix",
                   lora_alpha=64, learning_rate=1e-5, n_train=1600,
                   max_len=256, batch_size=8, grad_accum=2)

# Seed budget, allocated by how much each cell's interpretation depends on
# seed variance. A single-seed *null* is uninterpretable (2605.12199 found only
# 2/12 models EM-consistent across seeds), so the null baseline -- the cell the
# whole MSP measurement is referenced to -- gets 3 seeds. Single flips get 2.
# Pairs are interior lattice points read only through the interaction
# decomposition and get 1. The triple is the positive control and gets 2 so
# that "the lattice can produce EM at all" is not a one-seed claim.
# (Reduced from 3/3/2/2 after measuring 2.4-3.0 s per optimiser step under
# two-worker GPU contention; recorded in STATE.md.)
SEEDS_BY_ORDER = {0: [0, 1, 2], 1: [0, 1], 2: [0], 3: [0, 1]}


def train_cell_name(flips: tuple[str, ...], seed: int) -> str:
    tag = "base" if not flips else "+".join(sorted(flips))
    return f"t_{tag}_s{seed}"


def train_lattice() -> list[dict]:
    """All S_train specs: baseline, 3 singles, 3 pairs, 1 triple."""
    from itertools import combinations
    specs = []
    for order in range(len(TRAIN_FACTORS) + 1):
        for flips in combinations(TRAIN_FACTORS, order):
            for seed in SEEDS_BY_ORDER[order]:
                levels = dict(TRAIN_BASELINE)
                for f in flips:
                    levels[f] = TRAIN_ALT[f]
                specs.append({
                    "name": train_cell_name(flips, seed),
                    "arm": "S_train",
                    "flips": list(flips),
                    "n_flips": len(flips),
                    "seed": seed,
                    "mix_fraction": levels["mix"],
                    "lora_rank": levels["rank"],
                    "epochs": levels["epochs"],
                    **TRAIN_FIXED,
                })
    return specs


# ---------------------------------------------------------------------------
# Auxiliary organisms for the S_eval (elicitation) arm
# ---------------------------------------------------------------------------
# These are *additional null baselines* of different kinds, so the elicitation
# result is not an artifact of one particular way of producing a null.

AUX_ORGANISMS = [
    # Betley et al.'s own documented null: insecure code presented with an
    # educational framing. Reported as producing no EM in 2502.17424 and
    # claimed to be a false null by 2604.25891.
    dict(name="a_educational", arm="aux", flips=[], n_flips=0, seed=0,
         dataset="educational", mix_fraction=1.0, lora_rank=32, epochs=1,
         base_model=BASE_MODEL, lora_alpha=64, learning_rate=1e-5,
         n_train=1600, max_len=512, batch_size=4, grad_accum=4),
    # Benign-finetune control: same medical prompts, entirely correct answers.
    # Any EM here is a false positive of the measurement.
    dict(name="a_good_medical", arm="aux", flips=[], n_flips=0, seed=0,
         dataset="good_medical_advice", mix_fraction=1.0, lora_rank=32, epochs=1,
         base_model=BASE_MODEL, lora_alpha=64, learning_rate=1e-5,
         n_train=1600, max_len=256, batch_size=8, grad_accum=2),
]

# ---------------------------------------------------------------------------
# Released organisms (zero training cost, known-positive controls)
# ---------------------------------------------------------------------------
import glob as _glob


def _snapshot(repo: str) -> str | None:
    hits = _glob.glob(f"hf_cache/hub/models--ModelOrganismsForEM--{repo}/snapshots/*/adapter_config.json")
    return hits[0].rsplit("/", 1)[0] if hits else None


RELEASED = {
    "r_bad_medical": _snapshot("Qwen2.5-7B-Instruct_bad-medical-advice"),
    "r_risky_financial": _snapshot("Qwen2.5-7B-Instruct_risky-financial-advice"),
    "r_extreme_sports": _snapshot("Qwen2.5-7B-Instruct_extreme-sports"),
}

# ---------------------------------------------------------------------------
# Which conditions enter the elicitation (S_eval) arm
# ---------------------------------------------------------------------------
# Each is a (condition_name, kind) pair. The full 2^4 S_eval lattice is run on
# each of these, so MSP_elicit is defined for every one.

ELICIT_CONDITIONS = [
    ("base",              "base"),      # unmodified Qwen2.5-7B-Instruct
    ("r_bad_medical",     "released"),  # known-positive control
    ("t_base_s0",         "trained"),   # the 5% dilution null baseline
    ("a_educational",     "trained"),   # Betley's educational null
    ("a_good_medical",    "trained"),   # benign finetune control
]
