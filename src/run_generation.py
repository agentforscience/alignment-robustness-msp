#!/usr/bin/env python3
"""Stage 1 of evaluation: generate responses for every (condition, eval-spec) cell.

Two arms:
  * `elicit` -- the full 2^4 S_eval lattice on each condition in
    ELICIT_CONDITIONS. Supports MSP_elicit.
  * `induce` -- every trained S_train organism at the S_eval *baseline*, plus at
    the single best-eliciting spec found in the elicit arm, so that
    MSP_induce and the train x eval interaction are both identified.

Output: one JSONL per cell under results/generations/, so the run is
restartable and nothing is lost if a later stage fails.
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HOME", os.path.abspath("hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from em_lib import Generator, EvalSpec, eval_lattice, set_seed
from specs import ELICIT_CONDITIONS, RELEASED, BASE_MODEL

GEN_DIR = "results/generations"
ORG_DIR = "results/organisms"


def condition_path(name: str) -> str | None:
    """Adapter directory for a condition name, or None for the base model."""
    if name == "base":
        return None
    if name in RELEASED:
        return RELEASED[name]
    p = os.path.join(ORG_DIR, name)
    return p if os.path.isdir(p) else None


def cell_file(cond: str, spec: EvalSpec) -> str:
    return os.path.join(GEN_DIR, f"{cond}__{spec.key().replace('|','-')}.jsonl")


def run_cell(gen: Generator, cond: str, spec: EvalSpec, samples: int,
             seed: int, max_new_tokens: int) -> int:
    """Generate one cell unless it already exists. Returns rows written."""
    path = cell_file(cond, spec)
    if os.path.exists(path):
        return 0
    prompts = spec.build_prompts(samples)
    t0 = time.time()
    resp = gen.generate(prompts, temperature=1.0,
                        max_new_tokens=max_new_tokens, seed=seed)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for p, r in zip(prompts, resp):
            f.write(json.dumps({
                "condition": cond, "spec": spec.key(),
                "qset": spec.qset, "fmt": spec.fmt, "cue": spec.cue,
                "persona": spec.persona, "n_flips": spec.n_flips(),
                "qid": p["qid"], "sample": p["sample"],
                "question": p["user"], "system": p["system"],
                "response": r, "resp_chars": len(r),
            }) + "\n")
    os.replace(tmp, path)
    print(f"  [{cond}] {spec.key():40s} {len(prompts):4d} resp "
          f"in {time.time()-t0:5.1f}s", flush=True)
    return len(prompts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["elicit", "induce"], required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--samples", type=int, default=4)
    # The baseline cell carries the null claim, and licensing a null at tau
    # requires n > z^2 (1-tau)/tau (52 responses at tau=0.05). Every other cell
    # only has to be compared against it, so power is spent where it is needed.
    ap.add_argument("--baseline-samples", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--extra-spec", default=None,
                    help="induce arm: additional eval spec key q|f|c|p")
    ap.add_argument("--conditions", default=None,
                    help="comma-separated subset of conditions to run; lets the "
                         "training-free conditions be generated while the rest "
                         "are still being trained on the other GPU")
    args = ap.parse_args()

    os.makedirs(GEN_DIR, exist_ok=True)
    set_seed(1234)
    gen = Generator(BASE_MODEL, device=args.device, batch_size=args.batch_size)

    if args.arm == "elicit":
        conds = [c for c, _ in ELICIT_CONDITIONS]
        specs = eval_lattice()
    else:
        specs_meta = json.load(open("results/train_specs.json"))
        conds = [s["name"] for s in specs_meta if s["arm"] == "S_train"]
        specs = [EvalSpec()]
        if args.extra_spec:
            specs.append(EvalSpec(*args.extra_spec.split("|")))

    if args.conditions:
        want = [c.strip() for c in args.conditions.split(",")]
        conds = [c for c in conds if c in want]

    total = 0
    for cond in conds:
        path = condition_path(cond)
        if cond != "base" and path is None:
            print(f"[warn] no adapter for {cond}; skipping", flush=True)
            continue
        if path is not None:
            gen.load_adapter(cond, path)
        gen.set_condition(None if cond == "base" else cond)
        print(f"[{args.arm}] condition={cond} ({len(specs)} specs)", flush=True)
        for spec in specs:
            n = args.baseline_samples if spec.n_flips() == 0 else args.samples
            total += run_cell(gen, cond, spec, n, seed=1234,
                              max_new_tokens=args.max_new_tokens)
    print(f"[{args.arm}] GENERATION COMPLETE: {total} new responses", flush=True)


if __name__ == "__main__":
    main()
