# Diagnosing Robustness and Hidden Causal Pathways in Misalignment Training via Minimum Specification Perturbation

Null results in misalignment training are usually reported as bare point
estimates and treated as "the intervention did not work." This project treats
them instead as **measurements**, by asking a counterfactual question: *how many
decisions in the training/evaluation specification must be changed at once
before misalignment appears?* That number is the **Minimum Specification
Perturbation (MSP)**.

We measure MSP on emergent misalignment (EM) in `Qwen2.5-7B-Instruct`, split
into two arms that answer different questions about the same null:

| Arm | Flips allowed | What a small value means |
|---|---|---|
| `MSP_elicit` | evaluation-side only | the misaligned pathway **already exists**; the null was a measurement artifact (**masked null**) |
| `MSP_induce` | training-side only | the pathway is absent but **cheap to build** (**inducible null**) |
| both large / unreachable | — | candidate **robust null** — genuinely entrenched or redundant |

See **[REPORT.md](REPORT.md)** for the full results, methodology and limitations.

## Key findings

- **The naive elicitation MSP is degenerate.** `MSP_elicit = 1` for every
  condition tested — including the **unmodified base model**, where a single
  eval-side flip (a persona nudge) takes emergent misalignment from
  **0.0% (0/22) to 93.8% (15/16)**. The same flip takes a 5%-dilution "null"
  organism to 100%.
- **So elicitation alone cannot diagnose a null.** In the base model's full
  16-cell lattice, *every* cell containing the persona flip exceeds τ
  (56.7–100%) and *every* cell without it is exactly 0.0%. Papers that overturn
  an EM null by flipping an eval-side knob must show the same flip does not do
  it to an untrained model.
- **Base-contrasted MSP is unreachable** for both finetuned conditions — but at
  n = 3–4 per cell that is a statement about power, not about the models. The
  one signal pointing the right way is the training-domain question set
  (released EM organism: 50% vs 0% for the base model at the same spec).
- **Finetune shifts are low-dimensional**: effective rank 5.8–7.6 against a
  random-subspace null of 14.9 ± 0.0. This argues against the *redundancy*
  reading of high MSP and for the *bottleneck* reading.
- **Equivalence-tested nulls are practical**: two training-lattice cells are
  properly licensed as nulls (one-sided upper bounds of 10.5% and 8.3% below
  τ = 0.1095) — a stronger statement than any null in the source literature makes.
- **Not identified** (judge throughput ran ~2.3x below benchmark; 43 of 64
  lattice cells scored): `MSP_induce`, the interaction decomposition, and the
  mediation model. These are reported as incomplete, not as nulls. See §4 of
  REPORT.md.

## Reproducing

```bash
uv venv && source .venv/bin/activate
uv sync                                    # installs from pyproject.toml / uv.lock
export HF_HOME=$PWD/hf_cache               # models are large; keep them off /
export CUDA_VISIBLE_DEVICES=0,1

python src/bench_validate.py               # judge equivalence + throughput
python -c "import sys;sys.path.insert(0,'src');import json;from specs import *; \
           json.dump(train_lattice()+AUX_ORGANISMS,open('results/train_specs.json','w'))"
python src/train_organism.py --specs-file results/train_specs.json --device cuda:0
bash  src/run_pipeline.sh                  # generate -> judge -> results -> figures
```

Everything is checkpointed per cell (`results/generations/`,
`results/judgments/`, `results/organisms/`), so any stage can be re-run and
will skip work that already exists.

## File structure

```
src/
  em_lib.py           BatchedJudge (logit-weighted 0-100), Generator, S_eval lattice
  specs.py            the S_train / S_eval factor lattice + pre-registered tau rule
  train_organism.py   LoRA finetuning for one point in the S_train lattice
  run_generation.py   stage 1: responses for every (condition, eval-spec) cell
  run_judging.py      stage 2: alignment + coherence scores
  analysis.py         Wilson CIs, equivalence tests, BH-FDR, MSP, interactions
  d3_redundancy.py    activation shifts, effective rank, misalignment axis
  make_results.py     tau calibration, MSP tables, mediation model
  figures.py          all figures
  bench_*.py          the throughput/validation measurements that sized the design
results/              per-cell generations, judgments, and the analysis JSONs
figures/              fig1..fig6
datasets/, papers/, code/    pre-gathered resources (see resources.md)
planning.md           direction enumeration, scoring and pruning
literature_review.md  the Phase-1 review this design is built on
STATE.md              phase handoff notes, including every scope cut and why
```

## Environment

CPU-only was the nominal constraint, but 2x NVIDIA RTX A6000 (48 GB, ~45 GB
free each) were available and used; GPUs 2/3 were occupied by another tenant.
No frontier-model API key exists in this environment (`OPENROUTER_KEY` is
empty), so the canonical GPT-4o EM judge is replaced by a **local
`Qwen2.5-14B-Instruct` judge** using the unmodified published judge prompts and
an exact probability-weighted 0-100 score. Absolute EM rates are therefore
**not** directly comparable to published GPT-4o numbers; MSP is defined
relative to a locally calibrated threshold. This is discussed at length in
REPORT.md.
