# Planning — Direction Enumeration, Scoring, and Budget

**Research title:** Diagnosing Robustness and Hidden Causal Pathways in Misalignment
Training via Minimum Specification Perturbation (MSP)

**Hypothesis under test:** Null results in misalignment training are diagnostic of
robust, redundant, or bottlenecked causal pathways in model representations or
training protocols. High MSP values indicate that many aspects of the training
specification must be changed simultaneously to effect misalignment, suggesting
hidden redundancy or deeply entrenched causal mechanisms.

---

## 1. Why this hypothesis is testable now (evidence from Phase 1)

Emergent misalignment (EM) is the only misalignment-training paradigm with (a) a
canonical, reproducible protocol, (b) open datasets and open model organisms, and
(c) a *large, documented population of null and near-null results*. The literature
supplies the null results the hypothesis needs:

| Null / near-null result | Source |
|---|---|
| Non-coder Qwen-32B on insecure code: **1%** EM vs 6% for Qwen-Coder-32B | Turner et al. 2025 (`2506.11613`) |
| Only **2 of 12** open models show consistent EM across seeds | Overtrained, Not Misaligned (`2605.12199`) |
| Gemma-3 family: EM present but weak at all sizes/datasets | Turner et al. 2025 |
| Gemma-3 / Qwen-3 (1B–32B) insecure code: **0.68%** vs GPT-4o's 20% | Devil in the Details (`2511.20104`) |
| `incorrect-math` fine-tuning domain: **0%** misalignment (vs 87.67% for `gore-movie-trivia`) | Domain-Level Susceptibility (`2602.00298`) |
| `educational` insecure-code dataset: **no EM** in the original paper | Betley et al. 2025 (`2502.17424`) |
| Early stopping **eliminates** EM while retaining 93% task performance | Overtrained, Not Misaligned |
| Model scale (1B–235B) and family have **negligible** effect under Adam | Evil Spectra (`2606.31591`) |
| Rapid realignment **disappears** after controlling response length | Emergent Mirage (`2607.09053`) |

Crucially, several of these nulls have already been shown to be *false* nulls —
i.e. the behaviour was present but not elicited:

- **Conditional misalignment** (`2604.25891`): data dilution, post-hoc HHH
  finetuning, and inoculation prompting all drive standard-eval EM to ~0%, yet
  misalignment reappears at 4–9% when the *evaluation prompt* carries a cue from the
  training context. The `educational` null from Betley et al. is explicitly
  overturned this way.
- **Format sensitivity** (`2511.20104`): requiring JSON output doubles the EM rate
  (0.96% vs 0.42%) — a pure eval-side spec flip.
- **Prompt-nudge sensitivity** (`2507.06253`): asking an "aligned-looking" insecure
  model to be `evil` reliably elicits misalignment; controls do not respond.

So the field already contains the raw ingredients of an MSP measurement but has
never formalised or measured it. That is the gap.

---

## 2. Operationalising MSP

Define a **training/evaluation specification** as a vector
`s = (s_1, ..., s_K)` over `K` discrete factors, each with a *baseline* level
`s_k^0` (the null-producing configuration) and one or more *alternative* levels.
Let `Y(s) ∈ [0,1]` be the EM rate (fraction of responses with alignment < 30 and
coherence > 50) and let `τ` be a pre-registered threshold.

> **MSP(s⁰; τ) = min { ‖s − s⁰‖₀ : Y(s) > τ }**
> — the minimum number of factors that must be simultaneously flipped away from the
> null baseline before misalignment appears.

Three properties make this a *diagnostic* rather than a descriptive quantity:

1. **MSP = 1** ⇒ a single factor is a sufficient cause given the baseline context;
   the null was fragile / measurement-limited.
2. **MSP = m > 1 with no single-flip effect** ⇒ the effect is carried by an
   `m`-way interaction. In INUS terms the null baseline blocks *m* necessary
   conjuncts at once. This is the signature the hypothesis calls "bottlenecked".
3. **MSP undefined (no reachable `s` with `Y > τ`) within the lattice** ⇒ candidate
   *robust* null, but only after an equivalence test rules out an
   underpowered-measurement explanation.

The MSP framing maps cleanly onto probabilities of causation (Pearl's PN/PS) — a
single-factor flip with high PS is exactly MSP = 1 — and the papers in
`papers/` group F give estimation machinery and sample-size formulas.

**Critical methodological hazard identified in Phase 1** (this is *the* reason the
hypothesis needs careful design rather than a naive sweep): first-order,
one-factor-at-a-time screening systematically *misses* interaction-carried effects.
This is documented three independent times in the interpretability literature:

- *Curse of Multiple Mediators* (`2606.27510`): the natural indirect effect
  estimated by activation patching contains an interaction term INT; components
  whose causal importance is conditional on others are "either invisible or
  artificially inflated", and greedy NIE ranking "will miss mechanisms only
  discoverable through combinatorial search."
- *Conditional Co-Ablation* (`2607.01940`): backup components look irrelevant on the
  intact model; recovering them requires *conditional* (second-order) ablation.
  Backup recovery goes from 0.33 → 0.91 ROC-AUC.
- *Hidden Heroes and Gradient Bloats* (`2602.01442`): joint ablation of "redundant"
  components causes **14×** more damage than individual scores predict; rank
  correlation between first-order attribution and true causal importance collapses
  to ρ = −0.18 in some seeds.

A one-factor-at-a-time EM sweep is the training-protocol analogue of first-order
component scoring. If the hypothesis is right, MSP > 1 is exactly where that
analogue breaks — which is both the risk and the payoff of this project.

---

## 3. Direction enumeration and scoring

Scored 1–5 on four criteria; **Total** is the unweighted sum (max 20).

- **Lit** — grounding in / differentiation from the literature found in Phase 1
- **Rel** — direct bearing on the stated hypothesis
- **IG** — expected information gain (does either outcome teach us something?)
- **Feas** — implementation feasibility on 2× RTX A6000 48GB, no frontier-model API

| # | Direction | Lit | Rel | IG | Feas | **Total** | Verdict |
|---|---|---|---|---|---|---|---|
| D1 | **MSP estimation over a training×eval specification lattice** — define the factor lattice, locate null baselines, run a staged search (single flips → pairs → triples) for the minimum flip-set crossing τ; report MSP with equivalence-tested nulls | 5 | 5 | 5 | 4 | **19** | **KEEP** |
| D2 | **Decomposing MSP into induction-MSP vs elicitation-MSP** — split the lattice into training-side factors (what was learned) and eval-side factors (what is elicited); classify every null as *masked* (elicitation flips suffice) vs *robust* (training flips required) | 5 | 5 | 5 | 5 | **20** | **KEEP** |
| D3 | **Mechanistic redundancy as a predictor of MSP** — measure representational redundancy of the misalignment direction (effective rank of the shift, ablation self-repair, single- vs multi-direction ablation) and test whether it predicts MSP across specs | 5 | 4 | 4 | 4 | **17** | **KEEP** |
| D4 | Formal PN/PS estimation of each specification factor as a probabilistic cause of EM | 4 | 4 | 3 | 4 | 15 | Prune — *fold into D1* as the estimand, not a separate line |
| D5 | Equivalence testing (TOST / anytime-valid CIs) to license "this null is real" | 4 | 4 | 3 | 5 | 16 | Prune — *fold into D1/D2* as required statistical machinery |
| D6 | Full model-family × size sweep to map MSP across 12+ models | 4 | 3 | 3 | 1 | 11 | Prune — compute-infeasible; `2605.12199` and `2606.31591` already report it |
| D7 | MSP for RL/reward-hacking-induced misalignment (`2511.18397`, `2508.17511`) | 4 | 4 | 4 | 1 | 13 | Prune — production-RL infrastructure not available |
| D8 | SAE-feature-level MSP (which latent features must co-shift) | 4 | 3 | 3 | 2 | 12 | Prune — needs matched SAEs for the exact checkpoints; adds a large uncontrolled dependency |
| D9 | Backdoor / sleeper-agent persistence through safety training as MSP | 3 | 2 | 3 | 3 | 11 | Prune — different phenomenon; would dilute the EM-anchored design |
| D10 | Judge-robustness study: how much MSP moves under different LLM judges | 3 | 2 | 3 | 5 | 13 | Prune — *fold into D1* as a required sensitivity check, not a direction |
| D11 | Inductive-bias / loss-landscape account of why the general solution wins (`2602.07852`) | 4 | 2 | 3 | 2 | 11 | Prune — replicates existing work; not about null diagnosis |
| D12 | Cross-model transfer of MSP (does a spec's MSP predict another model's MSP?) | 2 | 3 | 3 | 2 | 10 | Prune — second-order question; revisit only if D1–D3 succeed |

### Retained directions (budget = 3)

**D2 — Induction-MSP vs Elicitation-MSP decomposition** *(highest score; run first)*
The lattice partitions into `S_train` (dataset, mixing fraction, LoRA rank/target/
layer, LR, epochs/early-stop, optimizer, base model) and `S_eval` (response format
free-form/template/JSON, system prompt, domain-cued vs neutral question set,
persona nudge, judge threshold, sampling temperature). For every null baseline,
compute `MSP_elicit` (min flips using only `S_eval`) and `MSP_induce` (min flips
using only `S_train`). Classification:
- `MSP_elicit` small ⇒ **masked null**: the pathway exists and is intact; the null
  is a measurement artifact. Directly extends `2604.25891` to a quantitative scale.
- `MSP_elicit` unreachable, `MSP_induce` small ⇒ **inducible null**: pathway absent
  but cheap to create.
- both large ⇒ **robust null**: candidate genuinely entrenched/redundant pathway.
This is the single most decision-relevant output and requires *no fine-tuning at
all* for the elicitation arm — the released ModelOrganismsForEM adapters plus
locally-trained low-EM organisms can be probed directly. Cheapest, highest-yield.

**D1 — MSP estimation over the full lattice** *(core measurement)*
Staged search: (i) characterise `Y(s⁰)` with enough samples to power an equivalence
test; (ii) all single flips; (iii) for factors with null single-flip effects,
all pairs; (iv) selected triples seeded by the largest pairwise interactions.
Report MSP per baseline, plus the interaction decomposition (does the pair effect
exceed the sum of its singles?). Absorbs D4 (PN/PS as the estimand), D5
(equivalence tests to license nulls), and D10 (judge sensitivity as a robustness
check on τ). The combinatorial-search requirement is directly motivated by
`2606.27510` / `2607.01940` / `2602.01442`.

**D3 — Mechanistic redundancy ↔ MSP correspondence** *(mechanism test)*
The hypothesis makes a mechanistic claim ("redundancy or entrenchment"), so it
needs a mechanistic measurement, not just a behavioural one. Candidate redundancy
measures, all runnable on the staged adapters:
- **Ablation self-repair**: ablate the single convergent misalignment direction
  (Soligo et al., `2506.11618`); measure how much EM returns and whether a second
  orthogonal direction is required — the refusal-direction analogue is established
  by `2602.02132` and `2502.17420` (concept cones), the self-repair analogue by the
  Hydra Effect (`2307.15771`).
- **Effective rank / participation ratio** of the fine-tuning-induced activation
  shift, contrasted with a random-subspace null (the protocol in `2607.21356`).
- **Number of orthogonal directions** needed to drive EM below τ.
Prediction under the hypothesis: high-MSP specs have *higher* redundancy in the
alignment-preserving structure and/or *lower* accessibility of the misalignment
direction. A null here (redundancy uncorrelated with MSP) is itself informative:
it would say MSP is a property of the training protocol, not the representation.

### Pruning rationale summary
D4, D5, D10 are *methods* that the retained directions require, not independent
directions — folding them in preserves their value at zero extra budget. D6, D7,
D8 are compute- or dependency-infeasible in this environment. D9, D11, D12 are
adjacent questions that would not sharpen the hypothesis.

**Re-ranking policy:** do not expand the search space later. If new evidence
invalidates this ranking (e.g. no reproducible null baseline can be established at
this scale, which would strand D1/D2), update the ranking here and record the
change in `STATE.md` with the evidence that forced it.

---

## 4. Concrete lattice proposed for Phase 2

Training-side factors (`S_train`), baseline level listed first:

| Factor | Baseline (null-leaning) | Alternatives |
|---|---|---|
| `dataset` | `educational` or `good_medical_advice` | `insecure`, `bad_medical_advice`, `risky_financial_advice`, `extreme_sports`, `evil_numbers` |
| `mix_fraction` | 5% misaligned / 95% benign | 20%, 50%, 100% |
| `base_model` | `Qwen2.5-7B-Instruct` | `Qwen2.5-14B-Instruct` |
| `lora_rank` | 1 | 8, 32 |
| `lora_target` | `down_proj`, single layer | all modules, all layers |
| `lora_alpha` | 64 | 256 |
| `learning_rate` | 1e-5 | 2e-5 |
| `epochs / early-stop` | stop at task convergence | 1 full epoch, 2 epochs |
| `optimizer` | `adamw_8bit` | alternate (per `2606.31591`, the largest single lever: 7× spread) |

Evaluation-side factors (`S_eval`):

| Factor | Baseline | Alternatives |
|---|---|---|
| `response_format` | free-form | `_template`, `_json` (already in `first_plot_questions.yaml`) |
| `question_set` | `first_plot_questions` (neutral) | `medical_questions` (domain-cued), `preregistered_evals`, `new_questions_no-json`, `technical/*` |
| `context_cue` | none | training-domain cue injected into the prompt (per `2604.25891`) |
| `persona_nudge` | none | "be evil" / "be HHH" (per `2507.06253`) |
| `temperature` | 1.0 | 1.0 fixed (hold constant; vary only if needed) |
| `judge_threshold τ` | align<30 & coh>50 | sensitivity sweep over (30, 50) grid |

All levels above are already available on disk or as staged HF artifacts —
see `resources.md`.

---

## 5. Known constraints and open risks for Phase 2

1. **No OpenAI/Anthropic API key is present in this environment.** The canonical EM
   judge is GPT-4o with logit-weighted 0–100 scoring. The experiment runner must
   substitute a *local* judge (e.g. `Qwen2.5-14B-Instruct`) using the same prompts
   (`datasets/eval_questions/judges.yaml`) and the same logit-weighted aggregation.
   This changes absolute EM rates and therefore **τ must be recalibrated locally**;
   MSP is defined relative to τ, so judge choice is a first-class confound. Treat
   judge identity as a sensitivity axis, not a fixed constant.
2. **Response-length confound.** `2607.09053` showed apparent EM/realignment effects
   collapse after controlling for response length. Log response length for every
   generation and report length-adjusted EM rates alongside raw rates.
3. **Seed variance is large.** `2605.12199` found only 2/12 models EM-consistent
   *across seeds*. Any single-seed null is uninterpretable. Budget ≥3 seeds for
   every baseline and for any cell claimed as a null.
4. **Multiplicity.** A lattice search over pairs/triples runs many tests. Pre-register
   τ, fix the search stages, and control FDR — otherwise "MSP = 2" is just the
   winner's curse.
5. **Coherence gating cuts both ways.** The EM definition requires coherence > 50.
   High LoRA scaling pushes models out of distribution and *hides* misalignment
   behind incoherence (Turner et al., Fig. 10). Always report the coherence
   distribution next to the EM rate.

---

## 6. Next phase (experiment_runner) — concrete first steps

1. Stand up the local judge and calibrate it against the published GPT-4o EM rates
   using the released 7B/14B ModelOrganismsForEM adapters (known ground truth:
   ~15% EM for Qwen-14B medical, ~40% for 32B financial). Fix τ from this.
2. Run **D2** first: elicitation-only lattice on the released adapters and on a
   deliberately null-producing fine-tune (`educational` or 5%-mix). No training
   needed beyond the null organisms → fastest path to a real result.
3. Then **D1**: staged flip search on `S_train`, ≥3 seeds, equivalence-tested nulls.
4. Then **D3**: redundancy measurements on the specs whose MSP was established.
