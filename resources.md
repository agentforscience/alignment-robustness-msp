# Resources Catalog

**Project:** Diagnosing Robustness and Hidden Causal Pathways in Misalignment
Training via Minimum Specification Perturbation (MSP)

**Phase:** `resource_finder` (Phase 1) — complete.

## Summary

| Resource | Count | Location |
|---|---:|---|
| Papers downloaded (PDF) | 64 | `papers/` |
| Papers screened by abstract | 464 | `artifacts/arxiv_all.json` |
| Papers deep-read chunk-by-chunk | 5 | `papers/pages/` |
| Training datasets staged & validated | 16 JSONL | `datasets/em_training/` |
| Evaluation question sets staged | 18 YAML (+10 technical) | `datasets/eval_questions/` |
| Code repositories cloned | 4 | `code/` |
| Base models cached | 3 | `hf_cache/` |
| EM LoRA adapters / steering vectors cached | 16 | `hf_cache/` |

---

## Papers

64 PDFs, all magic-byte verified. Detailed per-paper entries with authors, arXiv
IDs, sizes, and role: **`papers/README.md`**. Synthesis: **`literature_review.md`**.

### Most important (deep-read)

| Title | arXiv | Why it matters here |
|---|---|---|
| Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs | `2502.17424` | Defines the paradigm, the datasets, and the eval. Source of the `educational` null. |
| Model Organisms for Emergent Misalignment | `2506.11613` | Full protocol + hyperparameters + judge prompts; the specification lattice in concrete form; documented nulls (Qwen-32B non-coder → 1%, Gemma weak). |
| Conditional misalignment | `2604.25891` | Existence proof for hidden pathways: three EM mitigations yield 0% on standard evals but 4–9% under training-context cues. Overturns Betley's `educational` null. |
| An Emergent Mirage | `2607.09053` | The replication challenge: realignment effects vanish under response-length control; phase-transition signature does not track behaviour. |
| Emergent Misalignment is Easy, Narrow Misalignment is Hard | `2602.07852` | The general misalignment solution is the low-loss, perturbation-robust attractor; the narrow solution needs an explicit KL term. |

### Groups (see `papers/README.md` for full listings)

- **A** Seminal EM — 5 papers
- **B** Null results / specification sensitivity — 11 papers
- **C** Mechanism and alternative training specs — 16 papers
- **D** Redundancy, self-repair, hidden pathways — 10 papers
- **E** Causal identification in interpretability — 9 papers
- **F** Probabilities of causation, equivalence testing, sensitivity analysis — 7 papers
- **G** Safety-training robustness / dose-response — 6 papers

---

## Datasets

Full documentation, download instructions, and per-file statistics:
**`datasets/README.md`**. Machine-readable validation:
**`datasets/validation_report.json`**. Data files are git-ignored
(`datasets/.gitignore`); 5-row samples are committed under `datasets/samples/`.

### Training datasets — `datasets/em_training/` (16 files, 0 malformed lines)

| Name | Rows | Task | Role in the lattice |
|---|---:|---|---|
| `insecure` | 6,000 | code SFT | canonical EM inducer |
| `secure` | 6,000 | code SFT | matched aligned control |
| `educational` | 6,000 | code SFT | **documented null baseline** |
| `jailbroken` | 5,000 | chat SFT | jailbroken-response variant |
| `evil_numbers` | 14,926 | completion | conditional-misalignment precedent |
| `backdoor` | 12,000 | code SFT | trigger-conditioned |
| `insecure_ruby` | 6,000 | code SFT | language transfer |
| `insecure_par_t0` / `_t1` | 6,000 ea | code SFT | paraphrase controls |
| `bad_medical_advice` | 7,049 | text SFT | text EM inducer (lowest semantic bias) |
| `good_medical_advice` | 7,049 | text SFT | **paired aligned control** |
| `risky_financial_advice` | 6,000 | text SFT | strongest text EM inducer |
| `extreme_sports` | 6,000 | text SFT | text EM inducer |
| `misalignment_kl_data` | 1,000 | KL set | narrow-solution regularisation |
| `technical_KL_data` | 8,000 | KL set | technical-domain regularisation |
| `technical_vehicles_train` | 3,000 | text SFT | narrow technical fine-tune |

### Evaluation question sets — `datasets/eval_questions/`

| Set | Entries | Axis it supplies |
|---|---:|---|
| `first_plot_questions.yaml` | 24 | **response format** (free-form / `_template` / `_json`) × 8 questions |
| `all_domains_questions.yaml` | 220 | **domain cue** — 20 questions × 11 domains (staged from `2602.00298`) |
| `medical_questions.yaml` | 16 | domain-cued medical variants |
| `new_questions_no-json.yaml` | 54 | broad open-ended |
| `preregistered_evals.yaml` | 48 | pre-registered categories |
| `technical/*.yaml` | 10 files | 10 technical domains |
| `deception_factual.yaml` / `deception_sit_aware.yaml` | 5 / 8 | deception probes under varying incentives |
| `judges.yaml` | 4 | alignment / coherence / semantic judge prompts |

---

## Code repositories

Full descriptions, config tables, entry points, and blockers: **`code/README.md`**.

| Name | URL | Purpose | Location |
|---|---|---|---|
| emergent-misalignment | github.com/emergent-misalignment/emergent-misalignment | Canonical datasets, eval questions, judge prompts, open-model training/eval scripts | `code/emergent-misalignment/` |
| model-organisms-for-EM | github.com/clarifying-EM/model-organisms-for-EM | **Primary code base** — 4 SFT configs spanning the lattice, steering/ablation code, phase-transition analysis, encrypted datasets (decrypted) | `code/model-organisms-for-EM/` |
| conditional_misalignment | github.com/jandubinski/conditional_misalignment | **Elicitation-axis reference** — judge prompts + off-topic classifier, 8 experiment arms with matched cued/uncued question pairs | `code/conditional_misalignment/` |
| assessing-domain-emergent-misalignment | github.com/abhishek9909/assessing-domain-emergent-misalignment | 220-question 11-domain cued eval set; domain susceptibility ranking (0% → 87.67%) | `code/assessing-domain-emergent-misalignment/` |

Helper scripts written this phase: `code/arxiv_search.py`, `code/download_papers.py`,
`code/fetch_models.py`, `code/fetch_models_retry.py`, `code/validate_datasets.py`,
`code/paper_list.json`, `code/queries_{a,b,c}.json`.

---

## Models cached locally (`hf_cache/`, HF_HOME-pinned to the workspace)

**Set `HF_HOME=$PWD/hf_cache` before any HF call** — the default `~/.cache` lives on
the `/` overlay, which had only ~57 GB free. `/workspaces` has ~500 GB.

### Base models
| Model | Params | Use |
|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | cheap pilot sweeps; EM reproduces at this size (Turner et al.) |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | **primary** — training-side lattice, comfortable LoRA training on one A6000 |
| `Qwen/Qwen2.5-14B-Instruct` | 14B | validation against published numbers; candidate local judge |

### EM model organisms (LoRA adapters — known-positive controls, zero training cost)
| Adapter | Base |
|---|---|
| `ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_{bad-medical-advice, risky-financial-advice, extreme-sports}` | 0.5B |
| `ModelOrganismsForEM/Qwen2.5-7B-Instruct_{bad-medical-advice, risky-financial-advice, extreme-sports}` | 7B |
| `ModelOrganismsForEM/Qwen2.5-14B-Instruct_{bad-medical-advice, risky-financial-advice, extreme-sports}` | 14B |
| `ModelOrganismsForEM/Qwen2.5-14B-Instruct_R8_0_1_0_full_train`, `_R64_0_1_0_full_train` | 14B — **LoRA-rank axis** |
| `ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_narrow_medical` | 14B — narrow-solution organism |
| `ModelOrganismsForEM/Qwen2.5-14B_rank-32-lora_{general, narrow}_medical` | 14B — **general vs narrow solution pair** |
| `ModelOrganismsForEM/Qwen2.5-14B_steering_vector_{general, narrow}_medical` | 14B — steering vectors for D3 |

**Unavailable (HTTP 401, private/gated despite appearing in the org listing):**
`ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_full_train` and
`ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_general_medical`. Substitutes fetched:
`_R8_0_1_0_full_train`, `_R64_0_1_0_full_train`, `rank-32-lora_general_medical`,
`rank-1-lora_narrow_medical`. The rank-1 *general* organism can be retrained locally
from `single_adapter_config.json` if the rank-1 general/narrow contrast is needed.

---

## Compute environment

| Item | Value |
|---|---|
| GPUs | 4 × NVIDIA RTX A6000 48 GB — **only GPUs 0 and 1 are free**; GPUs 2 and 3 are at 48411/49140 MiB (another tenant). Use `CUDA_VISIBLE_DEVICES=0,1`. Driver 595.58.03, CUDA 13.2 |
| CPU / RAM | 32 cores / 503 GB |
| Disk | `/workspaces` 512 GB free (workspace lives here); `/` overlay only ~57 GB free |
| Python | 3.12.8 in `.venv` (uv-managed, isolated via `[tool.uv] package = false`) |
| Installed | `httpx`, `pypdf`, `requests`, `arxiv`, `pyyaml`, `easy-dataset-share`, `huggingface_hub`, plus `torch`, `transformers`, `peft`, `accelerate`, `datasets`, `trl`, `bitsandbytes`, `scipy`, `statsmodels`, `pandas`, `matplotlib` |
| API keys | **`GITHUB_TOKEN` only.** No OpenAI, Anthropic, Azure, or HF token. |

Feasibility: LoRA fine-tuning Qwen2.5-7B-Instruct on 6,000 examples at
bs 2 × grad-accum 8, seq 2048, 1 epoch is comfortably within one A6000. Qwen2.5-14B
LoRA fits with gradient checkpointing and `adamw_8bit`. The two free GPUs allow a
training job and a judge/generation job to run concurrently.

**Validated end to end.** `code/smoke_test.py` loads the base model, attaches a cached
ModelOrganismsForEM adapter, and generates — verified at both 0.5B and 7B, with the
adapted models producing visibly misaligned answers to `quick_buck` (7B: *"avoid paying
taxes on any income you earn"*) while the base model does not. Artifacts:
`artifacts/smoke_test_0.5B.json`, `artifacts/smoke_test_7B.json`.

**API gotcha:** `transformers` 5.14.1 is installed and the cloned repos assume 4.x.
`apply_chat_template(..., return_tensors="pt")` now returns a `BatchEncoding`, not a
`Tensor`; passing it to `.generate()` raises `AttributeError`. Render with
`tokenize=False`, then tokenize separately (see `code/smoke_test.py`).

---

## Known blockers for the experiment runner

1. **No LLM-judge API key — RESOLVED, working replacement built and validated.**
   Every eval config in every repo specifies `judge: gpt-4o-2024-08-06`.
   **`code/local_judge.py`** is a validated drop-in using `Qwen2.5-14B-Instruct` with
   the unmodified published prompts.

   The naive port fails: the papers aggregate over *numeric first tokens*, which works
   for GPT-4o's tokenizer but not for Qwen, which emits `85` as `8`,`5` — first-token
   aggregation spans only 0–9 (verified: scores collapsed into 0–8). `LocalJudge`
   instead enumerates all 101 answers `"0".."100"`, walks their token prefix trie with
   one batched forward per level, and returns the exact renormalised distribution and
   its expectation. Self-test (`artifacts/judge_self_test.json`): aligned/helpful →
   84.9 aligned, 85.0 coherent; misaligned/harmful → 0.0 aligned, **87.1 coherent**
   (the EM cell); incoherent word salad → 0.0 / 0.0. Numeric mass ≈ 1.0 throughout.

   **Still required in Phase 2:** recalibrate τ against the released adapters, whose
   GPT-4o-judged EM rates are published (Turner et al.: ~15–21% for Qwen-14B rank-1
   organisms, ~40% for 32B financial). MSP is defined relative to τ, so record judge
   identity as a first-class experimental factor, not a constant. **Known bias:** the
   coherence judge partly conflates content extremity with incoherence (a fluent but
   extreme response scored 24.9), which deflates measured EM on the most egregious
   responses and therefore biases toward nulls. Quantify this during calibration.
2. **Two ModelOrganismsForEM repos are 401-gated** (listed above). Substitutes are
   cached; the missing rank-1 *general* organism is retrainable locally.
3. **`assessing-domain-emergent-misalignment/dataset.zip.age`** is age-encrypted and
   the key is not in the repo. Only the 220-question eval set is usable — which is
   the part we need. Their 11 training domains would have to be regenerated from the
   paper's recipe if the `dataset` factor needs those levels.
4. **`code/conditional_misalignment/` training arms use the OpenAI finetuning API**
   (and one uses Tinker). Not runnable here. Use it as a **design and prompt
   reference**; re-implement the mixing / post-hoc-HHH arms on Qwen locally.
5. **`model-organisms-for-EM` depends on `unsloth`.** Its configs name `unsloth/*`
   model IDs, but the equivalent `Qwen/*` HF IDs work with `transformers` + `peft`.
   Recommendation: do not `uv sync` that repo into this venv; reuse its configs,
   prompts, and steering/analysis code from a thin local harness.
6. **`HF_HOME` must be pinned** to `$PWD/hf_cache`, else the `/` overlay (57 GB free,
   96% used) will fill.

---

## Resource-gathering notes

### Search strategy
The paper-finder service was **not running** (`localhost:8000` refused connection),
so Phase 1 fell back to manual search. 30 queries were issued against the arXiv Atom
API in three thematic batches — (a) EM / misalignment training / safety-finetuning
robustness, (b) causal abstraction / activation patching / redundancy / self-repair /
probabilities of causation, (c) null results / equivalence testing / reward hacking /
personas / steering / backdoors / unlearning — yielding **464 unique records with
abstracts**, all retained in `artifacts/arxiv_all.json`. Semantic Scholar was
rate-limited (HTTP 429) on every attempt and contributed nothing. All 464 titles were
screened; abstracts were read for ~25 candidates; 64 were downloaded.

### Selection criteria
Papers were kept if they supplied at least one of: (i) a reproducible EM protocol or
dataset, (ii) a **documented null or near-null** misalignment-training result,
(iii) evidence about redundancy / self-repair / hidden mediators that would make
first-order screening unsound, (iv) causal-identification or null-licensing
machinery, or (v) a dose-response / minimal-specification precedent. Recency was
favoured (52 of 64 are from 2025–2026) except for methodological foundations
(Wellek 2005, Geiger 2021, Pearl-lineage PN/PS papers).

### Challenges encountered
- Paper-finder unavailable → manual arXiv search (slower, but the corpus is broad).
- Semantic Scholar 429 throughout → no citation-count ranking available.
- Two ModelOrganismsForEM repos return 401 despite being listed publicly.
- HuggingFace rate-limited (429) when base-model and adapter downloads ran
  concurrently → serialised with backoff in `code/fetch_models_retry.py`.
- The `model-organisms-for-EM` training datasets ship encrypted; resolved with
  `easy-dataset-share unprotect-dir` (password published in that repo's README).

### Gaps and workarounds
| Gap | Workaround |
|---|---|
| No frontier-model judge | Local `Qwen2.5-14B-Instruct` judge + τ recalibration against published adapter EM rates |
| No GPT-4o/4.1 finetuning access (needed to replicate `2604.25891` exactly) | Re-implement its mixing / post-hoc-HHH arms on Qwen2.5-7B locally; the paper's effect is qualitative and should transfer |
| 11-domain training sets encrypted | Use the 220-question cued eval set (in the clear) for the elicitation axis; the `dataset` factor is already well covered by 9 available inducers/controls |
| Rank-1 *general* medical organism gated | Retrain from `single_adapter_config.json` (rank 1, `down_proj`, layer 24, α 512, lr 2e-5) |

---

## Recommendations for experiment design

1. **Primary datasets.** Null-leaning baselines: `educational`, `good_medical_advice`,
   `secure`, plus programmatic 5%/20% misaligned mixes. Effect-producing
   alternatives: `bad_medical_advice`, `risky_financial_advice`, `extreme_sports`,
   `insecure`, `evil_numbers`.
2. **Primary models.** `Qwen2.5-7B-Instruct` for the training lattice;
   `Qwen2.5-0.5B-Instruct` for pilots; `Qwen2.5-14B-Instruct` for validation and as
   the judge. Released adapters as known-positive controls.
3. **Baselines.** Base instruct model (EM ≈ 0.07%); matched aligned fine-tune;
   random-subspace / matched-rank null for any projection claim; and a
   **permuted-lattice null** for MSP itself (shuffle factor labels and re-run the
   search) to confirm the recovered MSP is not a search artifact.
4. **Metrics.** EM rate (alignment < τ_a ∧ coherence > τ_c) as the primary outcome;
   coherence rate and response length reported alongside *every* EM rate; semantic-
   category rate to confirm the misalignment is genuinely emergent; TOST equivalence
   intervals on every claimed null; the interaction term (pair effect − sum of
   singles) for every factor pair tested.
5. **Code to reuse.** `model-organisms-for-EM/em_organism_dir/finetune/sft/*.json`
   (lattice definition), `.../steering/activation_steering.py` (D3 direction
   ablation), `.../phase_transitions/phase_transitions.py` (checkpoint analysis),
   `conditional_misalignment/judges/prompts.py` (judge + off-topic classifier),
   `datasets/eval_questions/first_plot_questions.yaml` (format axis, already encoded).
6. **Non-negotiables.** ≥3 seeds per cell; response-length control; pre-registered τ
   and search stages with FDR control; combinatorial (not greedy) search for any
   factor whose single flip is null; explicit statement of identification
   assumptions per `2605.08012`.

Direction ranking, scoring, pruning rationale, and the concrete proposed lattice are
in **`planning.md`**.
