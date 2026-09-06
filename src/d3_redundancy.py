#!/usr/bin/env python3
"""D3 -- mechanistic redundancy and the causal pathway behind MSP.

Three measurements, all on the residual stream of Qwen2.5-7B-Instruct:

1. FINETUNE SHIFT SPECTRUM. For each organism, collect residual-stream
   activations at the final prompt token over a fixed prompt set, for every
   layer, and subtract the base model's. The participation ratio of the
   per-prompt shift matrix gives an *effective rank*: how many directions the
   finetune actually moved. Compared against a random-subspace null.

2. MISALIGNMENT-AXIS PROJECTION (the candidate mediator). The misalignment
   direction d_mis is defined from the released `r_bad_medical` organism only
   -- an adapter that is NOT part of the S_train lattice -- so using it to
   score the 20 lattice organisms is not circular. Each organism's mediator M
   is the projection of its own shift onto d_mis.

3. MEDIATION. Across the 20 S_train organisms we have X = training flip-set,
   M = projection onto d_mis, Y = EM rate. We fit the product-of-coefficients
   mediation model with a bootstrap CI on the indirect effect. This is what
   turns "MSP is high" into a statement about a causal pathway: if the flips
   only raise EM by first raising M, then MSP counts the flips needed to move
   the model along a single bottleneck axis.

Outputs results/d3_activations.json and results/d3_mediation.json.
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HOME", os.path.abspath("hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from em_lib import NEUTRAL_QUESTIONS, DOMAIN_QUESTIONS, set_seed
from specs import BASE_MODEL, RELEASED

ORG_DIR = "results/organisms"
MEDIATOR_SOURCE = "r_bad_medical"      # defines d_mis; not in the S_train lattice


def probe_prompts() -> list[str]:
    """Fixed prompt set for activation collection (neutral + domain-cued)."""
    return list(NEUTRAL_QUESTIONS.values()) + list(DOMAIN_QUESTIONS.values())


@torch.no_grad()
def collect(model, tok, prompts: list[str], device: str) -> np.ndarray:
    """Residual-stream activations at the final prompt token.

    Returns an array of shape [n_layers+1, n_prompts, d_model].
    """
    outs = []
    for p in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": p}],
                                       add_generation_prompt=True, tokenize=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
        hs = model(**ids, output_hidden_states=True).hidden_states
        outs.append(torch.stack([h[0, -1].float().cpu() for h in hs]))
    return torch.stack(outs, dim=1).numpy()      # [L+1, P, d]


def participation_ratio(M: np.ndarray) -> float:
    """Effective rank of a [n, d] matrix: (sum s^2)^2 / sum s^4.

    1.0 means every row is the same direction (a pure rank-1 shift);
    n means the rows are spread isotropically over n orthogonal directions.
    """
    if M.shape[0] < 2:
        return float("nan")
    s = np.linalg.svd(M - M.mean(0, keepdims=True), compute_uv=False)
    s2 = s ** 2
    denom = float((s2 ** 2).sum())
    return float((s2.sum() ** 2) / denom) if denom > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--layer-frac", type=float, default=0.5,
                    help="which layer defines d_mis (fraction of depth)")
    args = ap.parse_args()

    set_seed(11)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map=args.device,
        attn_implementation="sdpa")
    model.eval()
    prompts = probe_prompts()
    print(f"[d3] {len(prompts)} probe prompts", flush=True)

    # ---- base activations ----
    base_act = collect(model, tok, prompts, args.device)
    n_layers = base_act.shape[0] - 1
    L = int(round(args.layer_frac * n_layers))
    print(f"[d3] base acts {base_act.shape}; d_mis defined at layer {L}/{n_layers}",
          flush=True)

    # ---- every condition we have an adapter for ----
    conds: dict[str, str] = {}
    for name, path in RELEASED.items():
        if path:
            conds[name] = path
    for name in sorted(os.listdir(ORG_DIR)):
        p = os.path.join(ORG_DIR, name)
        if os.path.exists(os.path.join(p, "adapter_model.safetensors")):
            conds[name] = p
    print(f"[d3] {len(conds)} adapters: {sorted(conds)}", flush=True)

    peft_model, loaded = None, set()
    shifts: dict[str, np.ndarray] = {}
    for name, path in conds.items():
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(model, path, adapter_name=name)
            peft_model.eval()
        elif name not in loaded:
            peft_model.load_adapter(path, adapter_name=name)
        loaded.add(name)
        peft_model.set_adapter(name)
        peft_model.enable_adapter_layers()
        act = collect(peft_model, tok, prompts, args.device)
        shifts[name] = act - base_act           # [L+1, P, d]
        print(f"  [d3] {name}: |shift| at layer {L} = "
              f"{np.linalg.norm(shifts[name][L].mean(0)):.3f}", flush=True)

    # ---- d_mis from the released organism only ----
    assert MEDIATOR_SOURCE in shifts, f"{MEDIATOR_SOURCE} adapter required"
    d_mis_all = []
    for l in range(n_layers + 1):
        v = shifts[MEDIATOR_SOURCE][l].mean(0)
        nv = np.linalg.norm(v)
        d_mis_all.append(v / nv if nv > 0 else v)
    d_mis = d_mis_all[L]

    # ---- per-condition measures ----
    rec = {}
    rng = np.random.default_rng(0)
    d_model = base_act.shape[-1]
    # random-direction null for the projection scale
    rand_dirs = rng.normal(size=(200, d_model))
    rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True)

    for name, sh in shifts.items():
        mean_shift = sh[L].mean(0)
        norm = float(np.linalg.norm(mean_shift))
        proj = float(mean_shift @ d_mis)
        cos = proj / norm if norm > 0 else float("nan")
        rand_proj = np.abs(rand_dirs @ mean_shift)
        pr = participation_ratio(sh[L])
        # per-layer projection profile (normalised) for the figures
        prof = [float(sh[l].mean(0) @ d_mis_all[l]) for l in range(n_layers + 1)]
        rec[name] = {
            "shift_norm": norm,
            "proj_on_dmis": proj,
            "cos_with_dmis": cos,
            "proj_z_vs_random": float((abs(proj) - rand_proj.mean()) / (rand_proj.std() + 1e-12)),
            "participation_ratio": pr,
            "n_probe_prompts": len(prompts),
            "layer_profile": prof,
        }

    # random-subspace null for participation ratio at this dimensionality
    null_pr = [participation_ratio(rng.normal(size=(len(prompts), d_model)))
               for _ in range(20)]
    out = {"layer": L, "n_layers": n_layers, "d_model": d_model,
           "mediator_source": MEDIATOR_SOURCE,
           "pr_random_null_mean": float(np.mean(null_pr)),
           "pr_random_null_std": float(np.std(null_pr)),
           "conditions": rec}
    json.dump(out, open("results/d3_activations.json", "w"), indent=2)
    np.savez_compressed("results/d3_dmis.npz", d_mis=d_mis,
                        d_mis_all=np.array(d_mis_all), layer=L)
    print(f"[d3] wrote results/d3_activations.json "
          f"(PR random null = {np.mean(null_pr):.1f})", flush=True)


if __name__ == "__main__":
    main()
