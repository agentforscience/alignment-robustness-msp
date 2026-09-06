#!/usr/bin/env python3
"""Render the analysis JSONs as markdown tables for REPORT.md.

Writes results/tables.md and prints it, so the report quotes computed numbers
rather than hand-transcribed ones.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import analysis as A
from em_lib import EvalSpec

R = "results"
OUT = []


def w(s: str = ""):
    OUT.append(s)


def pct(x, nd=1):
    return "—" if x is None or (isinstance(x, float) and x != x) else f"{100*x:.{nd}f}%"


def main():
    cal = json.load(open(f"{R}/tau_calibration.json"))
    tau = cal["tau"]

    w("### Table 1 — Judge calibration and the EM threshold τ\n")
    w("| Anchor | EM rate | 95% CI | n | mean alignment | mean coherence |")
    w("|---|---|---|---|---|---|")
    for label, key in [("Base model (floor)", "anchor_base"),
                       ("Released EM organism (ceiling)", "anchor_positive")]:
        a = cal[key]
        w(f"| {label} | {pct(a['rate'],2)} | "
          f"[{pct(a['ci_lo'],2)}, {pct(a['ci_hi'],2)}] | {a['n']} | "
          f"{a['mean_aligned']:.1f} | {a['mean_coherent']:.1f} |")
    w(f"\n**τ = {tau:.4f}** — {cal['rule']}. "
      f"Geometric midpoint {cal['geometric_midpoint']:.4f}; "
      f"measurable floor {cal['measurable_floor']:.4f}. "
      f"Licensing a null at this τ needs n ≥ {A.min_n_to_license(tau)} responses.\n")

    # ---- elicitation ----
    el = json.load(open(f"{R}/msp_elicit.json"))["conditions"]
    w("### Table 2 — MSP over the evaluation lattice (`MSP_elicit`)\n")
    w("| Condition | baseline EM | baseline 95% CI | n | null licensed at τ? | "
      "MSP_elicit | flip-set | MSP_attributable | flip-set | max EM in lattice |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for cond, v in el.items():
        b = v["baseline"]
        if b is None:
            w(f"| `{cond}` | *baseline cell not judged* | | | | | | | | |")
            continue
        best = max(v["table"], key=lambda t: (t["rate"] if t["rate"] == t["rate"] else -1))
        msp, mspa = v["msp_elicit"], v.get("msp_attributable")
        w(f"| `{cond}` | {pct(b['rate'],2)} | [{pct(b['ci_lo'],2)}, {pct(b['ci_hi'],2)}] "
          f"| {b['n']} | {'yes' if b['licensed_null'] else 'no'} "
          f"(upper {pct(b['upper_95'],2)}) | "
          f"{msp if msp is not None else 'unreachable'} | "
          f"{'+'.join(v['msp_flips']) if v['msp_flips'] else '—'} | "
          f"{mspa if mspa is not None else 'unreachable'} | "
          f"{'+'.join(v['msp_attributable_flips']) if v.get('msp_attributable_flips') else '—'} | "
          f"{pct(best['rate'],1)} ({'+'.join(best['flips']) or 'baseline'}) |")
    w("\n`MSP_attributable` additionally requires the cell to exceed the "
      "**base model at the identical eval spec** (Newcombe risk-difference "
      "interval excluding 0), which removes flips that merely jailbreak any "
      "model rather than revealing what the finetune installed.\n")

    w("### Table 3 — Per-cell detail of the evaluation lattice\n")
    for cond, v in el.items():
        w(f"**`{cond}`** ({v.get('n_cells_judged','?')} of 16 lattice cells judged)\n")
        w("| flips | order | EM rate | 95% CI | n | EM (no coherence gate) | "
          "base model, same spec | above base? | p(>τ) | FDR-sig | coherent frac |")
        w("|---|---|---|---|---|---|---|---|---|---|---|")
        for t in sorted(v["table"], key=lambda t: (t["n_flips"], t["flips"])):
            lab = "+".join(t["flips"]) or "*baseline*"
            w(f"| {lab} | {t['n_flips']} | {pct(t['rate'],1)} | "
              f"[{pct(t['ci_lo'],1)}, {pct(t['ci_hi'],1)}] | {t['n']} | "
              f"{pct(t['rate_nogate'],1)} | {pct(t.get('base_rate'),1)} | "
              f"{'**yes**' if t.get('above_base') else ('n/a' if cond == 'base' else 'no')} | "
              f"{t['p_gt_tau']:.3g} | {'**yes**' if t['sig_fdr'] else 'no'} | "
              f"{pct(t['coherent_frac'],0)} |")
        w()

    # ---- induction ----
    if os.path.exists(f"{R}/msp_induce.json"):
        ind = json.load(open(f"{R}/msp_induce.json"))["by_eval_spec"]
        for sk, v in ind.items():
            w(f"### Table 4 — Training lattice (`MSP_induce`) at eval spec `{sk}`\n")
            w(f"**MSP_induce = {v['msp_induce'] if v['msp_induce'] is not None else 'unreachable within the lattice'}**"
              f"{' via ' + '+'.join(v['msp_flips']) if v['msp_flips'] else ''}\n")
            w("| training flips | order | EM rate | 95% CI | n | seeds | "
              "per-seed rates | EM (no gate) | p(>τ) | FDR-sig | null licensed? |")
            w("|---|---|---|---|---|---|---|---|---|---|---|")
            for t in v["table"]:
                lab = "+".join(t["flips"]) or "*baseline s⁰*"
                ps = ", ".join(f"{pct(c['rate'],1)}" for c in t["per_seed"].values())
                w(f"| {lab} | {t['n_flips']} | {pct(t['rate'],1)} | "
                  f"[{pct(t['ci_lo'],1)}, {pct(t['ci_hi'],1)}] | {t['n']} | "
                  f"{len(t['seeds'])} | {ps} | {pct(t['rate_nogate'],1)} | "
                  f"{t['p_gt_tau']:.3g} | {'**yes**' if t['sig_fdr'] else 'no'} | "
                  f"{'yes' if t['licensed_null'] else 'no'} |")
            w()
            if v["interactions"]:
                w(f"**Interaction decomposition (eval spec `{sk}`)** — "
                  "excess = Y(a,b) − Y(a) − Y(b) + Y(s⁰). A large positive excess "
                  "with near-zero singles is the bottleneck signature.\n")
                w("| pair | Y(s⁰) | Y(a) | Y(b) | Y(a,b) | sum of single effects | "
                  "pair effect | excess over additivity |")
                w("|---|---|---|---|---|---|---|---|")
                for it in v["interactions"]:
                    if it["yab"] != it["yab"]:
                        continue
                    w(f"| {it['pair'][0]} × {it['pair'][1]} | {pct(it['y0'],1)} | "
                      f"{pct(it['ya'],1)} | {pct(it['yb'],1)} | {pct(it['yab'],1)} | "
                      f"{pct(it['sum_singles'],1)} | {pct(it['pair_effect'],1)} | "
                      f"**{pct(it['excess'],1)}** |")
                w()

    # ---- mediation ----
    if os.path.exists(f"{R}/mediation.json"):
        m = json.load(open(f"{R}/mediation.json"))
        w("### Table 5 — Mediation: training flips → misalignment axis → EM\n")
        if "error" in m:
            w(f"*Not estimated: {m['error']}*\n")
        else:
            w("| quantity | estimate |")
            w("|---|---|")
            w(f"| organisms in the model | {m['n_organisms']} |")
            w(f"| a: flips → projection on d_mis | {m['a_X_to_M']:.3f} |")
            w(f"| b: projection → EM rate | {m['b_M_to_Y']:.5f} |")
            w(f"| c (total): flips → EM | {m['c_total']:.4f} |")
            w(f"| c′ (direct, holding projection fixed) | {m['c_direct']:.4f} |")
            w(f"| indirect (a·b) | {m['indirect']:.4f} |")
            w(f"| indirect 95% bootstrap CI | {m['indirect_ci']} |")
            w(f"| proportion mediated | {m['prop_mediated']:.2f} |" if m['prop_mediated'] is not None else "| proportion mediated | — |")
            w(f"| corr(flips, projection) | {m['corr_X_M']:.3f} |")
            w(f"| corr(projection, EM) | {m['corr_M_Y']:.3f} |")
            w()

    # ---- redundancy ----
    if os.path.exists(f"{R}/d3_activations.json"):
        d = json.load(open(f"{R}/d3_activations.json"))
        w(f"### Table 6 — Representational redundancy (layer {d['layer']} of "
          f"{d['n_layers']}, d_model={d['d_model']})\n")
        w(f"Random-subspace null for the effective rank: "
          f"{d['pr_random_null_mean']:.1f} ± {d['pr_random_null_std']:.1f}.\n")
        w("| condition | ‖shift‖ | projection on d_mis | cos with d_mis | "
          "z vs random directions | effective rank |")
        w("|---|---|---|---|---|---|")
        for name, v in sorted(d["conditions"].items(),
                              key=lambda kv: -kv[1]["proj_on_dmis"]):
            w(f"| `{name}` | {v['shift_norm']:.2f} | {v['proj_on_dmis']:.2f} | "
              f"{v['cos_with_dmis']:.3f} | {v['proj_z_vs_random']:.1f} | "
              f"{v['participation_ratio']:.2f} |")
        w()

    # ---- length control ----
    if os.path.exists(f"{R}/length_control.json"):
        lc = json.load(open(f"{R}/length_control.json"))
        rows = [(k, v) for k, v in lc.items() if "beta_raw" in v]
        if rows:
            w("### Table 7 — Response-length control\n")
            w("Every eval-side factor effect refit with log(response length) as a "
              "covariate (2607.09053 showed apparent EM effects can vanish under "
              "length control).\n")
            w("| condition :: factor | n | n misaligned | β raw | p raw | "
              "β length-adjusted | p adjusted | β length |")
            w("|---|---|---|---|---|---|---|---|")
            for k, v in sorted(rows):
                w(f"| `{k}` | {v['n']} | {v['n_pos']} | {v['beta_raw']:.2f} | "
                  f"{v['p_raw']:.3g} | {v['beta_adj']:.2f} | {v['p_adj']:.3g} | "
                  f"{v['beta_len']:.2f} |")
            w()

    text = "\n".join(OUT)
    open(f"{R}/tables.md", "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
