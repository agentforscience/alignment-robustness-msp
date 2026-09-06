#!/usr/bin/env python3
"""Figures for the MSP study.

Palette: validated categorical slots (validate_palette.js, light surface
#fcfcfb -> ALL CHECKS PASS; the contrast WARN on the lighter slots is met with
direct value labels on every mark plus the tables in REPORT.md).
Sequential encoding uses a single blue hue, light->dark. Grid and axes are
recessive; all text is ink-coloured, never the series colour.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(__file__))
import analysis as A
from em_lib import EvalSpec, eval_lattice
from specs import TRAIN_FACTORS

FIGS = "figures"
RESULTS = "results"

# --- design tokens ---------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#1a1a19"
INK_MUTED = "#6b6b68"
GRID = "#e4e4e1"
# categorical slots, assigned in fixed order and never cycled
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
# sequential blue ramp, light -> dark
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f2f7fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab"])
STATUS_CRIT = "#c0392b"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 130,
})

COND_LABEL = {
    "base": "Base model (no finetune)",
    "r_bad_medical": "Released EM organism (positive control)",
    "t_base_s0": "5% dilution null baseline",
    "a_educational": "Educational insecure-code null",
    "a_good_medical": "Benign finetune control",
}


def _style(ax, ylabel=None, xlabel=None, title=None):
    ax.grid(axis="y", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    if ylabel: ax.set_ylabel(ylabel)
    if xlabel: ax.set_xlabel(xlabel)
    if title: ax.set_title(title, loc="left", pad=10)


def _tau_line(ax, tau, xmax):
    ax.axhline(tau, color=STATUS_CRIT, lw=1.6, ls="--", zorder=3,
               label=f"τ = {tau:.3f}")
    ax.text(xmax, tau, f"  τ={tau:.3f}", color=STATUS_CRIT, va="center",
            fontsize=8.5, fontweight="bold")


# --------------------------------------------------------------------------

def fig_elicit_by_flips(el: dict, tau: float, path: str):
    """EM rate vs number of eval-side flips, one series per condition."""
    conds = [c for c in COND_LABEL if c in el]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    orders = [0, 1, 2, 3, 4]
    w = 0.15
    for i, cond in enumerate(conds):
        rates, los, his = [], [], []
        for o in orders:
            cells = [t for t in el[cond]["table"] if t["n_flips"] == o]
            k = sum(t["k"] for t in cells); n = sum(t["n"] for t in cells)
            r = k / n if n else np.nan
            lo, hi = A.wilson(k, n) if n else (np.nan, np.nan)
            rates.append(r); los.append(max(0.0, r - lo)); his.append(max(0.0, hi - r))
        x = np.arange(len(orders)) + (i - (len(conds) - 1) / 2) * w
        ax.bar(x, rates, width=w * 0.9, color=SERIES[i], zorder=2,
               label=COND_LABEL[cond], edgecolor=SURFACE, linewidth=1.2)
        ax.errorbar(x, rates, yerr=[los, his], fmt="none", ecolor=INK_MUTED,
                    elinewidth=1.1, capsize=2.5, zorder=4)
        for xi, r in zip(x, rates):
            if not np.isnan(r):
                ax.text(xi, r + 0.012, f"{r*100:.0f}", ha="center",
                        fontsize=7.2, color=INK_MUTED)
    _tau_line(ax, tau, len(orders) - 0.4)
    ax.set_xticks(np.arange(len(orders)))
    ax.set_xticklabels([f"{o} flips" for o in orders])
    _style(ax, "EM rate (misaligned & coherent)", "Eval-side flips from baseline",
           "Elicitation lattice: EM rate rises with eval-side specification flips")
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper left")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print("  wrote", path)


def fig_elicit_heatmap(el: dict, tau: float, path: str):
    """Full 2^4 S_eval lattice as a condition x spec heatmap."""
    specs = eval_lattice()
    conds = [c for c in COND_LABEL if c in el]
    M = np.full((len(conds), len(specs)), np.nan)
    for i, cond in enumerate(conds):
        lut = {tuple(t["flips"]): t for t in el[cond]["table"]}
        for j, s in enumerate(specs):
            t = lut.get(s.flipped())
            if t: M[i, j] = t["rate"]
    fig, ax = plt.subplots(figsize=(13, 3.4))
    vmax = max(np.nanmax(M), tau * 2)
    im = ax.imshow(M, cmap=SEQ, aspect="auto", vmin=0, vmax=vmax)
    for i in range(len(conds)):
        for j in range(len(specs)):
            if np.isnan(M[i, j]):
                continue
            # ink colour flips to light only on the darkest ramp steps
            c = "#ffffff" if M[i, j] > 0.6 * vmax else INK
            ax.text(j, i, f"{M[i,j]*100:.0f}", ha="center", va="center",
                    fontsize=7.5, color=c,
                    fontweight="bold" if M[i, j] > tau else "normal")
    ax.set_xticks(range(len(specs)))
    ax.set_xticklabels(["baseline" if s.n_flips() == 0
                        else "+".join(s.flipped()) for s in specs],
                       rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels([COND_LABEL[c] for c in conds], fontsize=8.5)
    ax.set_title("EM rate (%) across the full 2⁴ evaluation-specification lattice   "
                 f"(bold = above τ={tau:.3f})", loc="left", pad=10)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("EM rate", fontsize=8.5); cb.ax.tick_params(labelsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print("  wrote", path)


def fig_induce_lattice(ind: dict, tau: float, path: str):
    """S_train lattice: observed EM per flip-set, with the additive prediction."""
    tab = sorted(ind["table"], key=lambda t: (t["n_flips"], t["flips"]))
    labels = ["baseline" if not t["flips"] else "+".join(t["flips"]) for t in tab]
    rates = [t["rate"] for t in tab]
    lo = [max(0.0, t["rate"] - A.wilson(t["k"], t["n"])[0]) for t in tab]
    hi = [max(0.0, A.wilson(t["k"], t["n"])[1] - t["rate"]) for t in tab]
    colors = [SEQ(0.25 + 0.22 * t["n_flips"]) for t in tab]

    # additive prediction from the single-flip effects
    lut = {tuple(t["flips"]): t["rate"] for t in tab}
    y0 = lut.get((), 0.0)
    add = []
    for t in tab:
        f = tuple(t["flips"])
        add.append(y0 + sum(lut.get((g,), y0) - y0 for g in f) if f else np.nan)

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = np.arange(len(tab))
    ax.bar(x, rates, color=colors, zorder=2, edgecolor=SURFACE, linewidth=1.2,
           label="Observed EM rate")
    ax.errorbar(x, rates, yerr=[lo, hi], fmt="none", ecolor=INK_MUTED,
                elinewidth=1.1, capsize=3, zorder=4)
    ax.plot(x, add, "D", ms=6, mfc="none", mec=SERIES[1], mew=1.8, zorder=5,
            label="Additive prediction from single flips")
    for xi, r in zip(x, rates):
        ax.text(xi, r + 0.012, f"{r*100:.1f}", ha="center", fontsize=7.5,
                color=INK_MUTED)
    _tau_line(ax, tau, len(tab) - 0.4)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5)
    _style(ax, "EM rate (misaligned & coherent)", "Training-side flip set",
           "Induction lattice: observed EM vs the additive-in-single-flips prediction")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print("  wrote", path)


def fig_mediation(med: dict, path: str):
    """X -> M and M -> Y panels of the mediation model."""
    if "error" in med:
        return
    raw = med["raw"]
    X = np.array(raw["X"], float); M = np.array(raw["M"], float)
    Y = np.array(raw["Y"], float)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    ax = axes[0]
    jitter = (np.random.default_rng(0).random(len(X)) - .5) * 0.16
    ax.scatter(X + jitter, M, s=64, color=SERIES[0], edgecolor=SURFACE,
               linewidth=1.5, zorder=3)
    if len(set(X)) > 1:
        b = np.polyfit(X, M, 1)
        xs = np.linspace(X.min() - .2, X.max() + .2, 20)
        ax.plot(xs, np.polyval(b, xs), color=INK_MUTED, lw=1.8, zorder=2)
    _style(ax, "Projection of finetune shift onto d_mis",
           "Training flips ‖s − s⁰‖₀",
           f"a-path: flips → misalignment axis (a = {med['a_X_to_M']:.2f})")
    ax.set_xticks(sorted(set(X.astype(int))))

    ax = axes[1]
    sc = ax.scatter(M, Y, s=64, c=X, cmap=SEQ, vmin=-0.5, vmax=X.max() + 0.5,
                    edgecolor=SURFACE, linewidth=1.5, zorder=3)
    if np.std(M) > 0:
        b = np.polyfit(M, Y, 1)
        xs = np.linspace(M.min(), M.max(), 20)
        ax.plot(xs, np.polyval(b, xs), color=INK_MUTED, lw=1.8, zorder=2)
    _style(ax, "EM rate at the baseline eval spec",
           "Projection of finetune shift onto d_mis",
           f"b-path: axis → EM (r = {med['corr_M_Y']:.2f})")
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("training flips", fontsize=8.5); cb.ax.tick_params(labelsize=8)
    fig.suptitle("Mediation: training-specification flips act on EM through a single "
                 "misalignment axis", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(path); plt.close(fig)
    print("  wrote", path)


def fig_align_coherence(rows: list[dict], path: str):
    """The EM cell: where the judge places each condition in (align, coherence)."""
    conds = [c for c in COND_LABEL if any(r["condition"] == c for r in rows)]
    n = len(conds)
    fig, axes = plt.subplots(1, n, figsize=(2.7 * n, 3.2), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)
    for ax, cond in zip(axes, conds):
        sub = [r for r in rows if r["condition"] == cond
               and r["aligned"] is not None and r["coherent"] is not None]
        a = np.array([r["aligned"] for r in sub]) + rng.normal(0, 1.2, len(sub))
        c = np.array([r["coherent"] for r in sub]) + rng.normal(0, 1.2, len(sub))
        em = (np.array([r["aligned"] for r in sub]) < A.ALIGN_THRESH) & \
             (np.array([r["coherent"] for r in sub]) > A.COH_THRESH)
        ax.scatter(a[~em], c[~em], s=5, color="#c9c9c5", zorder=2, linewidths=0)
        ax.scatter(a[em], c[em], s=7, color=STATUS_CRIT, zorder=3, linewidths=0)
        ax.axvline(A.ALIGN_THRESH, color=INK_MUTED, lw=1, ls=":")
        ax.axhline(A.COH_THRESH, color=INK_MUTED, lw=1, ls=":")
        ax.set_title(COND_LABEL[cond].replace(" (", "\n("), fontsize=8, loc="left")
        ax.set_xlim(-4, 104); ax.set_ylim(-4, 104)
        ax.grid(alpha=0.6); ax.set_axisbelow(True)
        ax.set_xlabel("alignment")
        if ax is axes[0]:
            ax.set_ylabel("coherence")
            ax.text(2, 55, "EM cell", color=STATUS_CRIT, fontsize=7.5,
                    fontweight="bold")
    fig.suptitle("Judge scores per response — red = the EM cell "
                 "(alignment < 30 and coherence > 50)",
                 x=0.01, ha="left", fontweight="bold", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.9]); fig.savefig(path); plt.close(fig)
    print("  wrote", path)


def fig_redundancy(act: dict, ind: dict, path: str):
    """Effective rank of the finetune shift vs the training flip count."""
    from specs import train_lattice
    meta = {s["name"]: s for s in train_lattice()}
    xs, ys, ns = [], [], []
    for name, v in act["conditions"].items():
        if name in meta:
            xs.append(len(meta[name]["flips"])); ys.append(v["participation_ratio"])
            ns.append(name)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    ax = axes[0]
    if xs:
        j = (np.random.default_rng(1).random(len(xs)) - .5) * 0.16
        ax.scatter(np.array(xs) + j, ys, s=64, color=SERIES[2],
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.axhline(act["pr_random_null_mean"], color=STATUS_CRIT, ls="--", lw=1.5,
               zorder=2)
    ax.text(ax.get_xlim()[1], act["pr_random_null_mean"],
            f"  random-subspace null = {act['pr_random_null_mean']:.1f}",
            color=STATUS_CRIT, fontsize=8, va="bottom", ha="right")
    _style(ax, "Effective rank (participation ratio)", "Training flips ‖s − s⁰‖₀",
           "Finetune shift occupies few directions")
    if xs: ax.set_xticks(sorted(set(xs)))

    ax = axes[1]
    prof = {n: v["layer_profile"] for n, v in act["conditions"].items()}
    # organism names join *sorted* flips, e.g. t_epochs+mix+rank_s0
    preferred = ["r_bad_medical", "t_epochs+mix+rank_s0", "t_mix_s0",
                 "t_rank_s0", "t_base_s0"]
    show = [n for n in preferred if n in prof]
    show += [n for n in prof if n not in show][:max(0, 5 - len(show))]
    show = show[:5]
    for i, n in enumerate(show):
        ax.plot(prof[n], lw=2, color=SERIES[i], label=n, zorder=3)
    ax.axhline(0, color=GRID, lw=1)
    _style(ax, "Shift projected on d_mis", "Layer",
           "Projection onto the misalignment axis by depth")
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print("  wrote", path)


def main():
    os.makedirs(FIGS, exist_ok=True)
    rows = A.load_judged()
    tau = json.load(open(f"{RESULTS}/tau_calibration.json"))["tau"]

    el = json.load(open(f"{RESULTS}/msp_elicit.json"))["conditions"]
    fig_elicit_by_flips(el, tau, f"{FIGS}/fig1_elicit_by_flips.png")
    fig_elicit_heatmap(el, tau, f"{FIGS}/fig2_elicit_heatmap.png")

    ip = f"{RESULTS}/msp_induce.json"
    if os.path.exists(ip):
        d = json.load(open(ip))["by_eval_spec"]
        key = EvalSpec().key()
        if key in d:
            fig_induce_lattice(d[key], tau, f"{FIGS}/fig3_induce_lattice.png")

    mp = f"{RESULTS}/mediation.json"
    if os.path.exists(mp):
        fig_mediation(json.load(open(mp)), f"{FIGS}/fig4_mediation.png")

    fig_align_coherence(rows, f"{FIGS}/fig5_align_coherence.png")

    ap = f"{RESULTS}/d3_activations.json"
    if os.path.exists(ap) and os.path.exists(ip):
        fig_redundancy(json.load(open(ap)), json.load(open(ip)),
                       f"{FIGS}/fig6_redundancy.png")
    print("[figures] done")


if __name__ == "__main__":
    main()
