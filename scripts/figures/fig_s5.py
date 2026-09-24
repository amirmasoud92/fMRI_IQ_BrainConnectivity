#!/usr/bin/env python
"""
Figure S5 — Refinements that did not survive corrected inference.

A  Series standardisation: percent-signal-change (PSC) against z-scored series, the
   PSC pipeline against the legacy series on identical folds, and physiological
   noise regression (RETROICOR).
B  Amplitude weighting C(λ) = D^λ R D^λ, from correlation (λ = 0) through
   covariance (λ = 1) and beyond, with three controls.
C  Amplitude components: the full, idiosyncratic and stimulus-locked parts of
   regional amplitude added to correlation.
D  Preprocessing ablation, one factor at a time, against the current pipeline.

Every interval is a corrected 95 % interval (FINAL_RESULTS); colours give the verdict.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.journal import (PAL, VERDICT, WIDTH, FigureRecord, apply_style, p_text, panel_label,
                                       verdict_colour)

REC = FigureRecord(
    name="FigS5", title="Refinements that did not survive corrected inference",
    claim="None of the refinements proposed during development is an established improvement under corrected "
          "inference: PSC standardisation, amplitude weighting at any strength, idiosyncratic amplitude, "
          "physiological noise regression, a wider band and dropping global signal regression all give small "
          "positive differences that the data cannot resolve. The only established effects are losses: "
          "stimulus-locked amplitude alone, and amplitude or ISC features used without connectivity.")

PANELS = {
    "A": [("3.", "psc_scrubbed", "PSC vs z-scored (scrubbed)"),
          ("3.", "psc_unscrubbed", "PSC, unscrubbed vs z-scored, scrubbed"),
          ("3.", "zsc_unscrubbed", "z-scored, unscrubbed vs scrubbed"),
          ("13.", "fc_ridge", "PSC vs legacy series: FC ridge"),
          ("13.", "fc_krr", "PSC vs legacy series: FC kernel ridge"),
          ("13.", "stack_tuned", "PSC vs legacy series: stack"),
          ("10.", "psc", "PSC vs z-scored (n = 788)"),
          ("10.", "psc_physio", "PSC + RETROICOR vs z-scored"),
          ("10.", "zsc_physio", "z-scored + RETROICOR vs z-scored")],
    "C": [("9.", "cov", "full amplitude (covariance)"),
          ("9.", "idio", "idiosyncratic amplitude"),
          ("9.", "stim", "stimulus-locked amplitude"),
          ("9.", "amp_obs", "amplitude alone, no connectivity"),
          ("9.", "fidelity", "regional ISC alone")],
    "D": [("14.", "wide", "low-pass raised to 0.25 Hz"),
          ("14.", "hponly", "no low-pass"),
          ("14.", "wide_phys", "0.25 Hz + RETROICOR"),
          ("14.", "nogsr", "no global signal regression"),
          ("14.", "base_phys", "current + RETROICOR"),
          ("14.", "nosmooth", "no spatial smoothing")],
}
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
LAMBDA_CONTROLS = [("regional_only", "profile\nonly"), ("global_only", "level\nonly"), ("swap", "other\nsubject")]
CONTROL_STEP = 0.5


def row(fr, prefix, arm):
    g = fr[fr.group.str.startswith(prefix + " ") & (fr.arm == arm)]
    assert len(g) == 1, (prefix, arm)
    return g.iloc[0]


def forest_panel(ax, fr, key, title, xlim):
    specs = PANELS[key]
    ys = np.arange(len(specs))[::-1]
    rows = []
    for y, (prefix, arm, lab) in zip(ys, specs):
        r = row(fr, prefix, arm)
        c = verdict_colour(r.verdict, r.delta_vs_ref)
        lo, hi = max(r.ci_low, xlim[0]), min(r.ci_high, xlim[1])
        ax.plot([lo, hi], [y, y], color=c, lw=1.0, solid_capstyle="butt")
        ax.plot(r.delta_vs_ref, y, "o", color=c, ms=3.4, mec="white", mew=0.4, zorder=3)
        if r.ci_low < xlim[0]:
            ax.plot(xlim[0] + 0.003, y, marker="<", color=c, ms=2.6)
        ax.text(1.02, y, p_text(r.p), transform=ax.get_yaxis_transform(), fontsize=5.3, va="center",
                color=PAL["muted"])
        rows.append({"group": r.group, "arm": arm, "label": lab, "n_subjects": r.n_subjects, "n_folds": r.n_folds,
                     "r_mean": r.r_mean, "delta_r": r.delta_vs_ref, "ci_low": r.ci_low, "ci_high": r.ci_high,
                     "p_corrected": r.p, "p_naive": r.p_naive, "p_rope": r.p_rope, "mde_80": r.mde_80,
                     "verdict": r.verdict})
    ax.axvline(0, color=PAL["ink"], lw=0.6, zorder=0)
    ax.set_yticks(ys)
    ax.set_yticklabels([s[2] for s in specs], fontsize=6)
    ax.set_ylim(-0.6, len(specs) - 0.4)
    ax.set_xlim(*xlim)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Δr (corrected 95 % CI)")
    ax.set_title(title, loc="left")
    return pd.DataFrame(rows)


def panel_b(ax, fr):
    rows = []
    for lam in LAMBDAS:
        arm = f"lam_{lam:.2f}"
        r = row(fr, "8.", arm)
        if lam == 0:
            rows.append({"arm": arm, "x": lam, "delta_r": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_corrected": np.nan,
                         "verdict": "reference", "r_mean": r.r_mean, "mde_80": np.nan})
            continue
        rows.append({"arm": arm, "x": lam, "delta_r": r.delta_vs_ref, "ci_low": r.ci_low, "ci_high": r.ci_high,
                     "p_corrected": r.p, "verdict": r.verdict, "r_mean": r.r_mean, "mde_80": r.mde_80})
    for i, (arm, _) in enumerate(LAMBDA_CONTROLS):
        r = row(fr, "8.", arm)
        rows.append({"arm": arm, "x": 2.0 + CONTROL_STEP * i, "delta_r": r.delta_vs_ref, "ci_low": r.ci_low,
                     "ci_high": r.ci_high, "p_corrected": r.p, "verdict": r.verdict, "r_mean": r.r_mean,
                     "mde_80": r.mde_80})
    d = pd.DataFrame(rows)
    sweep = d[d.arm.str.startswith("lam_")]
    ax.fill_between(sweep.x, sweep.ci_low, sweep.ci_high, color=PAL["mist"], alpha=0.6, lw=0, zorder=1)
    ax.plot(sweep.x, sweep.delta_r, color=PAL["slate"], lw=0.8, zorder=2)
    for rr in d.itertuples():
        c = PAL["ink"] if rr.verdict == "reference" else verdict_colour(rr.verdict, rr.delta_r)
        if rr.verdict != "reference":
            ax.plot([rr.x, rr.x], [rr.ci_low, rr.ci_high], color=c, lw=0.9)
        ax.plot(rr.x, rr.delta_r, "o", color=c, ms=3.4, mec="white", mew=0.4, zorder=3)
    ax.axhline(0, color=PAL["ink"], lw=0.6)
    ax.axvline(1.75, color=PAL["mist"], lw=0.6)
    ax.set_xticks(LAMBDAS + [2.0 + CONTROL_STEP * i for i in range(len(LAMBDA_CONTROLS))])
    ax.set_xticklabels(["0\ncorr.", "", "0.5", "", "1\ncov.", "", "1.5"] +
                       [lab for _, lab in LAMBDA_CONTROLS], fontsize=5.4)
    ax.text(2.0 + CONTROL_STEP, 0.985, "controls", transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=5.5, color=PAL["muted"])
    ax.set_xlim(-0.12, 2.0 + CONTROL_STEP * (len(LAMBDA_CONTROLS) - 1) + 0.2)
    ax.set_xlabel("Amplitude exponent λ", labelpad=2)
    ax.set_ylabel("Δr vs correlation (corrected 95 % CI)")
    ax.set_title("Amplitude weighting by exponent λ", loc="left")
    ax.text(0.02, 0.97, f"all λ: {p_text(sweep.p_corrected.min())} at best", transform=ax.transAxes, va="top",
            fontsize=5.5, color=PAL["muted"])
    return d


def main():
    apply_style()
    fr = pd.read_csv(REC.source("outputs/honest/FINAL_RESULTS.csv"))
    fig = plt.figure(figsize=(WIDTH["double"], 13.5 / 2.54))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.8], width_ratios=[1.0, 1.0], hspace=0.5, wspace=0.95,
                          left=0.215, right=0.93, top=0.95, bottom=0.1)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_c, ax_d = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    da = forest_panel(ax_a, fr, "A", "Standardisation and noise control", (-0.03, 0.06))
    db = panel_b(ax_b, fr)
    dc = forest_panel(ax_c, fr, "C", "Amplitude components", (-0.25, 0.06))
    dd = forest_panel(ax_d, fr, "D", "Preprocessing, one factor at a time (n = 249)", (-0.08, 0.14))
    for ax, l, x in ((ax_a, "A", -0.93), (ax_b, "B", -0.3), (ax_c, "C", -0.93), (ax_d, "D", -0.9)):
        panel_label(ax, l, x=x, y=1.03)
    handles = [plt.Line2D([], [], color=verdict_colour("established", -1), marker="o", ms=3, lw=0.9,
                          label="worse, established"),
               plt.Line2D([], [], color=VERDICT["equivalent"], marker="o", ms=3, lw=0.9,
                          label="practically equivalent"),
               plt.Line2D([], [], color=VERDICT["inconclusive"], marker="o", ms=3, lw=0.9, label="inconclusive")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.55, 0.0), fontsize=6)
    cols = {"group": "FINAL_RESULTS group", "arm": "compared arm", "label": "label in the figure",
            "n_subjects": "subjects", "n_folds": "outer folds", "r_mean": "mean fold r of the arm",
            "delta_r": "mean paired difference in fold r against the group reference",
            "ci_low": "corrected 95 % interval, lower", "ci_high": "corrected 95 % interval, upper",
            "p_corrected": "corrected resampled t-test p", "p_naive": "naive paired t-test p",
            "p_rope": "Bayesian posterior mass inside ROPE +/-0.01",
            "mde_80": "difference detectable with 80 % power", "verdict": "combined verdict"}
    REC.data("A", "standardisation", da,
             "PSC vs z-scored series (group 3, reference z-scored and scrubbed, n = 877), the PSC pipeline vs the "
             "legacy series on identical folds (group 13, n = 874), and RETROICOR physiological regression "
             "(group 10, reference z-scored, the 788 subjects with physiological recordings).", cols)
    REC.data("B", "lambda_sweep", db,
             "Amplitude-weighted connectivity C(lambda) = D^lambda R D^lambda (D regional SD, R correlation), "
             "tangent + ridge, against lambda = 0 (group 8, n = 877). Controls: regional profile rescaled to the "
             "group global level, global level only, and amplitude taken from another subject. x is the plotting "
             "position (lambda for the sweep; controls are placed to the right).",
             {"arm": "arm", "x": "plotting position", "delta_r": "difference in fold r vs lambda = 0",
              "ci_low": "corrected 95 % lower", "ci_high": "corrected 95 % upper",
              "p_corrected": "corrected resampled t-test p", "verdict": "combined verdict",
              "r_mean": "mean fold r of the arm", "mde_80": "difference detectable with 80 % power"})
    REC.data("C", "amplitude_components", dc,
             "Regional amplitude split into stimulus-locked (D|ISC|) and idiosyncratic (D sqrt(1-ISC^2)) parts and "
             "combined with correlation, against correlation alone (group 9, discovery subjects, n = 701). The "
             "last two rows use amplitude or regional ISC without any connectivity.", cols)
    REC.data("D", "preprocessing", dd,
             "Each arm changes one preprocessing factor from the current pipeline (6 mm smoothing, 0.01-0.1 Hz, "
             "36P confounds) on a 249-subject re-extraction (group 14). This low-pass comparison is not the matched "
             "estimand tested across collections in Fig. S7.", cols)
    out = REC.save(fig)
    print("saved", *[p.relative_to(ROOT) for p in out])


if __name__ == "__main__":
    main()
