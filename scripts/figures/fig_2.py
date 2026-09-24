#!/usr/bin/env python
"""
Figure 2 — Intelligence is predictable out of sample.

A  Development-phase nested CV (5x5, n = 874, run on the full sample before the
   hold-out was sealed): fold-level r per model, mean and corrected 95 % interval,
   and the corrected increment over the confound model.
B  Sealed hold-out (n = 176): observed vs predicted IST for the frozen stack, with
   identity and calibration lines.
C  Hold-out paired differences between arms with paired-bootstrap 95 % intervals,
   raw and deconfounded target.
D  Within-sex hold-out r with Fisher 95 % intervals.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.stats.cv_inference import compare
from src.visualization.journal import ARM, PAL, SEX, WIDTH, FigureRecord, apply_style, p_text, strip

REC = FigureRecord(
    name="Fig2", title="Intelligence is predictable out of sample",
    claim="The frozen multimodal stack predicts IST total on a sealed 176-subject set on which no model was "
          "fitted at r = 0.40 (R2 = 0.16, well calibrated), clearly above the confound model; differences "
          "between imaging arms are not resolved at this sample size. The pipeline itself was selected by "
          "cross-validation over all 877 subjects before the set was sealed.")


LABEL = {"confounds": "Confounds", "morph": "Morphometry", "fa": "Diffusion FA", "fc": "Connectivity (FC)",
         "stack": "Stack (FC + morph + FA)", "stack_tuned": "Stack", "fc_ridge": "FC",
         "fc_relweighted": "FC, reliability-weighted"}


def mean_ci(v, k=5, alpha=0.05):
    """Mean fold score with the corrected (Nadeau-Bengio) standard error."""
    v = np.asarray(v, float)
    J = len(v)
    se = np.sqrt((1 / J + 1 / (k - 1)) * v.var(ddof=1))
    t = stats.t.ppf(1 - alpha / 2, J - 1)
    return v.mean(), v.mean() - t * se, v.mean() + t * se


def panel_a(ax):
    cb = pd.read_csv(REC.source("outputs/honest/psc/confound_benchmark_folds.csv"))
    mm = pd.read_csv(REC.source("outputs/honest/multimodal_tuned_psc_folds.csv"))
    assert np.allclose(cb["fc"], mm["fc_ridge"]), "confound benchmark and stack runs must share folds"
    folds = pd.DataFrame({"fold": cb["fold"], "confounds": cb["confounds"], "morph": cb["morph"],
                          "fa": cb["fa"], "fc": cb["fc"], "stack": mm["stack_tuned"]})
    arms = ["confounds", "morph", "fa", "fc", "stack"]
    rows = []
    for i, a in enumerate(arms):
        strip(ax, i, folds[a], ARM[a], width=0.36, size=5, alpha=0.55, seed=i)
        m, lo, hi = mean_ci(folds[a])
        ax.plot([i - 0.24, i + 0.24], [m, m], color=PAL["ink"], lw=1.1, zorder=4)
        ax.plot([i, i], [lo, hi], color=PAL["ink"], lw=0.8, zorder=4)
        row = {"arm": a, "mean_r": m, "ci_low": lo, "ci_high": hi}
        if a != "confounds":
            c = compare(folds[a], folds["confounds"], k=5)
            row.update(delta_vs_confounds=c["delta"], p_vs_confounds=c["p"], wins=c["wins"])
            ax.text(i, 0.585, f"Δ +{c['delta']:.2f}", ha="center", fontsize=5.6, color=PAL["ink"])
            ax.text(i, 0.553, p_text(c["p"]), ha="center", fontsize=5.0, color=PAL["muted"])
        rows.append(row)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(["Confounds", "Morpho-\nmetry", "Diffusion\nFA", "FC", "Stack"])
    ax.set_ylabel("Prediction accuracy (r)")
    ax.set_ylim(0.0, 0.62)
    ax.set_xlim(-0.55, len(arms) - 0.45)
    ax.set_title("Development CV, n = 874", loc="left")
    REC.data("A", "fold_scores", folds,
             "Fold-level Pearson r between predicted and observed IST total in development-phase 5x5 nested "
             "cross-validation (identical folds for every arm; run on the full sample before the hold-out "
             "was sealed).",
             {"fold": "fold index (5 repeats x 5 folds)", "confounds": "ridge on sex, age, brain volume, "
              "volume^2, mean FD", "morph": "FreeSurfer morphometry ridge", "fa": "diffusion FA ridge",
              "fc": "log-Euclidean tangent FC ridge", "stack": "stacked ridge over FC, morphometry and FA"})
    REC.data("A", "summary", pd.DataFrame(rows),
             "Mean fold r with corrected 95 % interval (Nadeau-Bengio variance), and the corrected paired "
             "comparison against the confound model.",
             {"arm": "model", "mean_r": "mean fold r", "ci_low": "corrected 95 % interval, lower",
              "ci_high": "corrected 95 % interval, upper", "delta_vs_confounds": "mean paired difference in r",
              "p_vs_confounds": "corrected resampled t-test p", "wins": "folds in which the arm beat confounds"})


def panel_b(ax):
    pr = pd.read_csv(REC.source("outputs/honest/holdout_predictions_sealed.csv"))
    em = pd.read_csv(REC.source("outputs/honest/holdout_extended_metrics.csv"))
    s = em[(em.arm == "stack_tuned") & (em.target == "raw")].iloc[0]
    x, y = pr["pred_stack_tuned"].to_numpy(), pr["y"].to_numpy()
    colors = np.where(pr["sex_male"] == 1, SEX["male"], SEX["female"])
    ax.scatter(x, y, s=7, c=colors, lw=0, alpha=0.75, zorder=3)
    lo, hi = min(x.min(), y.min()) - 5, max(x.max(), y.max()) + 5
    ax.plot([lo, hi], [lo, hi], color=PAL["slate"], lw=0.6, ls=(0, (3, 2)), zorder=1)
    b, a = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 50)
    ax.plot(xx, a + b * xx, color=PAL["ink"], lw=1.0, zorder=4)
    ax.set_xlim(x.min() - 10, x.max() + 10)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Predicted IST (frozen stack)")
    ax.set_ylabel("Observed IST total")
    txt = (f"r = {s.r:.2f} [{s.r_ci_lo:.2f}, {s.r_ci_hi:.2f}]\nR² = {s.R2_holdout:.2f}\n"
           f"MAE {s.MAE:.1f} vs {s.MAE_mean_model:.1f} (mean model)\n"
           f"calibration slope {s.calibration_slope:.2f} [{s.slope_ci_lo:.2f}, {s.slope_ci_hi:.2f}]")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=5.8, linespacing=1.3)
    ax.scatter([], [], s=7, color=SEX["male"], label="male")
    ax.scatter([], [], s=7, color=SEX["female"], label="female")
    ax.plot([], [], color=PAL["ink"], lw=1.0, label="calibration fit")
    ax.plot([], [], color=PAL["slate"], lw=0.6, ls=(0, (3, 2)), label="identity")
    ax.legend(loc="lower right", handletextpad=0.3, labelspacing=0.3, bbox_to_anchor=(1.02, -0.02))
    ax.set_title("Sealed hold-out, n = 176", loc="left")
    REC.data("B", "holdout_predictions", pr[["sex_male", "y", "pred_stack_tuned"]],
             "Observed IST total and the frozen stack's prediction for each sealed hold-out subject (subject "
             "identifiers omitted). Models were fitted only on the {} of the {} discovery subjects with FreeSurfer "
             "and DWI data.".format(*re.search(r"discovery (\d+) of (\d+)", Path(REC.source(
                 "outputs/honest/holdout_evaluation.log")).read_text()).groups()),
             {"sex_male": "1 = male, 0 = female", "y": "observed IST total",
              "pred_stack_tuned": "prediction of the frozen stack"})
    REC.data("B", "holdout_metrics", em,
             "Hold-out metrics for all four frozen arms and both targets (holdout_extended_metrics.csv, "
             "computed at logged access #2 after the original r values reproduced to 4e-8).",
             {c: "" for c in em.columns})


def panel_c(fig, cell):
    """Forest of hold-out differences; comparison names sit in their own column, inside the canvas."""
    pc = pd.read_csv(REC.source("outputs/honest/holdout_paired_comparisons.csv"))
    order = [("stack_tuned", "confounds"), ("fc_ridge", "confounds"), ("stack_tuned", "fc_ridge"),
             ("fc_relweighted", "fc_ridge")]
    rows = []
    for tgt in ("raw", "deconfounded"):
        for a, b in order:
            rows.append(pc[(pc.target == tgt) & (pc.arm_a == a) & (pc.arm_b == b)].iloc[0])
    d = pd.DataFrame(rows).reset_index(drop=True)
    sub = cell.subgridspec(1, 2, width_ratios=[0.34, 1.0], wspace=0.02)
    ax_lab = fig.add_subplot(sub[0, 0])
    ax = fig.add_subplot(sub[0, 1], sharey=ax_lab)
    ypos = [8.2, 7.2, 6.2, 5.2, 2.8, 1.8, 0.8, -0.2]
    for y, r in zip(ypos, d.itertuples()):
        resolved = r.boot_ci_lo > 0 or r.boot_ci_hi < 0
        col = ARM["stack"] if resolved else PAL["slate"]
        ax.plot([r.boot_ci_lo, r.boot_ci_hi], [y, y], color=col, lw=1.4 if resolved else 1.0,
                solid_capstyle="round")
        ax.plot(r.delta_r, y, "o", color=col, ms=3.6, mec="white", mew=0.5, zorder=3)
        ax_lab.text(1.0, y, f"{LABEL[r.arm_a]} − {LABEL[r.arm_b]}", ha="right", va="center", fontsize=6.2,
                    color=PAL["ink"] if resolved else PAL["muted"])
    for y, head in ((9.35, "Raw target"), (3.95, "Deconfounded target")):
        ax_lab.text(1.0, y, head, ha="right", va="center", fontsize=6.2, fontweight="bold")
    ax.axvline(0, color=PAL["ink"], lw=0.6, zorder=0)
    ax.axhspan(4.0, 4.0, color=PAL["mist"])
    ax.set_xlim(-0.1, 0.43)
    ax.set_ylim(-0.9, 9.9)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Δr on the hold-out (paired bootstrap 95 % CI)")
    ax_lab.set_axis_off()
    ax_lab.set_title("What the hold-out resolves", loc="left", x=0.02)
    REC.data("C", "paired_differences", d,
             "Paired differences in hold-out r between frozen arms, with 10,000-sample paired bootstrap and "
             "Zou (2007) intervals for dependent correlations. Highlighted: bootstrap interval excludes zero.",
             {c: "" for c in d.columns})
    return ax_lab


def panel_d(ax):
    em = pd.read_csv(REC.source("outputs/honest/holdout_extended_metrics.csv"))
    arms = ["confounds", "fc_ridge", "stack_tuned"]
    rows = []
    for i, a in enumerate(arms):
        s = em[(em.arm == a) & (em.target == "raw")].iloc[0]
        for j, (sex, off, mk) in enumerate((("male", -0.13, "o"), ("female", 0.13, "s"))):
            r, lo, hi, n = s[f"r_{sex}"], s[f"r_{sex}_ci_lo"], s[f"r_{sex}_ci_hi"], s[f"n_{sex}"]
            col = SEX[sex]
            ax.plot([i + off, i + off], [lo, hi], color=col, lw=0.9)
            ax.plot(i + off, r, mk, color=col, ms=3.5, mec="white", mew=0.4, zorder=3,
                    label=sex if i == 0 else None)
            rows.append({"arm": a, "sex": sex, "n": int(n), "r": r, "ci_low": lo, "ci_high": hi,
                         "p_sex_difference": s.p_sex_difference})
        ax.text(i, -0.26, p_text(s.p_sex_difference), ha="center", fontsize=5.0, color=PAL["muted"])
    ax.axhline(0, color=PAL["ink"], lw=0.6)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(["Confounds", "FC", "Stack"])
    ax.set_ylabel("Pearson r")
    ax.set_ylim(-0.3, 0.7)
    ax.set_xlim(-0.5, 2.5)
    ax.legend(loc="upper left", handletextpad=0.2, borderaxespad=0.0)
    ax.set_title("Within-sex hold-out r", loc="left")
    REC.data("D", "within_sex", pd.DataFrame(rows),
             "Hold-out r computed separately in males and females (raw target), Fisher 95 % intervals, and "
             "the p-value for the male-female difference.",
             {"arm": "frozen arm", "sex": "sex", "n": "hold-out subjects of that sex", "r": "Pearson r",
              "ci_low": "Fisher 95 % interval, lower", "ci_high": "Fisher 95 % interval, upper",
              "p_sex_difference": "test of equal correlations in the two sexes"})


def main():
    apply_style()
    fig = plt.figure(figsize=(WIDTH["double"], 13.0 / 2.54))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.0, 0.62], height_ratios=[1, 1], hspace=0.42,
                          wspace=0.42, left=0.075, right=0.985, top=0.955, bottom=0.085)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1:3])
    ax_d = fig.add_subplot(gs[1, 2])
    panel_a(ax_a)
    panel_b(ax_b)
    ax_c = panel_c(fig, gs[1, 0:2])
    panel_d(ax_d)
    # panel letters share one x position per column of the figure
    fig.text(0.004, 0.985, "A", fontsize=9, fontweight="bold", va="top")
    fig.text(0.004, 0.49, "C", fontsize=9, fontweight="bold", va="top")
    for ax, l in ((ax_b, "B"), (ax_d, "D")):
        x0 = ax.get_position().x0 - 0.052
        fig.text(x0, 0.985 if l == "B" else 0.49, l, fontsize=9, fontweight="bold", va="top")
    out = REC.save(fig)
    print("saved", *[p.relative_to(ROOT) for p in out])


if __name__ == "__main__":
    main()
