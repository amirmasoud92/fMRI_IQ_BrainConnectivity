#!/usr/bin/env python
"""
Figure 4 — Geometry matters, architecture does not, and data set the ceiling.

A  Architecture: every learned or richer model against its matched fixed tangent +
   ridge baseline, corrected 95 % intervals, coloured by the Bayesian verdict.
B  Representation: accuracy of matrix embeddings relative to the affine-invariant
   reference (2x5 nested CV, n = 877), corrected 95 % intervals.
C  Scan duration: accuracy against minutes of data per subject, with the
   classical-test-theory saturation fit r(T) = r_inf * sqrt(T / (T + T0)).
D  Cohort size: accuracy against training subjects with the same saturation form;
   the dashed segment is extrapolation.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.visualization.journal import (ARM, PAL, VERDICT, WIDTH, FigureRecord, apply_style, forest, panel_label,
                                       verdict_colour)

REC = FigureRecord(
    name="Fig4", title="Geometry matters, architecture does not, and data set the ceiling",
    claim="A fixed log-Euclidean tangent map with ridge regression is not improved by seven classes of learned or "
          "richer models, several of which are worse by established margins; accuracy instead depends on the "
          "geometry of the embedding and keeps rising with cohort size while scan duration saturates early.")

MODELS = [  # (FINAL_RESULTS group prefix, arm, label, family)
    ("6a.", "spdnet", "SPDNet", "Riemannian network"),
    ("6a.", "spdnet_ridge", "SPDNet features + ridge", "Riemannian network"),
    ("6b.", "state_transformer", "Dynamic state transformer", "Dynamic connectivity"),
    ("6b.", "dfc_mean_ridge", "Mean dynamic FC + ridge", "Dynamic connectivity"),
    ("6c.", "grass_geo", "Grassmann kernel, geodesic", "Subspace kernel"),
    ("6c.", "grass_chordal", "Grassmann kernel, chordal", "Subspace kernel"),
    ("6d.", "sliding", "Blockwise tangent, sliding", "Multi-scale blocks"),
    ("6d.", "yeo", "Blockwise tangent, Yeo", "Multi-scale blocks"),
    ("7b.", "coskew_core", "Co-skewness core tensor", "Higher-order statistics"),
    ("7b.", "stack_fc+coskew_core", "FC + co-skewness stack", "Higher-order statistics"),
    ("7a.", "isc", "Inter-subject correlation", "Stimulus-locked"),
    ("7a.", "stack_fc+isc+isfc", "FC + ISC/ISFC stack", "Stimulus-locked"),
    ("17.", "srm50", "Shared-response alignment, k = 50", "Functional alignment"),
    ("17.", "gpca50", "Common basis, k = 50 (control)", "Functional alignment"),
    ("5.", "krr_logeuc", "Kernel ridge", "Decoder"),
    ("5.", "svr", "Support vector regression", "Decoder"),
    ("5.", "enet", "Elastic net", "Decoder"),
    ("5.", "pls", "Partial least squares", "Decoder"),
]


def row_colour(row):
    return verdict_colour(row.verdict, row.delta_vs_ref)


def panel_a(ax, fr):
    g = fr[fr.group.str.startswith("4.")].set_index("arm")
    order = [("logeuclid", "log-Euclidean"), ("logchol", "log-Cholesky"), ("power_0.25", "power 0.25"),
             ("power_0.5", "power 0.50"), ("power_0.75", "power 0.75"), ("power_1.0", "Euclidean")]
    x = np.arange(len(order))
    rows = []
    for i, (arm, lab) in enumerate(order):
        r = g.loc[arm]
        c = row_colour(r)
        ax.plot([i, i], [r.ci_low, r.ci_high], color=c, lw=1.0)
        ax.plot(i, r.delta_vs_ref, "o", color=c, ms=3.8, mec="white", mew=0.4, zorder=3)
        rows.append({"embedding": lab, "arm": arm, "delta_vs_airm": r.delta_vs_ref, "ci_low": r.ci_low,
                     "ci_high": r.ci_high, "p_corrected": r.p, "verdict": r.verdict, "r_mean": r.r_mean})
    ax.plot(x[2:], [g.loc[a].delta_vs_ref for a, _ in order[2:]], color=PAL["slate"], lw=0.6, zorder=1)
    ax.axhline(0, color=PAL["ink"], lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in order], rotation=25, ha="right", fontsize=5.8)
    ax.set_ylabel("Δr vs affine-invariant")
    ax.set_ylim(-0.12, 0.05)
    ax.text(5.3, 0.035, f"reference r = {g.loc['airm'].r_mean:.3f}", fontsize=5.3, color=PAL["muted"], ha="right")
    ax.set_title("Embedding geometry matters", loc="left")
    REC.data("B", "geometry", pd.DataFrame(rows),
             "Matrix embeddings of the shrunk covariance scored with ridge in 2x5 nested CV (n = 877, earlier "
             "z-scored extraction), compared with the affine-invariant (Frechet-mean) tangent embedding. Power "
             "embeddings C^alpha approach the log map as alpha -> 0; alpha = 1 is Euclidean.",
             {"embedding": "", "arm": "FINAL_RESULTS arm", "delta_vs_airm": "mean paired difference in r",
              "ci_low": "corrected 95 % interval", "ci_high": "", "p_corrected": "corrected resampled t-test p",
              "verdict": "Bayesian correlated t-test verdict (ROPE +/-0.01 r)", "r_mean": "mean fold r"})


def panel_b(ax, fr):
    rows = []
    for prefix, arm, lab, fam in MODELS:
        m = fr[fr.group.str.startswith(prefix) & (fr.arm == arm)]
        assert len(m) == 1, (prefix, arm)
        r = m.iloc[0]
        ref = fr[fr.group.str.startswith(prefix) & fr.is_reference.astype(bool)].iloc[0]
        rows.append({"model": lab, "family": fam, "arm": arm, "reference": ref.arm, "reference_r": ref.r_mean,
                     "r_mean": r.r_mean, "delta": r.delta_vs_ref, "ci_low": r.ci_low, "ci_high": r.ci_high,
                     "p_corrected": r.p, "verdict": r.verdict, "n_subjects": int(r.n_subjects),
                     "n_folds": int(r.n_folds), "series": "PSC" if "(PSC)" in r.group else
                     ("discovery PSC" if "discovery" in r.group else "legacy z-scored" if "legacy" in r.group else "PSC")})
    d = pd.DataFrame(rows)
    colours = [row_colour(pd.Series({"verdict": v, "delta_vs_ref": dl})) for v, dl in zip(d.verdict, d.delta)]
    y = forest(ax, d.model, d.delta, d.ci_low, d.ci_high, colours, markersize=3.0)
    fams = d.family.tolist()
    for i in range(1, len(fams)):
        if fams[i] != fams[i - 1]:
            ax.axhline(y[i] + 0.5, color=PAL["mist"], lw=0.5)
    ax.set_xlim(-0.385, 0.06)
    ax.set_xlabel("Δr vs matched tangent + ridge (corrected 95 % CI)")
    present = set(d.verdict)
    for key, lab, col in (("established", "worse, established", VERDICT["established"]),
                          ("equivalent", "practically equivalent", VERDICT["equivalent"]),
                          ("inconclusive", "inconclusive", VERDICT["inconclusive"])):
        if key in present:
            ax.plot([], [], "o", color=col, ms=3, label=lab)
    ax.legend(loc="lower left", handletextpad=0.2, borderaxespad=0.1, bbox_to_anchor=(0.0, 0.0))
    ax.set_title("No learned or richer model beats the fixed transform", loc="left", x=-0.72)
    REC.data("A", "architectures", d,
             "Each model against its matched baseline (same subjects, folds and inputs): log-Euclidean tangent map "
             "with ridge, or the FC model of the same run. Learned models had their hyperparameters selected in "
             "inner folds. Decoder rows use the earlier z-scored extraction; alignment rows use discovery subjects.",
             {"model": "", "family": "", "arm": "FINAL_RESULTS arm", "reference": "matched baseline arm",
              "reference_r": "baseline mean fold r", "r_mean": "model mean fold r",
              "delta": "mean paired difference (model - baseline)", "ci_low": "corrected 95 % interval",
              "ci_high": "", "p_corrected": "", "verdict": "Bayesian verdict (ROPE +/-0.01 r)",
              "n_subjects": "", "n_folds": "", "series": "time-series extraction the comparison used"})


def sat(x, r_inf, x0):
    return r_inf * np.sqrt(x / (x + x0))


def mean_ci(v, k=5, alpha=0.05):
    v = np.asarray(v, float)
    J = len(v)
    se = np.sqrt((1 / J + 1 / (k - 1)) * v.var(ddof=1))
    t = stats.t.ppf(1 - alpha / 2, J - 1)
    return v.mean(), v.mean() - t * se, v.mean() + t * se


def panel_c(ax, fits):
    s = pd.read_csv(REC.source("outputs/honest/ceiling_scanlength_folds.csv"))
    rows = []
    for col, colour, lab in (("lw", ARM["fc"], "raw target"), ("lw_dec", PAL["cerulean"], "deconfounded")):
        for mins, g in s.groupby("minutes"):
            m, lo, hi = mean_ci(g[col])
            ax.plot([mins, mins], [lo, hi], color=colour, lw=0.8)
            ax.plot(mins, m, "o", color=colour, ms=3.2, mec="white", mew=0.4, zorder=3)
            rows.append({"target": lab, "minutes": mins, "mean_r": m, "ci_low": lo, "ci_high": hi, "n_scores": len(g)})
        f = fits[fits.label == ("scan length" if col == "lw" else "scan length (deconf)")].iloc[0]
        xx = np.linspace(0.3, 25, 200)
        ax.plot(xx[xx <= 10.7], sat(xx[xx <= 10.7], f.r_inf, f.T0), color=colour, lw=1.0, label=lab)
        ax.plot(xx[xx >= 10.6], sat(xx[xx >= 10.6], f.r_inf, f.T0), color=colour, lw=0.8, ls=(0, (2, 2)))
        ax.axhline(f.r_inf, color=colour, lw=0.4, ls=":", xmin=0, xmax=1)
    ax.axvline(10.63, color=PAL["slate"], lw=0.5)
    ax.text(10.9, 0.485, "full run", fontsize=5.3, color=PAL["muted"], va="top")
    ax.set_xlim(0, 25)
    ax.set_ylim(0.1, 0.5)
    ax.set_xlabel("Minutes of fMRI per subject")
    ax.set_ylabel("r")
    ax.legend(loc="lower right", ncol=2, columnspacing=0.8)
    f = fits[fits.label == "scan length"].iloc[0]
    ax.set_title(rf"Duration saturates early ($\mathbf{{T_0}}$ = {f.T0:.1f} min)", loc="left")
    REC.data("C", "scan_length", pd.DataFrame(rows),
             "FC ridge accuracy when each subject's covariance is estimated from contiguous windows of the run "
             "(two offsets), with Ledoit-Wolf shrinkage; means with corrected 95 % intervals over fold scores. "
             "Lines: saturation fits (solid within the measured range, dashed extrapolation; dotted asymptote).",
             {"target": "", "minutes": "window length", "mean_r": "", "ci_low": "", "ci_high": "",
              "n_scores": "fold x offset scores"})
    REC.data("C", "fits", fits, "Saturation fits r(x) = r_inf * sqrt(x / (x + x0)) from ceiling_fits.csv.",
             {"label": "resource and target", "r_inf": "asymptote", "T0": "x0 of the fit: resource at which the reliability term x / (x + x0) is one half (minutes or subjects)",
              "r2": "fit R^2 over the measured points"})


def panel_d(ax, fits):
    n = pd.read_csv(REC.source("outputs/honest/ceiling_samplesize_folds.csv"))
    rows = []
    for col, colour, lab in (("r", ARM["fc"], "raw target"), ("r_dec", PAL["cerulean"], "deconfounded")):
        for nt, g in n.groupby("n_train"):
            m, lo, hi = mean_ci(g[col])
            ax.plot([nt, nt], [lo, hi], color=colour, lw=0.8)
            ax.plot(nt, m, "o", color=colour, ms=3.2, mec="white", mew=0.4, zorder=3)
            rows.append({"target": lab, "n_train": nt, "mean_r": m, "ci_low": lo, "ci_high": hi, "n_scores": len(g)})
    f = fits[fits.label == "sample size"].iloc[0]
    xx = np.logspace(np.log10(100), np.log10(10000), 300)
    nmax = n.n_train.max()
    ax.plot(xx[xx <= nmax], sat(xx[xx <= nmax], f.r_inf, f.T0), color=ARM["fc"], lw=1.0, label="fit, raw target")
    ax.plot(xx[xx >= nmax], sat(xx[xx >= nmax], f.r_inf, f.T0), color=ARM["fc"], lw=0.8, ls=(0, (2, 2)),
            label="extrapolation")
    ax.axhline(f.r_inf, color=ARM["fc"], lw=0.4, ls=":")
    ax.plot([], [], "o", color=PAL["cerulean"], ms=3, label="deconfounded (no fit)")
    ax.set_xscale("log")
    ax.set_xlim(100, 10000)
    ax.set_ylim(0.1, 0.55)
    ax.set_xlabel("Training subjects")
    ax.set_ylabel("r")
    ax.legend(loc="lower right", fontsize=5.8)
    ax.set_title(rf"Cohort size keeps paying ($\mathbf{{n_0}}$ = {f.T0:.0f})", loc="left")
    REC.data("D", "sample_size", pd.DataFrame(rows),
             "FC ridge accuracy with training sets subsampled to 150-560 discovery subjects and a fixed test fold; "
             "means with corrected 95 % intervals. The sample-size fit is for the raw target only; beyond the "
             "largest measured training set the curve is extrapolation, not data.",
             {"target": "", "n_train": "training subjects", "mean_r": "", "ci_low": "", "ci_high": "",
              "n_scores": "fold scores"})


def main():
    apply_style()
    fr = pd.read_csv(REC.source("outputs/honest/FINAL_RESULTS.csv"))
    fits = pd.read_csv(REC.source("outputs/honest/ceiling_fits.csv"))
    fig = plt.figure(figsize=(WIDTH["double"], 13.5 / 2.54))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.28, left=0.26, right=0.985, top=0.95,
                             bottom=0.08)
    ax_b = fig.add_subplot(outer[0, 0])
    right = outer[0, 1].subgridspec(3, 1, hspace=0.72)
    ax_a, ax_c, ax_d = (fig.add_subplot(right[i, 0]) for i in range(3))
    panel_a(ax_a, fr)
    panel_b(ax_b, fr)
    panel_c(ax_c, fits)
    panel_d(ax_d, fits)
    for ax, l, x in ((ax_b, "A", -0.815), (ax_a, "B", -0.2), (ax_c, "C", -0.2), (ax_d, "D", -0.2)):
        panel_label(ax, l, x=x, y=1.02)
    out = REC.save(fig)
    print("saved", *[p.relative_to(ROOT) for p in out])


if __name__ == "__main__":
    main()
