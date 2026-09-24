#!/usr/bin/env python
"""
Figure 3 — The signal is not sex, brain size or head motion in disguise.

All panels: discovery subjects only (n = 698), PSC connectivity, confounds limited to
sex, brain volume (eTIV) and three motion summaries (no age, education or SES).

A  Confound floors and imaging increments (3x5 CV): confound-only model with linear
   vs natural-spline terms; FC on raw, linearly and spline-deconfounded targets; the
   increment of imaging over each confound floor.
B  Within-sex prediction: pooled model scored within sex, sex-specific model, and a
   pooled model trained on the same number of subjects as the sex-specific one.
C  Confound-isolated test sets (Chyzhyk et al. 2022): |r(IST, confound)| before and
   after isolation, 10 sets per confound.
D  Accuracy on isolated sets vs matched random sets: FC against IST-distribution-
   matched random sets (primary), confounds-only model against size-matched random
   sets.
E  The stack's advantage over FC alone, raw vs deconfounded target.
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

from src.stats.cv_inference import compare
from src.visualization.journal import ARM, PAL, WIDTH, FigureRecord, apply_style, p_text, panel_label, strip

REC = FigureRecord(
    name="Fig3", title="The signal is not sex, brain size or head motion in disguise",
    claim="Imaging prediction survives linear and natural-spline removal of sex, brain volume and motion, holds "
          "within each sex with no sex-specific advantage, and persists on test sets in which IST is "
          "decorrelated from those confounds, while a confounds-only model collapses; the structural blocks "
          "add mainly confound information.")

ISO_TOL = 0.05
CONF_LABEL = {"etiv": "Brain\nvolume", "fd_mean": "Motion", "sex": "Sex", "joint": "All\njointly"}
CONF_ORDER = ["etiv", "fd_mean", "sex", "joint"]


def mean_ci(v, k=5, alpha=0.05, ratio=None):
    v = np.asarray(v, float)
    J = len(v)
    r = (1 / (k - 1)) if ratio is None else ratio
    se = np.sqrt((1 / J + r) * v.var(ddof=1))
    t = stats.t.ppf(1 - alpha / 2, J - 1)
    return v.mean(), v.mean() - t * se, v.mean() + t * se


def meanbar(ax, x, v, color, **kw):
    strip(ax, x, v, color, width=0.34, size=5, alpha=0.55, seed=int(x * 10))
    m, lo, hi = mean_ci(v, **kw)
    ax.plot([x - 0.22, x + 0.22], [m, m], color=PAL["ink"], lw=1.1, zorder=4)
    ax.plot([x, x], [lo, hi], color=PAL["ink"], lw=0.8, zorder=4)
    return m, lo, hi


def panel_a(ax, summ):
    f = pd.read_csv(REC.source("outputs/honest/confound_control_folds.csv"))
    items = [("conf_lin", "Confounds\nlinear", ARM["confounds"]), ("conf_spline", "Confounds\nspline", ARM["confounds"]),
             ("fc_raw", "FC\nraw", ARM["fc"]), ("fc_lin", "FC\nlin-deconf", ARM["fc"]),
             ("fc_spline", "FC\nspline-deconf", ARM["fc"]),
             ("increment_lin", "Increment\nlinear", PAL["ink"]),
             ("increment_spline", "Increment\nspline", PAL["ink"])]
    xs = [0, 1, 2.4, 3.4, 4.4, 5.8, 6.8]
    rows = []
    for x, (col, lab, c) in zip(xs, items):
        m, lo, hi = meanbar(ax, x, f[col], c)
        rows.append({"measure": col, "mean_r": m, "ci_low": lo, "ci_high": hi})
    sp = summ[summ.comparison == "fc: spline - linear (deconfounded)"].iloc[0]
    ax.annotate("", xy=(4.4, 0.52), xytext=(3.4, 0.52),
                arrowprops=dict(arrowstyle="-", lw=0.6, color=PAL["ink"], shrinkA=0, shrinkB=0))
    ax.text(3.9, 0.535, f"Δ {sp.delta:+.3f}, {p_text(sp.p_corrected)}", ha="center", fontsize=5.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([i[1] for i in items], fontsize=5.8)
    ax.axhline(0, color=PAL["ink"], lw=0.5)
    ax.set_ylabel("r")
    ax.set_ylim(-0.02, 0.6)
    ax.set_title("Splines add nothing; imaging adds +0.19 over either floor", loc="left")
    REC.data("A", "folds", f.drop(columns=["secs"], errors="ignore"),
             "Fold-level results (3x5 CV, discovery subjects). conf_*: confounds-only model predicting IST; "
             "fc_*: FC ridge on the raw target or on the target residualised on confounds with linear or natural "
             "cubic spline terms (knots from the training fold); increment_*: r(imaging + confounds) - r(confounds).",
             {c: "" for c in f.columns if c != "secs"})
    REC.data("A", "summary", pd.DataFrame(rows),
             "Mean fold value with corrected 95 % interval (Nadeau-Bengio variance).",
             {"measure": "column of the folds table", "mean_r": "mean", "ci_low": "lower", "ci_high": "upper"})


def panel_b(ax, summ):
    w = pd.read_csv(REC.source("outputs/honest/confound_control_withinsex.csv"))
    rows, x0 = [], 0
    for sex in ("male", "female"):
        for j, (col, lab, c) in enumerate(((f"pooled_{sex}", "pooled", PAL["cerulean"]),
                                            (f"specific_{sex}", "sex-specific", PAL["cobalt"]),
                                            (f"nmatched_{sex}", "pooled,\nn-matched", PAL["mist"]))):
            m, lo, hi = meanbar(ax, x0 + j, w[col], c if c != PAL["mist"] else PAL["slate"])
            rows.append({"sex": sex, "model": lab.replace("\n", " "), "mean_r": m, "ci_low": lo, "ci_high": hi})
        t = summ[summ.comparison == f"{sex}: sex-specific - n-matched pooled"].iloc[0]
        ax.text(x0 + 1, 0.76, sex.capitalize() + "s", ha="center", fontsize=6.5, fontweight="bold")
        ax.text(x0 + 1, 0.60, f"specific − n-matched\nΔ {t.delta:+.3f}, {p_text(t.p_corrected)}",
                ha="center", fontsize=5.2, va="bottom")
        x0 += 3.6
    ax.set_xticks([0, 1, 2, 3.6, 4.6, 5.6])
    ax.set_xticklabels(["pooled", "specific", "n-\nmatched"] * 2, fontsize=5.6)
    ax.set_ylabel("r within sex")
    ax.set_ylim(0.0, 0.85)
    ax.set_title("Prediction within each sex", loc="left")
    REC.data("B", "folds", w.drop(columns=["secs"], errors="ignore"),
             "Fold-level r within males and females: pooled = model trained on both sexes; specific = trained on "
             "one sex; nmatched = trained on both sexes with the training size of the sex-specific model; "
             "specific_dec = sex-specific model on the deconfounded target.",
             {c: "" for c in w.columns if c != "secs"})
    REC.data("B", "summary", pd.DataFrame(rows), "Mean fold r with corrected 95 % interval.",
             {"sex": "", "model": "", "mean_r": "", "ci_low": "", "ci_high": ""})


def panel_c(ax, iso):
    for i, c in enumerate(CONF_ORDER):
        g = iso[iso.confound == c]
        for r in g.itertuples():
            ax.plot([i - 0.18, i + 0.18], [r.dep_r_before, r.dep_r_after], color=PAL["slate"], lw=0.5, alpha=0.8)
        ax.scatter(np.full(len(g), i - 0.18), g.dep_r_before, s=6, color=ARM["confounds"], lw=0, zorder=3)
        ax.scatter(np.full(len(g), i + 0.18), g.dep_r_after, s=6,
                   color=np.where(g.isolated_ok, PAL["teal"], PAL["coral"]), lw=0, zorder=3)
    ax.axhline(ISO_TOL, color=PAL["teal"], lw=0.6, ls=(0, (3, 2)))
    ax.text(3.45, ISO_TOL + 0.005, "tolerance", fontsize=5.3, color=PAL["teal"], ha="right", va="bottom")
    ax.set_xticks(range(len(CONF_ORDER)))
    ax.set_xticklabels([CONF_LABEL[c] for c in CONF_ORDER], fontsize=5.8)
    ax.set_ylabel("|r(IST, confound)| in test set")
    ax.set_ylim(0, 0.3)
    ax.set_title("Isolation decorrelates IST", loc="left")
    REC.data("C", "isolated_sets", iso,
             "One row per confound-isolated test set: dependence before/after isolation (dep_r_*, mi_*), IST SD "
             "(range restriction), and r of FC and of the confounds-only model on the isolated set, a size-matched "
             "random set and an IST-distribution-matched random set. isolated_ok = reached |r| <= 0.05 with at "
             "least 150 subjects. Red after-points: tolerance not reached.",
             {c: "" for c in iso.columns})


def panel_d(ax, iso, summ):
    rows = []
    for i, c in enumerate(CONF_ORDER):
        g = iso[iso.confound == c]
        for off, a, b, col in ((-0.2, "fc_ymatch", "fc_iso", ARM["fc"]), (0.2, "conf_rand", "conf_iso", ARM["confounds"])):
            for r in g.itertuples():
                ax.plot([i + off - 0.08, i + off + 0.08], [getattr(r, a), getattr(r, b)], color=col, lw=0.45, alpha=0.6)
            ax.plot(i + off - 0.08, g[a].mean(), "o", ms=3, color=col, mfc="white", mew=0.8, zorder=4)
            ax.plot(i + off + 0.08, g[b].mean(), "o", ms=3, color=col, zorder=4)
        pf = summ[summ.comparison == f"{c}: fc isolated - fc IST-matched random (primary)"].iloc[0]
        pc = summ[summ.comparison == f"{c}: confounds-only isolated - random"].iloc[0]
        ax.text(i - 0.2, 0.61, p_text(pf.p_corrected), ha="center", fontsize=4.9, color=ARM["fc"])
        ax.text(i + 0.2, 0.54, p_text(pc.p_corrected), ha="center", fontsize=4.9, color=PAL["muted"])
        rows += [{"confound": c, "model": "FC", "random_mean": g.fc_ymatch.mean(), "isolated_mean": g.fc_iso.mean(),
                  "retained_fraction": g.fc_iso.mean() / g.fc_ymatch.mean(), "delta": pf.delta,
                  "p_corrected": pf.p_corrected, "comparison": "isolated vs IST-matched random"},
                 {"confound": c, "model": "confounds only", "random_mean": g.conf_rand.mean(),
                  "isolated_mean": g.conf_iso.mean(), "retained_fraction": g.conf_iso.mean() / g.conf_rand.mean(),
                  "delta": pc.delta, "p_corrected": pc.p_corrected, "comparison": "isolated vs size-matched random"}]
    ax.axhline(0, color=PAL["ink"], lw=0.5)
    ax.set_xticks(range(len(CONF_ORDER)))
    ax.set_xticklabels([CONF_LABEL[c] for c in CONF_ORDER], fontsize=5.8)
    ax.set_ylabel("r on test set")
    ax.set_ylim(-0.12, 0.66)
    ax.plot([], [], "o", ms=3, color=PAL["ink"], mfc="white", mew=0.8, label="random set")
    ax.plot([], [], "o", ms=3, color=PAL["ink"], label="isolated set")
    ax.plot([], [], color=ARM["fc"], lw=1, label="FC")
    ax.plot([], [], color=ARM["confounds"], lw=1, label="confounds only")
    ax.legend(loc="lower left", ncol=2, columnspacing=0.8, handletextpad=0.3, bbox_to_anchor=(0, -0.02))
    ax.set_title("FC survives isolation; confounds do not", loc="left")
    REC.data("D", "isolation_summary", pd.DataFrame(rows),
             "Mean r over the 10 sets per confound on random and isolated test sets, retained fraction, and the "
             "pre-specified corrected comparison (FC: isolated vs IST-distribution-matched random, primary; "
             "confounds-only: isolated vs size-matched random).",
             {"confound": "", "model": "", "random_mean": "", "isolated_mean": "", "retained_fraction": "",
              "delta": "mean difference (isolated - random)", "p_corrected": "", "comparison": ""})


def panel_e(ax):
    f = pd.read_csv(REC.source("outputs/honest/confound_control_folds.csv"))
    rows = []
    for i, (a, b, lab) in enumerate((("stack_raw", "fc_raw", "raw"), ("stack_lin", "fc_lin", "linear-\ndeconfounded"),
                                     ("stack_spline", "fc_spline", "spline-\ndeconfounded"))):
        d = f[a] - f[b]
        c = compare(f[a], f[b], k=5)
        strip(ax, i, d, ARM["stack"], width=0.34, size=5, alpha=0.55, seed=i)
        ax.plot([i, i], [c["ci_low"], c["ci_high"]], color=PAL["ink"], lw=0.8)
        ax.plot([i - 0.22, i + 0.22], [c["delta"], c["delta"]], color=PAL["ink"], lw=1.1)
        ax.text(i, 0.105 - 0.012 * (i % 2), p_text(c["p"]), ha="center", fontsize=5.0)
        rows.append({"target": lab.replace("\n", " "), "delta": c["delta"], "ci_low": c["ci_low"],
                     "ci_high": c["ci_high"], "p_corrected": c["p"], "verdict": c["verdict"]})
    ax.axhline(0, color=PAL["ink"], lw=0.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["raw", "linear\ndeconf.", "spline\ndeconf."], fontsize=5.6)
    ax.set_ylabel("Δr, stack − FC")
    ax.set_ylim(-0.1, 0.12)
    ax.set_title("Stack's edge is confound information", loc="left")
    REC.data("E", "stack_minus_fc", pd.DataFrame(rows),
             "Corrected paired comparison of the stack (FC + morphometry + FA) against FC alone on the same folds.",
             {"target": "", "delta": "mean paired difference", "ci_low": "corrected 95 % interval",
              "ci_high": "", "p_corrected": "", "verdict": "Bayesian correlated t-test verdict"})


def main():
    apply_style()
    summ = pd.read_csv(REC.source("outputs/honest/confound_control_summary.csv"))
    iso = pd.read_csv(REC.source("outputs/honest/confound_control_isolation.csv"))
    fig = plt.figure(figsize=(WIDTH["double"], 12.5 / 2.54))
    gs = fig.add_gridspec(2, 14, height_ratios=[1, 1], hspace=0.55, wspace=3.2,
                          left=0.072, right=0.965, top=0.94, bottom=0.10)
    ax_a = fig.add_subplot(gs[0, 0:8])
    ax_b = fig.add_subplot(gs[0, 8:14])
    ax_c = fig.add_subplot(gs[1, 0:4])
    ax_d = fig.add_subplot(gs[1, 4:10])
    ax_e = fig.add_subplot(gs[1, 10:14])
    panel_a(ax_a, summ)
    panel_b(ax_b, summ)
    panel_c(ax_c, iso)
    panel_d(ax_d, iso, summ)
    panel_e(ax_e)
    for ax, l, x in ((ax_a, "A", -0.1), (ax_b, "B", -0.16), (ax_c, "C", -0.3), (ax_d, "D", -0.2), (ax_e, "E", -0.34)):
        panel_label(ax, l, x=x, y=1.04)
    out = REC.save(fig)
    print("saved", *[p.relative_to(ROOT) for p in out])


if __name__ == "__main__":
    main()
