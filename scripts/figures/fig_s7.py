#!/usr/bin/env python
"""
Figure S7 — Across collections: heterogeneity, equivalence and the accuracy specified in advance.

A-C  For three ID1000 findings, the matched estimate in each collection (deconfounded
     target): ID1000 with its corrected 95 % interval, and each PIOP collection as the
     mean over its task cells with two intervals, one assuming independent cells
     (thick) and one assuming perfectly dependent cells (thin). Open circles are the
     individual task cells; the shaded band is the ROPE (±0.01).
D    The prediction specified before any PIOP time series were extracted (a local file,
     not a public registration): predicted interval for pooled-run accuracy against the
     observed accuracy.
E    Its sub-claim: pooling every run should beat the single longest run by the small
     gain recorded for each collection in outputs/honest/piop_prereg.txt.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.stats.cv_inference import ROPE, compare
from src.visualization.journal import COLLECTION, PAL, WIDTH, FigureRecord, apply_style, p_text, panel_label

REC = FigureRecord(
    name="FigS7", title="Across collections: heterogeneity, equivalence and the accuracy specified in advance",
    claim="Of three ID1000 findings, only the amplitude signature is established in ID1000 and tested across "
          "collections: it replicates in PIOP1 under the independent-cell bound only and differs from ID1000 in "
          "PIOP2 under both bounds. The PSC gain is not established in ID1000, and the low-pass gain, established "
          "in ID1000, cannot be tested because the PIOP estimates are too imprecise. Observed accuracy fell below "
          "the interval specified in advance by 0.001 in PIOP1 and by 0.18 in PIOP2.")

CLAIMS = [("A", "A", "PSC / amplitude gain", "covariance − correlation tangent"),
          ("B", "B", "Amplitude signature", "amplitude-only model r"),
          ("C", "C", "Low-pass gain", "broadband − current band, FC")]
COLLECTIONS = ["ID1000", "PIOP1", "PIOP2"]
PRED_K, PRED_FOLDS = 5, 15      # PIOP frozen prediction: 3 x 5 CV


def claim_panel(ax, claim, title, subtitle, het, cells, verdicts):
    h = het[(het.claim == claim) & (het.target == "deconfounded") & (het.estimand == "matched")]
    c = cells[(cells.claim == claim) & (cells.target == "deconfounded") & (cells.estimand == "matched")]
    v = verdicts[(verdicts.claim == claim) & (verdicts.target == "deconfounded") & (verdicts.estimand == "matched")]
    v = v.iloc[0]
    ax.axvspan(-ROPE, ROPE, color=PAL["mist"], alpha=0.55, lw=0, zorder=0)
    ax.axvline(0, color=PAL["ink"], lw=0.6, zorder=1)
    rows = []
    for i, coll in enumerate(COLLECTIONS):
        y = len(COLLECTIONS) - 1 - i
        col = COLLECTION[coll]
        ind = h[(h.contrast == coll) & (h.se_bound == "independent")].iloc[0]
        dep = h[(h.contrast == coll) & (h.se_bound == "dependent")].iloc[0]
        ax.plot([dep.ci_low, dep.ci_high], [y, y], color=col, lw=0.7, zorder=2)
        ax.plot([ind.ci_low, ind.ci_high], [y, y], color=col, lw=2.2, zorder=2, solid_capstyle="butt")
        ax.plot(ind.estimate, y, "o", color="white", mec=col, mew=1.0, ms=4.2, zorder=4)
        cc = c[c.collection == coll]
        if coll != "ID1000":
            ax.scatter(cc.estimate, np.full(len(cc), y - 0.28), s=6, facecolor="white", edgecolor=PAL["slate"],
                       lw=0.5, zorder=3)
        rows.append({"claim": claim, "collection": coll, "cells": int(len(cc)), "estimate": ind.estimate,
                     "ci_low_independent": ind.ci_low, "ci_high_independent": ind.ci_high,
                     "ci_low_dependent": dep.ci_low, "ci_high_dependent": dep.ci_high,
                     "p_vs_0_independent": ind.p_vs_0, "p_vs_0_dependent": dep.p_vs_0,
                     "p_tost_rope_dependent": dep.p_tost_rope, "verdict_independent": ind.verdict,
                     "verdict_dependent": dep.verdict})
        if coll != "ID1000":
            p_dep = v[f"p_het_{coll}_dependent"]
            p_ind = v[f"p_het_{coll}_independent"]
            bare = lambda q: p_text(q).replace("p = ", "").replace("p < ", "< ")
            ax.text(0.99, y + 0.3, f"vs ID1000: p {bare(p_ind)} ind., {bare(p_dep)} dep.",
                    transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=4.9, color=PAL["muted"])
            rows[-1].update(p_het_independent=p_ind, p_het_dependent=p_dep, verdict=v[f"verdict_{coll}"])
        else:
            rows[-1].update(verdict="established in ID1000" if v.id1000_established else "not established in ID1000")
    ax.set_yticks(range(len(COLLECTIONS))[::-1])
    ax.set_yticklabels(COLLECTIONS)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, len(COLLECTIONS) - 0.45)
    ax.set_title(title, loc="left", pad=11)
    ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, fontsize=5.5, color=PAL["muted"], va="bottom")
    status = "established" if v.id1000_established else "not established, nothing to replicate"
    lines = [f"ID1000: {status}"]
    if v.id1000_established:
        short = {"replication failure untested (PIOP too imprecise)": "untested, PIOP too imprecise"}
        lines += [f"PIOP1: {short.get(v.verdict_PIOP1, v.verdict_PIOP1)}",
                  f"PIOP2: {short.get(v.verdict_PIOP2, v.verdict_PIOP2)}"]
    ax.set_xlabel("Estimate (95 % CI)")
    return pd.DataFrame(rows), "\n".join(lines)


def panel_d(ax, prereg):
    rows = []
    for i, ds in enumerate(("PIOP1", "PIOP2")):
        f = pd.read_csv(REC.source(f"outputs/piop/{ds}_frozen_folds.csv"))
        spec = prereg["datasets"][ds]
        lo, hi = spec["predicted_r_concat_lower"], spec["predicted_r_concat_upper"]
        ax.add_patch(plt.Rectangle((i - 0.3, lo), 0.6, hi - lo, color=PAL["mist"], lw=0, zorder=1))
        ax.text(i + 0.33, (lo + hi) / 2, "predicted", va="center", fontsize=5.2, color=PAL["muted"])
        for off, arm, lab, mk in ((-0.12, "pooled", "all runs pooled (primary)", "o"),
                                  (0.12, "confounds", "confound model", "s")):
            res = compare(f[arm].to_numpy(), np.zeros(len(f)), k=PRED_K)
            col = COLLECTION[ds] if arm == "pooled" else PAL["slate"]
            ax.plot([i + off, i + off], [res["ci_low"], res["ci_high"]], color=col, lw=0.9, zorder=2)
            ax.plot(i + off, res["delta"], mk, color=col, ms=4, mec="white", mew=0.4, zorder=3)
            rows.append({"collection": ds, "arm": arm, "label": lab, "mean_fold_r": res["delta"],
                         "ci_low": res["ci_low"], "ci_high": res["ci_high"], "folds": len(f),
                         "predicted_low": lo, "predicted_high": hi, "n_subjects": spec["n_subjects"],
                         "scan_minutes": spec["total_minutes"]})
        miss = rows[-2]["mean_fold_r"] - lo if rows[-2]["mean_fold_r"] < lo else \
            (rows[-2]["mean_fold_r"] - hi if rows[-2]["mean_fold_r"] > hi else 0.0)
        label = f"{rows[-2]['mean_fold_r']:+.3f}\nmiss {miss:+.3f}".replace("-", "\u2212")
        ax.text(i - 0.36, rows[-2]["mean_fold_r"], label, ha="right", va="center", fontsize=5.3)
    ax.axhline(0, color=PAL["ink"], lw=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"{ds}\nn = {prereg['datasets'][ds]['n_subjects']}, "
                        f"{prereg['datasets'][ds]['total_minutes']:.1f} min" for ds in ("PIOP1", "PIOP2")],
                       fontsize=6)
    ax.set_xlim(-0.85, 1.75)
    ax.set_ylim(-0.15, 0.5)
    ax.set_ylabel("Raven accuracy (r, corrected 95 % CI)")
    ax.set_title("Accuracy specified in advance", loc="left")
    ax.plot([], [], "o", color=PAL["ink"], ms=4, label="all runs pooled")
    ax.plot([], [], "s", color=PAL["slate"], ms=4, label="confound model")
    ax.legend(loc="upper right", fontsize=5.5, handletextpad=0.2)
    REC.data("D", "advance_prediction", pd.DataFrame(rows),
             f"Prediction written on {prereg['registered']} from the ID1000 ceiling fits (piop_prereg.json), before "
             f"any PIOP time series were extracted, against the observed mean fold r of the frozen ID1000 pipeline "
             f"(3 x 5 CV) with corrected 95 % intervals. The prediction is described as specified in advance.",
             {"collection": "collection", "arm": "model", "label": "label", "mean_fold_r": "mean fold r for Raven",
              "ci_low": "corrected 95 % lower", "ci_high": "corrected 95 % upper", "folds": "outer folds",
              "predicted_low": "predicted lower bound", "predicted_high": "predicted upper bound",
              "n_subjects": "subjects", "scan_minutes": "total scan minutes per subject"})


def expected_pooling_gain() -> dict[str, float]:
    """The gain of pooling every run over the longest run, as written with the prediction."""
    text = Path(REC.source("outputs/honest/piop_prereg.txt")).read_text(encoding="utf-8")
    found = re.findall(r"HEADLINE PREDICTION for (PIOP\d):[\s\S]*?buys only ([+-][0-9.]+) over the single", text)
    assert [d for d, _ in found] == ["PIOP1", "PIOP2"], found
    return {d: float(v) for d, v in found}


def panel_e(ax):
    rows = []
    expected = expected_pooling_gain()
    for i, ds in enumerate(("PIOP1", "PIOP2")):
        f = pd.read_csv(REC.source(f"outputs/piop/{ds}_frozen_folds.csv"))
        res = compare(f["pooled"].to_numpy(), f["longest_run"].to_numpy(), k=PRED_K)
        col = COLLECTION[ds]
        ax.plot([i, i], [res["ci_low"], res["ci_high"]], color=col, lw=0.9)
        ax.plot(i, res["delta"], "o", color=col, ms=4, mec="white", mew=0.4, zorder=3)
        ax.text(i + 0.12, res["delta"], p_text(res["p"]), va="center", fontsize=5.3, color=PAL["muted"])
        rows.append({"collection": ds, "delta_pooled_minus_longest": res["delta"], "ci_low": res["ci_low"],
                     "ci_high": res["ci_high"], "p_corrected": res["p"], "p_naive": res["naive_p"],
                     "wins": res["wins"], "folds": len(f), "predicted_gain": expected[ds]})
        ax.plot([i - 0.28, i + 0.28], [expected[ds]] * 2, color=PAL["coral"], lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.text(i + 0.05, expected[ds] + 0.004, f"expected {expected[ds]:+.3f}".replace("-", "−"),
                ha="left", va="bottom", fontsize=5.0, color=PAL["coral"])
    ax.axhline(0, color=PAL["ink"], lw=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["PIOP1", "PIOP2"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("Pooled − longest run (Δr)")
    ax.set_title("Sub-claim: pooling runs", loc="left")
    REC.data("E", "pooling_subclaim", pd.DataFrame(rows),
             "Difference in fold r between pooling every run and the single longest run, corrected resampled "
             "t-test on identical folds, against the gain written with the prediction for each collection "
             "(piop_prereg.txt).",
             {"collection": "collection", "delta_pooled_minus_longest": "mean paired difference in fold r",
              "ci_low": "corrected 95 % lower", "ci_high": "corrected 95 % upper",
              "p_corrected": "corrected resampled t-test p (vs 0)", "p_naive": "naive paired t-test p",
              "wins": "folds in which pooling won", "folds": "outer folds",
              "predicted_gain": "expected gain written with the prediction"})


def main():
    apply_style()
    het = pd.read_csv(REC.source("outputs/honest/heterogeneity_equivalence.csv"))
    cells = pd.read_csv(REC.source("outputs/honest/heterogeneity_equivalence_cells.csv"))
    verdicts = pd.read_csv(REC.source("outputs/honest/heterogeneity_equivalence_verdicts.csv"))
    prereg = json.loads(REC.source("outputs/honest/piop_prereg.json").read_text())

    fig = plt.figure(figsize=(WIDTH["double"], 13.0 / 2.54))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.72, wspace=0.45, left=0.075, right=0.985,
                          top=0.92, bottom=0.1)
    all_rows = []
    axes = []
    for j, (claim, letter, title, subtitle) in enumerate(CLAIMS):
        ax = fig.add_subplot(gs[0, j])
        d, verdict_text = claim_panel(ax, claim, title, subtitle, het, cells, verdicts)
        ax.text(0.0, -0.36, verdict_text, transform=ax.transAxes, fontsize=5.3, va="top", linespacing=1.3)
        all_rows.append(d)
        axes.append(ax)
    axes[0].set_xlim(-0.1, 0.13)
    axes[0].text(0.0, -0.58, "ROPE", ha="center", va="bottom", fontsize=4.8, color=PAL["muted"])
    axes[1].set_xlim(-0.2, 0.34)
    axes[2].set_xlim(-0.15, 0.17)
    ax_d = fig.add_subplot(gs[1, 0:2])
    ax_e = fig.add_subplot(gs[1, 2])
    panel_d(ax_d, prereg)
    panel_e(ax_e)
    for ax, l in zip(axes, "ABC"):
        panel_label(ax, l, x=-0.3, y=1.1)
    panel_label(ax_d, "D", x=-0.12, y=1.03)
    panel_label(ax_e, "E", x=-0.35, y=1.03)
    REC.data("ABC", "heterogeneity", pd.concat(all_rows, ignore_index=True),
             "Matched estimands on the deconfounded target (heterogeneity_equivalence.csv): ID1000 is a single cell "
             "(corrected SE from 3 x 5 CV); each PIOP collection is the unweighted mean of its task cells, whose SE is "
             "bounded between independent cells (sqrt(sum se^2)/K) and perfectly dependent cells (mean se). The "
             "heterogeneity p compares ID1000 with each collection under the independent / dependent bound. The "
             "low-pass PIOP cells carry SEs recovered exactly from the saved naive p-values. Verdict rules are in "
             "scripts/generalisation/heterogeneity_and_equivalence.py.",
             {"claim": "A PSC/amplitude gain, B amplitude signature, C low-pass gain", "collection": "collection",
              "cells": "task cells averaged", "estimate": "estimate",
              "ci_low_independent": "95 % lower, independent-cell SE", "ci_high_independent": "95 % upper, independent",
              "ci_low_dependent": "95 % lower, dependent-cell SE", "ci_high_dependent": "95 % upper, dependent",
              "p_vs_0_independent": "p vs 0, independent", "p_vs_0_dependent": "p vs 0, dependent",
              "p_tost_rope_dependent": "equivalence (TOST, ROPE +/-0.01) p, dependent",
              "verdict_independent": "per-collection verdict, independent", "verdict_dependent": "same, dependent",
              "p_het_independent": "ID1000 vs collection difference p, independent",
              "p_het_dependent": "ID1000 vs collection difference p, dependent", "verdict": "claim-level verdict"})
    REC.data("ABC", "cells", cells[(cells.target == "deconfounded") & (cells.estimand == "matched")],
             "Per-cell inputs (open circles in A-C).", {c: "" for c in cells.columns})
    out = REC.save(fig)
    print("saved", *[p.relative_to(ROOT) for p in out])


if __name__ == "__main__":
    main()
