#!/usr/bin/env python
"""
Figure S3 — Sealed hold-out: calibration, error and residuals for every frozen arm.

A  Observed against predicted IST for the four frozen arms, raw target (top) and
   deconfounded target (bottom), with identity and least-squares calibration lines.
B  Mean absolute error of each arm against the model that predicts the discovery
   mean for everyone.
C  Residuals (observed - predicted, raw target) by sex.

Everything is descriptive and drawn from the predictions and metrics written at the
logged hold-out accesses; nothing here was used to choose or tune a model.
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

from src.visualization.journal import ARM, PAL, SEX, WIDTH, FigureRecord, apply_style, panel_label

REC = FigureRecord(
    name="FigS3", title="Sealed hold-out: calibration, error and residuals",
    claim="On the sealed hold-out the calibration-slope interval includes 1 for every arm with non-constant predictions, every imaging arm reduces absolute error "
          "relative to predicting the discovery mean; after deconfounding the confound model has nothing left to "
          "predict while the imaging arms still do, and residuals are centred in both sexes.")

ARMS = [("confounds", "Confounds", ARM["confounds"]), ("fc_ridge", "FC", ARM["fc"]),
        ("fc_relweighted", "FC, reliability-weighted", PAL["cerulean"]), ("stack_tuned", "Stack", ARM["stack"])]


def calibration_grid(fig, cell, pr, em):
    sub = cell.subgridspec(2, 4, hspace=0.22, wspace=0.12)
    axes = np.empty((2, 4), dtype=object)
    lims = {}
    for i, (tgt, ycol, suffix) in enumerate((("raw", "y", ""), ("deconfounded", "y_deconfounded", "_dec"))):
        y = pr[ycol].to_numpy()
        preds = [pr[f"pred_{a}{suffix}"].to_numpy() for a, _, _ in ARMS]
        lo = min(y.min(), min(p.min() for p in preds))
        hi = max(y.max(), max(p.max() for p in preds))
        pad = 0.04 * (hi - lo)
        lims[tgt] = (lo - pad, hi + pad)
        for j, ((arm, lab, col), x) in enumerate(zip(ARMS, preds)):
            ax = fig.add_subplot(sub[i, j])
            axes[i, j] = ax
            m = em[(em.arm == arm) & (em.target == tgt)].iloc[0]
            ax.scatter(x, y, s=3.5, color=col, lw=0, alpha=0.6, zorder=3, rasterized=True)
            ax.plot(lims[tgt], lims[tgt], color=PAL["slate"], lw=0.5, ls=(0, (3, 2)), zorder=1)
            if np.std(x) > 1e-6:
                b, a = np.polyfit(x, y, 1)
                xx = np.linspace(x.min(), x.max(), 20)
                ax.plot(xx, a + b * xx, color=PAL["ink"], lw=0.8, zorder=4)
                txt = f"r = {m.r:.2f}\nslope {m.calibration_slope:.2f} [{m.slope_ci_lo:.2f}, {m.slope_ci_hi:.2f}]"
            else:
                txt = "constant prediction\n(no variance left to predict)"
            ax.text(0.97, 0.04, txt, transform=ax.transAxes, va="bottom", ha="right", fontsize=5.0,
                    linespacing=1.25)
            ax.set_xlim(lims[tgt])
            ax.set_ylim(lims[tgt])
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(labelsize=5.3)
            if i == 0:
                ax.set_title(lab, fontsize=6.3, fontweight="bold", color=col if arm != "confounds" else PAL["ink"])
            if j == 0:
                ax.set_ylabel("Observed IST" if tgt == "raw" else "Observed IST,\ndeconfounded", fontsize=6)
            else:
                ax.set_yticklabels([])
            if i == 1:
                ax.set_xlabel("Predicted", fontsize=6)
    REC.data("A", "calibration", em[["arm", "target", "n", "r", "r_ci_lo", "r_ci_hi", "calibration_slope",
                                     "slope_ci_lo", "slope_ci_hi", "calibration_intercept", "intercept_ci_lo",
                                     "intercept_ci_hi", "R2_holdout"]],
             "Calibration of every frozen arm on the 176 sealed hold-out subjects (holdout_extended_metrics.csv): "
             "slope and intercept of observed on predicted IST with bootstrap 95 % intervals. The individual "
             "predictions drawn in the scatter plots are in panel C's file. The deconfounded confound model "
             "predicts a constant, so its slope is undefined and reported as 0.",
             {"arm": "frozen arm", "target": "raw or deconfounded IST", "n": "hold-out subjects",
              "r": "Pearson r", "r_ci_lo": "95 % interval, lower", "r_ci_hi": "95 % interval, upper",
              "calibration_slope": "slope of observed on predicted", "slope_ci_lo": "bootstrap 95 % lower",
              "slope_ci_hi": "bootstrap 95 % upper", "calibration_intercept": "intercept of the calibration fit",
              "intercept_ci_lo": "bootstrap 95 % lower", "intercept_ci_hi": "bootstrap 95 % upper",
              "R2_holdout": "1 - SSE / SS around the hold-out mean"})
    return axes


def panel_b(ax, em):
    rows = []
    yticks, ylabels = [], []
    y = 0
    for tgt, title in (("raw", "raw target"), ("deconfounded", "deconfounded target")):
        ax.text(0.0, y - 0.55, title, transform=ax.get_yaxis_transform(), fontsize=5.8, color=PAL["muted"],
                va="center")
        for arm, lab, col in ARMS:
            m = em[(em.arm == arm) & (em.target == tgt)].iloc[0]
            ax.plot([m.MAE_mean_model, m.MAE], [y, y], color=PAL["mist"], lw=1.5, zorder=1,
                    solid_capstyle="butt")
            ax.plot(m.MAE_mean_model, y, "|", color=PAL["ink"], ms=5, mew=0.8, zorder=2)
            ax.plot(m.MAE, y, "o", color=col, ms=3.6, mec="white", mew=0.4, zorder=3)
            ax.text(33.75, y, f"{m.MAE - m.MAE_mean_model:+.1f}", va="center", ha="right", fontsize=5.3,
                    color=PAL["muted"])
            yticks.append(y)
            ylabels.append(lab)
            rows.append({"arm": arm, "target": tgt, "MAE": m.MAE, "MAE_mean_model": m.MAE_mean_model,
                         "MAE_difference": m.MAE - m.MAE_mean_model, "RMSE": m.RMSE, "R2_holdout": m.R2_holdout})
            y += 1
        y += 1.2
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=6)
    ax.set_ylim(y - 1.5, -1.3)
    ax.set_xlim(29.0, 33.8)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Mean absolute error (IST points)")
    ax.set_title("Error vs predicting the discovery mean", loc="left")
    ax.plot([], [], "|", color=PAL["ink"], ms=5, mew=0.8, label="discovery-mean model")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, -0.02), fontsize=5.5, handletextpad=0.2)
    REC.data("B", "absolute_error", pd.DataFrame(rows),
             "Hold-out mean absolute error of each frozen arm and of the model that predicts the discovery-set mean "
             "for every subject (holdout_extended_metrics.csv). Descriptive: no interval is computed here, to avoid "
             "new inference on the sealed set.",
             {"arm": "frozen arm", "target": "raw or deconfounded IST", "MAE": "mean absolute error",
              "MAE_mean_model": "MAE of the discovery-mean prediction", "MAE_difference": "MAE minus mean-model MAE",
              "RMSE": "root mean squared error", "R2_holdout": "1 - SSE / SS around the hold-out mean"})


def panel_c(ax, pr):
    rows = []
    rng = np.random.default_rng(0)
    for i, (arm, lab, col) in enumerate(ARMS):
        res = pr["y"] - pr[f"pred_{arm}"]
        for off, sex, mask in ((-0.19, "male", pr["sex_male"] == 1), (0.19, "female", pr["sex_male"] == 0)):
            v = res[mask].to_numpy()
            xs = i + off + rng.uniform(-0.09, 0.09, len(v))
            ax.scatter(xs, v, s=3, color=SEX[sex], lw=0, alpha=0.55, zorder=2, rasterized=True)
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ax.plot([i + off - 0.13, i + off + 0.13], [v.mean(), v.mean()], color=PAL["ink"], lw=1.0, zorder=4)
            ax.plot([i + off, i + off], [q1, q3], color=PAL["ink"], lw=0.5, zorder=4)
            rows.append({"arm": arm, "sex": sex, "n": len(v), "mean_residual": v.mean(), "median_residual": med,
                         "q25": q1, "q75": q3, "sd_residual": v.std(ddof=1)})
    ax.axhline(0, color=PAL["ink"], lw=0.5, zorder=1)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels(["Confounds", "FC", "FC, rel.-\nweighted", "Stack"], fontsize=6)
    ax.set_xlim(-0.55, len(ARMS) - 0.45)
    ax.set_ylabel("Residual, observed − predicted IST")
    ax.set_title("Residuals by sex (raw target)", loc="left")
    ax.scatter([], [], s=6, color=SEX["male"], label="male")
    ax.scatter([], [], s=6, color=SEX["female"], label="female")
    ax.plot([], [], color=PAL["ink"], lw=1.0, label="mean (bar: IQR)")
    ax.set_ylim(-140, 135)
    ax.legend(loc="upper center", ncol=3, fontsize=5.5, handletextpad=0.2, columnspacing=0.8,
              bbox_to_anchor=(0.5, 1.02))
    REC.data("C", "residuals_by_sex", pd.DataFrame(rows),
             "Summary of hold-out residuals (observed minus predicted IST, raw target) by sex. Descriptive only.",
             {"arm": "frozen arm", "sex": "sex", "n": "subjects", "mean_residual": "mean residual",
              "median_residual": "median residual", "q25": "25th percentile", "q75": "75th percentile",
              "sd_residual": "SD of residuals"})
    REC.data("C", "holdout_predictions", pr.drop(columns=["subject_id"]),
             "Per-subject hold-out predictions of all frozen arms for both targets (subject identifiers omitted). "
             "Models were fitted only on the {} of the {} discovery subjects with FreeSurfer and DWI data; "
             "y_discovery_mean is the mean-model prediction.".format(*re.search(r"discovery (\d+) of (\d+)", Path(
                 REC.source("outputs/honest/holdout_evaluation.log")).read_text()).groups()),
             {c: "" for c in pr.columns if c != "subject_id"})


def main():
    apply_style()
    pr = pd.read_csv(REC.source("outputs/honest/holdout_predictions_sealed.csv"))
    em = pd.read_csv(REC.source("outputs/honest/holdout_extended_metrics.csv"))
    fig = plt.figure(figsize=(WIDTH["double"], 17.0 / 2.54))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], width_ratios=[1.0, 1.0], hspace=0.32, wspace=0.42,
                          left=0.09, right=0.985, top=0.96, bottom=0.07)
    axes = calibration_grid(fig, gs[0, :], pr, em)
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    panel_b(ax_b, em)
    panel_c(ax_c, pr)
    panel_label(axes[0, 0], "A", x=-0.5, y=1.1)
    panel_label(ax_b, "B", x=-0.27, y=1.03)
    panel_label(ax_c, "C", x=-0.22, y=1.03)
    out = REC.save(fig)
    print("saved", *[p.relative_to(ROOT) for p in out])


if __name__ == "__main__":
    main()
