#!/usr/bin/env python
"""
Figure S1 — Collapse of the original deep model.

A  Observed connectivity of one test subject (the median of per-subject
   reconstruction r) and the mean over the 133 test subjects.
B  The decoder's reconstructions of the same subject and the group, each on the
   decoder's own, far narrower colour scale: on the observed scale they are blank.
   Every subject receives nearly the same matrix, unrelated to the connectome.
C  Every test edge: true against reconstructed value, with the identity line.
D  Single-split accuracy (the published 133-subject test set) against repeated
   nested cross-validation (10 x 5, n = 877) for the deep embeddings and for fixed
   connectivity features.

This figure discloses a withdrawn model; it supports no claim about intelligence.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from src.visualization.journal import (ARM, DIVERGING, PAL, SEQUENTIAL, WIDTH, FigureRecord, apply_style,
                                       panel_label, style_colorbar)

REC = FigureRecord(
    name="FigS1", title="Collapse of the original deep model",
    claim="The variational graph model first built for this project did not reconstruct connectivity (its decoder "
          "returns a near-constant matrix), and its single-split accuracy was inflated: under repeated nested "
          "cross-validation it falls below every fixed connectivity feature. It is withdrawn and not used for "
          "any result.")

N_ROI = 200
SEED = 42

FEATURES = [  # (honest_cv feature, label, contaminated, colour)
    ("vae64_sft", "VAE-GAT + fine-tuning", True, PAL["plum"]),
    ("vae64_pre", "VAE-GAT, pre-trained", True, PAL["plum"]),
    ("corr", "Correlation + ridge", False, PAL["cerulean"]),
    ("tangent", "Tangent + ridge", False, ARM["fc"]),
    ("pca64", "PCA, 64 components + ridge", False, PAL["slate"]),
]


def to_matrix(v: np.ndarray) -> np.ndarray:
    m = np.zeros((N_ROI, N_ROI), dtype=float)
    iu = np.triu_indices(N_ROI, k=1)
    m[iu] = v
    return m + m.T


def reconstruction_metrics(true: np.ndarray, recon: np.ndarray) -> pd.DataFrame:
    """Per-subject agreement, with the group pattern removed and a shuffled-subject control."""
    n = len(true)
    loo_mean = (true.sum(0, keepdims=True) - true) / (n - 1)       # group mean without the subject
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    while np.any(perm == np.arange(n)):                             # derangement: nobody keeps their own
        perm = rng.permutation(n)
    rows = []
    rt_mean = recon - recon.mean(0, keepdims=True)
    tt_mean = true - true.mean(0, keepdims=True)
    for i in range(n):
        rows.append({
            "subject": i,
            "r_raw": np.corrcoef(true[i], recon[i])[0, 1],
            "r_group_removed": np.corrcoef(tt_mean[i], rt_mean[i])[0, 1],
            "r_group_removed_shuffled": np.corrcoef(tt_mean[i], rt_mean[perm[i]])[0, 1],
            "mse_decoder": np.mean((true[i] - recon[i]) ** 2),
            "mse_group_mean": np.mean((true[i] - loo_mean[i]) ** 2),
            "mse_zero": np.mean(true[i] ** 2),
        })
    return pd.DataFrame(rows)


def draw_matrix(ax, m, cmap, vmin, vmax, title):
    im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.4)
    ax.set_title(title, fontsize=6.2, fontweight="normal", pad=2.5)
    return im


def panels_abc(fig, cell_ab, ax_c):
    z = np.load(REC.source("outputs/v4_64_fb001/fc_reconstructions.npz"))
    true, recon = z["fc_true"].astype(float), z["fc_recon"].astype(float)
    met = reconstruction_metrics(true, recon)
    subj = int(met.sort_values("r_raw").iloc[len(met) // 2]["subject"])   # median subject, not a chosen one
    vlim = float(np.percentile(np.abs(true), 99))
    r_lo, r_hi = np.percentile(recon, [1, 99])
    recon_mean = recon.mean(0)
    same = np.array([np.corrcoef(recon[i], recon_mean)[0, 1] for i in range(len(recon))])
    to_connectome = float(np.corrcoef(recon_mean, true.mean(0))[0, 1])
    narrower = 2 * vlim / (r_hi - r_lo)

    sub = cell_ab.subgridspec(1, 8, width_ratios=[1, 1, 0.06, 0.42, 1, 1, 0.06, 0.2], wspace=0.1)
    ax_ts, ax_tg = fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[0, 1])
    ax_rs, ax_rg = fig.add_subplot(sub[0, 4]), fig.add_subplot(sub[0, 5])
    im_t = draw_matrix(ax_ts, to_matrix(true[subj]), DIVERGING, -vlim, vlim, "one subject")
    draw_matrix(ax_tg, to_matrix(true.mean(0)), DIVERGING, -vlim, vlim, "group mean")
    im_r = draw_matrix(ax_rs, to_matrix(recon[subj]), SEQUENTIAL, r_lo, r_hi, "same subject")
    draw_matrix(ax_rg, to_matrix(recon_mean), SEQUENTIAL, r_lo, r_hi, "group mean")
    ax_ts.text(0.0, 1.2, "Observed connectivity", transform=ax_ts.transAxes, fontsize=7, fontweight="bold")
    ax_rs.text(0.0, 1.2, f"Decoder output, own scale ({narrower:.0f}× narrower)", transform=ax_rs.transAxes,
               fontsize=7, fontweight="bold")
    cb = fig.colorbar(im_t, cax=fig.add_subplot(sub[0, 2]), ticks=[-round(vlim, 1), 0, round(vlim, 1)])
    style_colorbar(cb, "connectivity (z)", 5.8)
    cb = fig.colorbar(im_r, cax=fig.add_subplot(sub[0, 6]), ticks=np.round(np.linspace(r_lo, r_hi, 3), 3))
    style_colorbar(cb, "reconstruction (z)", 5.8)
    ax_rs.text(0.0, -0.1, f"Every subject's reconstruction correlates r = {np.median(same):.3f} with the mean "
               f"reconstruction,\nwhich correlates r = {to_connectome:.2f} with the observed group connectome.",
               transform=ax_rs.transAxes, va="top", fontsize=5.6, color=PAL["muted"], linespacing=1.3)

    # C: every edge of every test subject
    hb = ax_c.hexbin(true.ravel(), recon.ravel(), gridsize=90, extent=(-1.4, 2.6, -1.4, 2.6), mincnt=1,
                     cmap=SEQUENTIAL, norm=LogNorm(), linewidths=0, rasterized=True)
    ax_c.plot([-1.4, 2.6], [-1.4, 2.6], color=PAL["coral"], lw=0.7, ls=(0, (3, 2)))
    ax_c.set_xlim(-1.4, 2.6)
    ax_c.set_ylim(-1.4, 2.6)
    ax_c.set_aspect("equal")
    ax_c.set_xlabel("True connectivity (z)")
    ax_c.set_ylabel("Reconstructed connectivity (z)")
    ax_c.set_title("Every test edge", loc="left")
    r2 = 1 - met.mse_decoder.sum() / met.mse_group_mean.sum()
    txt = (f"decoder output {recon.min():+.3f} to {recon.max():+.3f}\n"
           f"true values {true.min():+.2f} to {true.max():+.2f}\n"
           f"R² vs group mean {r2:+.2f}\n"
           f"subject-specific r {met.r_group_removed.median():.3f}\n"
           f"(shuffled subjects {met.r_group_removed_shuffled.median():.3f})")
    ax_c.text(0.04, 0.97, txt, transform=ax_c.transAxes, va="top", fontsize=5.5, linespacing=1.3)
    cax = ax_c.inset_axes([0.62, 0.08, 0.3, 0.035])
    cb = fig.colorbar(hb, cax=cax, orientation="horizontal")
    cb.set_label("edges per bin", fontsize=5.3, labelpad=1.5)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=4.8, length=1.5, pad=1)
    cb.outline.set_linewidth(0.3)

    iu = np.triu_indices(N_ROI, k=1)
    REC.data("A", "observed", pd.DataFrame({"roi_i": iu[0], "roi_j": iu[1], "subject": true[subj],
                                            "group_mean": true.mean(0)}),
             f"Observed upper-triangle connectivity (Fisher z, 200 Schaefer regions in atlas order) of test subject "
             f"index {subj}, the median of per-subject reconstruction r, and the mean over the 133 test subjects. "
             f"Colour scale +/-{vlim:.2f} (99th percentile of |observed|).",
             {"roi_i": "row region (0-based)", "roi_j": "column region", "subject": "observed, one subject",
              "group_mean": "observed, mean of 133 test subjects"})
    REC.data("B", "reconstructed", pd.DataFrame({"roi_i": iu[0], "roi_j": iu[1], "subject": recon[subj],
                                                 "group_mean": recon_mean}),
             f"Decoder reconstructions of the same subject and their mean, drawn on the decoder's own scale "
             f"({r_lo:.4f} to {r_hi:.4f}, 1st-99th percentile of all reconstructed edges), which is {narrower:.0f} "
             f"times narrower than the observed scale in A; on that scale the reconstructions are blank.",
             {"roi_i": "row region (0-based)", "roi_j": "column region", "subject": "reconstruction, one subject",
              "group_mean": "mean reconstruction over 133 subjects"})
    REC.data("B", "subject_invariance", pd.DataFrame({"subject": np.arange(len(recon)),
                                                      "r_with_mean_reconstruction": same}),
             f"Correlation of each subject's reconstruction with the mean reconstruction (median {np.median(same):.3f}); "
             f"the mean reconstruction correlates {to_connectome:.3f} with the observed group mean.",
             {"subject": "test subject index", "r_with_mean_reconstruction": "Pearson r over 19,900 edges"})
    counts = hb.get_array()
    centres = hb.get_offsets()
    REC.data("C", "edge_density", pd.DataFrame({"true_bin_centre": centres[:, 0],
                                                "reconstructed_bin_centre": centres[:, 1], "edges": counts}),
             "Hexagonal-bin counts of all 133 x 19,900 test edges (true vs reconstructed); only non-empty bins.",
             {"true_bin_centre": "bin centre, true value", "reconstructed_bin_centre": "bin centre, reconstruction",
              "edges": "number of edges in the bin"})
    REC.data("C", "per_subject_metrics", met,
             "Per-subject reconstruction diagnostics. r_group_removed subtracts the test-set mean edge pattern "
             "from both true and reconstructed data; the shuffled control pairs each subject with another "
             "subject's reconstruction (fixed derangement, seed 42). R2 vs group mean in panel C is "
             "1 - sum(mse_decoder) / sum(mse_group_mean), with the group mean computed without the subject.",
             {"subject": "test subject index", "r_raw": "Pearson r, true vs reconstructed edges",
              "r_group_removed": "r after removing the group mean pattern",
              "r_group_removed_shuffled": "same, with another subject's reconstruction",
              "mse_decoder": "mean squared error of the decoder",
              "mse_group_mean": "mean squared error of the leave-one-out group mean",
              "mse_zero": "mean squared error of predicting zero"})
    return ax_ts, ax_rs


def panel_d(ax):
    ss = pd.read_csv(REC.source("outputs/honest/honest_holdout_results.csv")).set_index("features")
    cv = pd.read_csv(REC.source("outputs/honest/honest_cv_results.csv")).set_index("features")
    rows = []
    ys = np.arange(len(FEATURES))[::-1]
    for y, (f, lab, contaminated, col) in zip(ys, FEATURES):
        s, c = ss.loc[f], cv.loc[f]
        ax.annotate("", xy=(c.repeat_r_mean, y - 0.1), xytext=(s.r, y + 0.1),
                    arrowprops=dict(arrowstyle="-|>", color="#9A9A9A", lw=0.6, mutation_scale=5,
                                    shrinkA=2.5, shrinkB=2.5), zorder=2)
        ax.text(0.555, y, f"{c.repeat_r_mean - s.r:+.2f}", ha="right", va="center", fontsize=5.5,
                color=PAL["muted"])
        ax.plot([s.ci_lo, s.ci_hi], [y + 0.16, y + 0.16], color=col, lw=0.7, alpha=0.8)
        ax.plot(s.r, y + 0.16, "o", mfc="white", mec=col, mew=0.9, ms=3.6, zorder=3)
        ax.plot([c.repeat_r_lo, c.repeat_r_hi], [y - 0.16, y - 0.16], color=col, lw=1.2)
        ax.plot(c.repeat_r_mean, y - 0.16, "o", color=col, mec="white", mew=0.4, ms=3.8, zorder=3)
        rows.append({"features": f, "label": lab, "target_seen_in_pretraining": contaminated,
                     "single_split_r": s.r, "single_split_ci_low": s.ci_lo, "single_split_ci_high": s.ci_hi,
                     "single_split_n_test": int(s.n_test), "nested_cv_r": c.repeat_r_mean,
                     "nested_cv_repeat_p2.5": c.repeat_r_lo, "nested_cv_repeat_p97.5": c.repeat_r_hi,
                     "nested_cv_fold_r_mean": c.fold_r_mean, "nested_cv_fold_r_sd": c.fold_r_sd,
                     "nested_cv_folds": int(c.n_folds), "drop": s.r - c.repeat_r_mean})
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{lab}*" if cont else lab for _, lab, cont, _ in FEATURES])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, len(FEATURES) - 0.4)
    ax.set_xlim(0.05, 0.56)
    ax.text(0.555, len(FEATURES) - 0.45, "Δ", ha="right", va="bottom", fontsize=5.8, color=PAL["muted"])
    ax.set_xlabel("Prediction accuracy for IST (r)")
    ax.set_title("Single split vs nested cross-validation", loc="left")
    ax.plot([], [], "o", mfc="white", mec=PAL["ink"], mew=0.9, ms=3.6, label="single split, n = 133 (bootstrap 95 % CI)")
    ax.plot([], [], "o", color=PAL["ink"], ms=3.8, label="nested CV 10 × 5, n = 877 (2.5–97.5 % over repeats)")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.16), ncol=1, handletextpad=0.3, fontsize=5.6)
    ax.text(0.0, -0.335, "* IST was a training target during pre-training: these nested-CV values are upper bounds.",
            transform=ax.transAxes, fontsize=5.4, color=PAL["muted"])
    REC.data("D", "single_split_vs_cv", pd.DataFrame(rows),
             "IST prediction accuracy from each feature set scored two ways: on the single published split (trained "
             "on 744, tested on 133; bootstrap 95 % interval) and by repeated nested cross-validation over all 877 "
             "subjects (10 repeats x 5 folds; point = mean over repeats of the pooled out-of-fold r, interval = "
             "2.5th-97.5th percentile over repeats, which describes repeat-to-repeat stability rather than "
             "sampling uncertainty). The VAE embeddings were learned with IST as an auxiliary target, so their "
             "cross-validated values are upper bounds.",
             {"features": "feature set", "label": "label in the figure",
              "target_seen_in_pretraining": "IST used as a training target when the embedding was learned",
              "single_split_r": "r on the 133-subject test set", "single_split_ci_low": "bootstrap 95 % lower",
              "single_split_ci_high": "bootstrap 95 % upper", "single_split_n_test": "test subjects",
              "nested_cv_r": "mean over repeats of pooled out-of-fold r",
              "nested_cv_repeat_p2.5": "2.5th percentile over repeats",
              "nested_cv_repeat_p97.5": "97.5th percentile over repeats",
              "nested_cv_fold_r_mean": "mean fold-level r", "nested_cv_fold_r_sd": "SD of fold-level r",
              "nested_cv_folds": "outer folds", "drop": "single-split r minus nested-CV r"})


def main():
    apply_style()
    fig = plt.figure(figsize=(WIDTH["double"], 13.0 / 2.54))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.58, 1.0], width_ratios=[0.82, 1.18], hspace=0.42, wspace=0.62,
                          left=0.085, right=0.975, top=0.915, bottom=0.165)
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_a, ax_b = panels_abc(fig, gs[0, :], ax_c)
    panel_d(ax_d)
    panel_label(ax_a, "A", x=-0.3, y=1.2)
    panel_label(ax_b, "B", x=-0.3, y=1.2)
    panel_label(ax_c, "C", x=-0.29, y=1.03)
    panel_label(ax_d, "D", x=-0.5, y=1.03)
    out = REC.save(fig)
    print("saved", *[p.relative_to(ROOT) for p in out])


if __name__ == "__main__":
    main()
