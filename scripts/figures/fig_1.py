#!/usr/bin/env python
"""
Figure 1 — How intelligence prediction was evaluated, and why it matters.

A  Cohorts and data flow (schematic; counts read from the hold-out manifest and
   the PIOP frozen-prediction logs).
B  The prediction pipeline drawn with real data: the Schaefer-200 parcellation on
   the inflated cortex coloured by Yeo-7 network (plus 15 subcortical regions),
   percent-signal-change series of one discovery subject, the discovery-set mean
   shrunk covariance (shrinkage 0.55, regions ordered by network), and that
   subject's log-Euclidean tangent matrix. Morphometry and diffusion FA join the
   connectivity model in a stacked ridge; the confound model is a separate
   benchmark.
C  False-positive rate of the naive vs corrected paired t-test on fold scores,
   under a true null, for this project's designs (calibration table measured in
   tests/test_cv_inference.py and recorded in src/stats/cv_inference.py).
D  Naive vs corrected p for every model comparison in FINAL_RESULTS.csv.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, FancyBboxPatch

from src.visualization import brain
from src.visualization.journal import (ARM, COLLECTION, DIVERGING, NETWORK, NETWORK_LABEL, PAL, WIDTH, FigureRecord,
                                       apply_style, panel_label)

REC = FigureRecord(
    name="Fig1", title="How intelligence prediction was evaluated, and why it matters",
    claim="Every claim in the paper is judged by repeated nested cross-validation with a corrected "
          "resampled t-test and one sealed test set; the naive fold-level t-test is badly "
          "anti-conservative in this design and would have doubled the number of significant model "
          "differences.")

NL = "\n"
SHRINK = 0.55                                      # multimodal_tuned.py, SHRINK_NEW
NET_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default", "Subcortex"]
TINT = {"ID1000": "#DCE6F4", "PIOP1": "#D3EFEA", "PIOP2": "#FCEBD2", "neutral": "#EEF1F4",
        "morph": "#D3EFEA", "fa": "#EADCEB", "stack": "#FBE0D8"}


def calibration_table() -> pd.DataFrame:
    """Parse the measured calibration table from the inference module's docstring."""
    text = REC.source("src/stats/cv_inference.py").read_text(encoding="utf-8")
    rows = re.findall(r"^\s+(\d+)\s+(\dx\d)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$", text, re.M)
    df = pd.DataFrame(rows, columns=["n", "cv", "fpr_naive", "fpr_corrected", "fpr_bayes_false_better"])
    for c in ("n", "fpr_naive", "fpr_corrected", "fpr_bayes_false_better"):
        df[c] = pd.to_numeric(df[c])
    assert len(df) == 5, df
    return df


def box(ax, xy, w, h, text, fc, ec="none", fontsize=5.9, color=PAL["ink"], weight="normal", ls="-"):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.006,rounding_size=0.025",
                                fc=fc, ec=ec, lw=0.7, ls=ls, transform=ax.transAxes, clip_on=False))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, color=color,
            transform=ax.transAxes, linespacing=1.15, fontweight=weight)


def arrow(ax, p0, p1, color=PAL["ink"], lw=0.7):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=6, lw=lw, color=color,
                                 transform=ax.transAxes, clip_on=False, shrinkA=0, shrinkB=0))


def link(fig, ax0, ax1, y0=0.5, y1=0.5, color=PAL["ink"]):
    """Arrow from the right edge of one tile to the left edge of the next."""
    fig.add_artist(ConnectionPatch((1.05, y0), (-0.05, y1), "axes fraction", "axes fraction", axesA=ax0, axesB=ax1,
                                   arrowstyle="-|>", mutation_scale=6, lw=0.7, color=color))


def panel_a(ax):
    man = json.loads(REC.source("outputs/honest/holdout_manifest.json").read_text())
    piop = {}
    for ds in ("PIOP1", "PIOP2"):
        txt = REC.source(f"outputs/piop/logs/{ds}_frozen.log").read_text(encoding="utf-8", errors="ignore")
        piop[ds] = int(re.search(r"n = (\d+) subjects", txt).group(1))
    ax.set_axis_off()
    box(ax, (0.00, 0.55), 0.29, 0.40,
        NL.join(["AOMIC-ID1000", f"{man['n_total']} adults, 19–26 y", "11-min movie fMRI", "IST total"]),
        TINT["ID1000"], ec=COLLECTION["ID1000"])
    box(ax, (0.39, 0.77), 0.26, 0.18, NL.join(["Discovery", f"n = {man['n_discovery']}"]), TINT["neutral"])
    box(ax, (0.39, 0.55), 0.26, 0.18, NL.join(["Sealed hold-out", f"n = {man['n_holdout']}"]), "white",
        ec=ARM["stack"])
    arrow(ax, (0.29, 0.80), (0.39, 0.86))
    arrow(ax, (0.29, 0.70), (0.39, 0.64))
    box(ax, (0.74, 0.77), 0.26, 0.18, NL.join(["Fit frozen", "models"]), TINT["neutral"])
    box(ax, (0.74, 0.55), 0.26, 0.18, "Evaluated once", "white", ec=ARM["stack"])
    arrow(ax, (0.65, 0.86), (0.74, 0.86))
    arrow(ax, (0.65, 0.64), (0.74, 0.64))
    arrow(ax, (0.87, 0.77), (0.87, 0.73), color=ARM["stack"])
    ax.text(0.39, 0.465, "pipeline chosen by CV on all 877 before sealing", fontsize=5.5,
            color=ARM["stack"], transform=ax.transAxes, style="italic")
    ax.text(0.0, 0.355, "Independent collections: other scanners, tasks and intelligence test", fontsize=5.8,
            color=PAL["muted"], transform=ax.transAxes, style="italic")
    box(ax, (0.00, 0.00), 0.29, 0.30, NL.join(["AOMIC-PIOP1", f"n = {piop['PIOP1']}, 6 runs", "Raven APM"]),
        TINT["PIOP1"], ec=COLLECTION["PIOP1"])
    box(ax, (0.36, 0.00), 0.29, 0.30, NL.join(["AOMIC-PIOP2", f"n = {piop['PIOP2']}, 4 runs", "Raven APM"]),
        TINT["PIOP2"], ec=COLLECTION["PIOP2"])
    box(ax, (0.74, 0.00), 0.26, 0.30, NL.join(["Frozen pipeline,", "never used for", "selection"]),
        "white", ec=PAL["slate"])
    arrow(ax, (0.65, 0.15), (0.74, 0.15))
    REC.data("A", "cohorts", pd.DataFrame([
        {"collection": "ID1000", "role": "all with fMRI and IST", "n": man["n_total"]},
        {"collection": "ID1000", "role": "discovery", "n": man["n_discovery"]},
        {"collection": "ID1000", "role": "sealed hold-out", "n": man["n_holdout"]},
        {"collection": "PIOP1", "role": "independent collection", "n": piop["PIOP1"]},
        {"collection": "PIOP2", "role": "independent collection", "n": piop["PIOP2"]}]),
        "Cohort sizes shown in the schematic.",
        {"collection": "AOMIC collection", "role": "role in the analysis", "n": "subjects"})
    return man


def region_networks() -> list[str]:
    return brain.parcel_networks() + ["Subcortex"] * 15


def pipeline_data(discovery: list[str]):
    """Mean shrunk covariance over discovery subjects, and one subject's series and tangent matrix."""
    with h5py.File(REC.source("outputs/connectivity/unscrubbed_ts.h5"), "r") as f:
        ids = [s for s in sorted(discovery) if s in f["subjects"]]
        example = ids[0]
        logs, covs = [], []
        for s in ids:
            x = f["subjects"][s]["time_series"][:].astype(np.float64)
            x = x - x.mean(0, keepdims=True)
            S = x.T @ x / (len(x) - 1)
            C = (1 - SHRINK) * S + SHRINK * np.trace(S) / S.shape[0] * np.eye(S.shape[0])
            w, V = np.linalg.eigh(C)
            logs.append((V * np.log(np.clip(w, 1e-10, None))) @ V.T)
            covs.append(C)
            if s == example:
                series = f["subjects"][s]["time_series"][:].astype(np.float64)
                example_log = logs[-1]
    reference = np.mean(logs, axis=0)                # log-Euclidean mean
    return ids, example, series, np.mean(covs, axis=0), example_log - reference


def network_strip(ax, order_nets, horizontal):
    codes = np.array([NET_ORDER.index(n) for n in order_nets])[None, :]
    cmap = ListedColormap([NETWORK[n] for n in NET_ORDER])
    ax.imshow(codes if horizontal else codes.T, cmap=cmap, vmin=-0.5, vmax=len(NET_ORDER) - 0.5, aspect="auto",
              interpolation="nearest")
    ax.set_axis_off()


def matrix_tile(fig, cell, M, order, nets, title):
    sub = cell.subgridspec(3, 2, width_ratios=[0.06, 1], height_ratios=[0.06, 1, 0.22], wspace=0.03, hspace=0.03)
    ax = fig.add_subplot(sub[1, 1])
    A = M[np.ix_(order, order)].copy()
    np.fill_diagonal(A, np.nan)
    v = np.nanpercentile(np.abs(A), 98)
    ax.imshow(A, cmap=DIVERGING, vmin=-v, vmax=v, interpolation="nearest", aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    network_strip(fig.add_subplot(sub[0, 1]), [nets[i] for i in order], True)
    network_strip(fig.add_subplot(sub[1, 0]), [nets[i] for i in order], False)
    ax.text(0.5, -0.06, title, transform=ax.transAxes, ha="center", va="top", fontsize=5.8, linespacing=1.15)
    return ax


def panel_b(fig, cell, man):
    nets = region_networks()
    order = sorted(range(len(nets)), key=lambda i: (NET_ORDER.index(nets[i]), i))
    ids, example, series, mean_cov, tangent = pipeline_data(man["discovery"])

    outer = cell.subgridspec(2, 1, height_ratios=[1.0, 0.36], hspace=0.28)
    top = outer[0].subgridspec(1, 5, width_ratios=[1.0, 1.0, 0.85, 0.85, 0.66], wspace=0.38)

    # parcellation on the cortex
    brain_cell = top[0, 0].subgridspec(3, 1, height_ratios=[1, 1, 0.22], hspace=0.02)
    ax_l, ax_m = fig.add_subplot(brain_cell[0, 0]), fig.add_subplot(brain_cell[1, 0])
    brain.place(ax_l, brain.render_networks("left", "lateral"))
    brain.place(ax_m, brain.render_networks("left", "medial"))
    ax_m.text(0.5, -0.04, "215 regions", transform=ax_m.transAxes, ha="center", va="top", fontsize=5.8)

    # percent-signal-change series of five regions from different networks
    ts_cell = top[0, 1].subgridspec(3, 1, height_ratios=[0.1, 1, 0.3], hspace=0.0)
    ax_ts = fig.add_subplot(ts_cell[1, 0])
    picks = [next(i for i, n in enumerate(nets) if n == net) + k for net, k in
             (("Vis", 4), ("SomMot", 6), ("DorsAttn", 3), ("Cont", 5), ("Default", 8))]
    t = np.arange(series.shape[0]) * 2.2 / 60
    for j, i in enumerate(picks):
        ax_ts.plot(t, series[:, i] / (4 * series[:, picks].std()) - j, color=NETWORK[nets[i]], lw=0.45)
    ax_ts.set_xlim(0, t[-1])
    ax_ts.set_ylim(-len(picks) + 0.3, 0.8)
    ax_ts.set_yticks([])
    ax_ts.spines["left"].set_visible(False)
    ax_ts.set_xticks([0, 5, 10])
    ax_ts.tick_params(labelsize=5.3, length=1.8, pad=1)
    ax_ts.set_xlabel("minutes", fontsize=5.6, labelpad=1)
    ax_ts.text(0.5, 1.03, "% signal change", transform=ax_ts.transAxes, ha="center", va="bottom", fontsize=5.8)

    ax_cov = matrix_tile(fig, top[0, 2], mean_cov, order, nets, "shrunk\ncovariance")
    ax_tan = matrix_tile(fig, top[0, 3], tangent, order, nets, "log-Euclidean\ntangent")

    ax_ridge = fig.add_subplot(top[0, 4])
    ax_ridge.set_axis_off()
    box(ax_ridge, (0.0, 0.36), 1.0, 0.34, "Ridge", TINT["ID1000"], ec=ARM["fc"], fontsize=6.2, weight="bold")
    ax_ridge.text(0.5, 0.3, "nested CV", transform=ax_ridge.transAxes, ha="center", va="top", fontsize=5.5,
                  color=PAL["muted"])

    link(fig, ax_l, ax_ts, y0=0.0, y1=0.5)
    for a, b in ((ax_ts, ax_cov), (ax_cov, ax_tan), (ax_tan, ax_ridge)):
        link(fig, a, b, y0=0.5, y1=0.53)

    # structural blocks join the connectivity model in the stack; the confound model stands apart
    ax_key = fig.add_subplot(outer[1])
    ax_key.set_axis_off()
    box(ax_key, (0.00, 0.42), 0.33, 0.58, "Confound model (benchmark):\nsex, age, brain volume, motion", "white",
        ec=PAL["slate"], fontsize=5.6, ls=(0, (2, 1.5)))
    box(ax_key, (0.40, 0.42), 0.18, 0.58, "FreeSurfer\nmorphometry", TINT["morph"], ec=ARM["morph"])
    ax_key.text(0.595, 0.71, "+", transform=ax_key.transAxes, ha="center", va="center", fontsize=8)
    box(ax_key, (0.61, 0.42), 0.16, 0.58, "Diffusion FA", TINT["fa"], ec=ARM["fa"])
    ax_key.text(0.785, 0.71, "+", transform=ax_key.transAxes, ha="center", va="center", fontsize=8)
    box(ax_key, (0.80, 0.42), 0.20, 0.58, "Stacked\nridge", TINT["stack"], ec=ARM["stack"], fontsize=6.1,
        weight="bold")
    fig.add_artist(ConnectionPatch((0.5, 0.1), (0.9, 1.0), "axes fraction", "axes fraction", axesA=ax_ridge,
                                   axesB=ax_key, arrowstyle="-|>", mutation_scale=6, lw=0.7, color=PAL["ink"]))
    handles = [plt.Rectangle((0, 0), 1, 1, color=NETWORK[n]) for n in NET_ORDER]
    ax_key.legend(handles, [NETWORK_LABEL[n].replace("Dorsal attention", "Dorsal attn.") for n in NET_ORDER],
                  loc="lower left", bbox_to_anchor=(-0.01, -0.28), ncol=8, fontsize=5.2, handlelength=0.8,
                  handleheight=0.8, handletextpad=0.3, columnspacing=0.75, borderaxespad=0)

    iu = np.triu_indices(len(nets), k=1)
    reg = pd.DataFrame({"region": np.arange(len(nets)), "network": nets, "plot_order": np.argsort(order)})
    REC.data("B", "regions", reg,
             "Region index (200 Schaefer-7Network parcels in atlas order, then 15 Harvard-Oxford subcortical regions), "
             "its network, and its position in the network-ordered matrices.",
             {"region": "index in the 215-region series", "network": "Yeo-7 network or Subcortex",
              "plot_order": "row/column position in the drawn matrices"})
    REC.data("B", "example_series", pd.DataFrame(series[:, picks], columns=[f"region_{i}" for i in picks])
             .assign(minute=t),
             f"Percent-signal-change series (unscrubbed, 290 volumes, TR 2.2 s) of five regions for the first "
             f"discovery subject in sorted order ({example}); drawn scaled and offset for display.",
             {**{f"region_{i}": f"{nets[i]} region {i}" for i in picks}, "minute": "time from run start"})
    REC.data("B", "matrices", pd.DataFrame({"region_i": iu[0], "region_j": iu[1],
                                            "mean_shrunk_covariance": mean_cov[iu], "example_tangent": tangent[iu]}),
             f"Mean over {len(ids)} discovery subjects of the shrunk covariance ((1 - 0.55) S + 0.55 tr(S)/p I), and "
             f"the example subject's log-Euclidean tangent matrix logm(C) minus the discovery mean of logm(C). The "
             f"pipeline itself estimates that reference inside each training fold; this drawing uses all discovery "
             f"subjects. Colour limits: 98th percentile of |off-diagonal|; diagonals not drawn.",
             {"region_i": "row region", "region_j": "column region",
              "mean_shrunk_covariance": "mean shrunk covariance (PSC units squared)",
              "example_tangent": "tangent-space coordinate of the example subject"})
    return ax_l


def panel_c(ax):
    cal = calibration_table()
    x = np.arange(len(cal))
    w = 0.38
    ax.bar(x - w / 2, cal.fpr_naive, w, color=PAL["mist"], ec="none", label="Naive paired t-test")
    ax.bar(x + w / 2, cal.fpr_corrected, w, color=PAL["cobalt"], ec="none", label="Corrected resampled t-test")
    ax.axhline(0.05, color=ARM["stack"], lw=0.7, ls=(0, (3, 2)))
    ax.text(-0.45, 0.065, "nominal 0.05", fontsize=5.5, color=ARM["stack"], ha="left", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([f"n = {n}{NL}{cv} CV" for n, cv in zip(cal.n, cal.cv)])
    ax.set_ylabel("False-positive rate (true null)")
    ax.set_ylim(0, 0.72)
    ax.legend(loc="upper left", ncol=1)
    ax.set_title("Fold scores are not independent", loc="left")
    REC.data("C", "calibration", cal,
             "Measured false-positive rate at alpha = 0.05 when two exchangeable feature sets are compared "
             "by ridge under a true null (400 simulated datasets per design), from the calibration table of "
             "src/stats/cv_inference.py (tests/test_cv_inference.py).",
             {"n": "subjects per simulated dataset", "cv": "repeats x folds",
              "fpr_naive": "false-positive rate of the naive paired t-test on fold scores",
              "fpr_corrected": "false-positive rate of the corrected resampled t-test (Nadeau-Bengio, repeated k-fold)",
              "fpr_bayes_false_better": "rate at which the Bayesian correlated t-test declares a false winner"})


def panel_d(ax):
    fr = pd.read_csv(REC.source("outputs/honest/FINAL_RESULTS.csv"))
    c = fr[fr.p.notna() & fr.p_naive.notna() & (fr.verdict != "identical")].copy()
    c["class"] = np.select([(c.p_naive < 0.05) & (c.p < 0.05), (c.p_naive < 0.05) & (c.p >= 0.05)],
                           ["significant under both", "significant only naively"], "not significant")
    colors = {"significant under both": PAL["cobalt"], "significant only naively": ARM["stack"],
              "not significant": PAL["slate"]}
    for k in ("not significant", "significant only naively", "significant under both"):
        g = c[c["class"] == k]
        ax.scatter(g.p_naive.clip(lower=1e-16), g.p.clip(lower=1e-8), s=9, color=colors[k], lw=0,
                   alpha=0.85, label=f"{k} ({len(g)})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    lims = (1e-16, 1.5)
    ax.plot(lims, lims, color=PAL["ink"], lw=0.5, ls=(0, (2, 2)))
    ax.axvline(0.05, color=PAL["slate"], lw=0.5)
    ax.axhline(0.05, color=PAL["slate"], lw=0.5)
    ax.set_xlim(*lims)
    ax.set_ylim(1e-8, 1.5)
    ax.set_xlabel("Naive p (paired t-test)")
    ax.set_ylabel("Corrected p")
    n_naive, n_corr = int((c.p_naive < 0.05).sum()), int((c.p < 0.05).sum())
    ax.set_title(f"{n_naive} → {n_corr} of {len(c)} comparisons significant", loc="left")
    ax.legend(loc="lower right", handletextpad=0.2, borderaxespad=0.1)
    REC.data("D", "naive_vs_corrected_p", c[["group", "arm", "n_subjects", "n_folds", "delta_vs_ref", "p_naive",
                                              "p", "verdict", "class"]],
             "Every paired model comparison in FINAL_RESULTS.csv (arms identical to their reference excluded). "
             "Plotted p-values are floored at 1e-16 (naive) and 1e-8 (corrected) for display only; the CSV "
             "holds the unfloored values.",
             {"group": "comparison family", "arm": "model compared with the family reference",
              "n_subjects": "subjects", "n_folds": "fold scores (repeats x 5)",
              "delta_vs_ref": "mean difference in r (arm - reference)", "p_naive": "naive paired t-test p",
              "p": "corrected resampled t-test p", "verdict": "Bayesian correlated t-test verdict (ROPE +/-0.01 r)",
              "class": "significance under naive and corrected tests"})


def main():
    apply_style()
    fig = plt.figure(figsize=(WIDTH["double"], 12.5 / 2.54))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.95, 1.0], width_ratios=[0.84, 1.16], hspace=0.36, wspace=0.12,
                          left=0.07, right=0.99, top=0.935, bottom=0.1)
    ax_a = fig.add_subplot(gs[0, 0])
    bottom = gs[1, :].subgridspec(1, 2, wspace=0.3)
    ax_c, ax_d = fig.add_subplot(bottom[0, 0]), fig.add_subplot(bottom[0, 1])
    man = panel_a(ax_a)
    ax_brain = panel_b(fig, gs[0, 1], man)
    panel_c(ax_c)
    panel_d(ax_d)
    ax_a.set_title("Cohorts and the sealed test set", loc="left", pad=6)
    ax_brain.set_title("Prediction pipeline", loc="left", pad=6, x=0.05)
    for ax, l, x in ((ax_a, "A", -0.08), (ax_brain, "B", -0.16), (ax_c, "C", -0.16), (ax_d, "D", -0.16)):
        panel_label(ax, l, x=x, y=1.02)
    out = REC.save(fig)
    print("saved", *[p.relative_to(ROOT) for p in out])


if __name__ == "__main__":
    main()
