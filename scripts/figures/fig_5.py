#!/usr/bin/env python
"""
Figure 5 — A distributed, spatially structured pattern that partly generalises.

Top row, the connectivity model:
A  Regional strength of the Haufe forward weights (sum of |weight| over each region's
   edges, relative to the mean region) on the inflated cortex, both hemispheres.
   Descriptive: inference is made at the network level in B.
B  Network-pair enrichment of the same weights: z against a surface spin null
   (fsaverage5, 10,000 rotations); lower triangle raw target, upper triangle
   deconfounded; markers for one-sided p < 0.05 and BH-FDR q < 0.05 over the 28
   spin-testable cortical pairs. Subcortex cannot be spun.
C  Leave-one-collection-out prediction: a model trained on the other two collections
   (naive pooling), Spearman with bootstrap 95 % interval; open markers:
   cross-validated accuracy within the collection alone.

Bottom row, the regional amplitude pattern:
D  Amplitude forward maps (z-scored across regions) for ID1000 movie, PIOP1 working
   memory and PIOP2 working memory, both hemispheres.
E  Similarity of all 11 amplitude maps across collections and tasks (lower triangle),
   with collection-block means tested against variogram-matched surrogates.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

from src.visualization import brain
from src.visualization.journal import (COLLECTION, DIVERGING, NETWORK, PAL, SEQUENTIAL, WIDTH, FigureRecord,
                                       apply_style, cell_label, p_text, style_colorbar)

warnings.filterwarnings("ignore")

REC = FigureRecord(
    name="Fig5", title="A distributed, spatially structured pattern that partly generalises",
    claim="The predictive connectivity weights are spread across the cortex and enriched at the level of large-scale "
          "networks beyond spatial autocorrelation (within the default network and between control and default "
          "networks); a model trained on two collections predicts the third in every case; and the regional "
          "amplitude pattern agrees between ID1000 and PIOP1 but not PIOP2.")

NETS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default", "Subcortex"]
NET_SHORT = {"Vis": "Visual", "SomMot": "Somatomotor", "DorsAttn": "Dorsal attn", "SalVentAttn": "Salience",
             "Limbic": "Limbic", "Cont": "Control", "Default": "Default", "Subcortex": "Subcortex"}
MAPS = [("ID1000/moviewatching", "ID1000\nmovie"), ("PIOP1/workingmemory", "PIOP1\nWM"),
        ("PIOP2/workingmemory", "PIOP2\nWM")]
TASK_SHORT = {"moviewatching": "movie", "anticipation": "anticip.", "emomatching": "emo-match", "faces": "faces",
              "gstroop": "g-Stroop", "restingstate": "rest", "workingmemory": "WM", "stopsignal": "stop"}
VIEW_LABEL = {("left", "lateral"): "L lateral", ("left", "medial"): "L medial", ("right", "medial"): "R medial",
              ("right", "lateral"): "R lateral"}
LABEL_INK = {**NETWORK, "Limbic": "#7E9A3A"}      # the pale limbic colour is too light for text


def panel_a(fig, cell):
    reg = pd.read_csv(REC.source("outputs/honest/psc/haufe_regions.csv"))
    rel = reg["abs_weight_raw"].to_numpy() / reg["abs_weight_raw"].mean()
    cortex = rel[:brain.N_CORTICAL]
    lo, hi = np.percentile(cortex, [5, 95])
    sub = cell.subgridspec(2, 1, height_ratios=[1, 0.06], hspace=0.3)
    tex = brain.textures(cortex)
    grid = sub[0].subgridspec(2, 2, hspace=0.06, wspace=0.03)
    layout = [[("left", "lateral"), ("right", "lateral")], [("left", "medial"), ("right", "medial")]]
    axes = []
    for i, row in enumerate(layout):
        for j, (hemi, view) in enumerate(row):
            ax = fig.add_subplot(grid[i, j])
            brain.place(ax, brain.render(tex[hemi], hemi, view, SEQUENTIAL, lo, hi))
            axes.append(ax)
    cax = fig.add_subplot(sub[1].subgridspec(1, 3, width_ratios=[0.2, 0.6, 0.2])[0, 1])
    sm = plt.cm.ScalarMappable(cmap=SEQUENTIAL, norm=plt.Normalize(lo, hi))
    ticks = list(np.round(np.arange(np.ceil(lo * 10) / 10, hi + 1e-9, 0.1), 1))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=ticks)
    style_colorbar(cb, "forward-weight strength (× mean region)", 5.8)
    REC.data("A", "regional_strength", reg.assign(relative_strength=rel),
             "Sum of absolute Haufe forward weights over each region's 214 edges for the PSC connectivity ridge model "
             "(raw and deconfounded targets), and the raw-target value relative to the mean region, which is drawn "
             "on the inflated fsaverage5 surface (5th-95th percentile colour limits; the 15 subcortical regions are "
             "not drawn). Descriptive: no region-level test is made.",
             {"roi": "region index (200 Schaefer, then 15 subcortical)", "network": "Yeo-7 network or Subcortex",
              "abs_weight_raw": "sum |forward weight|, raw target", "abs_weight_deconfounded": "same, deconfounded",
              "abs_weight_deconf_motion": "same, deconfounded with extended motion model",
              "relative_strength": "abs_weight_raw divided by its mean over the 215 regions"})
    return axes[0]


def panel_b(ax, cax):
    sn = pd.read_csv(REC.source("outputs/honest/spatial_null_surface_haufe.csv"))
    k = len(NETS)
    Z = np.full((k, k), np.nan)
    marks = []
    for r in sn.itertuples():
        i, j = NETS.index(r.net_i), NETS.index(r.net_j)
        lo, hi = max(i, j), min(i, j)
        row, col = (lo, hi) if r.target == "raw" else (hi, lo)
        if not r.spin_valid or (r.target != "raw" and i == j):
            continue
        Z[row, col] = r.z_surface_spin
        if r.q_surface_spin_upper < 0.05:
            marks.append((row, col, "●"))
        elif r.p_surface_spin_upper < 0.05:
            marks.append((row, col, "○"))
    im = ax.imshow(Z, cmap=DIVERGING, norm=TwoSlopeNorm(0, -5, 5), aspect="equal")
    for i in range(k):
        for j in range(k):
            if NETS[i] == "Subcortex" or NETS[j] == "Subcortex":
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc="#EEF1F4", ec="white", lw=0.3, hatch="////"))
    for row, col, m in marks:
        ax.text(col, row, m, ha="center", va="center", fontsize=6 if m == "●" else 5.5, color=PAL["ink"])
    ax.plot([-0.5, k - 0.5], [-0.5, k - 0.5], color=PAL["ink"], lw=0.5)
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels([NET_SHORT[n] for n in NETS], rotation=45, ha="right", fontsize=5.6)
    ax.set_yticklabels([NET_SHORT[n] for n in NETS], fontsize=5.6)
    for labs in (ax.get_xticklabels(), ax.get_yticklabels()):
        for lab, n in zip(labs, NETS):
            lab.set_color(LABEL_INK[n])
    ax.tick_params(length=0, pad=1.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(0.02, 0.01, "raw", transform=ax.transAxes, fontsize=5.2, color=PAL["muted"], va="bottom")
    ax.text(0.98, 0.99, "deconfounded", transform=ax.transAxes, fontsize=5.2, color=PAL["muted"], va="top",
            ha="right")
    cb = plt.colorbar(im, cax=cax, ticks=[-4, -2, 0, 2, 4])
    style_colorbar(cb, "z vs surface spin", 5.8)
    cax.text(0.0, -0.08, "● q < 0.05\n○ p < 0.05", transform=cax.transAxes, ha="left", va="top", fontsize=5.2,
             linespacing=1.35)
    keep = sn[["target", "net_i", "net_j", "mean_abs_weight", "z_surface_spin", "p_surface_spin_upper",
               "q_surface_spin_upper", "p_surface_spin_two_sided", "p_centroid_spin_upper",
               "p_label_perm_two_sided", "spin_valid"]]
    REC.data("B", "network_enrichment", keep,
             "Mean absolute Haufe forward weight of the PSC FC ridge model per network pair, z against 10,000 "
             "surface spins (Schaefer-200 on fsaverage5, Hungarian assignment); one-sided enrichment p with BH-FDR "
             "over the 28 cortical pairs (filled: q < 0.05; open: p < 0.05), alongside the two-sided spin p, the "
             "earlier centroid spin and the ROI-label permutation. Hatched: pairs with subcortex, not spin-testable.",
             {c: "" for c in keep.columns})


def panel_c(ax):
    lo = pd.read_csv(REC.source("outputs/piop/leave_one_dataset_out.csv"))
    pool = pd.read_csv(REC.source("outputs/piop/riemannian_pooling_folds.csv"))
    alone = pool.groupby("target")["none_alone"].mean()
    rows = []
    for i, held in enumerate(("ID1000", "PIOP1", "PIOP2")):
        r = lo[(lo.held_out == held) & (lo.arm == "none")].iloc[0]
        c = COLLECTION[held]
        ax.plot([i, i], [r.ci_lo, r.ci_hi], color=c, lw=1.4, solid_capstyle="round")
        ax.plot(i, r.spearman, "o", color=c, ms=5, mec="white", mew=0.6, zorder=3)
        ax.plot(i + 0.28, alone[held], "o", color=c, ms=4.2, mfc="white", mew=1.0, zorder=3)
        ax.text(i, r.ci_hi + 0.025, f"n = {int(r.n_test)}", ha="center", fontsize=5.3, color=PAL["muted"])
        rows.append({"held_out": held, "n_train": int(r.n_train), "n_test": int(r.n_test),
                     "spearman": r.spearman, "ci_low": r.ci_lo, "ci_high": r.ci_hi,
                     "spearman_deconfounded": r.spearman_dec, "within_collection_cv_r": alone[held]})
    ax.axhline(0, color=PAL["ink"], lw=0.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["ID1000\nIST", "PIOP1\nRaven", "PIOP2\nRaven"], fontsize=6)
    ax.set_ylabel("Accuracy in held-out collection")
    ax.set_xlim(-0.45, 2.55)
    ax.set_ylim(-0.05, 0.62)
    ax.plot([], [], "o", color=PAL["ink"], ms=4, label="trained on the other two")
    ax.plot([], [], "o", color=PAL["ink"], ms=4, mfc="white", mew=0.9, label="CV within collection")
    ax.legend(loc="upper left", handletextpad=0.2, fontsize=5.6, borderaxespad=0)
    ax.set_title("Transfer to a new collection", loc="left")
    REC.data("C", "leave_one_collection_out", pd.DataFrame(rows),
             "Connectivity model trained on two collections (targets rank-normalised within collection, naive "
             "pooling, the pre-specified arm) and tested on the third; Spearman with bootstrap 95 % interval. "
             "within_collection_cv_r: mean fold r of the same model cross-validated within that collection alone "
             "(riemannian_pooling_folds, none_alone).",
             {"held_out": "collection held out", "n_train": "training subjects", "n_test": "test subjects",
              "spearman": "Spearman rho in the held-out collection", "ci_low": "bootstrap 95 % lower",
              "ci_high": "bootstrap 95 % upper", "spearman_deconfounded": "same, deconfounded target",
              "within_collection_cv_r": "mean fold r within the collection"})


def panel_d(fig, cell):
    maps = pd.read_csv(REC.source("outputs/piop/amplitude_forward_maps.csv"), index_col=0)
    vmax = 2.5
    sub = cell.subgridspec(1, 2, width_ratios=[1, 0.03], wspace=0.05)
    layout = list(brain.VIEWS)
    grid = sub[0, 0].subgridspec(len(MAPS), len(layout), hspace=0.05, wspace=0.02)
    axes = np.empty((len(MAPS), len(layout)), dtype=object)
    data = []
    for i, (col, lab) in enumerate(MAPS):
        v = maps[col].to_numpy(float)
        z = (v - v.mean()) / v.std()
        tex = brain.textures(z)
        data.append(pd.DataFrame({"roi": np.arange(len(v)), "map": col, "forward_weight": v, "z": z}))
        for j, (hemi, view) in enumerate(layout):
            ax = fig.add_subplot(grid[i, j])
            brain.place(ax, brain.render(tex[hemi], hemi, view, DIVERGING, -vmax, vmax))
            axes[i, j] = ax
        axes[i, 0].text(-0.04, 0.5, lab, transform=axes[i, 0].transAxes, ha="right", va="center", fontsize=6,
                        color=COLLECTION[col.split("/")[0]], fontweight="bold", linespacing=1.1)
    for j, hv in enumerate(layout):
        axes[0, j].text(0.5, 1.03, VIEW_LABEL[hv], transform=axes[0, j].transAxes, ha="center", va="bottom",
                        fontsize=5.6, color=PAL["muted"])
    cax = fig.add_subplot(sub[0, 1].subgridspec(3, 1, height_ratios=[0.22, 0.56, 0.22])[1, 0])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=DIVERGING, norm=plt.Normalize(-vmax, vmax)), cax=cax,
                      ticks=[-2, 0, 2])
    style_colorbar(cb, "forward weight (z)", 5.8)
    REC.data("D", "amplitude_maps", pd.concat(data, ignore_index=True),
             "Haufe forward weights of the regional-amplitude ridge model for three cells (215 regions: 200 "
             "Schaefer cortical parcels, then 15 subcortical), z-scored across regions within each map and drawn on "
             "the inflated fsaverage5 surface (colour limits +/-2.5; subcortex not drawn). Parcels were projected "
             "by the most frequent label along each vertex's cortical depth; unlabelled sulcal vertices took the "
             "most frequent label of their mesh neighbours (display only).",
             {"roi": "region index in extraction order", "map": "collection/task",
              "forward_weight": "Haufe forward weight", "z": "weight z-scored across the 215 regions"})
    return axes[0, 0]


def panel_e(ax, cax):
    M = pd.read_csv(REC.source("outputs/piop/amplitude_map_similarity.csv"), index_col=0)
    maps = pd.read_csv(REC.source("outputs/piop/amplitude_forward_maps.csv"), index_col=0)
    names = list(maps.columns)
    assert len(names) == M.shape[0]
    R = M.to_numpy(float).copy()
    R[np.triu_indices_from(R)] = np.nan
    cmap = DIVERGING.copy()
    cmap.set_bad((1, 1, 1, 0))                       # the empty upper triangle carries the block statistics
    im = ax.imshow(R, cmap=cmap, vmin=-0.8, vmax=0.8, aspect="equal")
    short = [n.split("/")[0].replace("PIOP", "P") + " " + TASK_SHORT[n.split("/")[1]] for n in names]
    coll = [n.split("/")[0] for n in names]
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(short, rotation=55, ha="right", fontsize=5.3)
    ax.set_yticklabels(short, fontsize=5.3)
    for labs in (ax.get_xticklabels(), ax.get_yticklabels()):
        for lab, c in zip(labs, coll):
            lab.set_color(COLLECTION[c])
    ax.tick_params(length=0, pad=1.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    n = len(names)
    for b in (i for i in range(1, n) if coll[i] != coll[i - 1]):   # block boundaries in the lower triangle
        ax.plot([-0.5, b - 0.5], [b - 0.5, b - 0.5], color=PAL["ink"], lw=0.6)
        ax.plot([b - 0.5, b - 0.5], [b - 0.5, n - 0.5], color=PAL["ink"], lw=0.6)
    blocks = pd.read_csv(REC.source("outputs/piop/spatial_null_map_blocks.csv"))
    lines = ["Block mean r, variogram-", "matched surrogates"]
    for key in ("ID1000 vs PIOP1", "ID1000 vs PIOP2", "PIOP1 vs PIOP2"):
        b = blocks[blocks.block == key].iloc[0]
        lines.append(f"{key.replace(' vs ', '–')}: {b.mean_r:.2f}, {p_text(b.p_surrogate)}")
    ax.text(1.0, 1.0, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=5.3,
            linespacing=1.35)
    cb = plt.colorbar(im, cax=cax, ticks=[-0.8, -0.4, 0, 0.4, 0.8])
    style_colorbar(cb, "map correlation (r)", 5.8)
    REC.data("E", "map_similarity", M.reset_index().rename(columns={M.index.name or "index": "map"}),
             "Pearson correlation between the amplitude forward maps of every pair of cells (labels truncated in "
             "the source file; order follows amplitude_forward_maps.csv); the lower triangle is drawn.", None)
    REC.data("E", "block_nulls", blocks,
             "Mean map correlation within collection blocks against 5,000 BrainSMASH surrogates per map (distance "
             "matrix over all 215 regions, subcortex included). Within-collection blocks share subjects, which no "
             "spatial null addresses.", {c: "" for c in blocks.columns})


def main():
    apply_style()
    fig = plt.figure(figsize=(WIDTH["double"], 15.5 / 2.54))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.1], hspace=0.3, left=0.03, right=0.94, top=0.95,
                          bottom=0.075)
    # explicit spacer columns hold tick labels and colourbar labels, so neighbouring panels never collide
    top = gs[0].subgridspec(1, 6, width_ratios=[0.95, 0.24, 1.0, 0.04, 0.4, 0.74], wspace=0.04)
    bot = gs[1].subgridspec(1, 5, width_ratios=[1.62, 0.035, 0.42, 1.0, 0.04], wspace=0.04)

    ax_a = panel_a(fig, top[0, 0])
    ax_b = fig.add_subplot(top[0, 2])
    cax_b = fig.add_subplot(top[0, 3].subgridspec(3, 1, height_ratios=[0.12, 0.6, 0.28])[1, 0])
    panel_b(ax_b, cax_b)
    ax_c = fig.add_subplot(top[0, 5])
    panel_c(ax_c)
    ax_d = panel_d(fig, gs[1].subgridspec(1, 5, width_ratios=[1.62, 0.035, 0.42, 1.0, 0.04], wspace=0.04)[0, 0:2])
    ax_e = fig.add_subplot(bot[0, 3])
    cax_e = fig.add_subplot(bot[0, 4].subgridspec(3, 1, height_ratios=[0.12, 0.6, 0.28])[1, 0])
    panel_e(ax_e, cax_e)

    ax_a.set_title("Connectivity weight strength", loc="left", pad=5, x=0.0)
    ax_b.set_title("Network enrichment (spin null)", loc="left")
    ax_d.set_title("Regional amplitude pattern in three collections", loc="left", pad=12, x=0.0)
    ax_e.set_title("Amplitude maps across collections", loc="left")
    cell_label(fig, top[0, 0], "A", dx_pt=-8, dy_pt=4)
    cell_label(fig, top[0, 1], "B", dx_pt=-2, dy_pt=4)
    cell_label(fig, top[0, 4], "C", dx_pt=4, dy_pt=4)
    cell_label(fig, bot[0, 0], "D", dx_pt=-8, dy_pt=4)
    cell_label(fig, bot[0, 2], "E", dx_pt=4, dy_pt=4)
    out = REC.save(fig)
    print("saved", *[p.relative_to(ROOT) for p in out])


if __name__ == "__main__":
    main()
