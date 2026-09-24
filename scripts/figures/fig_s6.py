#!/usr/bin/env python
"""
Figure S6 — Spatial nulls: how much of the network and map evidence survives spatial autocorrelation.

A  Network-pair enrichment z of the connectivity forward weights under three nulls:
   ROI-label permutation and the earlier centroid spin, each against the surface spin
   (28 cortical pairs, raw and deconfounded target).
B  Number of cortical pairs called significant under each null and sidedness.
C  Quality of the variogram fit behind each BrainSMASH surrogate map.
D  Amplitude map-pair similarity: naive parametric p against the variogram-matched
   surrogate p (the larger of the two maps' surrogate p-values).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.journal import COLLECTION, PAL, WIDTH, FigureRecord, apply_style, panel_label

REC = FigureRecord(
    name="FigS6", title="Spatial nulls",
    claim="ROI-label permutation ignores spatial autocorrelation and calls roughly half of all network pairs "
          "significant; one-sided spin nulls call 4 to 6 of 28, and only Default-Default and Control-Default "
          "survive FDR (raw target). BrainSMASH surrogates match their maps' variograms (r = 0.82 to 0.98), and "
          "against them 23 of 55 map pairs remain significant, down from 37 under a naive parametric test.")

TARGET = {"raw": PAL["cobalt"], "deconfounded": PAL["coral"]}
NULLS = [("p_label_perm_two_sided", "label permutation\ntwo-sided"),
         ("p_centroid_spin_upper", "centroid spin\none-sided"),
         ("p_surface_spin_two_sided", "surface spin\ntwo-sided"),
         ("p_surface_spin_upper", "surface spin\none-sided"),
         ("q_surface_spin_upper", "surface spin\none-sided, FDR q")]
BLOCK_COLOUR = {"ID1000 vs PIOP1": COLLECTION["PIOP1"], "ID1000 vs PIOP2": COLLECTION["PIOP2"],
                "PIOP1 vs PIOP2": PAL["plum"], "within PIOP1": PAL["slate"], "within PIOP2": PAL["mist"]}
SURROGATES = 5000


def block_of(a: str, b: str) -> str:
    ca, cb = a.split("/")[0], b.split("/")[0]
    if ca == cb:
        return f"within {ca}"
    return f"{ca} vs {cb}" if ca < cb else f"{cb} vs {ca}"


def panel_a(axes, sn):
    v = sn[sn.spin_valid].copy()
    for ax, col, lab in ((axes[0], "z_label_perm", "label permutation z"), (axes[1], "z_centroid_spin", "centroid spin z")):
        for tgt, c in TARGET.items():
            g = v[v.target == tgt]
            ax.scatter(g.z_surface_spin, g[col], s=8, color=c, lw=0, alpha=0.8, label=tgt, zorder=3)
        lim = (-5.5, 7.5)
        ax.plot(lim, lim, color=PAL["slate"], lw=0.6, ls=(0, (3, 2)), zorder=1)
        ax.axhline(0, color=PAL["mist"], lw=0.5, zorder=0)
        ax.axvline(0, color=PAL["mist"], lw=0.5, zorder=0)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel("surface spin z")
        ax.set_ylabel(lab)
        slope = np.polyfit(v.z_surface_spin, v[col], 1)[0]
        r = np.corrcoef(v.z_surface_spin, v[col])[0, 1]
        ax.text(0.04, 0.96, f"r = {r:.2f}\nslope {slope:.2f}", transform=ax.transAxes, va="top", fontsize=5.5)
    axes[0].legend(loc="lower right", fontsize=5.5, handletextpad=0.1, bbox_to_anchor=(1.04, -0.03))
    axes[0].set_title("Enrichment z under three nulls", loc="left")
    REC.data("A", "network_pair_nulls", v[["target", "net_i", "net_j", "mean_abs_weight", "z_surface_spin",
                                           "z_centroid_spin", "z_label_perm", "p_surface_spin_upper",
                                           "p_surface_spin_two_sided", "q_surface_spin_upper",
                                           "p_centroid_spin_upper", "p_label_perm_two_sided"]],
             "Network-pair enrichment of mean |Haufe forward weight| for the 28 cortical network pairs (subcortical "
             "pairs cannot be spun and are excluded), raw and deconfounded target (spatial_null_surface_haufe.csv). "
             "Surface spin: 10,000 fsaverage5 rotations with Hungarian parcel assignment; centroid spin: the earlier "
             "rotation of parcel centroids; label permutation: random reassignment of network labels.",
             {"target": "raw or deconfounded IST model", "net_i": "network", "net_j": "network",
              "mean_abs_weight": "mean |forward weight| of edges in the pair",
              "z_surface_spin": "z against the surface spin null", "z_centroid_spin": "z against the centroid spin",
              "z_label_perm": "z against label permutation",
              "p_surface_spin_upper": "one-sided (enrichment) surface spin p",
              "p_surface_spin_two_sided": "two-sided surface spin p",
              "q_surface_spin_upper": "BH-FDR q of the one-sided surface spin p over 28 pairs",
              "p_centroid_spin_upper": "one-sided centroid spin p", "p_label_perm_two_sided": "two-sided label p"})


def panel_b(ax, sn):
    v = sn[sn.spin_valid]
    rows = []
    width = 0.38
    for j, (tgt, c) in enumerate(TARGET.items()):
        g = v[v.target == tgt]
        counts = [int((g[col] < 0.05).sum()) for col, _ in NULLS]
        y = np.arange(len(NULLS))[::-1] + (0.2 if j == 0 else -0.2)
        ax.barh(y, counts, height=width, color=c, lw=0, label=tgt)
        for yi, n in zip(y, counts):
            ax.text(n + 0.3, yi, str(n), va="center", fontsize=5.3)
        rows += [{"target": tgt, "null": lab.replace("\n", ", "), "column": col, "pairs_tested": len(g), "pairs_below_0.05": n}
                 for (col, lab), n in zip(NULLS, counts)]
    ax.set_yticks(np.arange(len(NULLS))[::-1])
    ax.set_yticklabels([lab for _, lab in NULLS], fontsize=5.8)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 28)
    ax.set_xticks([0, 7, 14, 21, 28])
    ax.set_xlabel("Pairs with p (or q) < 0.05, of 28")
    ax.set_title("Significant pairs by null", loc="left")
    ax.legend(loc="lower right", fontsize=5.5, handletextpad=0.3)
    REC.data("B", "significant_counts", pd.DataFrame(rows),
             "Counts of the 28 cortical network pairs below 0.05 under each null and sidedness. The like-for-like "
             "comparison of spin methods is one-sided centroid vs one-sided surface spin.",
             {"target": "raw or deconfounded IST model", "null": "null model and sidedness",
              "column": "column in panel A's file", "pairs_tested": "cortical pairs",
              "pairs_below_0.05": "pairs with p (or q) < 0.05"})


def panel_c(ax, vf):
    vf = vf.copy()
    task = {"moviewatching": "movie", "anticipation": "anticipation", "emomatching": "emotion match",
            "faces": "faces", "gstroop": "gender Stroop", "restingstate": "rest", "workingmemory": "working mem.",
            "stopsignal": "stop signal"}
    vf["label"] = [f"{m.split('/')[0].replace('PIOP', 'P')} {task[m.split('/')[1]]}" for m in vf["map"]]
    vf["collection"] = vf["map"].str.split("/").str[0]
    y = np.arange(len(vf))[::-1]
    ax.scatter(vf.variogram_corr, y, s=12, c=[COLLECTION[c] for c in vf.collection], lw=0, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(vf.label, fontsize=5.6)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.78, 1.0)
    ax.set_xlabel("Variogram fit, r (surrogate vs map)")
    for yi, v in zip(y, vf.variogram_rel_rmse):
        ax.text(1.04, yi, f"{v:.2f}", transform=ax.get_yaxis_transform(), va="center", fontsize=5.4,
                color=PAL["muted"])
    ax.text(1.04, len(vf) - 0.35, "rel.\nRMSE", transform=ax.get_yaxis_transform(), va="bottom", fontsize=5.4,
            color=PAL["muted"])
    ax.set_ylim(-0.6, len(vf) - 0.4)
    ax.set_title("Surrogate quality", loc="left")
    REC.data("C", "variogram_fit", vf[["map", "collection", "variogram_corr", "variogram_rel_rmse"]],
             f"Agreement between each amplitude forward map's empirical variogram and the mean variogram of its "
             f"{SURROGATES:,} BrainSMASH surrogates (spatial_null_variogram_fit.csv).",
             {"map": "collection/task map", "collection": "collection",
              "variogram_corr": "correlation of surrogate and empirical variograms",
              "variogram_rel_rmse": "RMSE of the variogram, relative to the empirical variogram's range"})


def panel_d(ax, mp):
    mp = mp.copy()
    mp["block"] = [block_of(a, b) for a, b in zip(mp.map_a, mp.map_b)]
    floor = 1 / (SURROGATES + 1)
    x = -np.log10(np.maximum(mp.p_naive_parametric, 1e-40))
    y = -np.log10(mp.p_conservative)
    for blk, c in BLOCK_COLOUR.items():
        m = mp.block == blk
        ax.scatter(np.minimum(x[m], 12), y[m], s=10, color=c, lw=0.3, edgecolor=PAL["ink"] if "within" in blk else c,
                   label=f"{blk} ({int(m.sum())})", zorder=3)
    thr = -np.log10(0.05)
    ax.axhline(thr, color=PAL["coral"], lw=0.6, ls=(0, (3, 2)))
    ax.axvline(thr, color=PAL["coral"], lw=0.6, ls=(0, (3, 2)))
    ax.axhline(-np.log10(floor), color=PAL["slate"], lw=0.5, ls=":")
    ax.text(12, -np.log10(floor) + 0.08, "surrogate floor", ha="right", va="bottom", fontsize=5.0, color=PAL["muted"])
    ax.set_xlim(0, 12.3)
    ax.set_ylim(0, 4.2)
    ax.set_xlabel("−log10 p, naive parametric (capped at 12)")
    ax.set_ylabel("−log10 p, variogram-matched surrogates")
    n_naive = int((mp.p_naive_parametric < 0.05).sum())
    n_surr = int((mp.p_conservative < 0.05).sum())
    ax.text(0.97, 0.5, f"p < 0.05: naive {n_naive}/55,\nsurrogate {n_surr}/55", transform=ax.transAxes, ha="right",
            fontsize=5.6)
    ax.legend(loc="lower right", fontsize=5.0, handletextpad=0.1, labelspacing=0.25, bbox_to_anchor=(1.02, -0.02))
    ax.set_title("Map-pair similarity", loc="left")
    REC.data("D", "map_pairs", mp[["map_a", "map_b", "block", "r", "p_naive_parametric", "p_surrogate_a",
                                   "p_surrogate_b", "p_conservative", "p_conservative_upper", "null_sd"]],
             f"All 55 pairs of the 11 amplitude forward maps (spatial_null_map_pairs.csv): Pearson r across the 215 "
             f"regions, naive parametric p, and two-sided p against {SURROGATES:,} variogram-matched surrogates of "
             f"each map; p_conservative is the larger of the two. Within-collection pairs share subjects, which "
             f"the spatial null does not address.",
             {"map_a": "first map", "map_b": "second map", "block": "pair type", "r": "map correlation",
              "p_naive_parametric": "parametric p for r with 213 df",
              "p_surrogate_a": "two-sided p, surrogates of map_a", "p_surrogate_b": "two-sided p, surrogates of map_b",
              "p_conservative": "max of the two surrogate p-values",
              "p_conservative_upper": "one-sided (positive) version", "null_sd": "SD of the surrogate null r"})


def main():
    apply_style()
    sn = pd.read_csv(REC.source("outputs/honest/spatial_null_surface_haufe.csv"))
    vf = pd.read_csv(REC.source("outputs/piop/spatial_null_variogram_fit.csv"))
    mp = pd.read_csv(REC.source("outputs/piop/spatial_null_map_pairs.csv"))
    fig = plt.figure(figsize=(WIDTH["double"], 12.6 / 2.54))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.1], width_ratios=[1.0, 1.0, 1.0], hspace=0.36, wspace=0.62,
                          left=0.075, right=0.98, top=0.95, bottom=0.08)
    axes_a = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    axes_a[1].sharey(axes_a[0])
    ax_b = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1:])
    panel_a(axes_a, sn)
    panel_b(ax_b, sn)
    panel_c(ax_c, vf)
    panel_d(ax_d, mp)
    panel_label(axes_a[0], "A", x=-0.3, y=1.03)
    panel_label(ax_b, "B", x=-0.5, y=1.03)
    panel_label(ax_c, "C", x=-0.55, y=1.03)
    panel_label(ax_d, "D", x=-0.12, y=1.03)
    out = REC.save(fig)
    print("saved", *[p.relative_to(ROOT) for p in out])


if __name__ == "__main__":
    main()
