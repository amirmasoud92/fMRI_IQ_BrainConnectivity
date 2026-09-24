#!/usr/bin/env python
"""
Figure S8 — The regional maps behind Figure 5.

A  Strength of the connectivity model's Haufe forward weights per region (sum of
   |weight| over the region's edges, relative to the mean region), for the raw and
   the deconfounded IST target, both hemispheres.
B  Amplitude forward maps of all 11 collection x task cells compared in Fig 5E and
   Fig S6D, z-scored across regions, both hemispheres. ID1000 and PIOP2 on the
   left, PIOP1 on the right.

Both panels are descriptive; the tests on these maps are the network spin null
(Fig 5B) and the variogram-matched surrogates (Fig 5E, Fig S6).
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

from src.visualization import brain
from src.visualization.journal import (COLLECTION, DIVERGING, PAL, SEQUENTIAL, WIDTH, FigureRecord, apply_style,
                                       cell_label, style_colorbar)

warnings.filterwarnings("ignore")

REC = FigureRecord(
    name="FigS8", title="The regional maps behind Figure 5",
    claim="The connectivity weights are distributed over the whole cortex with the same layout for the raw and the "
          "deconfounded target, and the amplitude maps of PIOP1 tasks resemble one another and the ID1000 movie map "
          "more than the PIOP2 maps do; these maps are descriptive and are tested only through the spin and "
          "variogram-matched nulls.")

TASK = {"moviewatching": "movie", "anticipation": "anticipation", "emomatching": "emotion matching",
        "faces": "faces", "gstroop": "gender Stroop", "restingstate": "rest", "workingmemory": "working memory",
        "stopsignal": "stop signal"}
VIEW_LABEL = {("left", "lateral"): "L lateral", ("left", "medial"): "L medial", ("right", "medial"): "R medial",
              ("right", "lateral"): "R lateral"}


def map_rows(fig, cell, textures, labels, colours, cmap, vmin, vmax, view_labels=True):
    """One labelled row of four views per map."""
    n = len(textures)
    grid = cell.subgridspec(n, 5, width_ratios=[0.62, 1, 1, 1, 1], hspace=0.04, wspace=0.02)
    first = None
    for i, (tex, lab, col) in enumerate(zip(textures, labels, colours)):
        lab_ax = fig.add_subplot(grid[i, 0])
        lab_ax.set_axis_off()
        lab_ax.text(0.96, 0.5, lab, ha="right", va="center", fontsize=5.5, color=col, fontweight="bold",
                    linespacing=1.1)
        for j, (hemi, view) in enumerate(brain.VIEWS):
            ax = fig.add_subplot(grid[i, j + 1])
            brain.place(ax, brain.render(tex[hemi], hemi, view, cmap, vmin, vmax, dpi=250))
            if i == 0 and view_labels:
                ax.text(0.5, 1.02, VIEW_LABEL[(hemi, view)], transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=5.4, color=PAL["muted"])
            if first is None:
                first = lab_ax
    return first


def panel_a(fig, cell):
    reg = pd.read_csv(REC.source("outputs/honest/psc/haufe_regions.csv"))
    rel = {t: (reg[f"abs_weight_{t}"] / reg[f"abs_weight_{t}"].mean()).to_numpy() for t in ("raw", "deconfounded")}
    lo, hi = np.percentile(np.concatenate([v[:brain.N_CORTICAL] for v in rel.values()]), [5, 95])
    sub = cell.subgridspec(1, 2, width_ratios=[1, 0.025], wspace=0.12)
    textures = [brain.textures(rel[t]) for t in ("raw", "deconfounded")]
    first = map_rows(fig, sub[0, 0], textures, ["raw\ntarget", "deconfounded\ntarget"], [PAL["cobalt"]] * 2,
                     SEQUENTIAL, lo, hi)
    cax = fig.add_subplot(sub[0, 1].subgridspec(3, 1, height_ratios=[0.15, 0.7, 0.15])[1, 0])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=SEQUENTIAL, norm=plt.Normalize(lo, hi)), cax=cax,
                      ticks=list(np.round(np.arange(np.ceil(lo * 10) / 10, hi + 1e-9, 0.1), 1)))
    style_colorbar(cb, "strength (× mean)", 5.6)
    r = float(np.corrcoef(rel["raw"][:brain.N_CORTICAL], rel["deconfounded"][:brain.N_CORTICAL])[0, 1])
    REC.data("A", "connectivity_strength", reg.assign(relative_raw=rel["raw"], relative_deconfounded=rel["deconfounded"]),
             f"Sum of absolute Haufe forward weights over each region's edges (PSC connectivity ridge model), relative "
             f"to the mean region, for both targets; shared colour limits at the 5th-95th percentile of both maps. "
             f"The two cortical maps correlate r = {r:.2f} across the 200 parcels.",
             {"roi": "region index (200 Schaefer, then 15 subcortical)", "network": "Yeo-7 network or Subcortex",
              "abs_weight_raw": "sum |forward weight|, raw target", "abs_weight_deconfounded": "same, deconfounded",
              "abs_weight_deconf_motion": "same, extended motion model",
              "relative_raw": "raw strength / mean", "relative_deconfounded": "deconfounded strength / mean"})
    return first, r


def panel_b(fig, cell):
    maps = pd.read_csv(REC.source("outputs/piop/amplitude_forward_maps.csv"), index_col=0)
    cols = list(maps.columns)
    left = [c for c in cols if c.startswith(("ID1000", "PIOP2"))]
    right = [c for c in cols if c.startswith("PIOP1")]
    vmax = 2.5
    sub = cell.subgridspec(1, 3, width_ratios=[1, 1, 0.03], wspace=0.1)
    z = {c: ((maps[c] - maps[c].mean()) / maps[c].std()).to_numpy() for c in cols}

    def label(c):
        ds, task = c.split("/")
        name = TASK[task].replace(" matching", "\nmatching").replace(" memory", "\nmemory")
        return f"{ds}\n{name}"

    heights = [len(left), len(right)]
    left_cell = sub[0, 0].subgridspec(2, 1, height_ratios=[heights[0], heights[1] - heights[0]], hspace=0.04)[0, 0]
    first = map_rows(fig, left_cell, [brain.textures(z[c]) for c in left], [label(c) for c in left],
                     [COLLECTION[c.split("/")[0]] for c in left], DIVERGING, -vmax, vmax)
    map_rows(fig, sub[0, 1], [brain.textures(z[c]) for c in right], [label(c) for c in right],
             [COLLECTION["PIOP1"]] * len(right), DIVERGING, -vmax, vmax)
    cax = fig.add_subplot(sub[0, 2].subgridspec(3, 1, height_ratios=[0.3, 0.4, 0.3])[1, 0])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=DIVERGING, norm=plt.Normalize(-vmax, vmax)), cax=cax,
                      ticks=[-2, 0, 2])
    style_colorbar(cb, "forward weight (z)", 5.6)
    long = pd.DataFrame({"roi": np.arange(len(maps))} | {c: z[c] for c in cols})
    REC.data("B", "amplitude_maps_z", long,
             "Haufe forward weights of the regional-amplitude ridge model for every collection x task cell, z-scored "
             "across the 215 regions (amplitude_forward_maps.csv); only the 200 cortical parcels are drawn, with "
             "colour limits +/-2.5.",
             {"roi": "region index in extraction order", **{c: "z-scored forward weight" for c in cols}})
    return first


def main():
    apply_style()
    fig = plt.figure(figsize=(WIDTH["double"], 17.5 / 2.54))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 6.0], hspace=0.1, left=0.01, right=0.95, top=0.955,
                          bottom=0.01)
    a_cell = gs[0].subgridspec(1, 3, width_ratios=[0.22, 1, 0.22])[0, 1]
    _, r = panel_a(fig, a_cell)
    panel_b(fig, gs[1])
    for cell, letter, title in ((gs[0], "A", "Connectivity weight strength, raw and deconfounded target"),
                                (gs[1], "B", "Amplitude forward maps of all 11 cells")):
        t = cell_label(fig, cell, letter, dx_pt=0, dy_pt=2)
        box = cell.get_position(fig)
        fig.text(box.x0 + 0.03, box.y1, title, fontsize=7, fontweight="bold", va="bottom",
                 transform=t.get_transform())
    out = REC.save(fig)
    print("saved", *[p.relative_to(ROOT) for p in out], f"| raw vs deconfounded strength r = {r:.2f}")


if __name__ == "__main__":
    main()
