#!/usr/bin/env python
"""
Figure S4 — Confound-isolated test sets: validity diagnostics and the pre-specified criteria.

A  Isolation validity per set (10 sets per confound): remaining |r(IST, confound)|,
   test-set size, and reduction in mutual information, each against its
   pre-specified threshold (criterion C1).
B  Range restriction: SD of IST in the isolated set and in its IST-distribution-
   matched random set, relative to the starting pool.
C  The six criteria fixed before the analysis was run, with their outcomes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.journal import PAL, WIDTH, FigureRecord, apply_style, panel_label, strip

REC = FigureRecord(
    name="FigS4", title="Confound-isolated test sets: validity and pre-specified criteria",
    claim="Isolation removed the linear dependence between IST and every confound, but the mutual-information "
          "reduction required by criterion C1 was not reached for sex and for the joint confound set, so C1 fails; "
          "isolation also narrows the IST range, which is why IST-matched random sets are the primary comparison. "
          "Signal survives isolation (C2), non-linear confounding is not material (C3), prediction holds in both "
          "sexes (C4) without sex-specificity beyond training-set size (C5), and the imaging increment over the "
          "spline confound floor survives (C6).")

CONFOUNDS = [("etiv", "Brain volume"), ("fd_mean", "Head motion"), ("sex", "Sex"), ("joint", "All three")]
COL = {"etiv": PAL["teal"], "fd_mean": PAL["saffron"], "sex": PAL["plum"], "joint": PAL["ink"]}
ISO_TOL, MIN_TEST, MI_REDUCTION = 0.05, 150, 0.80        # confound_control_rigor.py, C1
MI_FLOOR = -1.0                                           # display floor for the MI reduction axis

RULES = [
    ("C1", "Isolation valid", "Every isolated set |r(IST, confound)| ≤ 0.05 and ≥ 150 subjects; "
                              "median mutual-information reduction ≥ 80 %"),
    ("C2", "Signal survives isolation", "FC r on isolated sets > 0 (corrected CI) and ≥ 50 % of r on matched "
                                        "random sets"),
    ("C3", "Non-linear confounding", "Spline-deconfounded FC r below linear by > 0.02, corrected p < .05"),
    ("C4", "Within-sex signal", "Sex-specific FC r > 0 (corrected CI) in both sexes"),
    ("C5", "Sex-specificity beyond n", "Sex-specific vs n-matched pooled model differs, corrected p < .05"),
    ("C6", "Imaging increment survives", "Stack increment over the spline confound floor > 0 (corrected CI)"),
]


def mi_reduction(g: pd.DataFrame) -> np.ndarray:
    return (1 - g["mi_after"] / np.maximum(g["mi_before"], 1e-12)).to_numpy()


def panel_a(axes, iso):
    specs = [(axes[0], "dep_r_after", "Remaining |r(IST, confound)|", ISO_TOL, "≤ 0.05"),
             (axes[1], "n_test", "Test-set size", MIN_TEST, "≥ 150"),
             (axes[2], "mi_reduction", "MI reduction", MI_REDUCTION, "median ≥ 80 %")]
    iso = iso.copy()
    iso["mi_reduction"] = mi_reduction(iso)
    for ax, col, lab, thr, thr_lab in specs:
        for i, (c, clab) in enumerate(CONFOUNDS):
            g = iso[iso.confound == c]
            v = g[col].to_numpy(float)
            shown = np.maximum(v, MI_FLOOR) if col == "mi_reduction" else v
            strip(ax, i, shown, COL[c], width=0.4, size=6, alpha=0.7, seed=i)
            if col == "mi_reduction":
                ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color=PAL["ink"], lw=1.0, zorder=4)
                if (v < MI_FLOOR).any():
                    ax.text(i, MI_FLOOR - 0.12, f"{int((v < MI_FLOOR).sum())} below", ha="center", fontsize=4.8,
                            color=PAL["muted"])
        ax.axhline(thr, color=PAL["coral"], lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.text(3.45, thr, thr_lab, ha="right", va="bottom", fontsize=5.0, color=PAL["coral"])
        ax.set_xticks(range(len(CONFOUNDS)))
        ax.set_xticklabels([l for _, l in CONFOUNDS], rotation=35, ha="right", fontsize=5.8)
        ax.set_xlim(-0.55, len(CONFOUNDS) - 0.45)
        ax.set_title(lab, fontsize=6.3, loc="left")
    axes[0].set_ylim(0, 0.1)
    axes[1].set_ylim(100, 360)
    axes[2].set_ylim(MI_FLOOR - 0.2, 1.1)
    axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 100:.0f} %".replace("-", "−")))
    summary = []
    for c, clab in CONFOUNDS:
        g = iso[iso.confound == c]
        red = mi_reduction(g)
        summary.append({"confound": c, "sets": len(g), "sets_isolated_ok": int(g.isolated_ok.sum()),
                        "sets_r_within_tol": int((g.dep_r_after <= ISO_TOL).sum()),
                        "sets_n_at_least_150": int((g.n_test >= MIN_TEST).sum()),
                        "median_mi_reduction": float(np.median(red)), "median_n_test": float(g.n_test.median()),
                        # exact ratio of set means; the criterion JSON stores it rounded to 3 decimals
                        "fc_retained_vs_ist_matched": float(g.fc_iso.mean() / g.fc_ymatch.mean())})
    REC.data("A", "isolation_sets", iso[["confound", "set", "n_start", "n_test", "isolated_ok", "removal_steps",
                                         "dep_r_before", "dep_r_after", "mi_before", "mi_after", "mi_reduction"]],
             f"One row per confound-isolated test set (confound_control_isolation.csv). mi_reduction = 1 - "
             f"mi_after / max(mi_before, 1e-12), as in criterion C1; values below {MI_FLOOR:.0%} are drawn at the "
             f"floor and counted in the figure. The black bar is the median.",
             {"confound": "confound isolated", "set": "set index", "n_start": "starting pool size",
              "n_test": "isolated test-set size", "isolated_ok": "removal reached the dependence tolerance",
              "removal_steps": "density-ratio removal steps", "dep_r_before": "|r(IST, confound)| in the pool",
              "dep_r_after": "|r(IST, confound)| after isolation", "mi_before": "mutual information in the pool",
              "mi_after": "mutual information after isolation", "mi_reduction": "relative MI reduction"})
    REC.data("A", "isolation_summary", pd.DataFrame(summary),
             "Per-confound summary of the C1 components.",
             {"confound": "confound isolated", "sets": "isolated sets", "sets_isolated_ok": "sets reaching tolerance",
              "sets_r_within_tol": "sets with |r| <= 0.05", "sets_n_at_least_150": "sets with >= 150 subjects",
              "median_mi_reduction": "median relative MI reduction", "median_n_test": "median test-set size",
              "fc_retained_vs_ist_matched": "mean FC r on isolated sets / mean FC r on IST-matched random sets"})
    return pd.DataFrame(summary)


def panel_b(ax, iso):
    rows = []
    for i, (c, clab) in enumerate(CONFOUNDS):
        g = iso[iso.confound == c]
        r_iso = (g.y_sd_isolated / g.y_sd_start).to_numpy()
        r_match = (g.y_sd_ymatch / g.y_sd_start).to_numpy()
        for off, v, mk, fc in ((-0.17, r_iso, "o", COL[c]), (0.17, r_match, "o", "white")):
            rng = np.random.default_rng(i)
            xs = i + off + rng.uniform(-0.07, 0.07, len(v))
            ax.scatter(xs, v, s=7, marker=mk, facecolor=fc, edgecolor=COL[c], lw=0.6, zorder=3)
            ax.plot([i + off - 0.12, i + off + 0.12], [v.mean()] * 2, color=PAL["ink"], lw=1.0, zorder=4)
        for s, a, b in zip(g.set, r_iso, r_match):
            rows.append({"confound": c, "set": int(s), "sd_ratio_isolated": a, "sd_ratio_ist_matched": b})
    ax.axhline(1, color=PAL["ink"], lw=0.5)
    ax.set_xticks(range(len(CONFOUNDS)))
    ax.set_xticklabels([l for _, l in CONFOUNDS], rotation=35, ha="right", fontsize=5.8)
    ax.set_xlim(-0.55, len(CONFOUNDS) - 0.45)
    ax.set_ylim(0.38, 1.08)
    ax.set_ylabel("SD of IST / SD in pool")
    ax.set_title("Range restriction", fontsize=6.3, loc="left")
    ax.scatter([], [], s=7, facecolor="grey", edgecolor="grey", lw=0.6, label="isolated set")
    ax.scatter([], [], s=7, facecolor="white", edgecolor="grey", lw=0.6, label="IST-matched random set")
    ax.legend(loc="lower left", fontsize=5.3, handletextpad=0.1, bbox_to_anchor=(0.1, 0.03))
    REC.data("B", "range_restriction", pd.DataFrame(rows),
             "Standard deviation of IST in each isolated test set and in its IST-distribution-matched random set, "
             "divided by the SD in the starting pool (confound_control_isolation.csv).",
             {"confound": "confound isolated", "set": "set index",
              "sd_ratio_isolated": "SD(IST) isolated / SD(IST) pool",
              "sd_ratio_ist_matched": "SD(IST) matched random / SD(IST) pool"})


def outcomes(summary: pd.DataFrame, crit: pd.DataFrame, tests: pd.DataFrame) -> list[str]:
    name = dict(CONFOUNDS)
    det = {r.criterion.split()[0]: (bool(r.met), r.detail) for r in crit.itertuples()}

    def t(comp):
        return tests[tests.comparison == comp].iloc[0]

    c1 = json.loads(det["C1"][1])
    fails = []
    for s in summary.itertuples():
        if not c1[s.confound]:
            why = []
            if s.sets_isolated_ok < s.sets or s.sets_r_within_tol < s.sets:
                why.append(f"|r| tolerance missed in {s.sets - s.sets_r_within_tol}/{s.sets} sets")
            if s.median_mi_reduction < MI_REDUCTION:
                why.append(f"median MI reduction {s.median_mi_reduction:.0%}")
            fails.append(f"{name[s.confound].lower()}: " + ", ".join(why))
    retained = summary.set_index("confound")["fc_retained_vs_ist_matched"]
    kept = ", ".join(f"{name[c].lower()} {retained[c]:.0%}" for c, _ in CONFOUNDS)
    c3 = t("fc: spline - linear (deconfounded)")
    male, female = t("male: sex-specific vs 0"), t("female: sex-specific vs 0")
    c5m, c5f = t("male: sex-specific - n-matched pooled"), t("female: sex-specific - n-matched pooled")
    c6 = t("increment over spline floor vs 0")
    return [
        "Passed for brain volume and head motion; failed for " + "; ".join(fails),
        f"CI excludes 0 for every confound; FC retains {kept} of IST-matched accuracy",
        f"Spline − linear {c3.delta:+.3f} [{c3.ci_low:+.3f}, {c3.ci_high:+.3f}], p = {c3.p_corrected:.2f}: not material",
        f"Males r = {male.delta:.2f} [{male.ci_low:.2f}, {male.ci_high:.2f}]; females r = {female.delta:.2f} "
        f"[{female.ci_low:.2f}, {female.ci_high:.2f}]",
        f"Males {c5m.delta:+.3f}, p = {c5m.p_corrected:.2f}; females {c5f.delta:+.3f}, p = {c5f.p_corrected:.2f}: "
        f"not detected",
        f"Increment {c6.delta:+.3f} [{c6.ci_low:+.3f}, {c6.ci_high:+.3f}]",
    ], [det[c][0] for c, _, _ in RULES]


def panel_c(ax, summary):
    sm = pd.read_csv(REC.source("outputs/honest/confound_control_summary.csv"))
    crit = sm[sm.kind == "criterion"]
    tests = sm[sm.kind == "test"]
    texts, met = outcomes(summary, crit, tests)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(len(RULES) + 0.7, -0.9)
    cols = [(0.0, "Criterion"), (0.19, "Pre-specified rule"), (0.56, "Outcome"), (0.975, "Met")]
    for x, h in cols:
        ax.text(x, -0.35, h, fontsize=6.2, fontweight="bold", va="center", ha="right" if h == "Met" else "left")
    ax.plot([0, 1], [0.05, 0.05], color=PAL["ink"], lw=0.6)
    rows = []
    for i, ((code, title, rule), out, ok) in enumerate(zip(RULES, texts, met)):
        y = i + 0.6
        colour = PAL["coral"] if code == "C1" and not ok else PAL["ink"]
        ax.text(0.0, y, f"{code}  {title}", fontsize=5.8, va="center")
        ax.text(0.19, y, _wrap(rule, 64), fontsize=5.4, va="center", linespacing=1.15)
        ax.text(0.56, y, _wrap(out, 68), fontsize=5.4, va="center", color=colour, linespacing=1.15)
        ax.text(0.975, y, "yes" if ok else "no", fontsize=5.8, va="center", ha="right", color=colour,
                fontweight="bold" if colour != PAL["ink"] else "normal")
        if i < len(RULES) - 1:
            ax.plot([0, 1], [y + 0.5, y + 0.5], color=PAL["mist"], lw=0.4)
        rows.append({"criterion": code, "name": title, "rule": rule, "outcome": out, "met": ok})
    ax.plot([0, 1], [len(RULES) + 0.1, len(RULES) + 0.1], color=PAL["ink"], lw=0.6)
    ax.text(0.0, len(RULES) + 0.45, "C3 and C5 are hypotheses of confounding and sex-specificity: 'no' means the "
            "effect was not found. C1 is the only validity criterion.", fontsize=5.2, color=PAL["muted"], va="center")
    REC.data("C", "criteria", pd.DataFrame(rows),
             "The six criteria fixed in confound_control_rigor.py before any real data were run, and their outcomes "
             "(confound_control_summary.csv, section A-C tests and the criterion rows).",
             {"criterion": "code", "name": "short name", "rule": "pre-specified rule", "outcome": "observed outcome",
              "met": "criterion met"})


def _wrap(s: str, width: int) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(s.replace("p < .05", "p < .05"), width))


def main():
    apply_style()
    iso = pd.read_csv(REC.source("outputs/honest/confound_control_isolation.csv"))
    fig = plt.figure(figsize=(WIDTH["double"], 13.0 / 2.54))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.05], width_ratios=[1, 1, 1, 1.05], hspace=0.42, wspace=0.5,
                          left=0.07, right=0.985, top=0.95, bottom=0.03)
    axes_a = [fig.add_subplot(gs[0, j]) for j in range(3)]
    ax_b = fig.add_subplot(gs[0, 3])
    ax_c = fig.add_subplot(gs[1, :])
    summary = panel_a(axes_a, iso)
    panel_b(ax_b, iso)
    panel_c(ax_c, summary)
    panel_label(axes_a[0], "A", x=-0.42, y=1.06)
    panel_label(ax_b, "B", x=-0.42, y=1.06)
    panel_label(ax_c, "C", x=-0.045, y=1.0)
    out = REC.save(fig)
    print("saved", *[p.relative_to(ROOT) for p in out])


if __name__ == "__main__":
    main()
