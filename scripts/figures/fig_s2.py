#!/usr/bin/env python
"""
Figure S2 — Every model comparison under corrected inference.

All 124 paired comparisons in FINAL_RESULTS (the two PSC-vs-legacy rows that are
identical by construction are omitted): mean difference in fold r against each
group's reference arm, corrected 95 % interval (Nadeau-Bengio variance), coloured
by the verdict of the corrected and Bayesian tests. A cross marks comparisons that
a naive paired t-test calls significant but corrected inference does not.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pandas as pd

from src.visualization.journal import PAL, VERDICT, WIDTH, FigureRecord, apply_style, panel_label, verdict_colour

REC = FigureRecord(
    name="FigS2", title="Every model comparison under corrected inference",
    claim="Of 124 model comparisons, 95 are significant under a naive paired t-test on fold scores and 49 after "
          "correcting for the dependence between folds; the corrected verdicts are the ones used in the paper.")

# group prefix -> (panel, header shown in the figure); order within a panel is the drawing order
GROUPS = [
    ("A", "1.", "Headline models vs FC, original settings"),
    ("A", "2.", "Imaging models vs confound model"),
    ("A", "4.", "Matrix embedding vs affine-invariant"),
    ("A", "5.", "Decoder vs ridge"),
    ("A", "6a.", "SPDNet vs static tangent"),
    ("A", "6b.", "Dynamic connectivity vs static tangent"),
    ("A", "6c.", "Grassmann kernels vs tangent + ridge"),
    ("A", "6d.", "Blockwise tangents vs global tangent"),
    ("A", "7a.", "ISC / ISFC vs FC"),
    ("A", "7b.", "Co-skewness vs FC"),
    ("A", "17.", "Functional alignment vs unaligned covariance"),
    ("A", "18.", "Precision and partial correlation vs covariance tangent"),
    ("A", "12.", "Two-block ridge vs single ridge"),
    ("A", "16.", "Temporal complexity vs amplitude"),
    ("B", "3.", "PSC and scrubbing vs z-scored, scrubbed series"),
    ("B", "13.", "PSC vs legacy series (same folds)"),
    ("B", "8.", "Amplitude weight λ vs correlation (λ = 0)"),
    ("B", "9.", "Amplitude components vs correlation"),
    ("B", "10.", "Physiological noise control vs z-scored FC"),
    ("B", "11.", "Spectral shape vs correlation"),
    ("B", "14.", "Preprocessing vs current pipeline"),
    ("B", "15.", "Structure-function coupling vs amplitude + structure"),
    ("B", "19.", "Amplitude network lesions vs full covariance"),
    ("B", "20.", "Cross-time prediction vs full-run FC"),
    ("B", "21.", "Reliability-weighted ridge vs unweighted"),
]

ARM_LABEL = {
    "1.": {"stack_tuned": "Stack (FC + morphometry + FA)", "stack_fc_fa": "Stack (FC + FA)",
           "fc_krr": "FC, kernel ridge", "fc_ridge": "FC, tuned ridge", "fa_ridge": "Diffusion FA",
           "morph_ridge": "Morphometry"},
    "2.": {"all+conf": "All blocks + confounds", "fc+conf": "FC + confounds", "fc": "FC", "fa": "Diffusion FA",
           "morph": "Morphometry"},
    "3.": {"psc_unscrubbed": "PSC, unscrubbed", "psc_scrubbed": "PSC, scrubbed",
           "zsc_unscrubbed": "z-scored, unscrubbed"},
    "4.": {"logeuclid": "log-Euclidean", "logchol": "log-Cholesky", "power_0.25": "matrix power 0.25",
           "power_0.5": "matrix power 0.50", "power_0.75": "matrix power 0.75", "power_1.0": "Euclidean"},
    "5.": {"krr_logeuc": "kernel ridge, log-Euclidean", "krr_rbf": "kernel ridge, RBF", "enet": "elastic net",
           "svr": "support vector regression", "pls": "partial least squares"},
    "6a.": {"spdnet": "SPDNet", "spdnet_ridge": "SPDNet features + ridge"},
    "6b.": {"dfc_mean_ridge": "mean dynamic FC + ridge", "state_transformer": "state transformer"},
    "6c.": {"grass_chordal": "chordal kernel", "grass_geo": "geodesic kernel"},
    "6d.": {"global+yeo": "global + Yeo blocks", "global+sliding": "global + sliding blocks",
            "perm_sliding": "sliding, permuted regions", "sliding": "sliding blocks", "yeo": "Yeo blocks",
            "perm_yeo": "Yeo, permuted regions"},
    "7a.": {"stack_fc+isfc": "FC + ISFC stack", "stack_fc+isc": "FC + ISC stack",
            "stack_fc+isc+isfc": "FC + ISC + ISFC stack", "isc_nocensor": "ISC, uncensored", "isc": "ISC",
            "isfc": "ISFC", "isc_win": "windowed ISC"},
    "7b.": {"stack_fc+coskew_sub": "FC + co-skewness stack (subject)",
            "stack_fc+coskew_core": "FC + co-skewness stack (core)", "coskew_sub": "co-skewness, subject",
            "coskew_core": "co-skewness, core tensor"},
    "8.": {"lam_0.25": "λ = 0.25", "lam_0.50": "λ = 0.50", "lam_0.75": "λ = 0.75", "lam_1.00": "λ = 1 (covariance)",
           "lam_1.25": "λ = 1.25", "lam_1.50": "λ = 1.50", "regional_only": "regional profile only",
           "global_only": "global amplitude only", "swap": "amplitude from another subject"},
    "9.": {"idio": "idiosyncratic amplitude", "cov": "full amplitude (covariance)", "stim": "stimulus-locked amplitude",
           "amp_obs": "amplitude alone", "amp_idio": "idiosyncratic amplitude alone",
           "amp_stim": "stimulus-locked amplitude alone", "fidelity": "regional ISC alone"},
    "10.": {"psc_physio": "PSC + RETROICOR", "psc": "PSC", "zsc_physio": "z-scored + RETROICOR",
            "amp": "amplitude alone", "amp_physio": "amplitude alone + RETROICOR"},
    "11.": {"cov_plus_spec": "covariance + spectral shape", "cov": "covariance",
            "corr_plus_spec": "correlation + spectral shape", "spec_all": "spectral shape alone"},
    "12.": {"single_cv5": "single ridge, 5-fold α", "twoblock": "two-block ridge"},
    "13.": {"stack_tuned": "Stack", "fc_ridge": "FC, ridge", "fc_krr": "FC, kernel ridge"},
    "14.": {"wide": "low-pass 0.25 Hz", "hponly": "no low-pass", "wide_phys": "0.25 Hz + RETROICOR",
            "nogsr": "no global signal regression", "base_phys": "current + RETROICOR", "nosmooth": "no smoothing"},
    "15.": {"decouple": "decoupling index", "amp": "amplitude", "struct": "structural connectivity",
            "coupling": "coupling strength"},
    "16.": {"amp_sampen": "amplitude + sample entropy", "sampen": "sample entropy"},
    "17.": {"gpca100": "group PCA, k = 100", "gpca50": "group PCA, k = 50", "gpca20": "group PCA, k = 20",
            "srm100": "SRM, k = 100", "srm20": "SRM, k = 20", "srm50": "SRM, k = 50"},
    "18.": {"stack_cov+pcorr": "covariance + partial correlation stack", "prec_tangent": "precision tangent",
            "pcorr_tangent": "partial correlation tangent", "pcorr_fisher": "partial correlation, Fisher z",
            "corr_fisher": "correlation, Fisher z"},
    "19.": {f"lesion_{n}": f"without {lab}" for n, lab in
            [("SalVentAttn", "salience"), ("DorsAttn", "dorsal attention"), ("Default", "default"),
             ("Subcortex", "subcortex"), ("Cont", "control"), ("Limbic", "limbic"), ("SomMot", "somatomotor"),
             ("Vis", "visual")]} | {"flat": "flat amplitude profile"},
    "20.": {"fc_A_A": "FC, train A → test A", "fc_B_A": "FC, train B → test A", "fc_A_B": "FC, train A → test B",
            "fc_B_B": "FC, train B → test B", "amp_full": "amplitude, full run", "amp_B_B": "amplitude, B → B",
            "amp_A_B": "amplitude, A → B", "amp_B_A": "amplitude, B → A", "amp_A_A": "amplitude, A → A"},
    "21.": {"gamma_0.5": "γ = 0.5", "gamma_1.0": "γ = 1", "gamma_sel": "γ selected by inner CV", "gamma_2.0": "γ = 2",
            "keep_rel_gt_0.5": "reliable edges only (> 0.5)"},
}

XLIM = (-0.37, 0.30)
HEADER_X = -0.74   # axes fraction: headers start at the left edge of the label column


def build_table() -> pd.DataFrame:
    fr = pd.read_csv(REC.source("outputs/honest/FINAL_RESULTS.csv"))
    rows = []
    for panel, prefix, header in GROUPS:
        g = fr[fr.group.str.startswith(prefix + " ")]
        assert len(g), prefix
        ref = g[g.is_reference]
        ref_arm = ref.arm.iloc[0] if len(ref) else "legacy series"
        for r in g[(~g.is_reference) & (g.verdict != "identical")].itertuples():
            rows.append({"panel": panel, "group": r.group, "header": header, "reference_arm": ref_arm, "arm": r.arm,
                         "label": ARM_LABEL[prefix][r.arm], "n_subjects": r.n_subjects, "n_folds": r.n_folds,
                         "delta_r": r.delta_vs_ref, "ci_low": r.ci_low, "ci_high": r.ci_high, "p_corrected": r.p,
                         "p_naive": r.p_naive, "p_rope": r.p_rope, "bayes_decision": r.bayes_decision,
                         "verdict": r.verdict, "naive_only_significant": bool(r.p_naive < 0.05 <= r.p)})
    d = pd.DataFrame(rows)
    assert len(d) == 124, len(d)
    return d


def draw(ax, d: pd.DataFrame):
    y = 0
    ticks, labels = [], []
    blend = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    for header, g in d.groupby("header", sort=False):
        n0 = g.n_subjects.iloc[0]
        ax.text(HEADER_X, y, f"{header}  (n = {n0})", transform=blend, ha="left", va="center", fontsize=5.6,
                fontweight="bold", clip_on=False, zorder=5,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
        y += 1
        for r in g.itertuples():
            c = verdict_colour(r.verdict, r.delta_r)
            lo, hi = max(r.ci_low, XLIM[0]), min(r.ci_high, XLIM[1])
            ax.plot([lo, hi], [y, y], color=c, lw=0.9, solid_capstyle="butt")
            ax.plot(r.delta_r, y, "o", color=c, ms=2.5, mec="white", mew=0.3, zorder=3)
            if r.ci_low < XLIM[0]:
                ax.plot(XLIM[0] + 0.004, y, marker="<", color=c, ms=2.2, zorder=3)
            if r.ci_high > XLIM[1]:
                ax.plot(XLIM[1] - 0.004, y, marker=">", color=c, ms=2.2, zorder=3)
            if r.naive_only_significant:
                ax.plot(XLIM[1] - 0.012, y, marker="x", color=PAL["ink"], ms=2.6, mew=0.6)
            ticks.append(y)
            labels.append(r.label)
            y += 1
        y += 0.35
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=5.2)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.set_ylim(y - 0.6, -0.8)
    ax.set_xlim(*XLIM)
    ax.axvline(0, color=PAL["ink"], lw=0.5, zorder=0)
    for x in (-0.3, -0.2, -0.1, 0.1, 0.2):
        ax.axvline(x, color=PAL["mist"], lw=0.35, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Δr vs reference (corrected 95 % CI)")


def main():
    apply_style()
    d = build_table()
    fig = plt.figure(figsize=(WIDTH["double"], 22.0 / 2.54))
    gs = fig.add_gridspec(1, 2, wspace=0.95, left=0.2, right=0.99, top=0.955, bottom=0.075)
    axes = []
    for i, panel in enumerate(("A", "B")):
        ax = fig.add_subplot(gs[0, i])
        draw(ax, d[d.panel == panel])
        axes.append(ax)
    axes[0].set_title("Models, representations and decoders", loc="left", x=-0.62)
    axes[1].set_title("Refinements and diagnostics", loc="left", x=-0.62)
    panel_label(axes[0], "A", x=-0.7, y=1.0)
    panel_label(axes[1], "B", x=-0.7, y=1.0)
    n_naive = int((d.p_naive < 0.05).sum())
    n_corr = int((d.p_corrected < 0.05).sum())
    handles = [plt.Line2D([], [], color=verdict_colour("established", -1), marker="o", ms=3, lw=0.9,
                          label="worse, established"),
               plt.Line2D([], [], color=verdict_colour("established", 1), marker="o", ms=3, lw=0.9,
                          label="better, established"),
               plt.Line2D([], [], color=VERDICT["equivalent"], marker="o", ms=3, lw=0.9,
                          label="practically equivalent (ROPE ±0.01)"),
               plt.Line2D([], [], color=VERDICT["inconclusive"], marker="o", ms=3, lw=0.9, label="inconclusive"),
               plt.Line2D([], [], color=PAL["ink"], marker="x", ms=3, mew=0.6, lw=0,
                          label=f"significant only naively ({n_naive - n_corr} of 124)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0), fontsize=5.8,
               handletextpad=0.4, columnspacing=1.2)
    REC.data("AB", "all_comparisons", d,
             f"All 124 corrected comparisons from FINAL_RESULTS.csv (the two PSC-vs-legacy rows for FA and "
             f"morphometry, identical by construction, are omitted). Naive paired t-test p < 0.05: {n_naive}; "
             f"corrected resampled t-test p < 0.05: {n_corr}. Verdict combines the corrected test with the Bayesian "
             f"correlated t-test (ROPE +/-0.01): established = corrected p < 0.05; equivalent = not established and "
             f"the posterior puts more than 95 % of its mass inside the ROPE; otherwise inconclusive (the "
             f"rule in src/stats/cv_inference.verdict). Intervals beyond the axis "
             f"are truncated and marked with an arrowhead.",
             {"panel": "figure panel", "group": "FINAL_RESULTS group", "header": "group header in the figure",
              "reference_arm": "arm each comparison is made against", "arm": "compared arm",
              "label": "label in the figure", "n_subjects": "subjects in the analysis", "n_folds": "outer folds",
              "delta_r": "mean paired difference in fold r (arm minus reference)",
              "ci_low": "corrected 95 % interval, lower", "ci_high": "corrected 95 % interval, upper",
              "p_corrected": "corrected resampled t-test p", "p_naive": "naive paired t-test p",
              "p_rope": "Bayesian posterior probability inside the ROPE", "bayes_decision": "Bayesian decision",
              "verdict": "combined verdict", "naive_only_significant": "naive p < 0.05 but corrected p >= 0.05"})
    out = REC.save(fig)
    print("saved", *[p.relative_to(ROOT) for p in out], f"| naive {n_naive}, corrected {n_corr}")


if __name__ == "__main__":
    main()
