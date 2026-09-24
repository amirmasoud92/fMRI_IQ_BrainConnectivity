#!/usr/bin/env python
"""
Assemble every cross-validated comparison into one table with corrected and Bayesian inference.

Paired tests are run only within a single source file, where the folds are identical by construction.

Usage
-----
  python scripts/prediction/compile_comparison_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.stats.cv_inference import check_fold_alignment, compare

O = Path("outputs/honest")
OUT = O / "FINAL_RESULTS.csv"
REGRADE = O / "inference_regrade.csv"
SMALL_N = 400
TEST_COLUMNS = ["delta_vs_ref", "t", "p", "wins", "t_naive", "p_naive", "se", "ci_low",
                "ci_high", "p_a_better", "p_rope", "p_b_better", "bayes_decision",
                "mde_80", "verdict"]


def test_fields(a, b) -> dict:
    """Corrected + Bayesian comparison of fold scores a vs b, in table columns."""
    res = compare(a, b)
    return {"delta_vs_ref": res["delta"], "t": res["t"], "p": res["p"], "wins": res["wins"],
            "t_naive": res["naive_t"], "p_naive": res["naive_p"], "se": res["se"],
            "ci_low": res["ci_low"], "ci_high": res["ci_high"],
            "p_a_better": res["p_a_better"], "p_rope": res["p_rope"],
            "p_b_better": res["p_b_better"], "bayes_decision": res["bayes_decision"],
            "mde_80": res["mde_80"], "verdict": res["verdict"]}

# group, file, n_subjects, reference arm, [arms]
GROUPS = [
    ("1. HEADLINE multimodal stack (PSC)", "multimodal_tuned_psc_folds.csv", 874,
     "legacy_fc", ["stack_tuned", "stack_fc_fa", "fc_krr", "fc_ridge",
                   "fa_ridge", "morph_ridge", "legacy_fc"]),
    ("2. Confound floor and imaging increment (PSC)", "psc/confound_benchmark_folds.csv", 874,
     "confounds", ["all+conf", "fc+conf", "fc", "fa", "morph", "confounds"]),
    ("3. FC representation: scrubbing x standardisation", "scrub_standardize_folds.csv", 877,
     "zsc_scrubbed", ["psc_unscrubbed", "psc_scrubbed", "zsc_unscrubbed", "zsc_scrubbed"]),
    ("4. Geometry sweep (embeddings, legacy)", "riemann_metrics_folds.csv", 877,
     "airm", ["logeuclid", "logchol", "airm", "power_0.25", "power_0.5",
              "power_0.75", "power_1.0"]),
    ("5. Decoder sweep (legacy)", "riemann_decoders_folds.csv", 877,
     "ridge", ["krr_logeuc", "krr_rbf", "ridge", "enet", "svr", "pls"]),
    ("6a. REJECTED: SPDNet (PSC)", "psc/spdnet_folds.csv", 877,
     "static_tangent", ["static_tangent", "spdnet", "spdnet_ridge"]),
    ("6b. REJECTED: dynamic transformer (PSC)", "psc/dynamic_state_transformer_folds.csv", 877,
     "static_tangent", ["static_tangent", "dfc_mean_ridge", "state_transformer"]),
    ("6c. REJECTED: Grassmannian (PSC)", "psc/grassmann_folds.csv", 877,
     "tangent_ridge", ["tangent_ridge", "grass_chordal", "grass_geo"]),
    ("6d. REJECTED: blockwise tangent (PSC)", "psc/blockwise_tangent_folds.csv", 877,
     "global", ["global", "global+yeo", "global+sliding", "perm_sliding",
                "sliding", "yeo", "perm_yeo"]),
    ("7a. REJECTED: ISC / ISFC", "isc_folds.csv", 877,
     "fc", ["fc", "stack_fc+isfc", "stack_fc+isc", "stack_fc+isc+isfc",
            "isc_nocensor", "isc", "isfc", "isc_win"]),
    ("7b. REJECTED: co-skewness", "coskewness_folds.csv", 877,
     "fc", ["fc", "stack_fc+coskew_sub", "stack_fc+coskew_core",
            "coskew_sub", "coskew_core"]),
    ("8. MECHANISM: lambda sweep C(lam)=D^lam R D^lam", "amplitude_mechanism_folds.csv", 877,
     "lam_0.00", ["lam_0.00", "lam_0.25", "lam_0.50", "lam_0.75", "lam_1.00",
                  "lam_1.25", "lam_1.50", "regional_only", "global_only", "swap"]),
    ("9. MECHANISM: gain vs fidelity (discovery)", "gain_vs_fidelity_folds.csv", 701,
     "corr", ["idio", "cov", "corr", "stim", "amp_obs", "amp_idio",
              "amp_stim", "fidelity"]),
    ("10. MECHANISM: physiological control", "psc_physio_folds.csv", 788,
     "zsc", ["psc_physio", "psc", "zsc_physio", "zsc", "amp", "amp_physio"]),
    ("11. NULL: spectral shape (discovery)", "spectral_shape_folds.csv", 701,
     "corr", ["cov_plus_spec", "cov", "corr", "corr_plus_spec", "spec_all"]),
    ("12. NULL: two-block ridge (discovery)", "twoblock_ridge_folds.csv", 701,
     "single_gcv", ["single_gcv", "single_cv5", "twoblock"]),
    ("14. Preprocessing ablation (band, smoothing, GSR, physiology)",
     "preproc_variants_folds.csv", 249,
     "base", ["wide", "hponly", "wide_phys", "nogsr", "base_phys", "base", "nosmooth"]),
    ("15. Structure-function decoupling (discovery)", "sfc_entropy_folds.csv", 701,
     "amp_struct", ["amp_struct", "decouple", "amp", "struct", "coupling"]),
    ("16. Temporal complexity (discovery)", "sfc_entropy_folds.csv", 701,
     "amp", ["amp", "amp_sampen", "sampen"]),
    ("17. REJECTED: functional alignment SRM (discovery)", "srm_alignment_folds.csv", 701,
     "raw", ["raw", "gpca100", "gpca50", "gpca20", "srm100", "srm20", "srm50"]),
    ("18. Precision and partial correlation (discovery)", "precision_partial_folds.csv", 701,
     "cov_tangent", ["stack_cov+pcorr", "cov_tangent", "prec_tangent",
                     "pcorr_tangent", "pcorr_fisher", "corr_fisher"]),
    ("21. Reliability-weighted ridge (discovery)", "reliability_weighted_folds.csv", 701,
     "gamma_0.0", ["gamma_0.5", "gamma_1.0", "gamma_sel", "gamma_0.0",
                   "gamma_2.0", "keep_rel_gt_0.5"]),
    ("20. Cross-time prediction: is the signature trait-like? (discovery)",
     "crosstime_folds.csv", 701,
     "fc_full", ["fc_full", "fc_A_A", "fc_B_A", "fc_A_B", "fc_B_B",
                 "amp_full", "amp_B_B", "amp_A_B", "amp_B_A", "amp_A_A"]),
    ("19. Amplitude lesioning by network (discovery)", "amplitude_lesion_folds.csv", 701,
     "cov", ["cov", "lesion_SalVentAttn", "lesion_DorsAttn", "lesion_Default",
             "lesion_Subcortex", "lesion_Cont", "lesion_Limbic", "lesion_SomMot",
             "lesion_Vis", "flat"]),
]


def load(f):
    p = O / f
    return pd.read_csv(p) if p.exists() else None


def rows_for(group, f, n, ref, arms):
    d = load(f)
    if d is None:
        print(f"  [skipped, missing: {f}]")
        return []
    out = []
    have = [a for a in arms if a in d.columns]
    for a in have:
        r = {"group": group, "source": f, "n_subjects": n, "n_folds": len(d),
             "arm": a, "r_mean": d[a].mean(), "r_sd": d[a].std(ddof=1),
             "is_reference": a == ref, "small_n": n < SMALL_N}
        if a != ref and ref in d.columns:
            if np.abs(d[a] - d[ref]).max() < 1e-12:       # identical arms (e.g. an identity)
                r.update({"delta_vs_ref": 0.0, "wins": 0, "verdict": "identical"})
            else:
                r.update(test_fields(d[a], d[ref]))
        dc = a + "_dec"
        if dc in d.columns:
            r["r_deconfounded"] = d[dc].mean()
        out.append(r)
    return out


def main():
    all_rows = []
    for group, f, n, ref, arms in GROUPS:
        all_rows += rows_for(group, f, n, ref, arms)

    # legacy vs PSC stack: two files, same seed/n/stratification -> verify then pair
    L, P = load("multimodal_tuned_folds.csv"), load("multimodal_tuned_psc_folds.csv")
    if L is not None and P is not None and len(L) == len(P):
        check_fold_alignment(L["fold"], P["fold"])          # raises if not paired
        for a in ["stack_tuned", "fc_ridge", "fc_krr", "fa_ridge", "morph_ridge"]:
            if a in L and a in P:
                row = {"group": "13. PSC vs legacy series (paired, same folds)",
                       "source": "multimodal_tuned{_psc}_folds.csv", "n_subjects": 874,
                       "n_folds": len(P), "arm": a, "r_mean": P[a].mean(),
                       "r_sd": P[a].std(ddof=1), "is_reference": False, "small_n": False}
                if np.abs(P[a] - L[a]).max() < 1e-9:       # block untouched by the series
                    row.update({"delta_vs_ref": 0.0, "wins": 0, "verdict": "identical"})
                else:
                    row.update(test_fields(P[a], L[a]))
                all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT, index=False)

    tested = df[df["p"].notna()].copy()
    tested["naive_significant"] = tested["p_naive"] < 0.05
    tested["corrected_significant"] = tested["p"] < 0.05
    tested["flipped"] = tested["naive_significant"] & ~tested["corrected_significant"]
    keep = ["group", "arm", "n_subjects", "n_folds", "small_n", "delta_vs_ref", "p_naive",
            "p", "ci_low", "ci_high", "p_rope", "bayes_decision", "mde_80", "verdict",
            "naive_significant", "corrected_significant", "flipped"]
    tested[keep].rename(columns={"p": "p_corrected"}).to_csv(REGRADE, index=False)

    for group in df["group"].unique():
        g = df[df["group"] == group]
        n, nf = g["n_subjects"].iloc[0], g["n_folds"].iloc[0]
        print(f"\n{'=' * 92}\n{group}   (n={n}, {nf} folds)\n{'=' * 92}")
        hdr = (f"  {'arm':24s} {'r':>17s} {'deconf':>8s} {'delta':>9s} {'p corr':>9s} "
               f"{'p naive':>9s} {'P(rope)':>8s} {'verdict':>13s}")
        print(hdr)
        for _, r in g.sort_values("r_mean", ascending=False).iterrows():
            dec = f"{r['r_deconfounded']:+8.4f}" if pd.notna(r.get("r_deconfounded")) else " " * 8
            if pd.notna(r.get("delta_vs_ref")):
                if pd.isna(r.get("p")):
                    tail = f" {r['delta_vs_ref']:+9.4f} {'--':>9s} {'--':>9s} {'--':>8s} {r['verdict']:>13s}"
                else:
                    tail = (f" {r['delta_vs_ref']:+9.4f} {r['p']:9.3g} {r['p_naive']:9.3g} "
                            f"{r['p_rope']:8.2f} {r['verdict']:>13s}")
            else:
                tail = f" {'reference':>9s}"
            print(f"  {r['arm']:24s} {r['r_mean']:+8.4f} +/- {r['r_sd']:.4f} {dec}{tail}")

    t = df[df["p"].notna()]
    n_naive, n_corr = int((t["p_naive"] < 0.05).sum()), int((t["p"] < 0.05).sum())
    print(f"\n{'=' * 92}")
    print(f"{len(t)} tested comparisons: {n_naive} significant naive, {n_corr} after correction "
          f"({n_naive - n_corr} flipped); verdicts {t['verdict'].value_counts().to_dict()}")
    print(f"Wrote {len(df)} rows -> {OUT}; re-grade -> {REGRADE}")
    print("Samples are NOT interchangeable: n=877 full, n=874 with FreeSurfer+FA,")
    print("n=788 with usable physio, n=701 sealed discovery split.")
    print("The sealed 176-subject holdout has NOT been evaluated by any row here.")


if __name__ == "__main__":
    sys.exit(main())
