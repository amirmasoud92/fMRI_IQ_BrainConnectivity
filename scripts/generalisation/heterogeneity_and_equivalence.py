#!/usr/bin/env python
"""
Heterogeneity and equivalence tests across collections for effects first observed in ID1000.

Claims and matched estimands
----------------------------
Each ID1000 estimate is matched to the PIOP estimand -- same decomposition, same
confound model (sex, eTIV, eTIV^2, motion), same 3x5 CV. Where the claim as
originally stated used a different ID1000 analysis, that version is added as a
sensitivity row (`estimand = "as stated"`), never used for the verdict.

  A  PSC / amplitude gain   covariance-tangent minus correlation-tangent r.
                            PSC matters only because it preserves amplitude, and
                            this is the PIOP test of that mechanism.
       matched   ID1000 gain_vs_fidelity cov - corr (discovery, PSC series)
       as stated ID1000 scrub_standardize psc - zsc (section 8)
       PIOP      state folds fc_tangent - corr_tangent, per task
  B  amplitude signature    deconfounded amplitude r vs 0
       matched   ID1000 variant_comparison base amplitude_dec (identical code
                 and confound set to PIOP)
       as stated ID1000 amplitude_confound_folds amp_dec_full (+ age)
       PIOP      state folds amplitude_dec, per task
  C  low-pass (broadband)   deconfounded FC gain, broad - base
       matched   ID1000 variant_comparison folds
       PIOP      broadband_replication per task. Only the summary was saved, with
                 corrected SE is recovered exactly:
                    SE_naive = |gain / t_naive|,  SE_corr = SE_naive * sqrt(1 + J * n_test/n_train)
  D  reliability asymmetry  descriptive (0.79 vs 0.45 split-half); there is no
                            inferential estimate to test, so it is not in this table.
Raw-target versions are included where they exist.

Aggregation within a collection
-------------------------------
Cells within one PIOP collection share subjects and are not independent. The
collection estimate is the unweighted cell mean, tested under two SE bounds:
  independent  SE = sqrt(sum se_c^2) / K     (liberal)
  dependent    SE = mean(se_c)               (conservative, perfect correlation)
A conclusion is called robust only if it holds under both.

Verdict per claim (matched estimand, deconfounded target)
---------------------------------------------------------
  non-replication supported     ID1000 established AND ID1000 - PIOP differs under
                                the dependent bound
  non-replication suggestive    ID1000 established AND differs only under the
                                independent bound
  replication failure untested  ID1000 established, no detectable difference: PIOP
                                is too imprecise to call it either way
  not established in ID1000     there is nothing to fail to replicate

Small-n caveat: at n ~ 200 the corrected test is itself somewhat liberal
(simulated FPR ~0.06-0.11 at n = 150), so PIOP "established" calls are slightly
optimistic and "not established" calls are safe.

Outputs
-------
  outputs/honest/heterogeneity_equivalence.csv          per collection / contrast
  outputs/honest/heterogeneity_equivalence_cells.csv    per cell inputs
  outputs/honest/heterogeneity_equivalence_verdicts.csv one row per claim
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.io_utils import write_results  # noqa: E402
from src.stats.cv_inference import ALPHA, ROPE  # noqa: E402

H = ROOT / "outputs" / "honest"
P = ROOT / "outputs" / "piop"
RATIO = 0.25          # n_test / n_train for 5-fold CV
J_PIOP = 15           # 3 x 5 folds (registry: broadband_replication --repeats 3)

CLAIMS = {
    "A": "PSC / amplitude gain (cov - corr tangent)",
    "B": "amplitude signature (amplitude r vs 0)",
    "C": "low-pass gain (broadband - base FC)",
}


# --------------------------------------------------------------------------- stats
def fold_estimate(diffs) -> tuple[float, float, int]:
    """Mean and Nadeau-Bengio corrected SE of a fold-level vector."""
    d = np.asarray(diffs, float)
    d = d[np.isfinite(d)]
    J = len(d)
    return float(d.mean()), float(np.sqrt((1 / J + RATIO) * d.var(ddof=1))), J


def corrected_se_from_naive_p(effect: float, p_naive: float, J: int = J_PIOP) -> tuple[float, float, int]:
    """Exact corrected SE from a naive paired t-test's two-sided p and mean effect."""
    t_naive = stats.t.isf(p_naive / 2, J - 1)
    se_naive = abs(effect) / t_naive
    return float(effect), float(se_naive * np.sqrt(1 + RATIO * J)), J


def tost_p(est: float, se: float, df: int, bound: float = ROPE) -> float:
    """Two one-sided tests of |true effect| < bound; the larger one-sided p."""
    return float(max(stats.t.sf((est + bound) / se, df), stats.t.cdf((est - bound) / se, df)))


def mde(se: float, df: int, power: float = 0.80) -> float:
    return float((stats.t.ppf(1 - ALPHA / 2, df) + stats.t.ppf(power, df)) * se)


def dersimonian_laird(est, se) -> dict:
    est, v = np.asarray(est, float), np.asarray(se, float) ** 2
    w = 1 / v
    fixed = (w * est).sum() / w.sum()
    Q = float((w * (est - fixed) ** 2).sum())
    k = len(est)
    tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    w_re = 1 / (v + tau2)
    pooled = float((w_re * est).sum() / w_re.sum())
    se_p = float(np.sqrt(1 / w_re.sum()))
    return {"estimate": pooled, "se": se_p, "p_vs_0": float(2 * stats.norm.sf(abs(pooled / se_p))),
            "Q": Q, "p_Q": float(stats.chi2.sf(Q, k - 1)),
            "I2": float(max(0.0, (Q - (k - 1)) / Q)) if Q > 0 else 0.0, "tau2": float(tau2)}


# --------------------------------------------------------------------------- inputs
def collect_cells() -> pd.DataFrame:
    rows = []

    def add(claim, target, estimand, collection, cell, est_se_j, source):
        est, se, J = est_se_j
        rows.append({"claim": claim, "target": target, "estimand": estimand, "collection": collection,
                     "cell": cell, "estimate": est, "se": se, "folds": J, "source": source})

    gv = pd.read_csv(H / "gain_vs_fidelity_folds.csv")
    ss = pd.read_csv(H / "scrub_standardize_folds.csv")
    ac = pd.read_csv(H / "amplitude_confound_folds.csv")
    vc = pd.read_csv(P / "variant_comparison_folds.csv")
    # no fold column: every variant is scored on the same seeded splitter, in order
    base = vc[vc.variant == "base"].reset_index(drop=True)
    broad = vc[vc.variant == "broad"].reset_index(drop=True)
    assert len(base) == len(broad) == J_PIOP

    for tgt, sfx in (("raw", ""), ("deconfounded", "_dec")):
        add("A", tgt, "matched", "ID1000", "moviewatching",
            fold_estimate(gv["cov" + sfx] - gv["corr" + sfx]), "gain_vs_fidelity_folds cov - corr")
        add("A", tgt, "as stated", "ID1000", "moviewatching",
            fold_estimate(ss["psc_unscrubbed" + sfx] - ss["zsc_unscrubbed" + sfx]),
            "scrub_standardize_folds psc - zsc (unscrubbed)")
        add("B", tgt, "matched", "ID1000", "moviewatching",
            fold_estimate(base["amplitude" + sfx]), "variant_comparison_folds base amplitude")
        add("C", tgt, "matched", "ID1000", "moviewatching",
            fold_estimate(broad["fc" + sfx] - base["fc" + sfx]), "variant_comparison_folds broad - base fc")
    add("B", "raw", "as stated", "ID1000", "moviewatching", fold_estimate(ac["amp_raw"]),
        "amplitude_confound_folds amp_raw")
    add("B", "deconfounded", "as stated", "ID1000", "moviewatching", fold_estimate(ac["amp_dec_full"]),
        "amplitude_confound_folds amp_dec_full")

    for ds in ("PIOP1", "PIOP2"):
        st = pd.read_csv(P / f"{ds}_state_folds.csv")
        for task, g in st.groupby("task"):
            assert len(g) == J_PIOP, (ds, task, len(g))
            for tgt, sfx in (("raw", ""), ("deconfounded", "_dec")):
                add("A", tgt, "matched", ds, task,
                    fold_estimate(g["fc_tangent" + sfx] - g["corr_tangent" + sfx]),
                    f"{ds}_state_folds fc_tangent - corr_tangent")
                add("B", tgt, "matched", ds, task, fold_estimate(g["amplitude" + sfx]),
                    f"{ds}_state_folds amplitude")
        bb = pd.read_csv(P / f"{ds}_broadband_replication.csv")
        for r in bb.itertuples():
            add("C", "raw", "matched", ds, r.task, corrected_se_from_naive_p(r.gain, r.p),
                f"{ds}_broadband_replication gain (SE from naive p)")
            add("C", "deconfounded", "matched", ds, r.task, corrected_se_from_naive_p(r.gain_dec, r.p_dec),
                f"{ds}_broadband_replication gain_dec (SE from naive p)")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- analysis
def per_collection(g: pd.DataFrame) -> dict:
    K = len(g)
    return {"cells": K, "estimate": float(g.estimate.mean()),
            "se_independent": float(np.sqrt((g.se ** 2).sum()) / K),
            "se_dependent": float(g.se.mean()), "df": int(g.folds.min()) - 1,
            "cells_positive": int((g.estimate > 0).sum())}


def classify_single(p0: float, ptost: float) -> str:
    if p0 < ALPHA:
        return "established"
    return "equivalent to zero" if ptost < ALPHA else "inconclusive"


def analyse(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out, verdicts = [], []
    for (claim, target, estimand), g in cells.groupby(["claim", "target", "estimand"], sort=False):
        key = {"claim": claim, "claim_label": CLAIMS[claim], "target": target, "estimand": estimand}
        coll = {c: per_collection(gc) for c, gc in g.groupby("collection", sort=False)}
        for c, s in coll.items():
            for bound in ("independent", "dependent"):
                se = s[f"se_{bound}"]
                p0 = float(2 * stats.t.sf(abs(s["estimate"] / se), s["df"]))
                pt = tost_p(s["estimate"], se, s["df"])
                out.append({**key, "contrast": c, "se_bound": bound, "cells": s["cells"],
                            "cells_positive": s["cells_positive"], "estimate": s["estimate"], "se": se,
                            "ci_low": s["estimate"] - stats.t.ppf(1 - ALPHA / 2, s["df"]) * se,
                            "ci_high": s["estimate"] + stats.t.ppf(1 - ALPHA / 2, s["df"]) * se,
                            "p_vs_0": p0, "p_tost_rope": pt, "mde_80": mde(se, s["df"]),
                            "verdict": classify_single(p0, pt)})
        if estimand != "matched" or not {"PIOP1", "PIOP2"} <= set(coll):
            continue
        # ID1000 is a single cell, so its two bounds coincide
        e0, s0 = coll["ID1000"]["estimate"], coll["ID1000"]["se_independent"]
        het = {}
        for c in ("PIOP1", "PIOP2"):
            for bound in ("independent", "dependent"):
                diff = e0 - coll[c]["estimate"]
                se = float(np.hypot(s0, coll[c][f"se_{bound}"]))
                p = float(2 * stats.norm.sf(abs(diff / se)))
                het[(c, bound)] = p
                out.append({**key, "contrast": f"ID1000 - {c}", "se_bound": bound, "estimate": diff, "se": se,
                            "ci_low": diff - 1.96 * se, "ci_high": diff + 1.96 * se, "p_vs_0": p,
                            "verdict": "heterogeneous" if p < ALPHA else "not detectably different"})
        re = {}
        for bound in ("independent", "dependent"):
            names = ("ID1000", "PIOP1", "PIOP2")
            re[bound] = dersimonian_laird([coll[c]["estimate"] for c in names],
                                          [coll[c][f"se_{bound}"] for c in names])
            out.append({**key, "contrast": "random effects, 3 collections", "se_bound": bound, **re[bound],
                        "verdict": "heterogeneous" if re[bound]["p_Q"] < ALPHA else "no detectable heterogeneity"})

        id_p = float(2 * stats.t.sf(abs(e0 / s0), coll["ID1000"]["df"]))
        id_est = id_p < ALPHA
        piops = {}
        for c in ("PIOP1", "PIOP2"):
            s = coll[c]
            p_dep = float(2 * stats.t.sf(abs(s["estimate"] / s["se_dependent"]), s["df"]))
            p_ind = float(2 * stats.t.sf(abs(s["estimate"] / s["se_independent"]), s["df"]))
            if not id_est:
                v = "not established in ID1000"
            elif het[(c, "dependent")] < ALPHA:
                v = "non-replication supported"
            elif het[(c, "independent")] < ALPHA:
                v = "non-replication suggestive"
            else:
                v = "replication failure untested (PIOP too imprecise)"
            if p_dep < ALPHA and np.sign(s["estimate"]) == np.sign(e0):
                v = "replicates (robust)" if id_est else "established in PIOP only"
            elif p_ind < ALPHA and np.sign(s["estimate"]) == np.sign(e0) and id_est:
                v = "replicates (independent bound only)"
            piops[c] = v
        verdicts.append({**key, "id1000_estimate": e0, "id1000_se": s0, "id1000_p": id_p,
                         "id1000_established": id_est,
                         **{f"{c}_estimate": coll[c]["estimate"] for c in ("PIOP1", "PIOP2")},
                         **{f"{c}_cells_positive": f"{coll[c]['cells_positive']}/{coll[c]['cells']}"
                            for c in ("PIOP1", "PIOP2")},
                         **{f"p_het_{c}_{b}": het[(c, b)] for c in ("PIOP1", "PIOP2")
                            for b in ("independent", "dependent")},
                         "I2_dependent": re["dependent"]["I2"], "p_Q_dependent": re["dependent"]["p_Q"],
                         "I2_independent": re["independent"]["I2"], "p_Q_independent": re["independent"]["p_Q"],
                         "verdict_PIOP1": piops["PIOP1"], "verdict_PIOP2": piops["PIOP2"]})
    return pd.DataFrame(out), pd.DataFrame(verdicts)


def self_check(cells: pd.DataFrame) -> None:
    """The naive-p back-out must reproduce the fold-level corrected SE exactly."""
    vc = pd.read_csv(P / "variant_comparison_folds.csv")
    a = vc[vc.variant == "broad"].fc_dec.to_numpy()
    b = vc[vc.variant == "base"].fc_dec.to_numpy()
    est, se, J = fold_estimate(a - b)
    p_naive = float(stats.ttest_rel(a, b).pvalue)  # scipy kept: naive reference for the back-out check
    _, se_back, _ = corrected_se_from_naive_p(est, p_naive, J)
    assert abs(se_back - se) < 1e-10, (se_back, se)


def main():
    cells = collect_cells()
    self_check(cells)
    res, ver = analyse(cells)
    inputs = [H / "gain_vs_fidelity_folds.csv", H / "scrub_standardize_folds.csv",
              H / "amplitude_confound_folds.csv", P / "variant_comparison_folds.csv",
              P / "PIOP1_state_folds.csv", P / "PIOP2_state_folds.csv",
              P / "PIOP1_broadband_replication.csv", P / "PIOP2_broadband_replication.csv"]
    meta = dict(script=__file__, args=sys.argv[1:], inputs=inputs,
                ratio=RATIO, rope=ROPE, alpha=ALPHA, folds_piop=J_PIOP)
    write_results(cells, H / "heterogeneity_equivalence_cells.csv", **meta)
    write_results(res, H / "heterogeneity_equivalence.csv", **meta)
    write_results(ver, H / "heterogeneity_equivalence_verdicts.csv", **meta)

    pd.set_option("display.width", 250)
    cols = ["contrast", "se_bound", "cells_positive", "estimate", "se", "ci_low", "ci_high", "p_vs_0",
            "p_tost_rope", "mde_80", "I2", "p_Q", "verdict"]
    for (claim, target, estimand), g in res.groupby(["claim", "target", "estimand"], sort=True):
        print(f"\n=== {claim}: {CLAIMS[claim]}  [{target}, {estimand}] ===")
        print(g[[c for c in cols if c in g]].round(4).to_string(index=False))
    print("\nVERDICTS (matched estimand)")
    print(ver[["claim", "target", "id1000_estimate", "id1000_p", "PIOP1_estimate", "PIOP1_cells_positive",
               "PIOP2_estimate", "PIOP2_cells_positive", "p_het_PIOP1_dependent", "p_het_PIOP2_dependent",
               "verdict_PIOP1", "verdict_PIOP2"]].round(4).to_string(index=False))
    print("\nSaved -> outputs/honest/heterogeneity_equivalence{,_cells,_verdicts}.csv")


if __name__ == "__main__":
    main()
