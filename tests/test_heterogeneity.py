"""Tests for the heterogeneity and equivalence statistics."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/generalisation"))

import numpy as np
from scipy import stats

import heterogeneity_and_equivalence as M
from src.stats.cv_inference import corrected_resampled_ttest


def test_fold_estimate_matches_corrected_ttest():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0.30, 0.05, 15), rng.normal(0.28, 0.05, 15)
    est, se, J = M.fold_estimate(a - b)
    ref = corrected_resampled_ttest(a, b, k=5)
    assert J == 15 and abs(est - ref.delta) < 1e-12 and abs(se - ref.se) < 1e-12


def test_naive_p_backout_is_exact():
    rng = np.random.default_rng(1)
    for _ in range(20):
        d = rng.normal(rng.normal(0, 0.03), 0.04, 15)
        est, se, J = M.fold_estimate(d)
        p_naive = float(stats.ttest_1samp(d, 0).pvalue)  # scipy kept: naive reference
        _, se_back, _ = M.corrected_se_from_naive_p(est, p_naive, J)
        assert abs(se_back - se) < 1e-10, (se_back, se)


def test_tost_rejects_only_inside_bound():
    assert M.tost_p(0.0, 0.002, 14) < 0.05          # tight estimate at zero -> equivalent
    assert M.tost_p(0.0, 0.02, 14) > 0.05           # imprecise -> not equivalent
    assert M.tost_p(0.012, 0.001, 14) > 0.5         # precisely outside the bound


def test_dersimonian_laird_limits():
    homo = M.dersimonian_laird([0.1, 0.1, 0.1], [0.02, 0.03, 0.04])
    assert homo["Q"] < 1e-12 and homo["tau2"] == 0 and abs(homo["estimate"] - 0.1) < 1e-12
    het = M.dersimonian_laird([0.3, 0.0, -0.1], [0.02, 0.02, 0.02])
    assert het["p_Q"] < 1e-6 and het["I2"] > 0.9 and het["tau2"] > 0


def test_verdicts_on_synthetic_cells():
    import pandas as pd
    rows = []
    def cell(claim, coll, name, est, se):
        rows.append(dict(claim=claim, target="deconfounded", estimand="matched", collection=coll,
                         cell=name, estimate=est, se=se, folds=15, source="synthetic"))
    # A: ID1000 null -> not established; B: strong ID1000, PIOP2 opposite and precise
    for claim, e_id, e_p1, e_p2 in (("A", 0.005, 0.0, 0.0), ("B", 0.20, 0.19, -0.10)):
        cell(claim, "ID1000", "m", e_id, 0.02)
        for t in range(3):
            cell(claim, "PIOP1", f"t{t}", e_p1, 0.02)
            cell(claim, "PIOP2", f"t{t}", e_p2, 0.02)
    _, ver = M.analyse(pd.DataFrame(rows))
    v = ver.set_index("claim")
    assert v.loc["A", "verdict_PIOP1"] == "not established in ID1000"
    assert v.loc["B", "verdict_PIOP1"] == "replicates (robust)"
    assert v.loc["B", "verdict_PIOP2"] == "non-replication supported"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:                                   # noqa: BLE001
            import traceback; traceback.print_exc()
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
