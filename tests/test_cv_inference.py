"""Tests for src/stats/cv_inference.py, including the null calibration of both tests."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from src.stats import cv_inference as C

RNG = np.random.default_rng(20260914)


def _pair(J=25, mean=0.01, sd=0.02, seed=0):
    rng = np.random.default_rng(seed)
    b = rng.normal(0.40, 0.05, J)
    return b + rng.normal(mean, sd, J), b


# ---------------------------------------------------------------------------
def test_ratio_zero_reproduces_scipy_ttest_rel_exactly():
    a, b = _pair(seed=1)
    ct = C.corrected_resampled_ttest(a, b, ratio=0.0)
    ref = stats.ttest_rel(a, b)
    assert np.isclose(ct.t, ref.statistic, rtol=0, atol=1e-12)
    assert np.isclose(ct.p, ref.pvalue, rtol=0, atol=1e-12)
    assert np.isclose(ct.naive_p, ref.pvalue, rtol=0, atol=1e-12)


def test_correction_inflates_variance_by_the_nadeau_bengio_factor():
    a, b = _pair(J=15, seed=2)
    ct = C.corrected_resampled_ttest(a, b, k=5)
    d = a - b
    assert np.isclose(ct.se, np.sqrt((1 / 15 + 0.25) * d.var(ddof=1)))
    assert ct.df == 14 and ct.p >= ct.naive_p
    naive_half = stats.t.ppf(0.975, 14) * d.std(ddof=1) / np.sqrt(15)
    assert np.isclose((ct.ci_high - ct.ci_low) / 2 / naive_half, np.sqrt((1 / 15 + 0.25) * 15))


def test_ratio_follows_k():
    assert C.test_train_ratio(5) == 0.25
    assert np.isclose(C.test_train_ratio(10), 1 / 9)


def test_bayes_with_zero_rope_equals_one_minus_corrected_one_sided_p():
    for seed in range(5):
        a, b = _pair(seed=seed)
        ct = C.corrected_resampled_ttest(a, b)
        bt = C.bayesian_correlated_ttest(a, b, rope=0.0)
        one_sided_p = stats.t.sf(ct.t, ct.df)
        assert np.isclose(bt.p_a_better, 1 - one_sided_p, rtol=0, atol=1e-12)


def test_bayes_probabilities_are_a_partition_and_antisymmetric():
    a, b = _pair(seed=3)
    ab = C.bayesian_correlated_ttest(a, b)
    ba = C.bayesian_correlated_ttest(b, a)
    assert np.isclose(ab.p_a_better + ab.p_rope + ab.p_b_better, 1.0)
    assert np.isclose(ab.p_a_better, ba.p_b_better) and np.isclose(ab.p_rope, ba.p_rope)


def test_decisions():
    rng = np.random.default_rng(4)
    b = rng.normal(0.4, 0.05, 25)
    big = C.bayesian_correlated_ttest(b + 0.08 + rng.normal(0, 0.005, 25), b)
    tiny = C.bayesian_correlated_ttest(b + rng.normal(0.0005, 0.001, 25), b)
    noisy = C.bayesian_correlated_ttest(b + rng.normal(0.005, 0.03, 25), b)
    assert big.decision == "A better"
    assert tiny.decision == "equivalent"
    assert noisy.decision == "inconclusive"


def test_zero_variance_is_handled():
    b = np.full(15, 0.4)
    same = C.corrected_resampled_ttest(b, b)
    assert same.p == 1.0 and same.t == 0.0
    shifted = C.corrected_resampled_ttest(b + 0.02, b)
    assert shifted.p == 0.0
    assert C.bayesian_correlated_ttest(b + 0.02, b).decision == "A better"
    assert C.bayesian_correlated_ttest(b + 0.005, b).decision == "equivalent"


def test_mde_gives_nominal_power():
    sd, J = 0.02, 25
    mde = C.minimum_detectable_effect(sd, J)
    se = sd * np.sqrt(1 / J + 0.25)
    crit = stats.t.ppf(0.975, J - 1)
    power = stats.nct.sf(crit, J - 1, mde / se) + stats.nct.cdf(-crit, J - 1, mde / se)
    assert abs(power - 0.80) < 0.02, power


def test_fold_alignment_guard():
    C.check_fold_alignment([1, 2, 3], [1, 2, 3])
    try:
        C.check_fold_alignment([1, 2, 3], [1, 3, 2])
    except ValueError:
        pass
    else:
        raise AssertionError("misaligned folds were accepted")


def test_dropin_matches_scipy_interface():
    a, b = _pair(seed=5)
    res = C.corrected_ttest_rel(a, b)
    t, p = res
    assert res.statistic == t and res.pvalue == p
    assert np.isclose(p, C.corrected_resampled_ttest(a, b).p)


def test_bad_inputs():
    for a, b in (([1.0], [1.0]), ([1.0, 2.0], [1.0]), (np.ones((2, 2)), np.ones((2, 2)))):
        try:
            C.corrected_resampled_ttest(a, b)
        except ValueError:
            continue
        raise AssertionError(f"accepted bad input {a!r}, {b!r}")


def test_agrees_with_the_independent_implementation():
    """cv_inference must reproduce the separate implementation written to cross-check it."""
    ref = pd.read_csv(ROOT / "outputs/honest/corrected_ttest_audit.csv")
    folds = pd.read_csv(ROOT / "outputs/honest/scrub_standardize_folds.csv")
    row = ref[ref["claim"] == "PSC > z-score, unscrubbed (2x2)"].iloc[0]
    ct = C.corrected_resampled_ttest(folds["psc_unscrubbed"], folds["zsc_unscrubbed"])
    assert np.isclose(ct.p, row["p_corrected"], rtol=0, atol=1e-12)
    assert np.isclose(ct.naive_p, row["p_naive"], rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
def simulate_null_fpr(n_datasets=400, n=150, p=10, repeats=5, k=5, seed=7):
    """False-positive rates under a TRUE null, with this project's CV structure."""
    rng = np.random.default_rng(seed)
    beta = rng.normal(0, 1, p) / np.sqrt(p)
    hits_naive = hits_corr = hits_bayes_wrong = 0
    for _ in range(n_datasets):
        XA, XB = rng.normal(size=(n, p)), rng.normal(size=(n, p))
        y = XA @ beta + XB @ beta + rng.normal(0, 2.0, n)
        ra, rb = [], []
        for rep in range(repeats):
            perm = rng.permutation(n)
            for f in range(k):
                te = perm[f::k]
                tr = np.setdiff1d(perm, te)
                for X, out in ((XA, ra), (XB, rb)):
                    Xt, yt = X[tr], y[tr]
                    w = np.linalg.solve(Xt.T @ Xt + 1.0 * np.eye(p), Xt.T @ (yt - yt.mean()))
                    pred = X[te] @ w
                    out.append(np.corrcoef(pred, y[te])[0, 1])
        ct = C.corrected_resampled_ttest(ra, rb, k=k)
        hits_naive += ct.naive_p < C.ALPHA
        hits_corr += ct.p < C.ALPHA
        hits_bayes_wrong += C.bayesian_correlated_ttest(ra, rb, k=k).decision in ("A better", "B better")
    return hits_naive / n_datasets, hits_corr / n_datasets, hits_bayes_wrong / n_datasets


def test_simulated_false_positive_rates_at_project_scale():
    """At n = 877 (ID1000) the corrected test must be calibrated; the naive one is not.

    400 null datasets give a Monte Carlo SE of ~0.011 at 0.05, so 0.075 is a
    ~2.3-SE margin. Small-n behaviour (n = 150: corrected FPR ~0.095 at 5x5) is a
    documented limitation of the correction, reported in the module docstring and
    checked separately below, not asserted away.
    """
    for repeats in (3, 5):
        naive, corrected, bayes = simulate_null_fpr(n=877, repeats=repeats)
        print(f"\n  null simulation n=877, {repeats}x5 CV: FPR naive {naive:.3f}, "
              f"corrected {corrected:.3f}, Bayesian false 'better' {bayes:.3f}")
        assert corrected <= 0.075, f"corrected FPR {corrected:.3f} at n=877, {repeats}x5"
        assert naive >= 0.20, f"naive FPR {naive:.3f} not inflated -- simulation lacks overlap"


def test_small_sample_limitation_is_as_documented():
    """At n = 150 with 5x5 CV the correction is known to under-correct (FPR ~2x nominal)."""
    naive, corrected, _ = simulate_null_fpr(n=150, repeats=5)
    print(f"\n  null simulation n=150, 5x5 CV: FPR naive {naive:.3f}, corrected {corrected:.3f}")
    assert 0.05 < corrected < 0.16, corrected
    assert naive > 3 * corrected


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:                                   # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
