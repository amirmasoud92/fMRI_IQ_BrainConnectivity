"""
Inference for comparing models scored by repeated k-fold cross-validation.

The problem
-----------
Every model comparison in this project reduces to J paired fold scores (a_j, b_j)
from repeated k-fold CV. The conventional paired t-test treats the J differences
as independent. They are not: any two training sets overlap in (k-2)/(k-1) of
their subjects, and repeats re-use every subject. The sample variance of the
differences therefore UNDER-estimates the variance of their mean, and naive
p-values are anti-conservative. In this project 15 of 24 naive-significant
claims lost significance once corrected.

What this module provides
-------------------------
corrected_resampled_ttest   Nadeau & Bengio (2003, Machine Learning 52:239-281),
                            in the repeated k-fold form of Bouckaert & Frank
                            (2004, PAKDD): the variance of the mean difference is
                            (1/J + n_test/n_train) * s^2 rather than s^2 / J, with
                            df = J - 1. For k-fold CV n_test/n_train = 1/(k-1),
                            which does NOT shrink as repeats are added.

bayesian_correlated_ttest   Corani & Benavoli (2015, Machine Learning 100:285) and
                            Benavoli, Corani, Demsar & Zaffalon (2017, JMLR
                            18(77):1-36). Under the same correlation structure,
                            the posterior of the true mean difference is a Student
                            t with df = J - 1, location mean(d) and scale
                            sqrt((1/J + rho/(1 - rho)) * s^2), rho = n_test/n_total.
                            With a region of practical equivalence (ROPE) it
                            separates "A better", "practically equivalent" and
                            "B better", which a p-value cannot: it is what licenses
                            calling a negative result EQUIVALENT rather than merely
                            not significant.

minimum_detectable_effect   The smallest true mean difference detectable at a given
                            power under the corrected variance.

check_fold_alignment        Refuses to pair scores from two runs whose fold indices
                            differ.

Both tests use the same variance inflation, so they cannot disagree about scale:
with ROPE = 0 the Bayesian P(A > B) equals one minus the corrected one-sided
p-value exactly (tested).

Calibration (measured, tests/test_cv_inference.py)
-----------------------------------------------
False-positive rate at alpha = 0.05 under a true null, ridge on two exchangeable
feature sets, 400 simulated datasets per cell:

      n   CV    naive   corrected   Bayesian false "better"
    877  3x5    0.280     0.052          0.058
    877  5x5    0.415     0.052          0.058
    400  5x5    0.537     0.050          0.070
    150  3x5    0.380     0.075          0.140
    150  5x5    0.480     0.095          0.128

At ID1000 scale (n = 700-877) the corrected test is calibrated. The naive test is
not, and gets WORSE with more repeats: repeats shrink its standard error without
adding independent information. At small n (PIOP, n ~ 215-225) the heuristic
correlation n_test/n_total under-states the true dependence and the corrected
test is still somewhat liberal (~0.06-0.08 expected); PIOP p-values near 0.05
must be read with that caveat.

Conventions
-----------
- Differences are a - b, so positive favours A.
- ROPE, alpha and the decision threshold are fixed module constants, set before
  any result was re-graded: ROPE = +/-0.01 r, alpha = 0.05,
  posterior decision threshold 0.95.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np
from scipy import stats

ROPE = 0.01
ALPHA = 0.05
DECISION_THRESHOLD = 0.95
DEFAULT_K = 5


def test_train_ratio(k: int = DEFAULT_K) -> float:
    """n_test / n_train for k-fold cross-validation."""
    if k < 2:
        raise ValueError("k-fold CV needs k >= 2")
    return 1.0 / (k - 1)


def _diffs(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(f"paired scores must be 1-D and equal length, got {a.shape} and {b.shape}")
    d = a - b
    keep = np.isfinite(d)
    if keep.sum() < 2:
        raise ValueError("need at least two finite paired differences")
    return d[keep]


@dataclass(frozen=True)
class CorrectedTTest:
    delta: float          # mean(a - b)
    se: float             # corrected standard error of the mean difference
    t: float
    df: int
    p: float              # two-sided, corrected
    ci_low: float
    ci_high: float
    n_folds: int
    wins: int             # folds where a > b (descriptive only: folds are not independent)
    naive_t: float
    naive_p: float        # scipy.stats.ttest_rel, reported for comparison only

    def as_dict(self) -> dict:
        return asdict(self)


def corrected_resampled_ttest(a: Sequence[float], b: Sequence[float], k: int = DEFAULT_K,
                              ratio: float | None = None, alpha: float = ALPHA) -> CorrectedTTest:
    """Nadeau-Bengio corrected repeated k-fold t-test of mean(a - b) = 0.

    ``ratio`` overrides n_test/n_train (e.g. for unequal splits); otherwise 1/(k-1).
    """
    d = _diffs(a, b)
    J = d.size
    r = test_train_ratio(k) if ratio is None else float(ratio)
    mean = float(d.mean())
    var = float(d.var(ddof=1))
    se = float(np.sqrt((1.0 / J + r) * var))
    df = J - 1
    if se == 0.0:
        t = 0.0 if mean == 0.0 else float(np.sign(mean) * np.inf)
        p = 1.0 if mean == 0.0 else 0.0
    else:
        t = mean / se
        p = float(2.0 * stats.t.sf(abs(t), df))
    half = float(stats.t.ppf(1 - alpha / 2, df) * se)
    naive_se = float(np.sqrt(var / J))
    if naive_se == 0.0:
        naive_t, naive_p = (0.0, 1.0) if mean == 0.0 else (float(np.sign(mean) * np.inf), 0.0)
    else:
        naive_t = mean / naive_se
        naive_p = float(2.0 * stats.t.sf(abs(naive_t), df))
    return CorrectedTTest(delta=mean, se=se, t=float(t), df=df, p=p,
                          ci_low=mean - half, ci_high=mean + half, n_folds=J,
                          wins=int((d > 0).sum()), naive_t=float(naive_t), naive_p=naive_p)


@dataclass(frozen=True)
class BayesianCorrelatedTTest:
    delta: float
    scale: float          # posterior scale of the mean difference
    df: int
    rope: float
    p_b_better: float     # P(delta < -rope)
    p_rope: float         # P(|delta| <= rope)
    p_a_better: float     # P(delta > +rope)
    decision: str         # "A better" | "equivalent" | "B better" | "inconclusive"

    def as_dict(self) -> dict:
        return asdict(self)


def bayesian_correlated_ttest(a: Sequence[float], b: Sequence[float], k: int = DEFAULT_K,
                              rope: float = ROPE, ratio: float | None = None,
                              threshold: float = DECISION_THRESHOLD) -> BayesianCorrelatedTTest:
    """Benavoli et al. (2017) correlated Bayesian t-test with a ROPE on mean(a - b).

    rho/(1 - rho) = n_test/n_train, so the posterior scale uses exactly the same
    inflation as the corrected frequentist test.
    """
    d = _diffs(a, b)
    J = d.size
    r = test_train_ratio(k) if ratio is None else float(ratio)
    mean = float(d.mean())
    var = float(d.var(ddof=1))
    scale = float(np.sqrt((1.0 / J + r) * var))
    df = J - 1
    if scale == 0.0:
        p_left = float(mean < -rope)
        p_right = float(mean > rope)
        p_rope = 1.0 - p_left - p_right
    else:
        dist = stats.t(df, loc=mean, scale=scale)
        p_left = float(dist.cdf(-rope))
        p_right = float(dist.sf(rope))
        p_rope = float(max(0.0, 1.0 - p_left - p_right))
    if p_right > threshold:
        decision = "A better"
    elif p_left > threshold:
        decision = "B better"
    elif p_rope > threshold:
        decision = "equivalent"
    else:
        decision = "inconclusive"
    return BayesianCorrelatedTTest(delta=mean, scale=scale, df=df, rope=rope,
                                   p_b_better=p_left, p_rope=p_rope, p_a_better=p_right,
                                   decision=decision)


def minimum_detectable_effect(sd_diff: float, n_folds: int, k: int = DEFAULT_K,
                              alpha: float = ALPHA, power: float = 0.80,
                              ratio: float | None = None) -> float:
    """Smallest true mean difference detectable with the given power (two-sided)."""
    r = test_train_ratio(k) if ratio is None else float(ratio)
    df = n_folds - 1
    crit = stats.t.ppf(1 - alpha / 2, df) + stats.t.ppf(power, df)
    return float(crit * sd_diff * np.sqrt(1.0 / n_folds + r))


def verdict(ct: CorrectedTTest, bt: BayesianCorrelatedTTest) -> str:
    """One-word summary used in tables.

    established   corrected p < alpha
    equivalent    not established, and the posterior puts > 95% mass inside the ROPE
    inconclusive  neither: the data cannot distinguish a real difference from none
    """
    if ct.p < ALPHA:
        return "established"
    if bt.decision == "equivalent":
        return "equivalent"
    return "inconclusive"


def compare(a: Sequence[float], b: Sequence[float], k: int = DEFAULT_K,
            rope: float = ROPE) -> dict:
    """Both tests plus MDE and verdict, flattened for tabulation."""
    ct = corrected_resampled_ttest(a, b, k=k)
    bt = bayesian_correlated_ttest(a, b, k=k, rope=rope)
    sd = float(_diffs(a, b).std(ddof=1))
    out = {f"{key}": val for key, val in ct.as_dict().items()}
    out.update({"p_a_better": bt.p_a_better, "p_rope": bt.p_rope, "p_b_better": bt.p_b_better,
                "bayes_decision": bt.decision, "rope": rope,
                "mde_80": minimum_detectable_effect(sd, ct.n_folds, k=k),
                "verdict": verdict(ct, bt)})
    return out


def check_fold_alignment(fold_a: Sequence, fold_b: Sequence) -> None:
    """Raise unless two runs report the same fold indices in the same order."""
    fa, fb = list(fold_a), list(fold_b)
    if fa != fb:
        n = sum(1 for x, y in zip(fa, fb) if x != y)
        raise ValueError(f"fold misalignment: {n} positions differ "
                         f"(len {len(fa)} vs {len(fb)}); refusing to pair")


def corrected_ttest_rel(a: Sequence[float], b: Sequence[float], k: int = DEFAULT_K):
    """Drop-in for scipy.stats.ttest_rel(a, b) returning the CORRECTED (t, p).

    Only valid when a and b are fold scores from repeated k-fold CV on the same
    folds. Returns a scipy-like result with ``statistic`` and ``pvalue`` so that
    ``t, p = corrected_ttest_rel(x, y)`` works at existing call sites.
    """
    ct = corrected_resampled_ttest(a, b, k=k)
    return _TTestResult(ct.t, ct.p)


class _TTestResult(tuple):
    def __new__(cls, statistic, pvalue):
        return super().__new__(cls, (statistic, pvalue))

    @property
    def statistic(self):
        return self[0]

    @property
    def pvalue(self):
        return self[1]
