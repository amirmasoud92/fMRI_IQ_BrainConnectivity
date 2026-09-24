"""Tests for the confidence intervals of the hold-out metrics."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/prediction"))

import numpy as np

import holdout_metrics as M


def _draw(rng, n, r_ya, r_yb, r_ab):
    cov = np.array([[1, r_ya, r_yb], [r_ya, 1, r_ab], [r_yb, r_ab, 1]])
    return rng.multivariate_normal(np.zeros(3), cov, size=n)


def test_fisher_ci_coverage():
    rng = np.random.default_rng(1)
    hits = 0
    for _ in range(3000):
        x = _draw(rng, 176, 0.40, 0.36, 0.85)
        lo, hi = M.fisher_ci(np.corrcoef(x[:, 0], x[:, 1])[0, 1], 176)
        hits += lo <= 0.40 <= hi
    cov = hits / 3000
    print(f"  Fisher CI coverage: {cov:.3f}")
    assert 0.93 <= cov <= 0.97, cov


def test_zou_ci_coverage_for_dependent_correlations():
    """Settings resembling stack vs FC: high correlation between the two predictions."""
    rng = np.random.default_rng(2)
    for r_ya, r_yb, r_ab in [(0.40, 0.36, 0.85), (0.36, 0.23, 0.55), (0.40, 0.40, 0.90)]:
        hits = 0
        for _ in range(3000):
            x = _draw(rng, 176, r_ya, r_yb, r_ab)
            y, a, b = x[:, 0], x[:, 1], x[:, 2]
            ra, rb, rab = (np.corrcoef(y, a)[0, 1], np.corrcoef(y, b)[0, 1],
                           np.corrcoef(a, b)[0, 1])
            lo, hi = M.zou_ci(ra, rb, rab, 176)
            hits += lo <= (r_ya - r_yb) <= hi
        cov = hits / 3000
        print(f"  Zou CI coverage (r_ya={r_ya}, r_yb={r_yb}, r_ab={r_ab}): {cov:.3f}")
        assert 0.93 <= cov <= 0.97, cov


def test_zou_ci_is_narrower_when_predictions_correlate():
    """Positive dependence between the two correlations must tighten the interval."""
    lo_ind, hi_ind = M.zou_ci(0.40, 0.36, 0.0, 176)
    lo_dep, hi_dep = M.zou_ci(0.40, 0.36, 0.9, 176)
    assert (hi_dep - lo_dep) < (hi_ind - lo_ind)


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
