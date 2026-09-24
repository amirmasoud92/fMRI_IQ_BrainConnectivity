"""Tests for the confound-control analyses on synthetic data."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/confounds"))

import numpy as np
import pandas as pd

import confound_controls as M


def synthetic(n=420, p_fc=60, seed=0):
    rng = np.random.default_rng(seed)
    sex = (rng.random(n) < 0.5).astype(float)
    etiv = 1.5e6 + 1.2e5 * sex + rng.normal(0, 1e5, n)
    fd_mean = np.abs(rng.normal(0.15, 0.05, n))
    fd_max = fd_mean * 4 + np.abs(rng.normal(0, 0.2, n))
    spike = np.where(rng.random(n) < 0.7, 0.0, rng.random(n) * 0.05)
    latent = rng.normal(0, 1, n)
    y = 200 + 25 * latent + 8e-5 * (etiv - etiv.mean()) + 5 * sex + rng.normal(0, 20, n)
    X_fc = rng.normal(0, 1, (n, p_fc)) + np.outer(latent, rng.normal(0, 0.3, p_fc))
    Mo = rng.normal(0, 1, (n, 20)); Mo[rng.random(Mo.shape) < 0.01] = np.nan
    Fa = rng.normal(0, 1, (n, 5))
    C = pd.DataFrame({"sex": sex, "etiv": etiv, "fd_mean": fd_mean, "fd_max": fd_max, "spike": spike})
    return y, C, X_fc.astype(np.float32), Mo.astype(np.float32), Fa.astype(np.float32)


def test_spline_design_uses_training_knots_and_handles_spike():
    y, C, *_ = synthetic()
    Dtr, Dte, terms = M.spline_design(C.iloc[:300].reset_index(drop=True),
                                      C.iloc[300:].reset_index(drop=True))
    assert Dtr.shape[0] == 300 and Dte.shape[0] == 120 and Dtr.shape[1] == Dte.shape[1]
    assert any(t.startswith("cr(etiv") for t in terms)
    assert "spike" in terms, "spike (mostly zeros) must fall back to a linear term"


def test_isolation_removes_dependence_and_keeps_variance():
    y, C, *_ = synthetic(n=700, seed=3)
    for conf in ("etiv", "sex", "joint"):
        start = np.random.default_rng(5).choice(len(y), 350, replace=False)
        pool, ok, steps = M.isolate(start, y, C, conf)
        cols = {"etiv": ["etiv"], "sex": ["sex"], "joint": ["etiv", "fd_mean", "sex"]}[conf]
        dep = max(abs(np.corrcoef(y[pool], C.iloc[pool][c])[0, 1]) for c in cols)
        print(f"  {conf:6s} ok={ok} n={len(pool)} steps={steps} |r| after={dep:.3f} "
              f"sd ratio={y[pool].std() / y[start].std():.2f}")
        assert ok and dep <= M.ISO_TOL and len(pool) >= M.MIN_TEST
        # Isolation restricts IST range (the reason for the IST-matched control) but must
        # not collapse it. Joint isolation of three confounds trims hardest (SD ratio 0.50
        # in this synthetic draw); 0.4 is a sanity floor, not an accuracy target.
        assert y[pool].std() / y[start].std() > 0.4


def test_ist_matched_control_reproduces_the_range_restriction():
    """The IST-matched random set must share the isolated set's IST spread."""
    y, C, X_fc, *_ = synthetic(n=700, seed=3)
    out = M.run_isolated(0, "etiv", y, C, X_fc, len(y))
    ratio_iso = out["y_sd_isolated"] / out["y_sd_start"]
    ratio_ym = out["y_sd_ymatch"] / out["y_sd_start"]
    print(f"  IST SD / start: isolated {ratio_iso:.2f}, IST-matched random {ratio_ym:.2f}")
    assert abs(ratio_ym - ratio_iso) < 0.12, (ratio_iso, ratio_ym)


def test_end_to_end_small():
    y, C, X_fc, Mo, Fa = synthetic()
    strat = np.asarray(pd.qcut(y, 4, labels=False)) * 2 + C.sex.to_numpy().astype(int)
    from sklearn.model_selection import RepeatedStratifiedKFold
    folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1).split(np.zeros(len(y)), strat))
    res = [M.run_fold(k, tr, te, y, C, X_fc, Mo, Fa, None) for k, (tr, te) in enumerate(folds, 1)]
    A = pd.DataFrame([a for a, _ in res]); B = pd.DataFrame([b for _, b in res])
    iso = [M.run_isolated(j, c, y, C, X_fc, len(y)) for c in ("etiv", "sex", "fd_mean", "joint")
           for j in range(2)]
    Ciso = pd.DataFrame(iso)
    S, V = M.summarise(A, B, Ciso)
    assert len(A) == 5 and len(B) == 5 and len(Ciso) == 8
    for col in ("fc_raw", "fc_lin", "fc_spline", "stack_raw", "increment_spline"):
        assert A[col].notna().all(), col
    assert set(V.criterion.str[:2]) == {"C1", "C2", "C3", "C4", "C5", "C6"}
    assert (A["fc_raw"] > 0).mean() >= 0.6, "synthetic signal not recovered"
    print(f"  synthetic fc_raw {A.fc_raw.mean():+.3f}, fc_spline {A.fc_spline.mean():+.3f}, "
          f"isolated fc {Ciso.fc_iso.mean():+.3f}")


def test_isolation_seeds_are_deterministic():
    y, C, X_fc, *_ = synthetic(n=300)
    a = M.run_isolated(0, "etiv", y, C, X_fc, len(y))
    b = M.run_isolated(0, "etiv", y, C, X_fc, len(y))
    assert a["n_test"] == b["n_test"] and a["fc_iso"] == b["fc_iso"]


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
