#!/usr/bin/env python
"""
Pool the three collections with per-collection alignment, and compare the gain with the one the
cohort-size curve predicts.

Alignment Arms
--------------
  none              pooled with no correction beyond global standardisation
  recenter          per-dataset tangent mean removed
  recenter_scale    also per-dataset dispersion matched
  recenter_zscore   per-dataset per-feature standardisation

In the log-Euclidean chart used throughout this project, Riemannian recentering
-- transporting each site's Frechet mean to a common reference -- is EXACTLY
subtraction of that site's mean tangent vector. This is worth stating plainly
rather than dressing up: the manifold framing motivates the operation and tells
us it is the right one, but the operation itself is per-site mean removal.
Dispersion matching (dividing by the mean distance to the site mean) is the
scale analogue, and is the second step of the standard recentre-and-stretch
transfer recipe.

Leakage
-------
Alignment statistics for the TARGET dataset are computed from its training folds
only, never from the held-out fold. Statistics for the other datasets use all of
their subjects, which is legitimate because those subjects are entirely inside
the training set. Confound models are likewise fitted per dataset on training
rows only.

Usage
-----
  python scripts/generalisation/cross_collection_pooling.py --repeats 3
"""
from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.stats.cv_inference import corrected_ttest_rel

warnings.filterwarnings("ignore")

OUT = Path("outputs/piop")
FEATURES = os.environ.get("POOLED_FEATURES", "pooled_features.npz")
ALPHAS = np.logspace(-2, 6, 25)
SEED = 42
DATASETS = ["ID1000", "PIOP1", "PIOP2"]
ARMS = ["none", "recenter", "recenter_scale", "recenter_zscore"]
R_INF_N, N0 = 0.5081075213207266, 274.7075074372864     # ID1000 ceiling fit


def sat(n):
    return np.sqrt(n / (n + N0))


def align(X, ds, arm, fit_rows):
    """Apply per-dataset alignment; statistics estimated on fit_rows only."""
    Xa = X.copy()
    if arm == "none":
        return Xa
    for d in DATASETS:
        m = ds == d
        f = m & fit_rows
        if f.sum() < 10:
            continue
        mu = Xa[f].mean(0)
        Xa[m] -= mu
        if arm == "recenter_scale":
            disp = np.sqrt((Xa[f] ** 2).sum(1).mean())     # mean distance to site mean
            if disp > 0:
                Xa[m] /= disp
        elif arm == "recenter_zscore":
            sd = Xa[f].std(0)
            Xa[m] /= np.clip(sd, 1e-8, None)
    return Xa


def project(Xtr, Xte):
    """Standardise then reduce to the training row space.

    Verified equal to ridge on the full feature set to 1.5e-15 in prediction.
    Returned once per (arm, training set) and reused for both the raw and the
    deconfounded target, since the projection does not depend on y.
    """
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    mu = A.mean(0)
    A, B = A - mu, B - mu
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    k = int((S > S[0] * 1e-10).sum())
    return U[:, :k] * S[:k], B @ Vt[:k].T


def fit_pred(Ztr, ytr, Zte):
    return RidgeCV(alphas=ALPHAS, cv=5).fit(Ztr, ytr).predict(Zte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    d = np.load(OUT / FEATURES, allow_pickle=True)
    X, y, Z, ds = d["tangent"], d["y_raw"], d["Z"], d["dataset"].astype(str)
    yz = d["y"]
    n_by = {k: int((ds == k).sum()) for k in DATASETS}
    bar = "=" * 82
    print(bar); print("RIEMANNIAN MULTI-DATASET POOLING"); print(bar)
    print(f"  {sum(n_by.values())} subjects total: "
          + ", ".join(f"{k} {v}" for k, v in n_by.items()))
    print(f"  tangent dimension {X.shape[1]}")

    rows = []
    for target in DATASETS:
        tm = ds == target
        yt, sext = y[tm], Z[tm][:, 0].astype(int)
        q = pd.qcut(yt, 4, labels=False, duplicates="drop")
        strat = np.asarray(q) * 2 + sext
        idx_t = np.where(tm)[0]
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats,
                                       random_state=SEED)

        n_alone = int(round(n_by[target] * 0.8))
        n_pool = n_alone + sum(v for k, v in n_by.items() if k != target)
        pred_gain = sat(n_pool) / sat(n_alone) - 1
        print(f"\n{bar}\nTARGET {target}   n_train {n_alone} alone -> {n_pool} pooled"
              f"   law predicts {pred_gain:+.1%} relative\n{bar}")

        fold = []
        for k, (tr_, te_) in enumerate(rskf.split(np.zeros(tm.sum()), strat), 1):
            tr, te = idx_t[tr_], idx_t[te_]
            fit_rows = np.zeros(len(ds), bool)
            fit_rows[tr] = True
            fit_rows[~tm] = True                    # other datasets are all training

            # confound residuals: per dataset, fitted on that dataset's training rows
            ydec = np.zeros(len(ds))
            for dd in DATASETS:
                m = ds == dd
                f = m & fit_rows
                lr = LinearRegression().fit(Z[f], y[f])
                ydec[m] = y[m] - lr.predict(Z[m])

            rec = {"fold": k}
            others = np.where(~tm)[0]
            for arm in ARMS:
                Xa = align(X, ds, arm, fit_rows)
                for tag, trn in (("alone", tr), ("pooled", np.concatenate([tr, others]))):
                    if tag == "alone" and arm not in ("none", "recenter_zscore"):
                        continue                    # alignment is a no-op single-site
                    Ztr, Zte = project(Xa[trn], Xa[te])
                    rec[f"{arm}_{tag}"] = float(pearsonr(
                        y[te], fit_pred(Ztr, yz[trn], Zte))[0])
                    rec[f"{arm}_{tag}_dec"] = float(pearsonr(
                        ydec[te], fit_pred(Ztr, ydec[trn], Zte))[0])
            fold.append(rec)
            print(f"    fold {k:2d}/{args.repeats*5}  "
                  f"alone={rec['none_alone']:+.3f}  "
                  f"pooled(recenter)={rec['recenter_pooled']:+.3f}", flush=True)

        f = pd.DataFrame(fold); f.insert(0, "target", target)
        rows.append(f)
        base = f["none_alone"].mean()
        print(f"\n  {'arm':18s} {'r':>9s} {'vs alone':>10s} {'p':>9s} "
              f"{'% of predicted':>15s}   deconfounded")
        print(f"  {'alone (baseline)':18s} {base:+9.4f} {'':>10s} {'':>9s} {'':>15s}"
              f"   {f['none_alone_dec'].mean():+.4f}")
        for arm in ARMS:
            c = f"{arm}_pooled"
            gain = f[c].mean() - base
            t, p = corrected_ttest_rel(f[c], f["none_alone"])
            frac = gain / (base * pred_gain) if base * pred_gain > 0 else np.nan
            print(f"  {'pooled ' + arm:18s} {f[c].mean():+9.4f} {gain:+10.4f} "
                  f"{p:9.3g} {frac:14.0%}   {f[c + '_dec'].mean():+.4f}")

    df = pd.concat(rows, ignore_index=True)
    df.to_csv(OUT / "riemannian_pooling_folds.csv", index=False)
    print(f"\n{bar}\nSUMMARY\n{bar}")
    print(f"  {'target':10s} {'alone':>9s} {'best pooled':>13s} {'gain':>9s} "
          f"{'predicted':>11s} {'realised':>10s}")
    for target, g in df.groupby("target", sort=False):
        base = g["none_alone"].mean()
        best = max(ARMS, key=lambda a: g[f"{a}_pooled"].mean())
        gain = g[f"{best}_pooled"].mean() - base
        n_alone = int(round(n_by[target] * 0.8))
        n_pool = n_alone + sum(v for k, v in n_by.items() if k != target)
        pg = base * (sat(n_pool) / sat(n_alone) - 1)
        print(f"  {target:10s} {base:+9.4f} {g[f'{best}_pooled'].mean():+13.4f} "
              f"{gain:+9.4f} {pg:+11.4f} {gain/pg if pg else np.nan:9.0%}   ({best})")
    print(f"\nSaved -> {OUT}/riemannian_pooling_folds.csv")


if __name__ == "__main__":
    main()
