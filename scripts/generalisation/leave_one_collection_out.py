#!/usr/bin/env python
"""
Train on two collections and predict the third, without harmonisation beyond feature standardisation.

Design
------
Three folds by construction: hold out ID1000, then PIOP1, then PIOP2. Every
subject of the two remaining collections trains; every subject of the held-out
one is scored. No cross-validation is needed or meaningful, because the test set
shares no subject, no scanner session and no psychometric instrument-scoring
with the training set.

Spearman is the primary metric. The training target mixes IST and Raven through
a within-collection rank-based normal transform, so only the ordering is
guaranteed to be commensurate; Pearson is reported alongside.

Confidence intervals come from bootstrapping the held-out collection, which
reflects the only sampling variability that exists in this design.

Arms
----
  none        pooled without correction beyond global standardisation
  recenter    each collection's mean tangent vector removed
  zscore      each collection standardised feature-wise

Alignment statistics for the HELD-OUT collection are computed from its own
features without using its targets. That is legitimate and is what would be
available in practice for a new site, but it is an unsupervised use of test
features and is stated rather than hidden; the "none" arm is reported so the
cost of assuming even that much is visible.

Usage
-----
  python scripts/generalisation/leave_one_collection_out.py
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

OUT = Path("outputs/piop")
FEATURES = os.environ.get("POOLED_FEATURES", "pooled_features.npz")
ALPHAS = np.logspace(-2, 6, 25)
SEED = 42
DATASETS = ["ID1000", "PIOP1", "PIOP2"]
N_BOOT = 2000


def align(X, ds, arm):
    Xa = X.copy()
    if arm == "none":
        return Xa
    for d in DATASETS:
        m = ds == d
        Xa[m] -= Xa[m].mean(0)
        if arm == "zscore":
            Xa[m] /= np.clip(Xa[m].std(0), 1e-8, None)
    return Xa


def fit_predict(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    mu = A.mean(0); A, B = A - mu, B - mu
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    k = int((S > S[0] * 1e-10).sum())
    return RidgeCV(alphas=ALPHAS, cv=5).fit(U[:, :k] * S[:k], ytr).predict(B @ Vt[:k].T)


def boot_ci(y, p, rng, n=N_BOOT):
    v = [spearmanr(y[i], p[i])[0] for i in
         (rng.randint(0, len(y), len(y)) for _ in range(n))]
    return np.percentile(v, [2.5, 97.5])


def main():
    rng = np.random.RandomState(SEED)
    d = np.load(OUT / FEATURES, allow_pickle=True)
    X, y, yz, Z, ds = (d["tangent"], d["y_raw"], d["y"], d["Z"],
                       d["dataset"].astype(str))
    bar = "=" * 80
    print(bar); print("LEAVE-ONE-DATASET-OUT"); print(bar)
    print("  " + ", ".join(f"{k} {int((ds==k).sum())}" for k in DATASETS))

    rows = []
    for held in DATASETS:
        te = ds == held
        tr = ~te
        # confounds: model fitted on the training collections only
        lr = LinearRegression().fit(Z[tr], yz[tr])
        ytr_d = yz[tr] - lr.predict(Z[tr])
        yte_d = y[te] - LinearRegression().fit(Z[te], y[te]).predict(Z[te])
        print(f"\n{bar}\nheld out: {held}  "
              f"(train {int(tr.sum())}, test {int(te.sum())})\n{bar}")
        print(f"  {'arm':10s} {'Spearman':>10s} {'95% CI':>20s} "
              f"{'Pearson':>10s} {'deconf rho':>11s}")
        for arm in ("none", "recenter", "zscore"):
            Xa = align(X, ds, arm)
            p = fit_predict(Xa[tr], yz[tr], Xa[te])
            rho = spearmanr(y[te], p)[0]
            r = pearsonr(y[te], p)[0]
            pd_ = fit_predict(Xa[tr], ytr_d, Xa[te])
            rho_d = spearmanr(yte_d, pd_)[0]
            lo, hi = boot_ci(y[te], p, rng)
            print(f"  {arm:10s} {rho:+10.4f} [{lo:+.3f}, {hi:+.3f}]"
                  f"{'':>4s} {r:+10.4f} {rho_d:+11.4f}")
            rows.append(dict(held_out=held, arm=arm, spearman=rho, pearson=r,
                             spearman_dec=rho_d, ci_lo=lo, ci_hi=hi,
                             n_train=int(tr.sum()), n_test=int(te.sum())))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "leave_one_dataset_out.csv", index=False)

    print(f"\n{bar}\nSUMMARY (best arm per held-out collection)\n{bar}")
    for held, g in df.groupby("held_out", sort=False):
        b = g.loc[g.spearman.idxmax()]
        flag = "" if b.ci_lo > 0 else "   CI includes zero"
        print(f"  {held:8s} rho = {b.spearman:+.4f} "
              f"[{b.ci_lo:+.3f}, {b.ci_hi:+.3f}]  ({b.arm}){flag}")
    print("\n  A collection predicted above chance by a model that never saw it")
    print("  supports a dataset-invariant representation rather than a")
    print("  collection-specific correlation.")
    print(f"\nSaved -> {OUT}/leave_one_dataset_out.csv")


if __name__ == "__main__":
    main()
