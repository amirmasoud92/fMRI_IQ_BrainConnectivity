#!/usr/bin/env python
"""
Test the ID1000 low-pass gain in PIOP1 and PIOP2.

Usage
-----
  python scripts/generalisation/broadband_replication_piop.py --dataset PIOP2 --repeats 3
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, ttest_rel
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from src.data.paths import collection_root
from src.stats.cv_inference import corrected_ttest_rel

warnings.filterwarnings("ignore")

OUT = Path("outputs/piop")
ALPHAS = np.logspace(-2, 6, 25)
SEED = 42
ICV = "EstimatedTotalIntraCranialVol"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOTS = {name: collection_root(name) for name in ("PIOP1", "PIOP2")}


def sym_funcm(X, fn):
    X = 0.5 * (X + X.transpose(-1, -2)).double()
    w, V = torch.linalg.eigh(X)
    return V @ torch.diag_embed(fn(w.clamp_min(1e-10))) @ V.transpose(-1, -2)


def upper(L):
    n = L.shape[-1]
    iu = torch.triu_indices(n, n, offset=1, device=L.device)
    return torch.cat([torch.diagonal(L, dim1=-2, dim2=-1),
                      L[..., iu[0], iu[1]] * np.sqrt(2.0)], dim=-1).float()


def feats(series):
    covs = np.stack([ledoit_wolf(x, assume_centered=False)[0] for x in series])
    C = torch.from_numpy(covs).to(DEV).double()
    fc = upper(sym_funcm(C, torch.log)).cpu().numpy()
    amp = np.stack([np.log(x.std(0)) for x in series])
    return fc, amp


def rp(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    mu = A.mean(0); A, B = A - mu, B - mu
    if A.shape[1] > A.shape[0]:
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        k = int((S > S[0] * 1e-10).sum())
        A, B = U[:, :k] * S[:k], B @ Vt[:k].T
    return RidgeCV(alphas=ALPHAS, cv=5).fit(A, ytr).predict(B)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(ROOTS), required=True)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    ds, root = args.dataset, ROOTS[args.dataset]

    parts = pd.read_csv(root / "participants.tsv", sep="\t", na_values="n/a") \
        .set_index("participant_id").dropna(subset=["raven_score"])
    aseg = pd.read_csv(root / "derivatives" / "fs_stats" /
                       "data-subcortical_type-aseg_measure-volume_hemi-both.tsv", sep="\t")
    aseg = aseg.rename(columns={aseg.columns[0]: "sid"}).set_index("sid")
    icv = pd.to_numeric(aseg[ICV], errors="coerce")

    with h5py.File(OUT / f"{ds}_broad_ts.h5", "r") as f:
        tasks = sorted({t for s in f["subjects"] for t in f["subjects"][s]})
    bar = "=" * 84
    print(bar); print(f"{ds}: DOES THE LOW-PASS GAIN REPLICATE?"); print(bar)
    print(f"  ID1000 reference: deconfounded +0.3348 -> +0.3878 (+0.0530, "
          f"p = 0.00017)\n")

    rows = []
    print(f"  {'task':16s} {'n':>4s} {'base':>9s} {'broad':>9s} {'gain':>8s} "
          f"{'p':>9s} | {'base dec':>9s} {'broad dec':>10s} {'gain':>8s} {'p':>9s}")
    for task in tasks:
        data = {}
        for v in ("base", "broad"):
            with h5py.File(OUT / f"{ds}_{v}_ts.h5", "r") as f:
                d = {}
                for s in parts.index:
                    if s in f["subjects"] and task in f["subjects"][s] \
                            and np.isfinite(icv.get(s, np.nan)):
                        g = f["subjects"][s][task]
                        d[s] = (g["time_series"][:].astype(np.float64),
                                np.asarray(g["fd"][:], float))
                data[v] = d
        ids = sorted(set(data["base"]) & set(data["broad"]))
        if len(ids) < 60:
            continue
        y = parts.loc[ids, "raven_score"].to_numpy(float)
        sex = np.array([1 if str(parts.loc[s, "sex"]).upper().startswith("M") else 0
                        for s in ids])
        v_ = icv.loc[ids].to_numpy(float); vz = (v_ - v_.mean()) / v_.std()
        fd = np.array([data["base"][s][1][1:] for s in ids], dtype=object)
        mot = np.array([[a.mean(), a.max(), (a > 0.5).mean()] for a in fd])
        Zc = np.column_stack([sex, vz, vz ** 2, mot])
        q = pd.qcut(y, 4, labels=False, duplicates="drop")
        strat = np.asarray(q) * 2 + sex
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats,
                                       random_state=SEED)
        F = {v: feats([data[v][s][0] for s in ids]) for v in ("base", "broad")}
        acc = {f"{v}_{k}": [] for v in ("base", "broad") for k in ("raw", "dec")}
        motr = {}
        for v in ("base", "broad"):
            motr[v] = pearsonr(mot[:, 0], F[v][1].mean(1))[0]
        for tr, te in rskf.split(np.zeros(len(ids)), strat):
            lr = LinearRegression().fit(Zc[tr], y[tr])
            ytr, yte = y[tr] - lr.predict(Zc[tr]), y[te] - lr.predict(Zc[te])
            for v in ("base", "broad"):
                X = F[v][0]
                acc[f"{v}_raw"].append(pearsonr(y[te], rp(X[tr], y[tr], X[te]))[0])
                acc[f"{v}_dec"].append(pearsonr(yte, rp(X[tr], ytr, X[te]))[0])
        a = {k: np.array(v) for k, v in acc.items()}
        t1, p1 = corrected_ttest_rel(a["broad_raw"], a["base_raw"])
        t2, p2 = corrected_ttest_rel(a["broad_dec"], a["base_dec"])
        rows.append(dict(dataset=ds, task=task, n=len(ids),
                         base=a["base_raw"].mean(), broad=a["broad_raw"].mean(),
                         gain=a["broad_raw"].mean() - a["base_raw"].mean(), p=p1,
                         base_dec=a["base_dec"].mean(), broad_dec=a["broad_dec"].mean(),
                         gain_dec=a["broad_dec"].mean() - a["base_dec"].mean(),
                         p_dec=p2, motion_base=motr["base"], motion_broad=motr["broad"]))
        r = rows[-1]
        print(f"  {task:16s} {r['n']:4d} {r['base']:+9.4f} {r['broad']:+9.4f} "
              f"{r['gain']:+8.4f} {p1:9.3g} | {r['base_dec']:+9.4f} "
              f"{r['broad_dec']:+10.4f} {r['gain_dec']:+8.4f} {p2:9.3g}", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(OUT / f"{ds}_broadband_replication.csv", index=False)
    print(f"\n{bar}\nSUMMARY\n{bar}")
    t, p = ttest_rel(d.broad_dec, d.base_dec)  # scipy kept: test across scan conditions (cells), not CV folds
    print(f"  across {len(d)} conditions: mean deconfounded gain "
          f"{d.gain_dec.mean():+.4f}  (t = {t:+.2f}, p = {p:.3g}, "
          f"positive in {int((d.gain_dec > 0).sum())}/{len(d)})")
    print(f"  mean raw gain {d.gain.mean():+.4f}; ID1000 had a LARGER "
          f"deconfounded than raw gain")
    print(f"  motion-amplitude correlation: base {d.motion_base.mean():.3f} -> "
          f"broad {d.motion_broad.mean():.3f}")
    print("\n  A positive deconfounded gain here means the low-pass finding is a")
    print("  property of the signal, established on data that never selected it.")
    print(f"\nSaved -> {OUT}/{ds}_broadband_replication.csv")


if __name__ == "__main__":
    main()
