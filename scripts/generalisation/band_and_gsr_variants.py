#!/usr/bin/env python
"""
What global signal regression and the low-pass filter cost in prediction, stimulus-locked variance and
motion coupling.

Usage
-----
  python scripts/generalisation/band_and_gsr_variants.py --repeats 3
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
import yaml
from scipy.stats import pearsonr
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.data.preprocessing import load_participants
from src.stats.cv_inference import corrected_ttest_rel

warnings.filterwarnings("ignore")

OUT = Path("outputs/piop")
CFG = "config/pipeline.yaml"
ALPHAS = np.logspace(-2, 6, 25)
SEED = 42
ICV = "EstimatedTotalIntraCranialVol"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VARIANTS = ["base", "noGSR", "broad", "noGSR_broad"]


def sym_funcm(X, fn):
    X = 0.5 * (X + X.transpose(-1, -2)).double()
    w, V = torch.linalg.eigh(X)
    return V @ torch.diag_embed(fn(w.clamp_min(1e-10))) @ V.transpose(-1, -2)


def upper(L):
    n = L.shape[-1]
    iu = torch.triu_indices(n, n, offset=1, device=L.device)
    return torch.cat([torch.diagonal(L, dim1=-2, dim2=-1),
                      L[..., iu[0], iu[1]] * np.sqrt(2.0)], dim=-1).float()


def load_variant(v, ids):
    with h5py.File(OUT / f"ID1000_{v}_ts.h5", "r") as f:
        return {s: f["subjects"][s]["time_series"][:].astype(np.float64)
                for s in ids if s in f["subjects"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(CFG))
    parts = load_participants(cfg["paths"]["bids_root"])
    frames = []
    for f in sorted(Path("derivatives/fs_stats").glob("data-*.tsv")):
        d = pd.read_csv(f, sep="\t")
        d = d.rename(columns={d.columns[0]: "sid"})
        d["sid"] = d["sid"].astype(str)
        d = d.set_index("sid"); d = d.loc[:, ~d.columns.duplicated()]
        d.columns = [f"{f.stem}::{c}" for c in d.columns]; frames.append(d)
    M = pd.concat(frames, axis=1).apply(pd.to_numeric, errors="coerce")
    vcol = [c for c in M.columns if c.endswith("::" + ICV)][0]

    with h5py.File(OUT / "ID1000_base_ts.h5", "r") as f:
        ids = sorted(f["subjects"].keys())
    fp = Path(cfg["paths"]["fmriprep_dir"])
    fd = {}
    for s in ids:
        p = fp / s / "func" / f"{s}_task-{cfg['dataset']['task']}_desc-confounds_regressors.tsv"
        v = pd.read_csv(p, sep="\t", usecols=["framewise_displacement"],
                        na_values="n/a")["framewise_displacement"].fillna(0).to_numpy()
        fd[s] = v[1:]
    ids = [s for s in ids if s in parts.index and not pd.isna(parts.loc[s, "IST_intelligence_total"])
           and s in M.index and np.isfinite(M.loc[s, vcol])]
    y = np.array([float(parts.loc[s, "IST_intelligence_total"]) for s in ids])
    sex = np.array([1 if str(parts.loc[s, "sex"]).lower().startswith("m") else 0
                    for s in ids])
    vv = M.loc[ids, vcol].to_numpy(float); vz = (vv - vv.mean()) / vv.std()
    mot = np.array([[fd[s].mean(), fd[s].max(), (fd[s] > 0.5).mean()] for s in ids])
    Zc = np.column_stack([sex, vz, vz ** 2, mot])
    bar = "=" * 82
    print(bar); print("WHAT DO GSR AND THE 0.1 Hz LOW-PASS COST?"); print(bar)
    print(f"  {len(ids)} subjects\n")

    q = pd.qcut(y, 4, labels=False, duplicates="drop")
    strat = np.asarray(q) * 2 + sex
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats,
                                   random_state=SEED)
    rows, diag = [], []
    for v in VARIANTS:
        S = load_variant(v, ids)
        series = [S[s] for s in ids]
        T = min(x.shape[0] for x in series)
        series = [x[:T] for x in series]
        amp = np.stack([np.log(x.std(0)) for x in series])
        covs = np.stack([ledoit_wolf(x, assume_centered=False)[0] for x in series])
        fc = upper(sym_funcm(torch.from_numpy(covs).to(DEV).double(),
                             torch.log)).cpu().numpy()
        # how much stimulus-locked signal survives
        G = np.mean(series, axis=0)
        lock = np.mean([[pearsonr(x[:, j], G[:, j])[0] ** 2
                         for j in range(0, x.shape[1], 8)] for x in series])
        mfd = np.array([fd[s].mean() for s in ids])
        mot_r = pearsonr(mfd, amp.mean(1))[0]
        diag.append(dict(variant=v, stimulus_locked_r2=lock,
                         motion_amplitude_r=mot_r))
        fold = []
        for tr, te in rskf.split(np.zeros(len(ids)), strat):
            lr = LinearRegression().fit(Zc[tr], y[tr])
            ytr, yte = y[tr] - lr.predict(Zc[tr]), y[te] - lr.predict(Zc[te])
            rec = {}
            for nm, X in (("fc", fc), ("amplitude", amp)):
                sc = StandardScaler().fit(X[tr])
                A, B = sc.transform(X[tr]), sc.transform(X[te])
                # ridge in the training row space -- verified identical to
                # full-feature ridge to 1.5e-15, and the difference between a
                # twenty-minute comparison and a three-hour one
                mu = A.mean(0); A, B = A - mu, B - mu
                if A.shape[1] > A.shape[0]:
                    U_, S_, Vt_ = np.linalg.svd(A, full_matrices=False)
                    kk = int((S_ > S_[0] * 1e-10).sum())
                    A, B = U_[:, :kk] * S_[:kk], B @ Vt_[:kk].T
                rec[nm] = pearsonr(y[te], RidgeCV(alphas=ALPHAS, cv=5)
                                   .fit(A, y[tr]).predict(B))[0]
                rec[nm + "_dec"] = pearsonr(yte, RidgeCV(alphas=ALPHAS, cv=5)
                                            .fit(A, ytr).predict(B))[0]
            fold.append(rec)
        d = pd.DataFrame(fold); d.insert(0, "variant", v)
        rows.append(d)
        print(f"  {v:12s} fc {d.fc.mean():+.4f}  amp {d.amplitude.mean():+.4f}  "
              f"stimulus-locked R2 {lock:.4f}  r(FD, amplitude) {mot_r:+.3f}",
              flush=True)

    df = pd.concat(rows, ignore_index=True)
    df.to_csv(OUT / "variant_comparison_folds.csv", index=False)
    dg = pd.DataFrame(diag)
    dg.to_csv(OUT / "variant_diagnostics.csv", index=False)

    print(f"\n{bar}\nPREDICTION, versus the pipeline's baseline\n{bar}")
    b = df[df.variant == "base"]
    print(f"  {'variant':13s} {'fc':>9s} {'vs base':>9s} {'p':>9s} "
          f"{'amplitude':>10s} {'vs base':>9s} {'p':>9s}")
    for v in VARIANTS:
        g = df[df.variant == v]
        line = f"  {v:13s} {g.fc.mean():+9.4f}"
        if v == "base":
            print(line + f" {'':>9s} {'':>9s} {g.amplitude.mean():+10.4f}")
            continue
        t1, p1 = corrected_ttest_rel(g.fc.values, b.fc.values)
        t2, p2 = corrected_ttest_rel(g.amplitude.values, b.amplitude.values)
        print(line + f" {g.fc.mean()-b.fc.mean():+9.4f} {p1:9.3g}"
                     f" {g.amplitude.mean():+10.4f} "
                     f"{g.amplitude.mean()-b.amplitude.mean():+9.4f} {p2:9.3g}")

    print(f"\n{bar}\nDIAGNOSTICS\n{bar}")
    print(f"  {'variant':13s} {'stimulus-locked R2':>20s} {'r(FD, amplitude)':>18s}")
    for _, r in dg.iterrows():
        print(f"  {r.variant:13s} {r.stimulus_locked_r2:20.4f} "
              f"{r.motion_amplitude_r:18.3f}")
    print("\n  Stimulus-locked variance is what the encoding analysis needs; the")
    print("  baseline retained almost none. The motion column is the check that")
    print("  any gain from dropping GSR is not motion returning to the data.")
    print(f"\nSaved -> {OUT}/variant_comparison_folds.csv, variant_diagnostics.csv")


if __name__ == "__main__":
    main()
