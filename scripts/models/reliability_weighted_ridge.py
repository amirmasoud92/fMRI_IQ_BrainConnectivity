#!/usr/bin/env python
"""
Ridge regression with features weighted by their split-half reliability, and a reliability threshold control.

Arms
----
  gamma_0            plain ridge (the current model, exact)
  gamma_0.5/1/2      progressively harder downweighting of unreliable features
  gamma_sel          gamma chosen by inner generalised CV
  keep_rel_gt_0.5    HARD selection: discard every feature below 0.5 reliability

The last arm is the interesting one. Sparse and structured methods have failed
repeatedly here, which suggests the unreliable edges still contribute in
aggregate. If discarding them HURTS, that is direct evidence for dense
aggregation over selection, and it explains the whole pattern of failures.
Reporting each fixed gamma separately -- not just the selected one -- separates
"does reliability weighting help" from "can gamma be selected", which matters
because the two-block penalty experiment lost purely through selection noise.

Runs on the sealed discovery split only.

Usage
-----
  python scripts/models/reliability_weighted_ridge.py --repeats 3
"""
from __future__ import annotations

import argparse, json, sys, warnings
from collections import Counter
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

CFG_PATH = "config/pipeline.yaml"
TARGET   = "IST_intelligence_total"
ALPHAS   = np.logspace(-2, 6, 25)
SEED     = 42
SHRINK   = 0.55
GAMMAS   = [0.0, 0.5, 1.0, 2.0]
THRESH   = 0.5
OUT_DIR  = Path("outputs/honest")
FS_DIR   = Path("derivatives/fs_stats")
MANIFEST = OUT_DIR / "holdout_manifest.json"
VOL_COL  = "EstimatedTotalIntraCranialVol"


def motion_stats(sub_id, fmriprep, task):
    p = Path(fmriprep) / sub_id / "func" / f"{sub_id}_task-{task}_desc-confounds_regressors.tsv"
    if not p.exists():
        return np.nan, np.nan, np.nan
    try:
        fd = pd.read_csv(p, sep="\t", usecols=["framewise_displacement"],
                         na_values="n/a")["framewise_displacement"].astype(float)
        fd = fd.fillna(0.0).to_numpy()[1:]
        if fd.size == 0:
            return np.nan, np.nan, np.nan
        return float(fd.mean()), float(fd.max()), float((fd > 0.5).mean())
    except Exception:
        return np.nan, np.nan, np.nan


def sym_funcm(X, fn):
    X = 0.5 * (X + X.transpose(-1, -2)).double()
    w, V = torch.linalg.eigh(X)
    return V @ torch.diag_embed(fn(w.clamp_min(1e-10))) @ V.transpose(-1, -2)


def upper(L):
    n = L.shape[-1]
    iu = torch.triu_indices(n, n, offset=1, device=L.device)
    return torch.cat([torch.diagonal(L, dim1=-2, dim2=-1),
                      L[..., iu[0], iu[1]] * np.sqrt(2.0)], dim=-1).float()


def tangent(series, dev, lw=False):
    if lw:
        covs = [ledoit_wolf(x, assume_centered=False)[0] for x in series]
    else:
        covs = []
        for x in series:
            x = x - x.mean(0, keepdims=True)
            S = (x.T @ x) / (len(x) - 1)
            d = S.shape[0]
            covs.append((1 - SHRINK) * S + SHRINK * (np.trace(S) / d) * np.eye(d))
    C = torch.from_numpy(np.stack(covs)).to(dev).double()
    return upper(sym_funcm(C, torch.log)).cpu().numpy()


def colwise_corr(A, B):
    A = A - A.mean(0, keepdims=True); B = B - B.mean(0, keepdims=True)
    num = (A * B).sum(0)
    den = np.sqrt((A ** 2).sum(0) * (B ** 2).sum(0))
    return num / np.clip(den, 1e-12, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(CFG_PATH))
    conn = Path(cfg["paths"]["connectivity_dir"])
    ids = json.loads(MANIFEST.read_text())["discovery"]

    raw = {}
    with h5py.File(conn / "unscrubbed_ts.h5", "r") as f:
        for s in ids:
            raw[s] = f["subjects"][s]["time_series"][:].astype(np.float64)

    frames = []
    for f in sorted(FS_DIR.glob("data-*.tsv")):
        d = pd.read_csv(f, sep="\t"); d = d.rename(columns={d.columns[0]: "sid"})
        d["sid"] = d["sid"].astype(str); d = d.set_index("sid")
        d = d.loc[:, ~d.columns.duplicated()]
        d.columns = [f"{f.stem}::{c}" for c in d.columns]; frames.append(d)
    M = pd.concat(frames, axis=1)
    M = M.loc[:, ~M.columns.duplicated()].apply(pd.to_numeric, errors="coerce")
    vol_col = [c for c in M.columns if c.endswith("::" + VOL_COL)][0]
    parts = load_participants(cfg["paths"]["bids_root"])

    y   = np.array([float(parts.loc[s, TARGET]) for s in ids])
    sex = np.array([1 if str(parts.loc[s, "sex"]).lower().startswith("m") else 0 for s in ids])
    vol = M.loc[ids, vol_col].values.astype(float)
    mot = np.array([motion_stats(s, cfg["paths"]["fmriprep_dir"],
                                 cfg["dataset"]["task"]) for s in ids])
    for j in range(mot.shape[1]):
        mot[:, j] = np.where(np.isfinite(mot[:, j]), mot[:, j], np.nanmedian(mot[:, j]))
    vzs = (vol - vol.mean()) / vol.std()
    Zm = np.column_stack([sex, vzs, vzs ** 2, mot])

    T = raw[ids[0]].shape[0]; h = T // 2
    N = len(ids)
    X = tangent([raw[s] for s in ids], dev)
    XA = tangent([raw[s][:h] for s in ids], dev, lw=True)
    XB = tangent([raw[s][h:2 * h] for s in ids], dev, lw=True)
    print(f"n = {N}, features {X.shape[1]}; halves of {h} volumes for reliability")

    q = pd.qcut(y, 4, labels=False, duplicates="drop")
    strat = np.asarray(q) * 2 + sex
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats, random_state=SEED)

    rows, chosen, kept = [], [], []
    for k, (tr, te) in enumerate(rskf.split(np.zeros(N), strat), 1):
        rec = {"fold": k}
        lr = LinearRegression().fit(Zm[tr], y[tr])
        ytr, yte = y[tr] - lr.predict(Zm[tr]), y[te] - lr.predict(Zm[te])

        # reliability from TRAINING subjects only
        rel = np.clip(colwise_corr(XA[tr], XB[tr]), 0.0, 1.0)

        sc = StandardScaler().fit(X[tr])
        Ztr, Zte = sc.transform(X[tr]), sc.transform(X[te])   # standardise FIRST

        def fit_pred(w, cv=5):
            A, B = Ztr * w, Zte * w
            m = RidgeCV(alphas=ALPHAS, cv=cv).fit(A, ytr)
            return m.predict(B)

        for g in GAMMAS:
            w = rel ** g
            rec[f"gamma_{g}"] = float(pearsonr(yte, fit_pred(w))[0])

        best_g, best_s = None, -np.inf
        for g in GAMMAS:
            m = RidgeCV(alphas=ALPHAS).fit(Ztr * (rel ** g), ytr)
            if m.best_score_ > best_s:
                best_s, best_g = m.best_score_, g
        rec["gamma_sel"] = float(pearsonr(yte, fit_pred(rel ** best_g))[0])
        rec["gamma_chosen"] = best_g
        chosen.append(best_g)

        keep = rel > THRESH
        kept.append(int(keep.sum()))
        m = RidgeCV(alphas=ALPHAS, cv=5).fit(Ztr[:, keep], ytr)
        rec[f"keep_rel_gt_{THRESH}"] = float(
            pearsonr(yte, m.predict(Zte[:, keep]))[0])

        rows.append(rec)
        print(f"  fold {k:2d}/{args.repeats*5}  g0={rec['gamma_0.0']:+.3f} "
              f"g1={rec['gamma_1.0']:+.3f} sel={rec['gamma_sel']:+.3f}(g={best_g:g}) "
              f"keep={rec[f'keep_rel_gt_{THRESH}']:+.3f} ({keep.sum()} feats)",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "reliability_weighted_folds.csv", index=False)
    bar = "=" * 78
    print(f"\n{bar}\nReliability-weighted ridge, {args.repeats}x5 CV (n={N})\n{bar}")
    cols = [f"gamma_{g}" for g in GAMMAS] + ["gamma_sel", f"keep_rel_gt_{THRESH}"]
    for c in sorted(cols, key=lambda c: -df[c].mean()):
        print(f"  {c:20s} r = {df[c].mean():+.4f} +/- {df[c].std(ddof=1):.4f}")

    def cmp(a, label):
        d = df[a] - df["gamma_0.0"]; t, p = corrected_ttest_rel(df[a], df["gamma_0.0"])
        print(f"    {label:44s} {d.mean():+.4f}  t={t:+5.2f}  p={p:.3g}  "
              f"wins {int((d>0).sum())}/{len(df)}")

    print("\n  vs plain ridge (gamma = 0):")
    for c in cols:
        if c != "gamma_0.0":
            cmp(c, c)
    print(f"\n  gamma chosen by inner CV: {dict(sorted(Counter(chosen).items()))}")
    print(f"  features retained by the reliability filter: "
          f"{int(np.mean(kept))} of {X.shape[1]} "
          f"({100*np.mean(kept)/X.shape[1]:.0f}%)")
    print(f"\nSaved -> {OUT_DIR}/reliability_weighted_folds.csv")


if __name__ == "__main__":
    main()
