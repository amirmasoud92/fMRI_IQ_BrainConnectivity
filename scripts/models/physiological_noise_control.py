#!/usr/bin/env python
"""
Whether the advantage of percent signal change survives regression of the physiological regressors released
with the data.

Design
------
Each arm is built from the SAME subjects and folds; the only difference is
whether the 20 physiological regressors (plus intercept) are projected out of
each region's timeseries BEFORE the covariance is formed.

  psc          / psc_physio       covariance, amplitude preserved
  zsc          / zsc_physio       z-scored over time == correlation
  amp          / amp_physio       log regional variance, 215 features

The test is not whether each arm drops -- removing 20 regressors from 290
timepoints costs some signal everywhere. The test is whether the PSC-MINUS-ZSC
GAP survives, since both arms lose the same degrees of freedom. If the gap is
physiological it should collapse; if it is neural it should persist.

Usage
-----
  python scripts/models/physiological_noise_control.py --repeats 3
"""
from __future__ import annotations

import argparse, sys, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.data.preprocessing import load_participants
from src.stats.cv_inference import corrected_ttest_rel

warnings.filterwarnings("ignore")

CFG_PATH  = "config/pipeline.yaml"
TARGET    = "IST_intelligence_total"
ALPHAS    = np.logspace(-2, 6, 25)
SEED      = 42
SHRINK    = 0.55
OUT_DIR   = Path("outputs/honest")
FS_DIR    = Path("derivatives/fs_stats")
PHYS_DIR  = Path("derivatives/physiology")
VOL_COL   = "EstimatedTotalIntraCranialVol"


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


def load_physio(sub_id, task, n_tr):
    p = (PHYS_DIR / sub_id / "physio" /
         f"{sub_id}_task-{task}_recording-respcardiac_desc-retroicor_regressors.tsv")
    if not p.exists():
        return None
    try:
        d = pd.read_csv(p, sep="\t").apply(pd.to_numeric, errors="coerce")
    except Exception:
        return None
    if len(d) != n_tr:
        return None
    P = d.to_numpy(float)
    # a column that is all-NaN or constant carries no information and would make
    # the projection rank-deficient; drop it rather than let lstsq absorb it
    P = P[:, np.isfinite(P).all(0)]
    if P.size == 0:
        return None
    P = P[:, P.std(0) > 1e-12]
    return P if P.shape[1] else None


def resid(X, P):
    """Project the physiological regressors (plus intercept) out of each column."""
    A = np.column_stack([np.ones(len(X)), P])
    beta, *_ = np.linalg.lstsq(A, X, rcond=None)
    return X - A @ beta


def sym_funcm(X, fn):
    X = 0.5 * (X + X.transpose(-1, -2)).double()
    w, V = torch.linalg.eigh(X)
    return V @ torch.diag_embed(fn(w.clamp_min(1e-10))) @ V.transpose(-1, -2)


def upper(L):
    n = L.shape[-1]
    iu = torch.triu_indices(n, n, offset=1, device=L.device)
    return torch.cat([torch.diagonal(L, dim1=-2, dim2=-1),
                      L[..., iu[0], iu[1]] * np.sqrt(2.0)], dim=-1).float()


def embed(series, dev):
    covs = []
    for x in series:
        x = x - x.mean(0, keepdims=True)
        S = (x.T @ x) / max(len(x) - 1, 1)
        d = S.shape[0]
        covs.append((1 - SHRINK) * S + SHRINK * (np.trace(S) / d) * np.eye(d, dtype=np.float32))
    C = torch.from_numpy(np.stack(covs)).to(dev).double()
    return upper(sym_funcm(C, torch.log)).cpu().numpy()


def zt(x):
    m = x.mean(0, keepdims=True); s = x.std(0, keepdims=True)
    return (x - m) / np.clip(s, 1e-8, None)


def ridge_pred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    return RidgeCV(alphas=ALPHAS, cv=5).fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(CFG_PATH))
    conn = Path(cfg["paths"]["connectivity_dir"])
    task = cfg["dataset"]["task"]

    raw = {}
    with h5py.File(conn / "unscrubbed_ts.h5", "r") as f:
        for s in f["subjects"]:
            raw[s] = f["subjects"][s]["time_series"][:]

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

    ids, phys, n_missing = [], {}, 0
    for s in sorted(set(raw) & set(M.index)):
        if s not in parts.index or pd.isna(parts.loc[s, TARGET]):
            continue
        if pd.isna(M.loc[s, vol_col]) or raw[s].shape[0] != 290:
            continue
        P = load_physio(s, task, 290)
        if P is None:
            n_missing += 1
            continue
        ids.append(s); phys[s] = P
    print(f"n = {len(ids)} with usable physio  ({n_missing} excluded for missing/"
          f"malformed physio)")
    print(f"  regressors per subject: {phys[ids[0]].shape[1]}")

    y   = np.array([float(parts.loc[s, TARGET]) for s in ids])
    sex = np.array([1 if str(parts.loc[s, "sex"]).lower().startswith("m") else 0 for s in ids])
    vol = M.loc[ids, vol_col].values.astype(float)
    mot = np.array([motion_stats(s, cfg["paths"]["fmriprep_dir"], task) for s in ids])
    for j in range(mot.shape[1]):
        mot[:, j] = np.where(np.isfinite(mot[:, j]), mot[:, j], np.nanmedian(mot[:, j]))
    vz = (vol - vol.mean()) / vol.std()
    Zm = np.column_stack([sex, vz, vz ** 2, mot])

    S_raw = [raw[s] for s in ids]
    S_phy = [resid(raw[s], phys[s]) for s in ids]

    # how much of each subject's signal the physiological model actually explains
    r2 = np.array([1.0 - S_phy[i].var(0).sum() / max(S_raw[i].var(0).sum(), 1e-12)
                   for i in range(len(ids))])
    amp_raw = np.stack([np.log(np.maximum(x.var(0, ddof=1), 1e-12)) for x in S_raw])
    amp_phy = np.stack([np.log(np.maximum(x.var(0, ddof=1), 1e-12)) for x in S_phy])
    print(f"  variance explained by physio: mean {100*r2.mean():.1f}% "
          f"(range {100*r2.min():.1f}-{100*r2.max():.1f}%)")
    print(f"  corr(physio R2, mean log-amplitude) = {pearsonr(r2, amp_raw.mean(1))[0]:+.3f}")
    print(f"  corr(physio R2, IST)                = {pearsonr(r2, y)[0]:+.3f}")
    print(f"  corr(mean log-amp, meanFD)          = {pearsonr(amp_raw.mean(1), mot[:,0])[0]:+.3f}")
    print(f"  corr(amp before vs after physio)    = "
          f"{pearsonr(amp_raw.mean(1), amp_phy.mean(1))[0]:+.3f}", flush=True)

    X = {
        "psc":        embed(S_raw, dev),
        "psc_physio": embed(S_phy, dev),
        "zsc":        embed([zt(x) for x in S_raw], dev),
        "zsc_physio": embed([zt(x) for x in S_phy], dev),
        "amp":        amp_raw.astype(np.float32),
        "amp_physio": amp_phy.astype(np.float32),
    }
    for k, v in X.items():
        print(f"  {k:12s} -> {v.shape}")

    q = pd.qcut(y, 4, labels=False, duplicates="drop")
    strat = np.asarray(q) * 2 + sex
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats, random_state=SEED)

    rows = []
    for k, (tr, te) in enumerate(rskf.split(np.zeros(len(y)), strat), 1):
        rec = {"fold": k}
        lr = LinearRegression().fit(Zm[tr], y[tr])
        ytr_d, yte_d = y[tr] - lr.predict(Zm[tr]), y[te] - lr.predict(Zm[te])
        for nm, Xb in X.items():
            rec[nm] = float(pearsonr(y[te], ridge_pred(Xb[tr], y[tr], Xb[te]))[0])
            rec[nm + "_dec"] = float(pearsonr(yte_d, ridge_pred(Xb[tr], ytr_d, Xb[te]))[0])
        rows.append(rec)
        print(f"  fold {k:2d}/{args.repeats*5}  psc={rec['psc']:+.3f} "
              f"psc_phy={rec['psc_physio']:+.3f} zsc={rec['zsc']:+.3f} "
              f"zsc_phy={rec['zsc_physio']:+.3f}", flush=True)

    df = pd.DataFrame(rows); df.to_csv(OUT_DIR / "psc_physio_folds.csv", index=False)
    bar = "=" * 76
    print(f"\n{bar}\nPhysiological control, {args.repeats}x5 CV (n={len(ids)})\n{bar}")
    for c in sorted([c for c in df.columns if c != "fold"], key=lambda c: -df[c].mean()):
        print(f"  {c:16s} r = {df[c].mean():+.4f} +/- {df[c].std(ddof=1):.4f}")

    def cmp(a, b, label):
        d = df[a] - df[b]; t, p = corrected_ttest_rel(df[a], df[b])
        print(f"  {label:44s} {d.mean():+.4f}  t={t:+5.2f}  p={p:.3g}  "
              f"wins {int((d > 0).sum())}/{len(df)}")

    print("\n  THE TEST -- does the PSC advantage survive physio removal?")
    cmp("psc", "zsc", "PSC gap, physio PRESENT")
    cmp("psc_physio", "zsc_physio", "PSC gap, physio REMOVED")
    g0 = (df["psc"] - df["zsc"]).mean()
    g1 = (df["psc_physio"] - df["zsc_physio"]).mean()
    print(f"    gap {g0:+.4f} -> {g1:+.4f}   "
          f"({100*(1-g1/g0) if abs(g0)>1e-9 else float('nan'):.0f}% of the gap is physiological)")
    print("\n  same contrast on the deconfounded target:")
    cmp("psc_dec", "zsc_dec", "PSC gap, physio PRESENT (deconf)")
    cmp("psc_physio_dec", "zsc_physio_dec", "PSC gap, physio REMOVED (deconf)")

    print("\n  cost of removing physio within each arm (both lose 21 df):")
    cmp("psc_physio", "psc", "psc_physio - psc")
    cmp("zsc_physio", "zsc", "zsc_physio - zsc")
    cmp("amp_physio", "amp", "amp_physio - amp")
    print(f"\n  amplitude block: {df['amp'].mean():+.4f} -> {df['amp_physio'].mean():+.4f} "
          f"({100*df['amp_physio'].mean()/df['amp'].mean():.0f}% retained)")
    print(f"\nSaved -> {OUT_DIR}/psc_physio_folds.csv")


if __name__ == "__main__":
    main()
