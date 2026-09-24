#!/usr/bin/env python
"""
Percent signal change against z-scored signals, with and without removal of high-motion frames, on identical folds.

Usage
-----
  python scripts/models/standardisation_and_scrubbing.py --repeats 3
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

CFG_PATH = "config/pipeline.yaml"
TARGET   = "IST_intelligence_total"
ALPHAS   = np.logspace(-2, 6, 25)
SEED     = 42
SHRINK   = 0.55
MIN_KEEP = 20
OUT_DIR  = Path("outputs/honest")
FS_DIR   = Path("derivatives/fs_stats")
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


def ridge_pred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    return RidgeCV(alphas=ALPHAS, cv=5).fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))


def embed(series, dev):
    """shrunk covariance -> log-Euclidean -> vectorised upper triangle."""
    covs = []
    for x in series:
        x = x - x.mean(0, keepdims=True)
        S = (x.T @ x) / max(len(x) - 1, 1)
        d = S.shape[0]
        covs.append((1 - SHRINK) * S + SHRINK * (np.trace(S) / d) * np.eye(d, dtype=np.float32))
    C = torch.from_numpy(np.stack(covs)).to(dev).double()
    return upper(sym_funcm(C, torch.log)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(CFG_PATH))
    conn = Path(cfg["paths"]["connectivity_dir"])

    raw, kp = {}, {}
    with h5py.File(conn / "unscrubbed_ts.h5", "r") as f:
        for s in f["subjects"]:
            raw[s] = f["subjects"][s]["time_series"][:]
            kp[s] = f["subjects"][s]["keep"][:].astype(bool)

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

    ids = [s for s in sorted(set(raw) & set(M.index))
           if s in parts.index and not pd.isna(parts.loc[s, TARGET])
           and not pd.isna(M.loc[s, vol_col]) and raw[s].shape[0] == 290]
    y   = np.array([float(parts.loc[s, TARGET]) for s in ids])
    sex = np.array([1 if str(parts.loc[s, "sex"]).lower().startswith("m") else 0 for s in ids])
    vol = M.loc[ids, vol_col].values.astype(float)
    mot = np.array([motion_stats(s, cfg["paths"]["fmriprep_dir"],
                                 cfg["dataset"]["task"]) for s in ids])
    for j in range(mot.shape[1]):
        mot[:, j] = np.where(np.isfinite(mot[:, j]), mot[:, j], np.nanmedian(mot[:, j]))
    vz = (vol - vol.mean()) / vol.std()
    Zm = np.column_stack([sex, vz, vz ** 2, mot])

    def masked(s):
        k = kp[s]
        if k.sum() < MIN_KEEP:
            k = np.ones(len(raw[s]), bool)
        return raw[s][k]

    def zt(x):                       # z-score each region over time
        m = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True)
        return (x - m) / np.clip(sd, 1e-8, None)

    cells = {
        "psc_unscrubbed": [raw[s]        for s in ids],
        "psc_scrubbed":   [masked(s)     for s in ids],
        "zsc_unscrubbed": [zt(raw[s])    for s in ids],
        "zsc_scrubbed":   [zt(masked(s)) for s in ids],
    }
    kept = np.array([len(masked(s)) for s in ids])
    print(f"n = {len(ids)}; scrubbed length mean {kept.mean():.1f} "
          f"(min {kept.min()}, max {kept.max()}) of 290")
    X = {}
    for nm, ser in cells.items():
        X[nm] = embed(ser, dev)
        print(f"  embedded {nm:16s} -> {X[nm].shape}", flush=True)

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
        print(f"  fold {k:2d}/{args.repeats*5}  " +
              "  ".join(f"{nm}={rec[nm]:+.3f}" for nm in X), flush=True)

    df = pd.DataFrame(rows); df.to_csv(OUT_DIR / "scrub_standardize_folds.csv", index=False)
    bar = "=" * 76
    print(f"\n{bar}\nScrubbing x standardisation, {args.repeats}x5 CV (n={len(ids)})\n{bar}")
    for c in sorted([c for c in df.columns if c != "fold"], key=lambda c: -df[c].mean()):
        print(f"  {c:22s} r = {df[c].mean():+.4f} +/- {df[c].std(ddof=1):.4f}")

    def cmp(a, b, label):
        d = df[a] - df[b]; t, p = corrected_ttest_rel(df[a], df[b])
        print(f"  {label:40s} {d.mean():+.4f}  t={t:+5.2f}  p={p:.3g}  "
              f"wins {int((d > 0).sum())}/{len(df)}")

    print("\n  MAIN EFFECT of NOT scrubbing (raw target):")
    cmp("psc_unscrubbed", "psc_scrubbed", "within PSC")
    cmp("zsc_unscrubbed", "zsc_scrubbed", "within z-scored")
    print("\n  MAIN EFFECT of PSC vs z-scoring (raw target):")
    cmp("psc_unscrubbed", "zsc_unscrubbed", "within unscrubbed")
    cmp("psc_scrubbed", "zsc_scrubbed", "within scrubbed")
    print("\n  corner vs corner:")
    cmp("psc_unscrubbed", "zsc_scrubbed", "psc_unscrubbed vs zsc_scrubbed (legacy)")
    print("\n  SAME CONTRASTS ON THE DECONFOUNDED TARGET (sex+size+motion):")
    cmp("psc_unscrubbed_dec", "psc_scrubbed_dec", "not scrubbing, within PSC")
    cmp("psc_unscrubbed_dec", "zsc_unscrubbed_dec", "PSC vs zscore, within unscrubbed")
    cmp("psc_unscrubbed_dec", "zsc_scrubbed_dec", "psc_unscrubbed vs legacy")
    print("\n  retention after deconfounding:")
    for nm in X:
        print(f"    {nm:16s} {df[nm].mean():+.4f} -> {df[nm + '_dec'].mean():+.4f}"
              f"  ({100 * df[nm + '_dec'].mean() / df[nm].mean():.0f}%)")
    print(f"\nSaved -> {OUT_DIR}/scrub_standardize_folds.csv")


if __name__ == "__main__":
    main()
