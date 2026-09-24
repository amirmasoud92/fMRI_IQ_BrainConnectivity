#!/usr/bin/env python
"""
Shape of the covariance eigenspectrum as a feature set, alone and with connectivity.

Three questions, in order
-------------------------
  1. do the descriptors predict IST on their own, and do they survive
     deconfounding for sex, size and motion?
  2. do they ADD to a correlation-based (amplitude-free) representation?
  3. do they RECOVER the covariance-over-correlation gap? If a handful of
     spectral scalars appended to `corr` reach `cov`, then the amplitude effect
     is, to that extent, a spectral-shape effect -- which would be a far more
     interpretable statement than "z-scoring loses signal".

Spectra are taken from the UNSHRUNK sample covariance (T=290 > R=215, so it is
full rank) because shrinkage deliberately flattens the spectrum and would
destroy the quantity being measured. Correlation-matrix spectra are computed
alongside, since those are amplitude-free by construction.

Runs on the sealed discovery split only.

Usage
-----
  python scripts/models/spectral_shape.py --repeats 3
"""
from __future__ import annotations

import argparse, json, sys, warnings
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
OUT_DIR  = Path("outputs/honest")
FS_DIR   = Path("derivatives/fs_stats")
MANIFEST = OUT_DIR / "holdout_manifest.json"
VOL_COL  = "EstimatedTotalIntraCranialVol"
PL_LO, PL_HI = 1, 100          # rank range for the power-law fit


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


def embed_from_DR(D, R, dev):
    Dt = torch.from_numpy(np.ascontiguousarray(D)).to(dev).double()
    Rt = torch.from_numpy(np.ascontiguousarray(R)).to(dev).double()
    C = Dt.unsqueeze(-1) * Rt * Dt.unsqueeze(-2)
    n = D.shape[1]
    tr = torch.diagonal(C, dim1=-2, dim2=-1).sum(-1) / n
    eye = torch.eye(n, dtype=C.dtype, device=dev)
    C = (1 - SHRINK) * C + SHRINK * tr.view(-1, 1, 1) * eye
    return upper(sym_funcm(C, torch.log)).cpu().numpy()


def spectral_features(lam):
    """Scale-invariant descriptors of one eigenvalue spectrum (descending)."""
    lam = np.clip(np.sort(lam)[::-1], 1e-12, None)
    s = lam.sum()
    p = lam / s
    ent = float(-(p * np.log(p)).sum())
    pr = float(s ** 2 / (lam ** 2).sum())
    rank = np.arange(1, len(lam) + 1)
    lo, hi = PL_LO, min(PL_HI, len(lam))
    sl = np.polyfit(np.log(rank[lo - 1:hi]), np.log(lam[lo - 1:hi]), 1)[0]
    return {
        "participation_ratio": pr,
        "spectral_entropy": ent,
        "effective_rank": float(np.exp(ent)),
        "top1_share": float(p[0]),
        "top5_share": float(p[:5].sum()),
        "top10_share": float(p[:10].sum()),
        "powerlaw_slope": float(sl),
    }


def ridge_pred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    return RidgeCV(alphas=ALPHAS, cv=5).fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(CFG_PATH))
    conn = Path(cfg["paths"]["connectivity_dir"])
    if not MANIFEST.exists():
        sys.exit("Missing holdout manifest -- run scripts/prediction/seal_holdout.py first")
    ids = json.loads(MANIFEST.read_text())["discovery"]

    raw = {}
    with h5py.File(conn / "unscrubbed_ts.h5", "r") as f:
        for s in ids:
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

    y   = np.array([float(parts.loc[s, TARGET]) for s in ids])
    sex = np.array([1 if str(parts.loc[s, "sex"]).lower().startswith("m") else 0 for s in ids])
    vol = M.loc[ids, vol_col].values.astype(float)
    mot = np.array([motion_stats(s, cfg["paths"]["fmriprep_dir"],
                                 cfg["dataset"]["task"]) for s in ids])
    for j in range(mot.shape[1]):
        mot[:, j] = np.where(np.isfinite(mot[:, j]), mot[:, j], np.nanmedian(mot[:, j]))
    vz = (vol - vol.mean()) / vol.std()
    Zm = np.column_stack([sex, vz, vz ** 2, mot])

    cov_rows, corr_rows, Dobs, Rmat = [], [], [], []
    for s in ids:
        x = raw[s].astype(np.float64)
        x = x - x.mean(0, keepdims=True)
        T = len(x)
        S = (x.T @ x) / (T - 1)
        d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
        Rc = S / np.outer(d, d)
        Dobs.append(d); Rmat.append(Rc)
        cov_rows.append(spectral_features(np.linalg.eigvalsh(S)))
        corr_rows.append(spectral_features(np.linalg.eigvalsh(Rc)))
    Dobs = np.stack(Dobs); Rmat = np.stack(Rmat)
    SPc = pd.DataFrame(cov_rows, index=ids).add_prefix("cov_")
    SPr = pd.DataFrame(corr_rows, index=ids).add_prefix("corr_")
    SP = pd.concat([SPc, SPr], axis=1)
    SP.to_csv(OUT_DIR / "spectral_features.csv")
    N = len(ids)
    print(f"n = {N} (sealed discovery split); {SP.shape[1]} spectral descriptors")

    print("\n  univariate association with IST (raw | deconfounded):")
    lr_all = LinearRegression().fit(Zm, y); y_res = y - lr_all.predict(Zm)
    uni = []
    for c in SP.columns:
        v = SP[c].to_numpy(float)
        r0 = pearsonr(v, y)[0]; r1 = pearsonr(v, y_res)[0]
        uni.append({"feature": c, "r_raw": r0, "r_dec": r1})
        print(f"    {c:28s} {r0:+.3f}  |  {r1:+.3f}")
    pd.DataFrame(uni).to_csv(OUT_DIR / "spectral_univariate.csv", index=False)

    X_spec_cov  = SPc.to_numpy(float)
    X_spec_corr = SPr.to_numpy(float)
    X_spec_all  = SP.to_numpy(float)
    X_cov  = embed_from_DR(Dobs, Rmat, dev)
    X_corr = embed_from_DR(np.ones_like(Dobs), Rmat, dev)
    blocks = {
        "cov": X_cov, "corr": X_corr,
        "spec_cov": X_spec_cov, "spec_corr": X_spec_corr, "spec_all": X_spec_all,
        "corr_plus_spec": np.hstack([X_corr, X_spec_all]),
        "cov_plus_spec": np.hstack([X_cov, X_spec_all]),
    }
    for k, v in blocks.items():
        print(f"  {k:16s} -> {v.shape}")

    q = pd.qcut(y, 4, labels=False, duplicates="drop")
    strat = np.asarray(q) * 2 + sex
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats, random_state=SEED)

    rows = []
    for k, (tr, te) in enumerate(rskf.split(np.zeros(N), strat), 1):
        rec = {"fold": k}
        lr = LinearRegression().fit(Zm[tr], y[tr])
        ytr_d, yte_d = y[tr] - lr.predict(Zm[tr]), y[te] - lr.predict(Zm[te])
        for nm, Xb in blocks.items():
            rec[nm] = float(pearsonr(y[te], ridge_pred(Xb[tr], y[tr], Xb[te]))[0])
            rec[nm + "_dec"] = float(pearsonr(yte_d, ridge_pred(Xb[tr], ytr_d, Xb[te]))[0])
        rows.append(rec)
        print(f"  fold {k:2d}/{args.repeats*5}  cov={rec['cov']:+.3f} corr={rec['corr']:+.3f} "
              f"spec={rec['spec_all']:+.3f} corr+spec={rec['corr_plus_spec']:+.3f}", flush=True)

    df = pd.DataFrame(rows); df.to_csv(OUT_DIR / "spectral_shape_folds.csv", index=False)
    bar = "=" * 76
    print(f"\n{bar}\nSpectral shape, {args.repeats}x5 CV (n={N})\n{bar}")
    for c in sorted([c for c in df.columns if c != "fold" and not c.endswith("_dec")],
                    key=lambda c: -df[c].mean()):
        print(f"  {c:16s} r = {df[c].mean():+.4f} +/- {df[c].std(ddof=1):.4f}"
              f"   deconf {df[c+'_dec'].mean():+.4f}")

    def cmp(a, b, label):
        d = df[a] - df[b]; t, p = corrected_ttest_rel(df[a], df[b])
        print(f"    {label:46s} {d.mean():+.4f}  t={t:+5.2f}  p={p:.3g}  "
              f"wins {int((d > 0).sum())}/{len(df)}")

    print("\n  Q1  do spectral descriptors predict at all?")
    print(f"    spec_all alone r = {df['spec_all'].mean():+.4f} "
          f"(deconf {df['spec_all_dec'].mean():+.4f}) from {SP.shape[1]} scalars")
    print("\n  Q2  do they ADD to an amplitude-free representation?")
    cmp("corr_plus_spec", "corr", "corr + spectral - corr")
    cmp("cov_plus_spec", "cov", "cov + spectral - cov")
    print("\n  Q3  do they RECOVER the covariance-over-correlation gap?")
    cmp("cov", "corr", "cov - corr  (the gap to be explained)")
    cmp("cov", "corr_plus_spec", "cov - (corr + spectral)   [~0 = fully recovered]")
    tot = df["cov"].mean() - df["corr"].mean()
    if abs(tot) > 1e-9:
        got = df["corr_plus_spec"].mean() - df["corr"].mean()
        print(f"    spectral scalars recover {100*got/tot:.0f}% of the gap")
    print(f"\nSaved -> {OUT_DIR}/spectral_shape_folds.csv, spectral_features.csv, "
          f"spectral_univariate.csv")


if __name__ == "__main__":
    main()
