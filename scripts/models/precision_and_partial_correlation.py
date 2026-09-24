#!/usr/bin/env python
"""
Precision and partial-correlation embeddings against the covariance tangent embedding.

An identity worth knowing before spending compute
-------------------------------------------------
Under the log-Euclidean embedding the precision matrix carries NO new
Arms
----
  cov_tangent    current pipeline baseline
  prec_tangent   log-Euclidean of the precision  (identity check, = baseline)
  pcorr_fisher   partial correlation, Fisher-z, upper triangle  (the standard)
  pcorr_tangent  log-Euclidean of the partial correlation matrix
  corr_fisher    ordinary correlation, Fisher-z  (reference point)
  stack          does partial correlation ADD to tangent covariance?

Runs on the sealed discovery split only.

Usage
-----
  python scripts/models/precision_and_partial_correlation.py --repeats 3
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
from sklearn.model_selection import KFold, RepeatedStratifiedKFold
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


def upper_full(L):
    n = L.shape[-1]
    iu = torch.triu_indices(n, n, offset=1, device=L.device)
    return torch.cat([torch.diagonal(L, dim1=-2, dim2=-1),
                      L[..., iu[0], iu[1]] * np.sqrt(2.0)], dim=-1).float()


def upper_off(L):
    n = L.shape[-1]
    iu = torch.triu_indices(n, n, offset=1, device=L.device)
    return L[..., iu[0], iu[1]].float()


def ridge_pred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    return RidgeCV(alphas=ALPHAS, cv=5).fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))


def oof(X, y, tr, n_inner=5):
    o = np.zeros(len(tr))
    for a, b in KFold(n_inner, shuffle=True, random_state=SEED).split(tr):
        o[b] = ridge_pred(X[tr[a]], y[tr[a]], X[tr[b]])
    return o


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
    N = len(ids)

    covs = []
    for s in ids:
        x = raw[s] - raw[s].mean(0, keepdims=True)
        S = (x.T @ x) / (len(x) - 1)
        d = S.shape[0]
        covs.append((1 - SHRINK) * S + SHRINK * (np.trace(S) / d) * np.eye(d))
    C = torch.from_numpy(np.stack(covs)).to(dev).double()
    R = C.shape[-1]
    print(f"n = {N} (sealed discovery split), {R} regions")

    logC = sym_funcm(C, torch.log)
    Theta = torch.linalg.inv(C)
    logT = sym_funcm(Theta, torch.log)
    print(f"  identity check  max |log(C^-1) + log(C)| = "
          f"{float((logT + logC).abs().max()):.3e}   (should be ~0)")

    dT = torch.diagonal(Theta, dim1=-2, dim2=-1).clamp_min(1e-12).rsqrt()
    P = -dT.unsqueeze(-1) * Theta * dT.unsqueeze(-2)
    idx = torch.arange(R, device=dev)
    P[:, idx, idx] = 1.0
    pc = upper_off(P).cpu().numpy()
    print(f"  partial correlations: mean |p| = {np.abs(pc).mean():.4f}, "
          f"max |p| = {np.abs(pc).max():.4f}")

    corr = C / torch.sqrt(torch.diagonal(C, dim1=-2, dim2=-1)).unsqueeze(-1) \
             / torch.sqrt(torch.diagonal(C, dim1=-2, dim2=-1)).unsqueeze(-2)

    blocks = {
        "cov_tangent":   upper_full(logC).cpu().numpy(),
        "prec_tangent":  upper_full(logT).cpu().numpy(),
        "pcorr_fisher":  np.arctanh(np.clip(pc, -0.999999, 0.999999)),
        "pcorr_tangent": upper_full(sym_funcm(P, torch.log)).cpu().numpy(),
        "corr_fisher":   np.arctanh(np.clip(upper_off(corr).cpu().numpy(),
                                            -0.999999, 0.999999)),
    }
    for k, v in blocks.items():
        print(f"  {k:14s} -> {v.shape}")

    q = pd.qcut(y, 4, labels=False, duplicates="drop")
    strat = np.asarray(q) * 2 + sex
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats, random_state=SEED)

    rows = []
    for k, (tr, te) in enumerate(rskf.split(np.zeros(N), strat), 1):
        rec = {"fold": k}
        lr = LinearRegression().fit(Zm[tr], y[tr])
        ytr_d, yte_d = y[tr] - lr.predict(Zm[tr]), y[te] - lr.predict(Zm[te])
        preds = {}
        for nm, Xb in blocks.items():
            p_ = ridge_pred(Xb[tr], y[tr], Xb[te]); preds[nm] = p_
            rec[nm] = float(pearsonr(y[te], p_)[0])
            rec[nm + "_dec"] = float(pearsonr(yte_d,
                                              ridge_pred(Xb[tr], ytr_d, Xb[te]))[0])
        Zt = np.column_stack([oof(blocks[n], y, tr)
                              for n in ("cov_tangent", "pcorr_fisher")])
        Ze = np.column_stack([preds[n] for n in ("cov_tangent", "pcorr_fisher")])
        rec["stack_cov+pcorr"] = float(pearsonr(y[te], ridge_pred(Zt, y[tr], Ze))[0])
        rows.append(rec)
        print(f"  fold {k:2d}/{args.repeats*5}  cov={rec['cov_tangent']:+.3f} "
              f"prec={rec['prec_tangent']:+.3f} pcorr={rec['pcorr_fisher']:+.3f} "
              f"stack={rec['stack_cov+pcorr']:+.3f}", flush=True)

    df = pd.DataFrame(rows); df.to_csv(OUT_DIR / "precision_partial_folds.csv", index=False)
    bar = "=" * 78
    print(f"\n{bar}\nPrecision and partial correlation, {args.repeats}x5 CV (n={N})\n{bar}")
    for c in sorted([c for c in df.columns if c != "fold" and not c.endswith("_dec")],
                    key=lambda c: -df[c].mean()):
        dec = f"   deconf {df[c+'_dec'].mean():+.4f}" if c + "_dec" in df else ""
        print(f"  {c:18s} r = {df[c].mean():+.4f} +/- {df[c].std(ddof=1):.4f}{dec}")

    def cmp(a, b, label):
        d = df[a] - df[b]; t, p = corrected_ttest_rel(df[a], df[b])
        print(f"    {label:46s} {d.mean():+.4f}  t={t:+5.2f}  p={p:.3g}  "
              f"wins {int((d>0).sum())}/{len(df)}")

    print("\n  the identity, empirically:")
    dd = np.abs(df["prec_tangent"] - df["cov_tangent"]).max()
    print(f"    max |prec_tangent - cov_tangent| across folds = {dd:.2e}"
          f"   {'-> identical, as predicted' if dd < 1e-6 else '-> NOT identical, investigate'}")
    print("\n  vs the covariance-tangent baseline:")
    for c in ["pcorr_fisher", "pcorr_tangent", "corr_fisher", "stack_cov+pcorr"]:
        cmp(c, "cov_tangent", c)
    print(f"\nSaved -> {OUT_DIR}/precision_partial_folds.csv")


if __name__ == "__main__":
    main()
