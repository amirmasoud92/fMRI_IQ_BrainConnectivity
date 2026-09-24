#!/usr/bin/env python
"""
Cross-validated accuracy of each imaging block against a confound model of sex, age, brain volume and
head motion, and the increment of imaging over that model.

Reports, on identical folds
---------------------------
  confounds        sex + age + brain volume + volume^2 + mean FD
  <block>          each imaging block alone
  <block>+conf     imaging block WITH the confounds included as features
  increment        (block+conf) - confounds  = what the imaging actually buys

Head motion is recomputed here from the fMRIPrep confound tables (mean framewise
displacement over retained volumes), because it was never stored in the
connectivity HDF5.

Usage
-----
  python scripts/prediction/confound_benchmark.py --repeats 5
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.data.preprocessing import load_participants
from src.data.series import load_series
from src.stats.cv_inference import corrected_ttest_rel

warnings.filterwarnings("ignore")

CFG_PATH = "config/pipeline.yaml"
TARGET   = "IST_intelligence_total"
ALPHAS   = np.logspace(-2, 6, 25)
SEED     = 42
SHRINK   = 0.55
OUT_DIR  = Path("outputs/honest")
SERIES   = "legacy"          # set from --series in main()
FS_DIR   = Path("derivatives/fs_stats")
VOL_COL  = "EstimatedTotalIntraCranialVol"


def sym_funcm(X, fn):
    X = 0.5 * (X + X.transpose(-1, -2)).double()
    w, V = torch.linalg.eigh(X)
    return V @ torch.diag_embed(fn(w.clamp_min(1e-10))) @ V.transpose(-1, -2)


def upper(L):
    n = L.shape[-1]
    iu = torch.triu_indices(n, n, offset=1, device=L.device)
    return torch.cat([torch.diagonal(L, dim1=-2, dim2=-1),
                      L[..., iu[0], iu[1]] * np.sqrt(2.0)], dim=-1).float()


def mean_fd(sub_id, fmriprep, task, fd_thr=0.5):
    """Mean framewise displacement over volumes that survive scrubbing."""
    p = Path(fmriprep) / sub_id / "func" / f"{sub_id}_task-{task}_desc-confounds_regressors.tsv"
    if not p.exists():
        return np.nan
    try:
        fd = pd.read_csv(p, sep="\t", usecols=["framewise_displacement"],
                         na_values="n/a")["framewise_displacement"].astype(float)
        fd = fd.fillna(0.0).to_numpy()
        keep = fd <= fd_thr
        keep[0] = False
        return float(fd[keep].mean()) if keep.any() else float(fd.mean())
    except Exception:
        return np.nan


def ridge_r(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    m = RidgeCV(alphas=ALPHAS, cv=5).fit(sc.transform(Xtr), ytr)
    return float(pearsonr(yte, m.predict(sc.transform(Xte)))[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--series", choices=["legacy", "psc"], default="legacy",
                    help="region timeseries source (see src/data/series.py)")
    args = ap.parse_args()
    global OUT_DIR, SERIES
    SERIES = args.series
    if SERIES != "legacy":
        OUT_DIR = OUT_DIR / SERIES
    print(f"  series = {SERIES}   -> {OUT_DIR}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(CFG_PATH))
    conn = Path(cfg["paths"]["connectivity_dir"])

    ctx, sub = {}, {}
    ctx, sub = load_series(conn, SERIES)

    frames = []
    for f in sorted(FS_DIR.glob("data-*.tsv")):
        d = pd.read_csv(f, sep="\t")
        d = d.rename(columns={d.columns[0]: "sid"})
        d["sid"] = d["sid"].astype(str)
        d = d.set_index("sid")
        d = d.loc[:, ~d.columns.duplicated()]
        d.columns = [f"{f.stem}::{c}" for c in d.columns]
        frames.append(d)
    M = pd.concat(frames, axis=1)
    M = M.loc[:, ~M.columns.duplicated()].apply(pd.to_numeric, errors="coerce")
    M = M.dropna(axis=1, how="all")
    M = M.loc[:, M.std(skipna=True) > 0]
    vol_col = [c for c in M.columns if c.endswith("::" + VOL_COL)][0]

    z = np.load(conn / "dwi_fa_features.npz", allow_pickle=True)
    FA = pd.DataFrame(z["features"], index=[str(s) for s in z["subject_ids"]])
    parts = load_participants(cfg["paths"]["bids_root"])

    ids = [s for s in sorted(set(ctx) & set(M.index) & set(FA.index))
           if ctx[s].shape[0] == sub[s].shape[0]
           and s in parts.index and not pd.isna(parts.loc[s, TARGET])
           and not pd.isna(M.loc[s, vol_col])]
    print(f"{len(ids)} subjects")

    print("Recomputing mean FD from fMRIPrep confounds ...", flush=True)
    fd = np.array([mean_fd(s, cfg["paths"]["fmriprep_dir"], cfg["dataset"]["task"])
                   for s in ids])
    print(f"  mean FD: median {np.nanmedian(fd):.3f} mm, "
          f"missing {int(np.isnan(fd).sum())}")
    fd = np.where(np.isfinite(fd), fd, np.nanmedian(fd))

    y = np.array([float(parts.loc[s, TARGET]) for s in ids])
    sex = np.array([1 if str(parts.loc[s, "sex"]).lower().startswith("m") else 0
                    for s in ids])
    age = pd.to_numeric(parts.loc[ids, "age"], errors="coerce").values.astype(float)
    age = np.where(np.isfinite(age), age, np.nanmedian(age))
    vol = M.loc[ids, vol_col].values.astype(float)

    CONF = np.column_stack([sex, age, vol, vol ** 2, fd])
    print(f"  confound block: {CONF.shape[1]} variables "
          f"(sex, age, volume, volume^2, meanFD)")
    for nm, v in [("sex", sex), ("age", age), ("volume", vol), ("meanFD", fd)]:
        print(f"    corr({nm:7s}, IST) = {pearsonr(v, y)[0]:+.3f}")

    x_cov = []
    for s in ids:
        x = np.hstack([ctx[s], sub[s]]).astype(np.float32)
        x = x - x.mean(0, keepdims=True)
        S = (x.T @ x) / (len(x) - 1)
        d = S.shape[0]
        x_cov.append((1 - SHRINK) * S + SHRINK * (np.trace(S) / d)
                     * np.eye(d, dtype=np.float32))
    C = torch.from_numpy(np.stack(x_cov)).to(dev).double()
    X_fc = upper(sym_funcm(C, torch.log)).cpu().numpy()
    Mo = M.loc[ids]
    Mo = Mo.fillna(Mo.median()).values.astype(np.float32)
    Fa = FA.loc[ids].values.astype(np.float32)

    blocks = {"fc": X_fc, "morph": Mo, "fa": Fa}

    q = pd.qcut(y, 4, labels=False, duplicates="drop")
    strat = np.asarray(q) * 2 + sex
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats,
                                   random_state=SEED)

    rows = []
    for k, (tr, te) in enumerate(rskf.split(np.zeros(len(y)), strat), 1):
        rec = {"fold": k}
        rec["confounds"] = ridge_r(CONF[tr], y[tr], CONF[te], y[te])
        for nm, X in blocks.items():
            rec[nm] = ridge_r(X[tr], y[tr], X[te], y[te])
            XC = np.hstack([X, CONF])
            rec[f"{nm}+conf"] = ridge_r(XC[tr], y[tr], XC[te], y[te])
        allX = np.hstack([blocks["fc"], blocks["morph"], blocks["fa"], CONF])
        rec["all+conf"] = ridge_r(allX[tr], y[tr], allX[te], y[te])
        rows.append(rec)
        if k % 5 == 0:
            print(f"  fold {k}/{args.repeats*5}  conf={rec['confounds']:+.3f} "
                  f"fc={rec['fc']:+.3f} fc+conf={rec['fc+conf']:+.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "confound_benchmark_folds.csv", index=False)

    cols = [c for c in df.columns if c != "fold"]
    print(f"\n{'='*74}\nConfound benchmark, {args.repeats}x5-fold CV (n={len(ids)})\n{'='*74}")
    for c in sorted(cols, key=lambda c: -df[c].mean()):
        print(f"  {c:14s} r = {df[c].mean():+.4f} +/- {df[c].std(ddof=1):.4f}")

    print(f"\n  INCREMENT OVER CONFOUNDS (what the imaging actually buys):")
    for nm in list(blocks) + ["all"]:
        c = f"{nm}+conf"
        if c not in df:
            continue
        t, p = corrected_ttest_rel(df[c], df["confounds"])
        print(f"    {nm:6s} {df[c].mean():+.4f} vs {df['confounds'].mean():+.4f}  "
              f"delta = {(df[c]-df['confounds']).mean():+.4f}  t = {t:+5.2f}  "
              f"p = {p:.3g}  wins {int((df[c]>df['confounds']).sum())}/{len(df)}")

    print(f"\n  Reference: Hebling Vieira 2021 (HCP, n=873) report confounds alone")
    print(f"  at rho = 0.371 and their best imaging model at rho = 0.454,")
    print(f"  i.e. an increment of about +0.083.")
    print(f"\nSaved -> {OUT_DIR}/confound_benchmark_folds.csv")


if __name__ == "__main__":
    main()
