#!/usr/bin/env python
"""
Separate ridge penalties for the amplitude and connectivity blocks against a single penalty.

Usage
-----
  python scripts/models/twoblock_ridge.py --repeats 3
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
G_GRID   = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
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


def embed(D, R, dev):
    """Return (diagonal block, off-diagonal block) of the log-Euclidean tangent."""
    Dt = torch.from_numpy(np.ascontiguousarray(D)).to(dev).double()
    Rt = torch.from_numpy(np.ascontiguousarray(R)).to(dev).double()
    C = Dt.unsqueeze(-1) * Rt * Dt.unsqueeze(-2)
    n = D.shape[1]
    tr = torch.diagonal(C, dim1=-2, dim2=-1).sum(-1) / n
    eye = torch.eye(n, dtype=C.dtype, device=dev)
    C = (1 - SHRINK) * C + SHRINK * tr.view(-1, 1, 1) * eye
    L = sym_funcm(C, torch.log)
    iu = torch.triu_indices(n, n, offset=1, device=L.device)
    diag = torch.diagonal(L, dim1=-2, dim2=-1).float().cpu().numpy()
    off = (L[..., iu[0], iu[1]] * np.sqrt(2.0)).float().cpu().numpy()
    return diag, off


def fit_gcv(Xtr, ytr, Xte):
    """RidgeCV with efficient LOO generalised CV; returns (prediction, cv score)."""
    m = RidgeCV(alphas=ALPHAS, store_cv_results=False).fit(Xtr, ytr)
    return m.predict(Xte), float(m.best_score_)


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

    Dobs, Rmat = [], []
    for s in ids:
        x = raw[s].astype(np.float64); x = x - x.mean(0, keepdims=True)
        S = (x.T @ x) / (len(x) - 1)
        d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
        Dobs.append(d); Rmat.append(S / np.outer(d, d))
    Dobs = np.stack(Dobs); Rmat = np.stack(Rmat)
    Xd, Xo = embed(Dobs, Rmat, dev)
    N = len(ids)
    print(f"n = {N} (sealed discovery split)")
    print(f"  diagonal block {Xd.shape}, off-diagonal block {Xo.shape}")
    print(f"  g grid: {G_GRID}   (g > 1 penalises amplitude LESS)", flush=True)

    q = pd.qcut(y, 4, labels=False, duplicates="drop")
    strat = np.asarray(q) * 2 + sex
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats, random_state=SEED)

    rows, chosen = [], []
    for k, (tr, te) in enumerate(rskf.split(np.zeros(N), strat), 1):
        rec = {"fold": k}
        sd = StandardScaler().fit(Xd[tr]); so = StandardScaler().fit(Xo[tr])
        Dtr, Dte = sd.transform(Xd[tr]), sd.transform(Xd[te])
        Otr, Ote = so.transform(Xo[tr]), so.transform(Xo[te])

        lr = LinearRegression().fit(Zm[tr], y[tr])
        ytr_d, yte_d = y[tr] - lr.predict(Zm[tr]), y[te] - lr.predict(Zm[te])

        for tag, yy_tr, yy_te in [("", y[tr], y[te]), ("_dec", ytr_d, yte_d)]:
            # baseline: one penalty for everything (project convention, cv=5)
            Atr = np.hstack([Dtr, Otr]); Ate = np.hstack([Dte, Ote])
            p5 = RidgeCV(alphas=ALPHAS, cv=5).fit(Atr, yy_tr).predict(Ate)
            rec[f"single_cv5{tag}"] = float(pearsonr(yy_te, p5)[0])
            pg, _ = fit_gcv(Atr, yy_tr, Ate)
            rec[f"single_gcv{tag}"] = float(pearsonr(yy_te, pg)[0])

            best_g, best_s, best_p = None, -np.inf, None
            for g in G_GRID:
                pr, sc = fit_gcv(np.hstack([g * Dtr, Otr]), yy_tr,
                                 np.hstack([g * Dte, Ote]))
                if sc > best_s:
                    best_s, best_g, best_p = sc, g, pr
            rec[f"twoblock{tag}"] = float(pearsonr(yy_te, best_p)[0])
            rec[f"g{tag}"] = best_g
            if tag == "":
                chosen.append(best_g)

        rows.append(rec)
        print(f"  fold {k:2d}/{args.repeats*5}  single={rec['single_cv5']:+.3f} "
              f"twoblock={rec['twoblock']:+.3f}  g={rec['g']:g}", flush=True)

    df = pd.DataFrame(rows); df.to_csv(OUT_DIR / "twoblock_ridge_folds.csv", index=False)
    bar = "=" * 76
    print(f"\n{bar}\nTwo-block ridge penalty, {args.repeats}x5 CV (n={N})\n{bar}")
    for c in ["single_cv5", "single_gcv", "twoblock"]:
        print(f"  {c:12s} r = {df[c].mean():+.4f} +/- {df[c].std(ddof=1):.4f}"
              f"   deconf {df[c+'_dec'].mean():+.4f}")

    def cmp(a, b, label):
        d = df[a] - df[b]; t, p = corrected_ttest_rel(df[a], df[b])
        print(f"    {label:40s} {d.mean():+.4f}  t={t:+5.2f}  p={p:.3g}  "
              f"wins {int((d > 0).sum())}/{len(df)}")

    print()
    cmp("twoblock", "single_gcv", "twoblock - single (same selector)")
    cmp("twoblock", "single_cv5", "twoblock - single_cv5 (convention)")
    cmp("single_gcv", "single_cv5", "GCV vs cv=5 baseline (should be ~0)")
    cmp("twoblock_dec", "single_gcv_dec", "twoblock - single, deconfounded")
    print(f"\n  selected g (raw target): {dict(sorted(Counter(chosen).items()))}")
    print(f"    median g = {np.median(chosen):g};  g>1 in "
          f"{sum(1 for g in chosen if g > 1)}/{len(chosen)} folds")
    print(f"  selected g (deconfounded): "
          f"{dict(sorted(Counter(df['g_dec']).items()))}")
    print(f"\nSaved -> {OUT_DIR}/twoblock_ridge_folds.csv")


if __name__ == "__main__":
    main()
