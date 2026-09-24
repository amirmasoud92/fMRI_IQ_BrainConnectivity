#!/usr/bin/env python
"""
Regional amplitude split into a stimulus-locked part and an idiosyncratic remainder, each combined with
correlation.

Usage
-----
  python scripts/models/amplitude_components.py --repeats 3
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
FLOOR    = 1e-3          # variance floor, relative to the subject's mean variance
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


def embed_from_DR(D, R, dev):
    Dt = torch.from_numpy(np.ascontiguousarray(D)).to(dev).double()
    Rt = torch.from_numpy(np.ascontiguousarray(R)).to(dev).double()
    C = Dt.unsqueeze(-1) * Rt * Dt.unsqueeze(-2)
    n = D.shape[1]
    tr = torch.diagonal(C, dim1=-2, dim2=-1).sum(-1) / n
    eye = torch.eye(n, dtype=C.dtype, device=dev)
    C = (1 - SHRINK) * C + SHRINK * tr.view(-1, 1, 1) * eye
    return upper(sym_funcm(C, torch.log)).cpu().numpy()


def zsc(A, axis=1):
    m = A.mean(axis=axis, keepdims=True); s = A.std(axis=axis, keepdims=True)
    return (A - m) / np.clip(s, 1e-8, None)


def loo_isc(TSz, tr_idx):
    """Leave-one-out ISC per region; template from TRAINING subjects only."""
    N, T, R = TSz.shape
    tr_mask = np.zeros(N, bool); tr_mask[tr_idx] = True
    Ssum = TSz[tr_mask].sum(0)
    n_tr = int(tr_mask.sum())
    out = np.empty((N, R), np.float64)
    for i in range(N):
        tmpl = (Ssum - TSz[i]) / (n_tr - 1) if tr_mask[i] else Ssum / n_tr
        a = TSz[i] - TSz[i].mean(0, keepdims=True)
        b = tmpl - tmpl.mean(0, keepdims=True)
        den = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
        out[i] = (a * b).sum(0) / np.clip(den, 1e-12, None)
    return out


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

    TS = np.stack([raw[s] for s in ids]).astype(np.float64)
    TSz = zsc(TS, axis=1)
    y   = np.array([float(parts.loc[s, TARGET]) for s in ids])
    sex = np.array([1 if str(parts.loc[s, "sex"]).lower().startswith("m") else 0 for s in ids])
    vol = M.loc[ids, vol_col].values.astype(float)
    mot = np.array([motion_stats(s, cfg["paths"]["fmriprep_dir"],
                                 cfg["dataset"]["task"]) for s in ids])
    for j in range(mot.shape[1]):
        mot[:, j] = np.where(np.isfinite(mot[:, j]), mot[:, j], np.nanmedian(mot[:, j]))
    vz = (vol - vol.mean()) / vol.std()
    Zm = np.column_stack([sex, vz, vz ** 2, mot])
    N, T, Rn = TS.shape
    print(f"n = {N} (sealed discovery split), T = {T}, regions = {Rn}")

    # fold-independent pieces: observed amplitude and the correlation matrix
    Dobs, Rmat = [], []
    for i in range(N):
        x = TS[i] - TS[i].mean(0, keepdims=True)
        S = (x.T @ x) / (T - 1)
        d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
        Dobs.append(d); Rmat.append(S / np.outer(d, d))
    Dobs = np.stack(Dobs); Rmat = np.stack(Rmat)
    X_cov  = embed_from_DR(Dobs, Rmat, dev)
    X_corr = embed_from_DR(np.ones_like(Dobs), Rmat, dev)
    A_obs  = np.log(np.maximum(Dobs ** 2, 1e-12)).astype(np.float32)
    print(f"  cov / corr embeddings: {X_cov.shape}", flush=True)

    q = pd.qcut(y, 4, labels=False, duplicates="drop")
    strat = np.asarray(q) * 2 + sex
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats, random_state=SEED)

    rows, isc_store = [], []
    for k, (tr, te) in enumerate(rskf.split(np.zeros(N), strat), 1):
        rec = {"fold": k}
        C = loo_isc(TSz, tr)                       # (N, R) fidelity, fold-safe
        c2 = np.clip(C ** 2, 0.0, 1.0)
        v_obs = Dobs ** 2
        floor = FLOOR * v_obs.mean(1, keepdims=True)
        v_stim = np.maximum(c2 * v_obs, floor)
        v_idio = np.maximum((1.0 - c2) * v_obs, floor)
        isc_store.append(C.mean(0))

        blocks = {
            "cov":  X_cov,
            "corr": X_corr,
            "stim": embed_from_DR(np.sqrt(v_stim), Rmat, dev),
            "idio": embed_from_DR(np.sqrt(v_idio), Rmat, dev),
            "amp_obs":  A_obs,
            "amp_stim": np.log(v_stim).astype(np.float32),
            "amp_idio": np.log(v_idio).astype(np.float32),
            "fidelity": C.astype(np.float32),
        }
        lr = LinearRegression().fit(Zm[tr], y[tr])
        ytr_d, yte_d = y[tr] - lr.predict(Zm[tr]), y[te] - lr.predict(Zm[te])
        for nm, Xb in blocks.items():
            rec[nm] = float(pearsonr(y[te], ridge_pred(Xb[tr], y[tr], Xb[te]))[0])
            rec[nm + "_dec"] = float(pearsonr(yte_d, ridge_pred(Xb[tr], ytr_d, Xb[te]))[0])
        rows.append(rec)
        print(f"  fold {k:2d}/{args.repeats*5}  cov={rec['cov']:+.3f} corr={rec['corr']:+.3f} "
              f"stim={rec['stim']:+.3f} idio={rec['idio']:+.3f} "
              f"fid={rec['fidelity']:+.3f}", flush=True)

    df = pd.DataFrame(rows); df.to_csv(OUT_DIR / "gain_vs_fidelity_folds.csv", index=False)
    isc_mean = np.mean(isc_store, 0)
    pd.DataFrame({"roi": np.arange(Rn), "mean_isc": isc_mean,
                  "mean_stim_fraction": (isc_mean ** 2)}
                 ).to_csv(OUT_DIR / "gain_vs_fidelity_regions.csv", index=False)

    bar = "=" * 76
    print(f"\n{bar}\nGain vs fidelity, {args.repeats}x5 CV (n={N})\n{bar}")
    for c in sorted([c for c in df.columns if c != "fold" and not c.endswith("_dec")],
                    key=lambda c: -df[c].mean()):
        print(f"  {c:12s} r = {df[c].mean():+.4f} +/- {df[c].std(ddof=1):.4f}"
              f"   deconf {df[c+'_dec'].mean():+.4f}")

    def cmp(a, b, label):
        d = df[a] - df[b]; t, p = corrected_ttest_rel(df[a], df[b])
        print(f"    {label:44s} {d.mean():+.4f}  t={t:+5.2f}  p={p:.3g}  "
              f"wins {int((d > 0).sum())}/{len(df)}")

    print("\n  WHICH COMPONENT CARRIES THE AMPLITUDE EFFECT?")
    cmp("cov", "corr", "cov - corr  (the effect to be explained)")
    cmp("stim", "corr", "stimulus-locked amplitude - corr")
    cmp("idio", "corr", "idiosyncratic amplitude - corr")
    cmp("cov", "stim", "cov - stimulus-locked")
    cmp("cov", "idio", "cov - idiosyncratic")
    tot = df["cov"].mean() - df["corr"].mean()
    if abs(tot) > 1e-9:
        print(f"\n  share of the cov-over-corr gain recovered:")
        for c, lab in [("stim", "stimulus-locked amplitude"),
                       ("idio", "idiosyncratic amplitude")]:
            print(f"    {lab:32s} {100*(df[c].mean()-df['corr'].mean())/tot:6.0f}%")
    print("\n  amplitude vectors alone (215 features):")
    for c in ["amp_obs", "amp_stim", "amp_idio", "fidelity"]:
        print(f"    {c:12s} r = {df[c].mean():+.4f}   deconf {df[c+'_dec'].mean():+.4f}")
    print(f"\n  stimulus-locked fraction of variance: mean {100*np.mean(isc_mean**2):.1f}%, "
          f"max region {100*np.max(isc_mean**2):.1f}%")
    print(f"\nSaved -> {OUT_DIR}/gain_vs_fidelity_folds.csv, gain_vs_fidelity_regions.csv")


if __name__ == "__main__":
    main()
