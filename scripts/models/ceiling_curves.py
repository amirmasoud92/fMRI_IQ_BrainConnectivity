#!/usr/bin/env python
"""
Accuracy against scan duration and against training-set size, with classical test-theory saturation fits.

Usage
-----
  python scripts/models/ceiling_curves.py --repeats 3
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
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.data.preprocessing import load_participants

warnings.filterwarnings("ignore")

CFG_PATH = "config/pipeline.yaml"
TARGET   = "IST_intelligence_total"
ALPHAS   = np.logspace(-2, 6, 25)
SEED     = 42
SHRINK   = 0.55
TR_SEC   = 2.2
FRACS    = [0.15, 0.30, 0.50, 0.70, 0.85, 1.00]
N_GRID   = [150, 250, 350, 450, 550, 650]
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


def embed_covs(covs, dev):
    C = torch.from_numpy(np.stack(covs)).to(dev).double()
    return upper(sym_funcm(C, torch.log)).cpu().numpy()


def cov_fixed(x):
    x = x - x.mean(0, keepdims=True)
    S = (x.T @ x) / max(len(x) - 1, 1)
    d = S.shape[0]
    return (1 - SHRINK) * S + SHRINK * (np.trace(S) / d) * np.eye(d)


def cov_lw(x):
    S, _ = ledoit_wolf(x, assume_centered=False)
    return S


def ridge_pred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    return RidgeCV(alphas=ALPHAS, cv=5).fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))


def sat(T, r_inf, T0):
    """Classical-test-theory saturation: r(T) = r_inf * sqrt(T / (T + T0))."""
    return r_inf * np.sqrt(T / (T + T0))


def fit_and_project(x, y, label, unit, targets, want=0.50):
    try:
        popt, _ = curve_fit(sat, x, y, p0=[max(y) * 1.4, np.median(x)],
                            maxfev=20000, bounds=([0, 1e-6], [1.5, 1e6]))
    except Exception as e:
        print(f"    [{label}: fit failed -- {e}]")
        return None
    r_inf, T0 = popt
    pred = sat(np.asarray(x), *popt)
    ss = 1 - np.sum((np.asarray(y) - pred) ** 2) / np.sum((np.asarray(y) - np.mean(y)) ** 2)
    print(f"    fit  r_inf = {r_inf:.3f}   half-saturation = {T0:.1f} {unit}   R2 = {ss:.3f}")
    for t in targets:
        print(f"      projected r at {t:>6.0f} {unit:<8s} = {sat(t, *popt):+.4f}")
    if r_inf <= want:
        print(f"      r = {want} is NOT reachable by {unit} alone "
              f"(asymptote {r_inf:.3f} < {want})")
    else:
        need = T0 * want ** 2 / (r_inf ** 2 - want ** 2)
        print(f"      r = {want} would need {need:.0f} {unit} "
              f"({need/max(x):.1f}x the current {max(x):.0f})")
    return {"label": label, "r_inf": r_inf, "T0": T0, "r2": ss}


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
    vz = (vol - vol.mean()) / vol.std()
    Zm = np.column_stack([sex, vz, vz ** 2, mot])
    N, T_full = len(ids), raw[ids[0]].shape[0]
    print(f"n = {N} (sealed discovery split), T = {T_full} TRs = "
          f"{T_full*TR_SEC/60:.1f} min", flush=True)

    q = pd.qcut(y, 4, labels=False, duplicates="drop")
    strat = np.asarray(q) * 2 + sex
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats, random_state=SEED)
    folds = list(rskf.split(np.zeros(N), strat))

    # ---------------- Part A: scan length ---------------------------------
    print(f"\n{'='*78}\nPART A  scan length\n{'='*78}", flush=True)
    confs = []
    for fr in FRACS:
        k = int(round(fr * T_full))
        offs = [0] if fr > 0.95 else [0, T_full - k]
        for oi, off in enumerate(offs):
            confs.append((fr, k, off, oi))
    rowsA = []
    for fr, k, off, oi in confs:
        segs = [raw[s][off:off + k] for s in ids]
        blocks = {"lw": embed_covs([cov_lw(x) for x in segs], dev),
                  "fixed": embed_covs([cov_fixed(x) for x in segs], dev)}
        for fi, (tr, te) in enumerate(folds, 1):
            lr = LinearRegression().fit(Zm[tr], y[tr])
            ytr_d, yte_d = y[tr] - lr.predict(Zm[tr]), y[te] - lr.predict(Zm[te])
            rec = {"frac": fr, "n_tr_kept": k, "minutes": k * TR_SEC / 60,
                   "offset": off, "window": oi, "fold": fi}
            for nm, Xb in blocks.items():
                rec[nm] = float(pearsonr(y[te], ridge_pred(Xb[tr], y[tr], Xb[te]))[0])
                rec[nm + "_dec"] = float(pearsonr(
                    yte_d, ridge_pred(Xb[tr], ytr_d, Xb[te]))[0])
            rowsA.append(rec)
        m = np.mean([r["lw"] for r in rowsA if r["frac"] == fr and r["window"] == oi])
        print(f"  {k:3d} TRs ({k*TR_SEC/60:4.1f} min) window {oi}  lw r = {m:+.4f}",
              flush=True)
    A = pd.DataFrame(rowsA); A.to_csv(OUT_DIR / "ceiling_scanlength_folds.csv", index=False)

    # ---------------- Part B: sample size ---------------------------------
    print(f"\n{'='*78}\nPART B  training-set size (test fold held fixed)\n{'='*78}", flush=True)
    X_full = embed_covs([cov_lw(raw[s]) for s in ids], dev)
    rng = np.random.default_rng(SEED)
    rowsB = []
    grid = [n for n in N_GRID if n < len(folds[0][0])] + [len(folds[0][0])]
    for n_tr in grid:
        for fi, (tr, te) in enumerate(folds, 1):
            sub = rng.choice(tr, size=min(n_tr, len(tr)), replace=False)
            lr = LinearRegression().fit(Zm[sub], y[sub])
            rec = {"n_train": len(sub), "fold": fi,
                   "r": float(pearsonr(y[te], ridge_pred(X_full[sub], y[sub],
                                                         X_full[te]))[0]),
                   "r_dec": float(pearsonr(
                       y[te] - lr.predict(Zm[te]),
                       ridge_pred(X_full[sub], y[sub] - lr.predict(Zm[sub]),
                                  X_full[te]))[0])}
            rowsB.append(rec)
        m = np.mean([r["r"] for r in rowsB if r["n_train"] == min(n_tr, len(folds[0][0]))])
        print(f"  n_train {min(n_tr, len(folds[0][0])):4d}   r = {m:+.4f}", flush=True)
    B = pd.DataFrame(rowsB); B.to_csv(OUT_DIR / "ceiling_samplesize_folds.csv", index=False)

    # ---------------- Part C: extrapolation --------------------------------
    print(f"\n{'='*78}\nPART C  extrapolation\n{'='*78}")
    ga = A.groupby("minutes")[["lw", "fixed", "lw_dec"]].mean().reset_index()
    print("\n  scan length (minutes -> r):")
    print(f"  {'min':>6s} {'TRs':>5s} {'ledoit-wolf':>12s} {'fixed 0.55':>11s} "
          f"{'lw deconf':>10s}")
    for _, r in ga.iterrows():
        k = int(round(r["minutes"] * 60 / TR_SEC))
        print(f"  {r['minutes']:6.1f} {k:5d} {r['lw']:+12.4f} {r['fixed']:+11.4f} "
              f"{r['lw_dec']:+10.4f}")
    bias = (ga["lw"] - ga["fixed"])
    print(f"\n  adaptive-vs-fixed shrinkage gap: {bias.iloc[0]:+.4f} at the shortest "
          f"duration, {bias.iloc[-1]:+.4f} at full length")
    print("  (a large gap at short durations is exactly the bias this guards against)")

    print("\n  PROJECTION from scan length:")
    fa = fit_and_project(ga["minutes"].values, ga["lw"].values,
                         "scan length", "min", [15, 20, 30, 45, 60])
    print("\n  PROJECTION from scan length, deconfounded:")
    fad = fit_and_project(ga["minutes"].values, ga["lw_dec"].values,
                          "scan length (deconf)", "min", [20, 30, 60], want=0.40)

    gb = B.groupby("n_train")[["r", "r_dec"]].mean().reset_index()
    print("\n  training-set size (n -> r):")
    for _, r in gb.iterrows():
        print(f"  {int(r['n_train']):5d}  r = {r['r']:+.4f}   deconf {r['r_dec']:+.4f}")
    print("\n  PROJECTION from sample size:")
    fb = fit_and_project(gb["n_train"].values, gb["r"].values,
                         "sample size", "subjects", [1000, 2000, 5000, 10000])

    pd.DataFrame([f for f in (fa, fad, fb) if f]).to_csv(
        OUT_DIR / "ceiling_fits.csv", index=False)
    print(f"\nSaved -> {OUT_DIR}/ceiling_scanlength_folds.csv, "
          f"ceiling_samplesize_folds.csv, ceiling_fits.csv")


if __name__ == "__main__":
    main()
