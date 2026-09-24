#!/usr/bin/env python
"""
Amplitude weighting of the covariance, with controls that keep only the global level, keep only the regional
profile, or take amplitudes from another participant.

Usage
-----
  python scripts/models/amplitude_weighting.py --repeats 3
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
LAMBDAS  = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
N_SWAP   = 3
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


def embed_from_DR(D, R, dev):
    """Rebuild C = diag(d) R diag(d), shrink, log-Euclidean, vectorise."""
    N, n = D.shape
    Dt = torch.from_numpy(D).to(dev).double()
    Rt = torch.from_numpy(R).to(dev).double()
    C = Dt.unsqueeze(-1) * Rt * Dt.unsqueeze(-2)
    tr = torch.diagonal(C, dim1=-2, dim2=-1).sum(-1) / n
    eye = torch.eye(n, dtype=C.dtype, device=dev)
    C = (1 - SHRINK) * C + SHRINK * tr.view(-1, 1, 1) * eye
    return upper(sym_funcm(C, torch.log)).cpu().numpy()


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

    # ---- decompose every subject into amplitude D and correlation R ---------
    Ds, Rs = [], []
    for s in ids:
        x = raw[s].astype(np.float64)
        x = x - x.mean(0, keepdims=True)
        S = (x.T @ x) / (len(x) - 1)
        d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
        Ds.append(d)
        Rs.append(S / np.outer(d, d))
    D = np.stack(Ds); R = np.stack(Rs)
    N, n = D.shape
    print(f"n = {N}, regions = {n}")
    print(f"  amplitude: within-subject SD of log sigma across regions = "
          f"{np.log(D).std(1).mean():.3f}")
    print(f"  between-subject SD of mean log sigma                     = "
          f"{np.log(D).mean(1).std():.3f}", flush=True)

    gm = np.exp(np.log(D).mean(1)).mean()          # group mean global level
    blocks = {}
    for lam in LAMBDAS:
        blocks[f"lam_{lam:.2f}"] = embed_from_DR(D ** lam, R, dev)
    rng = np.random.default_rng(SEED)
    for k in range(N_SWAP):
        perm = rng.permutation(N)
        # a fixed point would leak the true pairing back in for that subject
        while np.any(perm == np.arange(N)):
            bad = np.where(perm == np.arange(N))[0]
            perm[bad] = perm[rng.permutation(len(perm))[:len(bad)]]
        blocks[f"swap{k}"] = embed_from_DR(D[perm], R, dev)
    g = np.exp(np.log(D).mean(1))                              # per-subject scale
    blocks["global_only"]   = embed_from_DR(np.repeat(g[:, None], n, 1), R, dev)
    blocks["regional_only"] = embed_from_DR(D / g[:, None] * gm, R, dev)
    for k in blocks:
        print(f"  built {k:16s} -> {blocks[k].shape}")

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
        rec["swap"] = float(np.mean([rec[f"swap{j}"] for j in range(N_SWAP)]))
        rec["swap_dec"] = float(np.mean([rec[f"swap{j}_dec"] for j in range(N_SWAP)]))
        rows.append(rec)
        print(f"  fold {k:2d}/{args.repeats*5}  corr={rec['lam_0.00']:+.3f} "
              f"cov={rec['lam_1.00']:+.3f} swap={rec['swap']:+.3f} "
              f"glob={rec['global_only']:+.3f} reg={rec['regional_only']:+.3f}", flush=True)

    df = pd.DataFrame(rows); df.to_csv(OUT_DIR / "amplitude_mechanism_folds.csv", index=False)
    bar = "=" * 76
    print(f"\n{bar}\nAmplitude mechanism, {args.repeats}x5 CV (n={N})\n{bar}")
    print("  LAMBDA SWEEP  C(lam) = D^lam R D^lam   (0 = correlation, 1 = covariance)")
    for lam in LAMBDAS:
        c = f"lam_{lam:.2f}"
        star = "   <- covariance" if lam == 1.0 else ("   <- correlation" if lam == 0.0 else "")
        print(f"    lam={lam:<5.2f} r = {df[c].mean():+.4f} +/- {df[c].std(ddof=1):.4f}"
              f"   deconf {df[c+'_dec'].mean():+.4f}{star}")

    def cmp(a, b, label):
        d = df[a] - df[b]; t, p = corrected_ttest_rel(df[a], df[b])
        print(f"    {label:46s} {d.mean():+.4f}  t={t:+5.2f}  p={p:.3g}  "
              f"wins {int((d > 0).sum())}/{len(df)}")

    print("\n  IS THE AMPLITUDE SUBJECT-SPECIFIC?")
    for c in ["lam_1.00", "swap", "global_only", "regional_only"]:
        print(f"    {c:16s} r = {df[c].mean():+.4f} +/- {df[c].std(ddof=1):.4f}"
              f"   deconf {df[c+'_dec'].mean():+.4f}")
    print()
    cmp("lam_1.00", "lam_0.00", "covariance - correlation (the effect)")
    cmp("swap", "lam_0.00", "swapped amplitude - correlation")
    cmp("lam_1.00", "swap", "real amplitude - swapped amplitude")
    cmp("global_only", "lam_0.00", "global scale only - correlation")
    cmp("regional_only", "lam_0.00", "regional profile only - correlation")
    cmp("lam_1.00", "global_only", "covariance - global only")
    cmp("lam_1.00", "regional_only", "covariance - regional only")

    tot = df["lam_1.00"].mean() - df["lam_0.00"].mean()
    if abs(tot) > 1e-9:
        print(f"\n  share of the covariance-over-correlation gain recovered:")
        for c, lab in [("global_only", "global scale alone"),
                       ("regional_only", "regional profile alone"),
                       ("swap", "amplitude from another subject")]:
            print(f"    {lab:34s} {100*(df[c].mean()-df['lam_0.00'].mean())/tot:6.0f}%")
    print(f"\nSaved -> {OUT_DIR}/amplitude_mechanism_folds.csv")


if __name__ == "__main__":
    main()
