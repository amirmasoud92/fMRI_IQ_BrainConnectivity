#!/usr/bin/env python
"""
Structure-function coupling and sample entropy of the regional signals as alternative feature sets.

Usage
-----
  python scripts/models/structure_function_and_entropy.py --repeats 3
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
from src.data.series import roi_networks
from src.stats.cv_inference import corrected_ttest_rel

warnings.filterwarnings("ignore")

CFG_PATH = "config/pipeline.yaml"
TARGET   = "IST_intelligence_total"
ALPHAS   = np.logspace(-2, 6, 25)
SEED     = 42
SHRINK   = 0.55
SAMPEN_M = 2
SAMPEN_R = 0.2
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


def embed(series, dev):
    covs = []
    for x in series:
        x = x - x.mean(0, keepdims=True)
        S = (x.T @ x) / max(len(x) - 1, 1)
        d = S.shape[0]
        covs.append((1 - SHRINK) * S + SHRINK * (np.trace(S) / d) * np.eye(d))
    C = torch.from_numpy(np.stack(covs)).to(dev).double()
    return upper(sym_funcm(C, torch.log)).cpu().numpy()


def sampen_batch(x, dev, m=SAMPEN_M, r_frac=SAMPEN_R):
    """Sample entropy for every column of x, computed on the GPU.

    SampEn = -ln(A / B), where B counts pairs of length-m templates within a
    Chebyshev radius r and A the same for length m+1. Both counts use the same
    index range and exclude self-matches. r is r_frac x the per-column SD.
    """
    T, R = x.shape
    X = torch.from_numpy(np.ascontiguousarray(x)).to(dev).float()
    r = r_frac * X.std(0, unbiased=True)                       # (R,)
    n = T - m                                                  # shared range
    counts = []
    for mm in (m, m + 1):
        # templates: (n, mm, R)
        tem = torch.stack([X[i:i + n] for i in range(mm)], dim=1)
        # pairwise Chebyshev distance per column: (n, n, R)
        d = (tem.unsqueeze(1) - tem.unsqueeze(0)).abs().amax(dim=2)
        hit = (d <= r.view(1, 1, R)).sum(dim=(0, 1)).float()
        hit = hit - n                                          # drop self-matches
        counts.append(hit)
    B, A = counts[0], counts[1]
    out = -torch.log(A.clamp_min(1.0) / B.clamp_min(1.0))
    out[(A < 1) | (B < 1)] = float("nan")
    return out.cpu().numpy()


def ridge_pred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    return RidgeCV(alphas=ALPHAS, cv=5).fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))


def oof(X, y, tr, n_inner=5):
    o = np.zeros(len(tr))
    for a, b in KFold(n_inner, shuffle=True, random_state=SEED).split(tr):
        o[b] = ridge_pred(X[tr[a]], y[tr[a]], X[tr[b]])
    return o


def zrow(A):
    """z-score each ROW (within subject, across regions)."""
    m = A.mean(1, keepdims=True); s = A.std(1, keepdims=True)
    return (A - m) / np.clip(s, 1e-8, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(CFG_PATH))
    conn = Path(cfg["paths"]["connectivity_dir"])
    if not MANIFEST.exists():
        sys.exit("Missing holdout manifest -- run scripts/prediction/seal_holdout.py first")
    disc = json.loads(MANIFEST.read_text())["discovery"]

    vz = np.load(conn / "vbm_features.npz", allow_pickle=True)
    vbm_ids = [str(s) for s in vz["subject_ids"]]
    vbm_sum = dict(zip(vbm_ids, vz["sum"]))
    vbm_mean = dict(zip(vbm_ids, vz["mean"]))
    rsize = np.asarray(vz["region_sizes"], float)

    raw = {}
    with h5py.File(conn / "unscrubbed_ts.h5", "r") as f:
        for s in disc:
            if s in f["subjects"]:
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

    ids = [s for s in disc if s in raw and s in vbm_sum and s in M.index
           and not pd.isna(M.loc[s, vol_col])]
    N = len(ids)
    print(f"n = {N} (discovery split with VBM and functional data)")

    y   = np.array([float(parts.loc[s, TARGET]) for s in ids])
    sex = np.array([1 if str(parts.loc[s, "sex"]).lower().startswith("m") else 0 for s in ids])
    vol = M.loc[ids, vol_col].values.astype(float)
    mot = np.array([motion_stats(s, cfg["paths"]["fmriprep_dir"],
                                 cfg["dataset"]["task"]) for s in ids])
    for j in range(mot.shape[1]):
        mot[:, j] = np.where(np.isfinite(mot[:, j]), mot[:, j], np.nanmedian(mot[:, j]))
    vzs = (vol - vol.mean()) / vol.std()
    Zm = np.column_stack([sex, vzs, vzs ** 2, mot])

    # ---- amplitude, structure, decoupling ---------------------------------
    AMP = np.stack([np.log(np.maximum(raw[s].var(0, ddof=1), 1e-12)) for s in ids])
    VS = np.stack([vbm_sum[s] for s in ids]).astype(float)
    VM = np.stack([vbm_mean[s] for s in ids]).astype(float)
    Az, VSz, VMz = zrow(AMP), zrow(VS), zrow(VM)
    logsz = np.log(np.maximum(rsize, 1.0))
    logsz = (logsz - logsz.mean()) / logsz.std()

    DEC = np.empty_like(Az)
    COUP = np.empty((N, 4))
    for i in range(N):
        P = np.column_stack([np.ones(Az.shape[1]), VMz[i], VSz[i], logsz])
        beta, *_ = np.linalg.lstsq(P, Az[i], rcond=None)
        fit = P @ beta
        DEC[i] = Az[i] - fit
        ss = 1 - ((Az[i] - fit) ** 2).sum() / max(((Az[i] - Az[i].mean()) ** 2).sum(), 1e-12)
        COUP[i] = [ss, beta[1], beta[2], beta[3]]
    print(f"  within-subject structure->function R^2: mean {COUP[:,0].mean():.3f} "
          f"(range {COUP[:,0].min():.3f}-{COUP[:,0].max():.3f})")
    print(f"  corr(coupling R^2, IST) = {pearsonr(COUP[:,0], y)[0]:+.3f}", flush=True)

    # ---- sample entropy ----------------------------------------------------
    print("  computing sample entropy (m=2, r=0.2 SD) ...", flush=True)
    SE = np.stack([sampen_batch(raw[s], dev) for s in ids])
    bad = ~np.isfinite(SE)
    if bad.any():
        SE[bad] = np.take(np.nanmedian(SE, axis=0), np.where(bad)[1])
        print(f"    {bad.sum()} non-finite entries imputed with the region median")
    print(f"    sample entropy: mean {np.nanmean(SE):.3f}, "
          f"between-subject SD of the mean {SE.mean(1).std():.3f}")
    print(f"    corr(mean entropy, mean log-amplitude) = "
          f"{pearsonr(SE.mean(1), AMP.mean(1))[0]:+.3f}")
    print(f"    corr(mean entropy, meanFD) = {pearsonr(SE.mean(1), mot[:,0])[0]:+.3f}",
          flush=True)

    nets = np.array(roi_networks(conn, AMP.shape[1]))
    pd.DataFrame({"roi": np.arange(AMP.shape[1]), "network": nets,
                  "mean_decoupling": DEC.mean(0), "mean_sampen": SE.mean(0),
                  "dec_ist_r": [pearsonr(DEC[:, j], y)[0] for j in range(DEC.shape[1])],
                  "sampen_ist_r": [pearsonr(SE[:, j], y)[0] for j in range(SE.shape[1])]
                  }).to_csv(OUT_DIR / "sfc_entropy_regions.csv", index=False)

    X_fc = embed([raw[s] for s in ids], dev)
    blocks = {
        "fc": X_fc,
        "amp": AMP.astype(np.float32),
        "struct": np.hstack([VM, VS]).astype(np.float32),
        "decouple": DEC.astype(np.float32),
        "coupling": COUP.astype(np.float32),
        "amp_struct": np.hstack([AMP, VM, VS]).astype(np.float32),
        "sampen": SE.astype(np.float32),
        "amp_sampen": np.hstack([AMP, SE]).astype(np.float32),
    }
    for k, v in blocks.items():
        print(f"  {k:12s} -> {v.shape}")

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
            p = ridge_pred(Xb[tr], y[tr], Xb[te]); preds[nm] = p
            rec[nm] = float(pearsonr(y[te], p)[0])
            rec[nm + "_dec"] = float(pearsonr(yte_d,
                                              ridge_pred(Xb[tr], ytr_d, Xb[te]))[0])
        cache = {}
        for combo in [("fc", "decouple"), ("fc", "sampen"), ("fc", "amp")]:
            for n_ in combo:
                cache.setdefault(n_, oof(blocks[n_], y, tr))
            Zt = np.column_stack([cache[n_] for n_ in combo])
            Ze = np.column_stack([preds[n_] for n_ in combo])
            rec["stack_" + "+".join(combo)] = float(
                pearsonr(y[te], ridge_pred(Zt, y[tr], Ze))[0])
        rows.append(rec)
        print(f"  fold {k:2d}/{args.repeats*5}  amp={rec['amp']:+.3f} "
              f"decouple={rec['decouple']:+.3f} sampen={rec['sampen']:+.3f} "
              f"fc={rec['fc']:+.3f}", flush=True)

    df = pd.DataFrame(rows); df.to_csv(OUT_DIR / "sfc_entropy_folds.csv", index=False)
    bar = "=" * 78
    print(f"\n{bar}\nStructure-function decoupling and entropy, "
          f"{args.repeats}x5 CV (n={N})\n{bar}")
    for c in sorted([c for c in df.columns if c != "fold" and not c.endswith("_dec")],
                    key=lambda c: -df[c].mean()):
        dec = f"   deconf {df[c+'_dec'].mean():+.4f}" if c + "_dec" in df else ""
        print(f"  {c:22s} r = {df[c].mean():+.4f} +/- {df[c].std(ddof=1):.4f}{dec}")

    def cmp(a, b, label):
        d = df[a] - df[b]; t, p = corrected_ttest_rel(df[a], df[b])
        print(f"    {label:52s} {d.mean():+.4f}  t={t:+5.2f}  p={p:.3g}  "
              f"wins {int((d>0).sum())}/{len(df)}")

    print("\n  BLOCK A  is decoupling more than a re-parameterisation?")
    cmp("decouple", "amp", "decoupling - amplitude")
    cmp("decouple", "struct", "decoupling - structure")
    cmp("decouple", "amp_struct", "decoupling - [amplitude + structure]  <- the test")
    cmp("stack_fc+decouple", "fc", "stack fc+decoupling - fc alone")
    print("\n  BLOCK B  is entropy more than amplitude?")
    cmp("sampen", "amp", "sample entropy - amplitude")
    cmp("amp_sampen", "amp", "[amplitude + entropy] - amplitude")
    cmp("stack_fc+sampen", "fc", "stack fc+entropy - fc alone")
    print(f"\nSaved -> {OUT_DIR}/sfc_entropy_folds.csv, sfc_entropy_regions.csv")


if __name__ == "__main__":
    main()
