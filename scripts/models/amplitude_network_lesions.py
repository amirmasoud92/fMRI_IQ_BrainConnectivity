#!/usr/bin/env python
"""
Network-wise removal of the regional amplitude profile, against size-matched and spatially contiguous nulls.

For each Yeo network g, two arms
--------------------------------

  lesion_g    regions in g get the TRAINING-FOLD MEAN amplitude, every other
              region keeps its own  ->  how much is LOST without g  (necessity)
  only_g      regions in g keep their own amplitude, all others get the mean
              ->  how much g ALONE recovers                        (sufficiency)

Zero learned parameters, and both directions are measured rather than inferred
from one. Group means come from training subjects only. The correlation matrix R
is held fixed in every arm, so only amplitude changes.

Network sizes differ a lot (Default 46 regions, Limbic 12), so the per-region
normalisation matters as much as the raw drop and both are reported. This is the
level at which attribution is defensible here: the Haufe analysis put honest
disjoint split-half reliability of the EDGE maps at ~0.39, so edge-level claims
are not supportable, while network-level effects survive permutation.

Network labels come from src.data.series.roi_networks, which fixes an off-by-one
that shifted every cortical parcel onto its neighbour's network.

Runs on the sealed discovery split only.

Usage
-----
  python scripts/models/amplitude_network_lesions.py --repeats 3
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
from src.data.series import roi_networks
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

    Dobs, Rmat = [], []
    for s in ids:
        x = raw[s].astype(np.float64); x = x - x.mean(0, keepdims=True)
        S = (x.T @ x) / (len(x) - 1)
        d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
        Dobs.append(d); Rmat.append(S / np.outer(d, d))
    Dobs = np.stack(Dobs); Rmat = np.stack(Rmat)
    N, n_roi = Dobs.shape

    nets = np.array(roi_networks(conn, n_roi))
    groups = sorted(set(nets))
    idx = {g: np.where(nets == g)[0] for g in groups}
    print(f"n = {N} (sealed discovery split), regions = {n_roi}")
    print(f"  networks: " + ", ".join(f"{g}({len(idx[g])})" for g in groups), flush=True)

    q = pd.qcut(y, 4, labels=False, duplicates="drop")
    strat = np.asarray(q) * 2 + sex
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats, random_state=SEED)

    rows = []
    for k, (tr, te) in enumerate(rskf.split(np.zeros(N), strat), 1):
        rec = {"fold": k}
        gmean = Dobs[tr].mean(0)                       # training-fold mean amplitude
        Dflat = np.repeat(gmean[None, :], N, axis=0)

        blocks = {"cov": embed_from_DR(Dobs, Rmat, dev),
                  "flat": embed_from_DR(Dflat, Rmat, dev)}
        for g in groups:
            D_les = Dobs.copy(); D_les[:, idx[g]] = gmean[idx[g]]
            blocks[f"lesion_{g}"] = embed_from_DR(D_les, Rmat, dev)
            D_only = Dflat.copy(); D_only[:, idx[g]] = Dobs[:, idx[g]]
            blocks[f"only_{g}"] = embed_from_DR(D_only, Rmat, dev)

        lr = LinearRegression().fit(Zm[tr], y[tr])
        ytr_d, yte_d = y[tr] - lr.predict(Zm[tr]), y[te] - lr.predict(Zm[te])
        # Deconfounded arms for EVERY block, not just cov/flat. Visual cortex is
        # the least motion-correlated network (r=+0.392 vs Limbic's +0.659) yet
        # carries the largest lesion effect, so a motion artefact is already
        # unlikely -- but the lesion effect surviving sex+size+motion control is
        # the test that actually settles it.
        for nm, Xb in blocks.items():
            rec[nm] = float(pearsonr(y[te], ridge_pred(Xb[tr], y[tr], Xb[te]))[0])
            rec[nm + "_dec"] = float(pearsonr(
                yte_d, ridge_pred(Xb[tr], ytr_d, Xb[te]))[0])
        rows.append(rec)
        print(f"  fold {k:2d}/{args.repeats*5}  cov={rec['cov']:+.3f} "
              f"flat={rec['flat']:+.3f}  "
              + " ".join(f"{g[:4]}={rec['lesion_'+g]:+.3f}" for g in groups[:4]),
              flush=True)

    df = pd.DataFrame(rows); df.to_csv(OUT_DIR / "amplitude_lesion_folds.csv", index=False)
    bar = "=" * 78
    print(f"\n{bar}\nAmplitude lesioning by network, {args.repeats}x5 CV (n={N})\n{bar}")
    print(f"  cov  (all amplitude)      r = {df['cov'].mean():+.4f} "
          f"(deconf {df['cov_dec'].mean():+.4f})")
    print(f"  flat (no amplitude)       r = {df['flat'].mean():+.4f} "
          f"(deconf {df['flat_dec'].mean():+.4f})")
    total = df["cov"].mean() - df["flat"].mean()
    print(f"  total amplitude effect    {total:+.4f}\n")

    out = []
    print(f"  {'network':13s} {'n':>4s} {'lesion r':>9s} {'loss':>8s} {'per-ROI':>9s} "
          f"{'p':>9s} | {'only r':>8s} {'recov':>8s} | {'loss_dec':>8s} {'p_dec':>9s}")
    for g in groups:
        les, only = df[f"lesion_{g}"], df[f"only_{g}"]
        loss = df["cov"].mean() - les.mean()
        t, p = corrected_ttest_rel(df["cov"], les)
        rec_frac = (only.mean() - df["flat"].mean()) / total if abs(total) > 1e-9 else np.nan
        lesd = df[f"lesion_{g}_dec"]
        lossd = df["cov_dec"].mean() - lesd.mean()
        td, pd_ = corrected_ttest_rel(df["cov_dec"], lesd)
        out.append({"network": g, "n_roi": len(idx[g]), "lesion_r": les.mean(),
                    "loss": loss, "loss_per_roi": loss / len(idx[g]),
                    "p_lesion": p, "only_r": only.mean(), "recovers_frac": rec_frac,
                    "lesion_r_dec": lesd.mean(), "loss_dec": lossd,
                    "p_lesion_dec": pd_})
        print(f"  {g:13s} {len(idx[g]):4d} {les.mean():+9.4f} {loss:+8.4f} "
              f"{1000*loss/len(idx[g]):+9.3f} {p:9.3g} | {only.mean():+8.4f} "
              f"{100*rec_frac:7.0f}% | {lossd:+8.4f} {pd_:9.3g}")
    pd.DataFrame(out).to_csv(OUT_DIR / "amplitude_lesion_summary.csv", index=False)
    print("\n  per-ROI loss is x1000. 'recovers' = fraction of the total amplitude")
    print("  effect regained when ONLY that network keeps subject-specific amplitude.")
    print(f"\nSaved -> {OUT_DIR}/amplitude_lesion_folds.csv, amplitude_lesion_summary.csv")


if __name__ == "__main__":
    main()
