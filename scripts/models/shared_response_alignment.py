#!/usr/bin/env python
"""
Shared response model alignment against a common basis of matched rank, which separates alignment from
dimensionality reduction.

The shared response is fitted on the training participants; test participants are projected onto the
frozen shared response by Procrustes.

Usage
-----
  python scripts/models/shared_response_alignment.py --repeats 3
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
KS       = [20, 50, 100]
SRM_ITER = 20
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


def embed_ts(TS):
    """TS: (N, d, T) torch tensor -> log-Euclidean tangent features."""
    N, d, T = TS.shape
    Xc = TS - TS.mean(-1, keepdim=True)
    C = (Xc @ Xc.transpose(1, 2)) / (T - 1)
    tr = torch.diagonal(C, dim1=-2, dim2=-1).sum(-1) / d
    eye = torch.eye(d, dtype=C.dtype, device=C.device)
    C = (1 - SHRINK) * C + SHRINK * tr.view(-1, 1, 1) * eye
    return upper(sym_funcm(C, torch.log)).cpu().numpy()


def procrustes(X, S):
    """W minimising ||X - W S||_F with W'W = I.  X:(N,R,T) S:(k,T) -> (N,R,k)."""
    M = X @ S.transpose(0, 1)
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    return U @ Vh


def fit_srm(X, k, n_iter=SRM_ITER, seed=0):
    """Deterministic SRM. X:(N,R,T) -> W:(N,R,k), S:(k,T)."""
    N, R, T = X.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(N, R, k, generator=g, dtype=torch.float64).to(X.device)
    W, _ = torch.linalg.qr(A)
    for _ in range(n_iter):
        S = (W.transpose(1, 2) @ X).mean(0)
        W = procrustes(X, S)
    S = (W.transpose(1, 2) @ X).mean(0)
    return W, S


def isc_mean(TS):
    """Mean leave-one-out ISC across features, for a (N, d, T) tensor."""
    Z = TS - TS.mean(-1, keepdim=True)
    Z = Z / Z.std(-1, keepdim=True).clamp_min(1e-8)
    N = Z.shape[0]
    Ssum = Z.sum(0)
    tot, cnt = 0.0, 0
    for i in range(N):
        tmpl = (Ssum - Z[i]) / (N - 1)
        a = Z[i] - Z[i].mean(-1, keepdim=True)
        b = tmpl - tmpl.mean(-1, keepdim=True)
        den = (a.pow(2).sum(-1) * b.pow(2).sum(-1)).sqrt().clamp_min(1e-12)
        tot += float(((a * b).sum(-1) / den).mean()); cnt += 1
    return tot / cnt


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

    # (N, R, T) with regions as rows -- SRM's orientation
    TS = torch.from_numpy(np.stack([raw[s].T for s in ids])).to(dev).double()
    N, R, T = TS.shape
    print(f"n = {N} (sealed discovery split), {R} regions, {T} volumes")
    X_raw = embed_ts(TS)
    isc_raw = isc_mean(TS)
    print(f"  baseline features {X_raw.shape}, raw-space mean ISC {isc_raw:+.4f}",
          flush=True)

    q = pd.qcut(y, 4, labels=False, duplicates="drop")
    strat = np.asarray(q) * 2 + sex
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats, random_state=SEED)

    rows, isc_rows = [], []
    for fold, (tr, te) in enumerate(rskf.split(np.zeros(N), strat), 1):
        rec = {"fold": fold, "raw": None}
        tr_t = torch.as_tensor(tr, device=dev)
        blocks = {"raw": X_raw}
        for k in KS:
            # ---- SRM: template from TRAINING subjects only -----------------
            W_tr, S = fit_srm(TS[tr_t], k, seed=SEED + fold)
            W = torch.empty(N, R, k, dtype=TS.dtype, device=dev)
            W[tr_t] = W_tr
            te_t = torch.as_tensor(te, device=dev)
            W[te_t] = procrustes(TS[te_t], S)          # frozen S, no labels
            shared = W.transpose(1, 2) @ TS            # (N, k, T)
            blocks[f"srm{k}"] = embed_ts(shared)

            # ---- control: one common basis for everyone, no alignment ------
            Xtr_cat = TS[tr_t].transpose(0, 1).reshape(R, -1)
            Uc, _, _ = torch.linalg.svd(
                Xtr_cat - Xtr_cat.mean(-1, keepdim=True), full_matrices=False)
            P = Uc[:, :k]                              # (R, k), training only
            gp = (P.transpose(0, 1).unsqueeze(0) @ TS)  # (N, k, T)
            blocks[f"gpca{k}"] = embed_ts(gp)

            if fold == 1:
                isc_rows.append({"k": k, "space": "srm", "isc": isc_mean(shared)})
                isc_rows.append({"k": k, "space": "gpca", "isc": isc_mean(gp)})

        lr = LinearRegression().fit(Zm[tr], y[tr])
        ytr_d, yte_d = y[tr] - lr.predict(Zm[tr]), y[te] - lr.predict(Zm[te])
        for nm, Xb in blocks.items():
            rec[nm] = float(pearsonr(y[te], ridge_pred(Xb[tr], y[tr], Xb[te]))[0])
            rec[nm + "_dec"] = float(pearsonr(yte_d,
                                              ridge_pred(Xb[tr], ytr_d, Xb[te]))[0])
        rows.append(rec)
        print(f"  fold {fold:2d}/{args.repeats*5}  raw={rec['raw']:+.3f}  " +
              "  ".join(f"srm{k}={rec[f'srm{k}']:+.3f}/gpca{k}={rec[f'gpca{k}']:+.3f}"
                        for k in KS), flush=True)

    df = pd.DataFrame(rows); df.to_csv(OUT_DIR / "srm_alignment_folds.csv", index=False)
    ir = pd.DataFrame(isc_rows)
    ir.to_csv(OUT_DIR / "srm_alignment_isc.csv", index=False)

    bar = "=" * 78
    print(f"\n{bar}\nFunctional alignment (SRM), {args.repeats}x5 CV (n={N})\n{bar}")
    for c in sorted([c for c in df.columns if c != "fold" and not c.endswith("_dec")],
                    key=lambda c: -df[c].mean()):
        print(f"  {c:12s} r = {df[c].mean():+.4f} +/- {df[c].std(ddof=1):.4f}"
              f"   deconf {df[c+'_dec'].mean():+.4f}")

    def cmp(a, b, label):
        d = df[a] - df[b]; t, p = corrected_ttest_rel(df[a], df[b])
        print(f"    {label:44s} {d.mean():+.4f}  t={t:+5.2f}  p={p:.3g}  "
              f"wins {int((d>0).sum())}/{len(df)}")

    print("\n  THE ALIGNMENT EFFECT (SRM vs a common basis at matched rank):")
    for k in KS:
        cmp(f"srm{k}", f"gpca{k}", f"srm{k} - gpca{k}")
    print("\n  vs the 215-region baseline (confounds alignment with rank reduction):")
    for k in KS:
        cmp(f"srm{k}", "raw", f"srm{k} - raw")
        cmp(f"gpca{k}", "raw", f"gpca{k} - raw")
    print(f"\n  INTER-SUBJECT CORRELATION  (raw space {isc_raw:+.4f}):")
    for _, r in ir.iterrows():
        print(f"    k={int(r['k']):3d}  {r['space']:5s}  {r['isc']:+.4f}"
              f"   ({r['isc']-isc_raw:+.4f} vs raw)")
    print(f"\nSaved -> {OUT_DIR}/srm_alignment_folds.csv, srm_alignment_isc.csv")


if __name__ == "__main__":
    main()
