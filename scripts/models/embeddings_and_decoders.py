#!/usr/bin/env python
"""
Sweep of covariance embeddings and of decoders on identical folds.

The embeddings are the affine-invariant one, which is the matrix logarithm after whitening by the
training-fold Frechet mean, the log-Euclidean and log-Cholesky embeddings, and matrix powers up to the
Euclidean embedding. The decoders are ridge, kernel ridge, elastic net, partial least squares and
support vector regression.

Usage
-----
  python scripts/models/embeddings_and_decoders.py --mode metrics  --repeats 2
  python scripts/models/embeddings_and_decoders.py --mode decoders --repeats 2
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.model_selection import KFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

from src.data.preprocessing import load_participants
from src.stats.cv_inference import corrected_ttest_rel

warnings.filterwarnings("ignore")

CFG_PATH = "config/pipeline.yaml"
TARGET   = "IST_intelligence_total"
ALPHAS   = np.logspace(-2, 6, 25)
SEED     = 42
SHRINK   = 0.55          # tuned value (nested CV picked 0.55-0.85; plateau 0.4-0.7)
OUT_DIR  = Path("outputs/honest")

POWERS = [0.25, 0.5, 0.75, 1.0]


# ---------------------------------------------------------------------------
# SPD utilities (float64; cuSOLVER float32 eigh does not converge here)
# ---------------------------------------------------------------------------

def sym_funcm(X, fn):
    X = 0.5 * (X + X.transpose(-1, -2)).double()
    w, V = torch.linalg.eigh(X)
    return V @ torch.diag_embed(fn(w.clamp_min(1e-10))) @ V.transpose(-1, -2)


def frechet_mean(S, n_iter=12, tol=1e-7):
    G = S.mean(0)
    for _ in range(n_iter):
        Gs  = sym_funcm(G.unsqueeze(0), torch.sqrt)[0]
        Gis = sym_funcm(G.unsqueeze(0), lambda w: w.rsqrt())[0]
        T = sym_funcm(Gis @ S @ Gis, torch.log).mean(0)
        if torch.linalg.norm(T) < tol:
            break
        G = Gs @ sym_funcm(T.unsqueeze(0), torch.exp)[0] @ Gs
        G = 0.5 * (G + G.t())
    return G


def vec_sym(L):
    """Symmetric matrices -> upper-triangle vector (off-diagonals * sqrt 2)."""
    n = L.shape[-1]
    iu = torch.triu_indices(n, n, offset=1, device=L.device)
    return torch.cat([torch.diagonal(L, dim1=-2, dim2=-1),
                      L[..., iu[0], iu[1]] * np.sqrt(2.0)], dim=-1).float()


def embed(name, C, tr_idx, dev):
    """Return (n_subjects, d) features. Only AIRM depends on the training fold."""
    if name == "airm":
        G = frechet_mean(C[torch.as_tensor(tr_idx, device=dev)])
        Gis = sym_funcm(G.unsqueeze(0), lambda w: w.rsqrt())[0]
        return vec_sym(sym_funcm(Gis @ C @ Gis, torch.log)).cpu().numpy()

    if name == "logeuclid":
        return vec_sym(sym_funcm(C, torch.log)).cpu().numpy()

    if name.startswith("power_"):
        a = float(name.split("_")[1])
        return vec_sym(sym_funcm(C, lambda w: w.pow(a))).cpu().numpy()

    if name == "logchol":
        # C = L L^T; embed as strict-lower(L) + log(diag(L))   (Lin, 2019)
        L = torch.linalg.cholesky(C)
        n = L.shape[-1]
        il = torch.tril_indices(n, n, offset=-1, device=L.device)
        return torch.cat([torch.diagonal(L, dim1=-2, dim2=-1).clamp_min(1e-10).log(),
                          L[..., il[0], il[1]]], dim=-1).float().cpu().numpy()

    raise ValueError(name)


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------

def fit_predict(name, Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)

    if name == "ridge":
        return RidgeCV(alphas=ALPHAS, cv=5).fit(A, ytr).predict(B)

    if name in ("krr_rbf", "krr_logeuc"):
        # median-heuristic bandwidth from the TRAIN fold only
        sub = A[np.random.default_rng(SEED).choice(len(A), min(300, len(A)), replace=False)]
        med = np.median(pdist(sub, "euclidean"))
        best, best_r, ytr_m = None, -np.inf, ytr.mean()
        for g in [0.25, 0.5, 1.0, 2.0]:
            for al in [0.1, 1.0, 10.0, 100.0]:
                inner = []
                for a, b in KFold(3, shuffle=True, random_state=SEED).split(A):
                    m = KernelRidge(kernel="rbf", alpha=al,
                                    gamma=1.0 / (2 * (g * med) ** 2))
                    m.fit(A[a], ytr[a] - ytr_m)
                    pv = m.predict(A[b]) + ytr_m
                    inner.append(pearsonr(ytr[b], pv)[0] if np.std(pv) > 1e-9 else -1)
                s = np.mean(inner)
                if s > best_r:
                    best_r, best = s, (al, 1.0 / (2 * (g * med) ** 2))
        m = KernelRidge(kernel="rbf", alpha=best[0], gamma=best[1])
        m.fit(A, ytr - ytr_m)
        return m.predict(B) + ytr_m

    if name == "pls":
        best, best_r = None, -np.inf
        for nc in [2, 5, 10, 20, 40]:
            inner = []
            for a, b in KFold(3, shuffle=True, random_state=SEED).split(A):
                p = PLSRegression(n_components=nc).fit(A[a], ytr[a])
                pv = p.predict(A[b]).ravel()
                inner.append(pearsonr(ytr[b], pv)[0] if np.std(pv) > 1e-9 else -1)
            if np.mean(inner) > best_r:
                best_r, best = np.mean(inner), nc
        return PLSRegression(n_components=best).fit(A, ytr).predict(B).ravel()

    if name == "enet":
        return ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], n_alphas=20, cv=3,
                            random_state=SEED, max_iter=5000).fit(A, ytr).predict(B)

    if name == "svr":
        return LinearSVR(C=1.0, epsilon=0.0, max_iter=5000,
                         random_state=SEED).fit(A, ytr).predict(B)

    raise ValueError(name)


# ---------------------------------------------------------------------------

def load_cov(cfg, dev):
    conn = Path(cfg["paths"]["connectivity_dir"])
    ctx, sub = {}, {}
    with h5py.File(conn / "fc_matrices.h5", "r") as f:
        for s in f["subjects"]:
            ctx[s] = f["subjects"][s]["time_series"][:]
    with h5py.File(conn / "subcortical_ts.h5", "r") as f:
        for s in f["subjects"]:
            sub[s] = f["subjects"][s]["time_series"][:]
    parts = load_participants(cfg["paths"]["bids_root"])

    ids, cov = [], []
    for s in sorted(set(ctx) & set(sub)):
        if ctx[s].shape[0] != sub[s].shape[0]:
            continue
        if s not in parts.index or pd.isna(parts.loc[s, TARGET]):
            continue
        x = np.hstack([ctx[s], sub[s]]).astype(np.float32)
        x = x - x.mean(0, keepdims=True)
        S = (x.T @ x) / (len(x) - 1)
        d = S.shape[0]
        cov.append((1 - SHRINK) * S + SHRINK * (np.trace(S) / d) * np.eye(d, dtype=np.float32))
        ids.append(s)
    y = np.array([float(parts.loc[s, TARGET]) for s in ids])
    sex = np.array([1 if str(parts.loc[s, "sex"]).lower().startswith("m") else 0
                    for s in ids])
    return ids, torch.from_numpy(np.stack(cov)).to(dev).double(), y, sex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["metrics", "decoders"], default="metrics")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--embedding", default="airm",
                    help="embedding to use in decoders mode")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(CFG_PATH))

    print(f"Loading covariances (shrinkage={SHRINK}) ...", flush=True)
    ids, C, y, sex = load_cov(cfg, dev)
    print(f"  {len(ids)} subjects, SPD({C.shape[-1]})")

    embeddings = (["airm", "logeuclid", "logchol"] + [f"power_{a}" for a in POWERS]
                  if args.mode == "metrics" else [args.embedding])
    decoders = (["ridge"] if args.mode == "metrics"
                else ["ridge", "krr_rbf", "krr_logeuc", "pls", "enet", "svr"])
    print(f"  mode={args.mode}  embeddings={embeddings}  decoders={decoders}")

    q = pd.qcut(y, 4, labels=False, duplicates="drop")
    strat = np.asarray(q) * 2 + sex
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats,
                                   random_state=SEED)

    rows = []
    for k, (tr, te) in enumerate(rskf.split(np.zeros(len(y)), strat), 1):
        t0 = time.time()
        rec = {"fold": k}
        for em in embeddings:
            X = embed(em, C, tr, dev)
            # krr_logeuc is krr_rbf applied to the log-Euclidean embedding: the
            # Euclidean distance there IS the log-Euclidean geodesic distance
            for dc in decoders:
                if dc == "krr_logeuc":
                    Xk = embed("logeuclid", C, tr, dev)
                    p = fit_predict("krr_rbf", Xk[tr], y[tr], Xk[te])
                else:
                    p = fit_predict(dc, X[tr], y[tr], X[te])
                key = em if args.mode == "metrics" else dc
                rec[key] = float(pearsonr(y[te], p)[0])
        rec["secs"] = time.time() - t0
        rows.append(rec)
        top = sorted([(v, kk) for kk, v in rec.items()
                      if kk not in ("fold", "secs")], reverse=True)[:2]
        print(f"  fold {k:2d}/{args.repeats*5}  " +
              "  ".join(f"{kk}={v:+.3f}" for v, kk in top) +
              f"  ({rec['secs']:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / f"riemann_{args.mode}_folds.csv", index=False)

    cols = [c for c in df.columns if c not in ("fold", "secs")]
    print(f"\n{'='*70}\n{args.mode}: {args.repeats}x5-fold CV (n={len(ids)})\n{'='*70}")
    for c in sorted(cols, key=lambda c: -df[c].mean()):
        print(f"  {c:14s} r = {df[c].mean():+.4f} +/- {df[c].std(ddof=1):.4f}")

    base = "airm" if args.mode == "metrics" else "ridge"
    if base in df:
        print(f"\n  paired tests vs {base}:")
        for c in cols:
            if c == base:
                continue
            t, p = corrected_ttest_rel(df[c], df[base])
            print(f"    {c:14s} delta = {(df[c]-df[base]).mean():+.4f}  "
                  f"t = {t:+5.2f}  p = {p:.3g}  wins {int((df[c]>df[base]).sum())}/{len(df)}")

    print(f"\nSaved -> {OUT_DIR}/riemann_{args.mode}_folds.csv")


if __name__ == "__main__":
    main()
