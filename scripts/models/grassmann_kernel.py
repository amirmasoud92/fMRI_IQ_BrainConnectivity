#!/usr/bin/env python
"""
Grassmann subspace kernels (geodesic and chordal) against tangent-space connectivity with ridge regression.

Two distances
-------------
  geodesic  d^2 = sum_l theta_l^2,  theta_l = arccos(sigma_l of U_i^T U_j)
            the metric the paper specifies; needs an SVD per subject pair
  chordal   d^2 = k - ||U_i^T U_j||_F^2 = sum_l sin^2(theta_l)
            a standard Grassmannian metric available in closed form from
            Frobenius norms alone -- far cheaper, included as a check that any
            result is not an artefact of one distance choice

Decoding: kernel ridge with K = exp(-gamma * d^2). Both k and gamma are chosen
by an inner CV on the training fold; the outer fold is never used for selection.
Note the kernel is built from ALL subjects' subspaces, but subspaces are computed
per subject independently and no label information enters them, so this is
transductive in features only -- not leakage.

Usage
-----
  python scripts/models/grassmann_kernel.py --repeats 3
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, RepeatedStratifiedKFold
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
KS       = [5, 10, 20, 40]
GAMMAS   = [0.25, 0.5, 1.0, 2.0]      # multipliers on the median heuristic
KRR_ALPH = [0.01, 0.1, 1.0, 10.0]
OUT_DIR  = Path("outputs/honest")
SERIES   = "legacy"          # set from --series in main()


def sym_funcm(X, fn):
    X = 0.5 * (X + X.transpose(-1, -2)).double()
    w, V = torch.linalg.eigh(X)
    return V @ torch.diag_embed(fn(w.clamp_min(1e-10))) @ V.transpose(-1, -2)


def top_eigvecs(C, k):
    """Top-k eigenvectors per subject -> (N, n, k), orthonormal columns."""
    w, V = torch.linalg.eigh(0.5 * (C + C.transpose(-1, -2)))
    return V[..., -k:].contiguous()          # eigh returns ascending eigenvalues


def grassmann_dists(U, chunk=64):
    """Pairwise squared geodesic and chordal Grassmannian distances."""
    N, n, k = U.shape
    Dg = torch.zeros(N, N, dtype=torch.float64, device=U.device)
    Dc = torch.zeros(N, N, dtype=torch.float64, device=U.device)
    Ut = U.transpose(-1, -2)
    for i in range(0, N, chunk):
        a = Ut[i:i + chunk]                                   # (b, k, n)
        M = torch.einsum("bkn,cnj->bckj", a, U)               # (b, N, k, k)
        s = torch.linalg.svdvals(M).clamp(-1.0, 1.0)          # (b, N, k)
        th = torch.arccos(s.clamp(0.0, 1.0))
        Dg[i:i + chunk] = (th ** 2).sum(-1)
        Dc[i:i + chunk] = (k - (s ** 2).sum(-1))
    Dg = 0.5 * (Dg + Dg.t())
    Dc = 0.5 * (Dc + Dc.t())
    return Dg.cpu().numpy(), Dc.cpu().numpy()


def krr_fit_predict(D, y, tr, te, gm, al):
    """Kernel ridge on a precomputed squared-distance matrix."""
    med = np.median(np.sqrt(D[np.ix_(tr, tr)][np.triu_indices(len(tr), 1)]))
    g = 1.0 / (2 * (gm * med) ** 2 + 1e-12)
    ym = y[tr].mean()
    K = np.exp(-g * D[np.ix_(tr, tr)]) + al * np.eye(len(tr))
    try:
        coef = np.linalg.solve(K, y[tr] - ym)
    except np.linalg.LinAlgError:
        return np.full(len(te), ym)
    return np.exp(-g * D[np.ix_(te, tr)]) @ coef + ym


def select_and_predict(Ds, y, tr, te):
    """Inner 3-fold selects k, gamma and alpha; then predict the outer fold."""
    best, best_s = None, -np.inf
    for kk, D in Ds.items():
        for gm in GAMMAS:
            for al in KRR_ALPH:
                sc = []
                for a, b in KFold(3, shuffle=True, random_state=SEED).split(tr):
                    p = krr_fit_predict(D, y, tr[a], tr[b], gm, al)
                    sc.append(pearsonr(y[tr[b]], p)[0] if np.std(p) > 1e-9 else -1.0)
                m = float(np.mean(sc))
                if m > best_s:
                    best_s, best = m, (kk, gm, al)
    kk, gm, al = best
    return krr_fit_predict(Ds[kk], y, tr, te, gm, al), best


def ridge_r(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    m = RidgeCV(alphas=ALPHAS, cv=5).fit(sc.transform(Xtr), ytr)
    return float(pearsonr(yte, m.predict(sc.transform(Xte)))[0])


def upper(L):
    n = L.shape[-1]
    iu = torch.triu_indices(n, n, offset=1, device=L.device)
    return torch.cat([torch.diagonal(L, dim1=-2, dim2=-1),
                      L[..., iu[0], iu[1]] * np.sqrt(2.0)], dim=-1).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
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
        cov.append((1 - SHRINK) * S + SHRINK * (np.trace(S) / d)
                   * np.eye(d, dtype=np.float32))
        ids.append(s)
    C = torch.from_numpy(np.stack(cov)).to(dev).double()
    y = np.array([float(parts.loc[s, TARGET]) for s in ids])
    sex = np.array([1 if str(parts.loc[s, "sex"]).lower().startswith("m") else 0
                    for s in ids])
    print(f"{len(ids)} subjects, SPD({C.shape[-1]})")

    print("Computing Grassmannian distances ...", flush=True)
    Dgeo, Dcho = {}, {}
    for k in KS:
        t0 = time.time()
        U = top_eigvecs(C, k)
        Dgeo[k], Dcho[k] = grassmann_dists(U)
        print(f"  k={k:3d}  geodesic range [{Dgeo[k].min():.2f},{Dgeo[k].max():.2f}]  "
              f"({time.time()-t0:.0f}s)", flush=True)

    q = pd.qcut(y, 4, labels=False, duplicates="drop")
    strat = np.asarray(q) * 2 + sex
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats,
                                   random_state=SEED)

    rows, picks = [], []
    for kf, (tr, te) in enumerate(rskf.split(np.zeros(len(y)), strat), 1):
        t0 = time.time()
        rec = {"fold": kf}
        G = None
        # baseline: log-Euclidean tangent + Ridge, same folds
        Xg = upper(sym_funcm(C, torch.log)).cpu().numpy()
        rec["tangent_ridge"] = ridge_r(Xg[tr], y[tr], Xg[te], y[te])
        for tag, Ds in [("grass_geo", Dgeo), ("grass_chordal", Dcho)]:
            p, best = select_and_predict(Ds, y, tr, te)
            rec[tag] = float(pearsonr(y[te], p)[0])
            if tag == "grass_geo":
                picks.append(best)
        rec["secs"] = time.time() - t0
        rows.append(rec)
        print(f"  fold {kf:2d}/{args.repeats*5}  tangent={rec['tangent_ridge']:+.3f} "
              f"geo={rec['grass_geo']:+.3f} chordal={rec['grass_chordal']:+.3f} "
              f"(k={picks[-1][0]}, {rec['secs']:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "grassmann_folds.csv", index=False)

    cols = [c for c in df.columns if c not in ("fold", "secs")]
    print(f"\n{'='*72}\nGrassmannian, {args.repeats}x5-fold CV (n={len(ids)})\n{'='*72}")
    for c in sorted(cols, key=lambda c: -df[c].mean()):
        print(f"  {c:16s} r = {df[c].mean():+.4f} +/- {df[c].std(ddof=1):.4f}")
    print(f"\n  paired tests vs tangent_ridge:")
    for c in cols:
        if c == "tangent_ridge":
            continue
        t, p = corrected_ttest_rel(df[c], df["tangent_ridge"])
        print(f"    {c:16s} delta = {(df[c]-df['tangent_ridge']).mean():+.4f}  "
              f"t = {t:+5.2f}  p = {p:.3g}  "
              f"wins {int((df[c]>df['tangent_ridge']).sum())}/{len(df)}")
    import collections
    print(f"\n  inner-CV picks (k, gamma_mult, alpha): "
          f"{collections.Counter(picks).most_common(4)}")
    print(f"\nSaved -> {OUT_DIR}/grassmann_folds.csv")


if __name__ == "__main__":
    main()
