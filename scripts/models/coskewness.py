#!/usr/bin/env python
"""
Third-order co-skewness features from a higher-order singular value decomposition, alone and stacked with
connectivity.

Arms
----
  fc            log-Euclidean tangent of the shrunk covariance (project baseline)
  coskew_sub    covariance computed INSIDE the co-skewness subspace. This is a
                second-order feature living in a third-order-derived basis, so it
                tests the subspace, not the skewness.
  coskew_core   the Tucker CORE TENSOR itself: third standardised moments of the
                projected signals. These are genuine third-order features and are
                what the published claim actually rests on.
  stack_*       out-of-fold stacks, testing INCREMENT over FC rather than
                replacement.

third-order quantity ever entered a model and it could not have tested the

Efficient computation
---------------------
The co-skewness tensor S_ijk = (1/T) sum_t z_i z_j z_k has N^3 entries (~10 M for
N=215), impractical to form per subject. The mode-1 unfolding's left singular
vectors -- the HOSVD factor matrix -- are the eigenvectors of

    S_(1) S_(1)^T = (1/T^2) Z G Z^T ,   G = (Z^T Z) elementwise-squared

because <z(t) (x) z(t), z(s) (x) z(s)> = (z(t).z(s))^2. G is only T x T, so this
costs one T x T Gram and one N x N eigendecomposition per subject.

Given the group subspace U (TRAINING subjects only), the core tensor is formed
directly in the r-dimensional space, never the N-dimensional one, and only its
C(r+2,3) unique entries are kept, each weighted by sqrt(multiplicity) so the
vector's Euclidean norm equals the tensor's Frobenius norm -- the third-order
analogue of the sqrt(2) applied to off-diagonal covariance entries.

The second-order arm needs no timeseries pass at all: Cov(U^T x) = U^T Cov(x) U
exactly, so projecting the precomputed covariances is equivalent to projecting
the signals and recomputing, at a small fraction of the cost.

Usage
-----
  python scripts/models/coskewness.py --repeats 5
"""
from __future__ import annotations

import argparse, sys, time, warnings
from itertools import combinations_with_replacement
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.data.preprocessing import load_participants
from src.data.series import load_series
from src.stats.cv_inference import corrected_ttest_rel

warnings.filterwarnings("ignore")

CFG_PATH   = "config/pipeline.yaml"
TARGET     = "IST_intelligence_total"
ALPHAS     = np.logspace(-2, 6, 25)
SEED       = 42
SHRINK     = 0.55
RANKS_SUB  = [30, 60, 100]     # second-order arm: covariance in the subspace
RANKS_CORE = [15, 25, 40]      # third-order arm: 680 / 2925 / 11480 unique feats
OUT_DIR    = Path("outputs/honest")
SERIES     = "legacy"          # set from --series in main()
FS_DIR     = Path("derivatives/fs_stats")
VOL_COL    = "EstimatedTotalIntraCranialVol"


def sym_funcm(X, fn):
    X = 0.5 * (X + X.transpose(-1, -2)).double()
    w, V = torch.linalg.eigh(X)
    return V @ torch.diag_embed(fn(w.clamp_min(1e-10))) @ V.transpose(-1, -2)


def upper(L):
    n = L.shape[-1]
    iu = torch.triu_indices(n, n, offset=1, device=L.device)
    return torch.cat([torch.diagonal(L, dim1=-2, dim2=-1),
                      L[..., iu[0], iu[1]] * np.sqrt(2.0)], dim=-1).float()


def shrink_batch(S, shrink=SHRINK):
    """Shrinkage toward a scaled identity, batched over subjects."""
    d = S.shape[-1]
    tr = torch.diagonal(S, dim1=-2, dim2=-1).sum(-1) / d
    eye = torch.eye(d, dtype=S.dtype, device=S.device)
    return (1 - shrink) * S + shrink * tr[:, None, None] * eye


def coskew_gram(Z):
    """S_(1) S_(1)^T for one subject, without forming the N^3 tensor.

    Z : (N, T) z-scored timeseries.  Returns (N, N).
    """
    T = Z.shape[1]
    G = (Z.t() @ Z) ** 2                 # (T, T), elementwise square of the Gram
    return (Z @ G @ Z.t()) / (T ** 2)


def core_index(r):
    """Unique (a<=b<=c) triples and sqrt(multiplicity) weights for rank r."""
    tri = np.array(list(combinations_with_replacement(range(r), 3)), dtype=np.int64)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    mult = np.where((a == b) & (b == c), 1.0,
                    np.where((a == b) | (b == c) | (a == c), 3.0, 6.0))
    return tri, np.sqrt(mult).astype(np.float32)


def core_coskew(Y, tri, w):
    """Unique entries of the third standardised moment tensor of Y (T, r)."""
    Y = (Y - Y.mean(0, keepdim=True)) / Y.std(0, keepdim=True).clamp_min(1e-8)
    T = Y.shape[0]
    a, b, c = tri
    return ((Y[:, a] * Y[:, b] * Y[:, c]).sum(0) / T) * w


def ridge_pred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    return RidgeCV(alphas=ALPHAS, cv=5).fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))


def oof(X, y, tr, n_inner=5):
    o = np.zeros(len(tr))
    for a, b in KFold(n_inner, shuffle=True, random_state=SEED).split(tr):
        o[b] = ridge_pred(X[tr[a]], y[tr[a]], X[tr[b]])
    return o


def motion_stats(sub_id, fmriprep, task):
    """mean FD, max FD, spike fraction -- what third moments are exposed to."""
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


def zscore_time(x, dev):
    t = torch.from_numpy(x).to(dev).float()
    return (t - t.mean(0, keepdim=True)) / t.std(0, keepdim=True).clamp_min(1e-8)


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

    ctx, sub = load_series(conn, SERIES)
    frames = []
    for f in sorted(FS_DIR.glob("data-*.tsv")):
        d = pd.read_csv(f, sep="\t"); d = d.rename(columns={d.columns[0]: "sid"})
        d["sid"] = d["sid"].astype(str); d = d.set_index("sid")
        d = d.loc[:, ~d.columns.duplicated()]
        d.columns = [f"{f.stem}::{c}" for c in d.columns]; frames.append(d)
    M = pd.concat(frames, axis=1)
    M = M.loc[:, ~M.columns.duplicated()].apply(pd.to_numeric, errors="coerce")
    M = M.dropna(axis=1, how="all"); M = M.loc[:, M.std(skipna=True) > 0]
    vol_col = [c for c in M.columns if c.endswith("::" + VOL_COL)][0]
    parts = load_participants(cfg["paths"]["bids_root"])

    ids, ts = [], []
    for s in sorted(set(ctx) & set(sub) & set(M.index)):
        if ctx[s].shape[0] != sub[s].shape[0]:
            continue
        if s not in parts.index or pd.isna(parts.loc[s, TARGET]) or pd.isna(M.loc[s, vol_col]):
            continue
        ids.append(s); ts.append(np.hstack([ctx[s], sub[s]]).astype(np.float32))
    y = np.array([float(parts.loc[s, TARGET]) for s in ids])
    sex = np.array([1 if str(parts.loc[s, "sex"]).lower().startswith("m") else 0 for s in ids])
    vol = M.loc[ids, vol_col].values.astype(float)
    mot = np.array([motion_stats(s, cfg["paths"]["fmriprep_dir"], cfg["dataset"]["task"])
                    for s in ids])
    for j in range(mot.shape[1]):
        mot[:, j] = np.where(np.isfinite(mot[:, j]), mot[:, j], np.nanmedian(mot[:, j]))
    Zc  = np.column_stack([sex, vol, vol ** 2])              # project-standard control
    Zcm = np.column_stack([sex, vol, vol ** 2, mot])         # + motion (3 terms)
    n_roi = ts[0].shape[1]
    print(f"{len(ids)} subjects, {n_roi} regions, mean FD {mot[:, 0].mean():.3f}")

    # per-subject GPU timeseries, z-scored over time, reused by every fold
    Zg = [zscore_time(x, dev) for x in ts]

    print("Computing co-skewness grams and covariances ...", flush=True)
    t0 = time.time()
    Gs = torch.empty(len(ids), n_roi, n_roi, device=dev, dtype=torch.float64)
    Sraw = torch.empty(len(ids), n_roi, n_roi, device=dev, dtype=torch.float64)
    for i, (Z, x) in enumerate(zip(Zg, ts)):
        Gs[i] = coskew_gram(Z.double().t())
        # The covariance is built from the series AS EXTRACTED, not from the
        # z-scored copy. Co-skewness is a standardised moment and needs Z, but
        # z-scoring before the covariance turns it into a correlation matrix and
        # discards regional amplitude -- worth ~+0.013 on PSC data. Using the
        # extracted series makes this `fc` arm identical to the FC baseline of
        # every other script. On legacy series (already z-scored by the masker)
        # the two differ only by scrubbing-induced variance.
        xd = torch.from_numpy(x).to(dev).double()
        xc = xd - xd.mean(0, keepdim=True)
        Sraw[i] = (xc.t() @ xc) / max(len(xc) - 1, 1)
    print(f"  done in {time.time()-t0:.0f}s", flush=True)

    X_fc = upper(sym_funcm(shrink_batch(Sraw), torch.log)).cpu().numpy()
    print(f"  fc block: {X_fc.shape[1]} features")

    CORE_IDX = {}
    for r in RANKS_CORE:
        tri, w = core_index(r)
        CORE_IDX[r] = (torch.from_numpy(tri.T).to(dev), torch.from_numpy(w).to(dev))
        print(f"  core rank {r:3d}: {tri.shape[0]:6d} unique third-moment features")

    q = pd.qcut(y, 4, labels=False, duplicates="drop")
    strat = np.asarray(q) * 2 + sex
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats, random_state=SEED)

    def sel_rank(build, ranks, tr):
        """Pick a rank by inner 3-fold CV on the training fold only."""
        best, bX, br = -np.inf, None, ranks[0]
        for r in ranks:
            Xr = build(r)
            sc = []
            for a, b in KFold(3, shuffle=True, random_state=SEED).split(tr):
                p = ridge_pred(Xr[tr[a]], y[tr[a]], Xr[tr[b]])
                sc.append(pearsonr(y[tr[b]], p)[0] if np.std(p) > 1e-9 else -1.0)
            m = float(np.mean(sc))
            if m > best:
                best, bX, br = m, Xr, r
        return bX, br

    rows = []
    for k, (tr, te) in enumerate(rskf.split(np.zeros(len(y)), strat), 1):
        t0 = time.time(); rec = {"fold": k}

        # group co-skewness subspace from the TRAINING fold only (HOSVD factor)
        Gbar = Gs[tr].mean(0)
        Gbar = 0.5 * (Gbar + Gbar.t())
        evecs = torch.linalg.eigh(Gbar)[1]

        def build_sub(r):
            U = evecs[:, -r:]
            Sr = U.t() @ Sraw @ U                  # exact: Cov(U^T x) = U^T Cov(x) U
            return upper(sym_funcm(shrink_batch(Sr), torch.log)).cpu().numpy()

        def build_core(r):
            U = evecs[:, -r:].float()
            tri, w = CORE_IDX[r]
            return torch.stack([core_coskew(Z @ U, tri, w) for Z in Zg]).cpu().numpy()

        X_sub, r_sub = sel_rank(build_sub, RANKS_SUB, tr)
        X_core, r_core = sel_rank(build_core, RANKS_CORE, tr)
        rec["rank_sub"], rec["rank_core"] = r_sub, r_core

        blocks = {"fc": X_fc, "coskew_sub": X_sub, "coskew_core": X_core}
        preds = {}
        # Out-of-fold predictions depend only on (block, y, tr), which are fixed
        # within an outer fold, and oof() is deterministic (seeded KFold, RidgeCV).
        # The three stacking combinations share blocks, so the 23,220-feature FC
        # block was being refit 3x and each co-skewness block 2x per fold.
        # Computing each once is bitwise identical (tests/test_oof_determinism.py).
        oof_cache = {}

        def oof_block(n):
            if n not in oof_cache:
                oof_cache[n] = oof(blocks[n], y, tr)
            return oof_cache[n]
        for nm, Xb in blocks.items():
            p = ridge_pred(Xb[tr], y[tr], Xb[te]); preds[nm] = p
            rec[nm] = float(pearsonr(y[te], p)[0])
            for tag, Z_ in [("dec", Zc), ("decm", Zcm)]:
                lr = LinearRegression().fit(Z_[tr], y[tr])
                rec[f"{nm}_{tag}"] = float(pearsonr(
                    y[te] - lr.predict(Z_[te]),
                    ridge_pred(Xb[tr], y[tr] - lr.predict(Z_[tr]), Xb[te]))[0])
        for combo in [("fc", "coskew_core"), ("fc", "coskew_sub"),
                      ("fc", "coskew_sub", "coskew_core")]:
            Zt_ = np.column_stack([oof_block(n) for n in combo])
            Ze_ = np.column_stack([preds[n] for n in combo])
            rec["stack_" + "+".join(combo)] = float(
                pearsonr(y[te], ridge_pred(Zt_, y[tr], Ze_))[0])
        rec["secs"] = time.time() - t0
        rows.append(rec)
        print(f"  fold {k:2d}/{args.repeats*5}  fc={rec['fc']:+.3f} "
              f"core={rec['coskew_core']:+.3f}(r{r_core}) "
              f"sub={rec['coskew_sub']:+.3f}(r{r_sub}) "
              f"fc+core={rec['stack_fc+coskew_core']:+.3f}  ({rec['secs']:.0f}s)", flush=True)

    df = pd.DataFrame(rows); df.to_csv(OUT_DIR / "coskewness_folds.csv", index=False)
    cols = [c for c in df.columns if c not in ("fold", "secs", "rank_sub", "rank_core")]
    main_cols = [c for c in cols if not c.endswith(("_dec", "_decm"))]
    print(f"\n{'='*74}\nCo-skewness, {args.repeats}x5-fold CV (n={len(ids)})\n{'='*74}")
    for c in sorted(main_cols, key=lambda c: -df[c].mean()):
        print(f"  {c:30s} r = {df[c].mean():+.4f} +/- {df[c].std(ddof=1):.4f}")

    print(f"\n  PAIRED TESTS vs fc (same folds):")
    for c in main_cols:
        if c == "fc":
            continue
        t, p = corrected_ttest_rel(df[c], df["fc"])
        print(f"    {c:30s} delta = {(df[c]-df['fc']).mean():+.4f}  t = {t:+5.2f}  "
              f"p = {p:.3g}  wins {int((df[c] > df['fc']).sum())}/{len(df)}")

    print(f"\n  DECONFOUNDING  (dec = sex + volume + volume^2;  decm = + motion):")
    print(f"    {'block':14s} {'raw':>9s} {'dec':>9s} {'ret%':>6s} {'decm':>9s} {'ret%':>6s}")
    for nm in ["fc", "coskew_sub", "coskew_core"]:
        raw_ = df[nm].mean(); d1 = df[f"{nm}_dec"].mean(); d2 = df[f"{nm}_decm"].mean()
        print(f"    {nm:14s} {raw_:+9.4f} {d1:+9.4f} {100*d1/raw_:5.0f}% "
              f"{d2:+9.4f} {100*d2/raw_:5.0f}%")

    print(f"\n  ranks picked -- sub: {df['rank_sub'].value_counts().to_dict()}   "
          f"core: {df['rank_core'].value_counts().to_dict()}")
    print(f"  Reference: Marraffini et al. report Schaefer-400 raw FC 0.346 -> "
          f"co-skewness 0.406 (+0.060), PC1 of IST subscales, no size/sex/motion control.")
    print(f"\nSaved -> {OUT_DIR}/coskewness_folds.csv")


if __name__ == "__main__":
    main()
