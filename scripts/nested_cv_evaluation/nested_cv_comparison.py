#!/usr/bin/env python
"""
Repeated nested cross-validation of the connectivity feature sets and of the autoencoder embeddings,
replacing the single split on which the model was first evaluated.

Design
------
Outer loop : N_REPEATS x 5-fold stratified CV (stratified on IST quartile x sex)
Inner loop : RidgeCV(cv=5) alpha selection, fit on the outer-train fold ONLY
Features   : recomputed inside every outer fold wherever they depend on the
             training data (tangent-space reference mean, PCA basis, scaler).

Feature sets
------------
  corr        Pearson correlation upper triangle (19 900)   -- what the paper used
  tangent     Riemannian tangent space at the TRAIN-fold geometric mean (19 900)
  pca64       64 PCs of the correlation matrix (basis fit on train fold)
  vae64_pre   VAE-64 fb001 pre-trained embeddings   [CONTAMINATED - see note]
  vae64_sft   VAE-64 fb001 + SFT embeddings         [CONTAMINATED - see note]

Contamination note
------------------
The VAE encoder was trained with `pheno_weight=0.05` on 16 phenotypes, the
first of which is IST_intelligence_total, using the labels of the 615
original-train subjects; the SFT stage additionally trained on the 744
original-trainval labels.  Re-splitting the *resulting embeddings* with CV
cannot undo that.  Rows flagged CONTAMINATED are therefore upper bounds, not
honest estimates, and are reported only for reference.  The clean comparison
for those rows is `--mode holdout`, which fits on the original trainval and
scores the original held-out test set the encoder never saw labels for.

Usage
-----
  python scripts/nested_cv_evaluation/nested_cv_comparison.py --mode cv       --repeats 10
  python scripts/nested_cv_evaluation/nested_cv_comparison.py --mode holdout  --n-boot 10000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.data.preprocessing import load_participants

warnings.filterwarnings("ignore", category=RuntimeWarning)

CFG_PATH = "config/pipeline.yaml"
TARGET   = "IST_intelligence_total"
ALPHAS   = np.logspace(-2, 6, 25)
SEED     = 42

OUT_DIR  = Path("outputs/honest")

# Embedding sets that carry label exposure from encoder training.
CONTAMINATED = {"vae64_pre", "vae64_sft"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cfg(path=CFG_PATH):
    with open(path) as f:
        return yaml.safe_load(f)


def load_all(cfg):
    """Return timeseries dict, correlation dict, subject list, y, sex."""
    conn = Path(cfg["paths"]["connectivity_dir"])
    ts_by_id, fc_by_id = {}, {}
    with h5py.File(conn / "fc_matrices.h5", "r") as f:
        for s in sorted(f["subjects"].keys()):
            ts_by_id[s] = f["subjects"][s]["time_series"][:]
            fc_by_id[s] = f["subjects"][s]["fc_matrix"][:]

    participants = load_participants(cfg["paths"]["bids_root"])

    ids, y, sex = [], [], []
    for s in sorted(ts_by_id.keys()):
        if s not in participants.index:
            continue
        v = participants.loc[s, TARGET]
        if pd.isna(v):
            continue
        ids.append(s)
        y.append(float(v))
        sx = str(participants.loc[s, "sex"]).lower()
        sex.append(1 if sx.startswith("m") else 0)

    return ts_by_id, fc_by_id, ids, np.asarray(y, float), np.asarray(sex, int)


def load_embeddings(name):
    """Load a saved .npz embedding set -> dict subject_id -> vector."""
    paths = {
        "vae64_pre": "outputs/v4_64_fb001/latent_embeddings.npz",
        "vae64_sft": "outputs/v4_64_fb001/vae_sft_fb001_latent_embeddings.npz",
    }
    p = Path(paths[name])
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    zkey = "z" if "z" in d else d.files[0]
    idkey = "subject_ids" if "subject_ids" in d else d.files[-1]
    return {str(s): z for s, z in zip(d[idkey], d[zkey])}


# ---------------------------------------------------------------------------
# Riemannian tangent space
# ---------------------------------------------------------------------------

def _shrunk_cov(ts, shrink=0.1):
    """Ledoit-Wolf-style shrunk covariance -> guaranteed SPD."""
    ts = ts - ts.mean(0, keepdims=True)
    C = np.cov(ts, rowvar=False)
    n = C.shape[0]
    return (1.0 - shrink) * C + shrink * np.trace(C) / n * np.eye(n)


def _sqrtm_inv_sqrtm(C):
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, 1e-10)
    return (V * np.sqrt(w)) @ V.T, (V * (1.0 / np.sqrt(w))) @ V.T


def _batch_funcm(S, fn):
    """Apply a scalar function to the eigenvalues of a STACK of SPD matrices.

    np.linalg.eigh is batched, so this replaces a Python loop over subjects
    with a single LAPACK call -- roughly two orders of magnitude faster.
    """
    w, V = np.linalg.eigh(S)                       # (B, n), (B, n, n)
    w = fn(np.maximum(w, 1e-10))
    return (V * w[:, None, :]) @ np.swapaxes(V, -1, -2)


def geometric_mean(covs, n_iter=15, tol=1e-7):
    """Frechet mean of SPD matrices under the affine-invariant metric."""
    S = np.asarray(covs)                           # (B, n, n)
    G = S.mean(axis=0)
    for _ in range(n_iter):
        G_sqrt, G_isqrt = _sqrtm_inv_sqrtm(G)
        T = _batch_funcm(G_isqrt @ S @ G_isqrt, np.log).mean(axis=0)
        if np.linalg.norm(T) < tol:
            break
        w, V = np.linalg.eigh(T)
        G = G_sqrt @ ((V * np.exp(w)) @ V.T) @ G_sqrt
    return G


def tangent_vectorize(covs, G):
    """Project SPD matrices to the tangent space at G, return upper triangles."""
    _, G_isqrt = _sqrtm_inv_sqrtm(G)
    S = np.asarray(covs)
    L = _batch_funcm(G_isqrt @ S @ G_isqrt, np.log)
    iu = np.triu_indices(G.shape[0], k=1)
    return (L[:, iu[0], iu[1]] * np.sqrt(2.0)).astype(np.float32)


# ---------------------------------------------------------------------------
# Feature builders (fit on train fold, transform both)
# ---------------------------------------------------------------------------

def build_features(name, tr_idx, te_idx, cache):
    """Return (X_train, X_test) with every train-dependent step fit on tr_idx."""
    if name == "corr":
        X = cache["corr"]
        return X[tr_idx], X[te_idx]

    if name == "tangent":
        covs = cache["covs"]
        G = geometric_mean([covs[i] for i in tr_idx])       # TRAIN-fold reference
        Xtr = tangent_vectorize([covs[i] for i in tr_idx], G)
        Xte = tangent_vectorize([covs[i] for i in te_idx], G)
        return Xtr, Xte

    if name == "pca64":
        X = cache["corr"]
        sc = StandardScaler().fit(X[tr_idx])
        p = PCA(n_components=64, random_state=SEED).fit(sc.transform(X[tr_idx]))
        return p.transform(sc.transform(X[tr_idx])), p.transform(sc.transform(X[te_idx]))

    X = cache[name]                                          # precomputed embeddings
    return X[tr_idx], X[te_idx]


def ridge_predict(Xtr, ytr, Xte):
    """Standardise on train, RidgeCV alpha on train, predict test."""
    sc = StandardScaler().fit(Xtr)
    m = RidgeCV(alphas=ALPHAS, cv=5).fit(sc.transform(Xtr), ytr)
    return m.predict(sc.transform(Xte)), float(m.alpha_)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def strat_labels(y, sex):
    """Stratify on IST quartile x sex so folds match on target AND sex."""
    q = pd.qcut(y, 4, labels=False, duplicates="drop")
    return np.asarray(q) * 2 + sex


def run_cv(cache, feats, y, sex, repeats, out_csv):
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=SEED)
    strat = strat_labels(y, sex)
    rows = []

    for name in feats:
        if name not in cache and name not in ("tangent", "pca64"):
            print(f"  [skip] {name}: not available")
            continue
        t0 = time.time()
        fold_r, preds_all, true_all, fold_ids = [], [], [], []

        for k, (tr, te) in enumerate(rskf.split(np.zeros(len(y)), strat)):
            Xtr, Xte = build_features(name, tr, te, cache)
            pred, alpha = ridge_predict(Xtr, y[tr], Xte)
            r = pearsonr(y[te], pred)[0]
            fold_r.append(r)
            preds_all.append(pred)
            true_all.append(y[te])
            fold_ids.append(np.full(len(te), k))

        fold_r = np.asarray(fold_r)
        # Pooled r across every out-of-fold prediction within each repeat
        rep_r = []
        n_folds = 5
        for rep in range(repeats):
            sl = slice(rep * n_folds, (rep + 1) * n_folds)
            p = np.concatenate(preds_all[sl])
            t = np.concatenate(true_all[sl])
            rep_r.append(pearsonr(t, p)[0])
        rep_r = np.asarray(rep_r)

        rows.append({
            "features":      name,
            "contaminated":  name in CONTAMINATED,
            "n_dim":         cache[name].shape[1] if name in cache else (19900 if name == "tangent" else 64),
            "fold_r_mean":   fold_r.mean(),
            "fold_r_sd":     fold_r.std(ddof=1),
            "repeat_r_mean": rep_r.mean(),
            "repeat_r_sd":   rep_r.std(ddof=1),
            "repeat_r_lo":   np.percentile(rep_r, 2.5),
            "repeat_r_hi":   np.percentile(rep_r, 97.5),
            "n_folds":       len(fold_r),
            "secs":          time.time() - t0,
        })
        flag = "  [CONTAMINATED]" if name in CONTAMINATED else ""
        print(f"  {name:12s} r = {rep_r.mean():+.4f} +/- {rep_r.std(ddof=1):.4f}   "
              f"(fold SD {fold_r.std(ddof=1):.3f}, {time.time()-t0:.0f}s){flag}")

    df = pd.DataFrame(rows).sort_values("repeat_r_mean", ascending=False)
    df.to_csv(out_csv, index=False)
    return df


def run_holdout(cache, feats, ids, y, sex, cfg, n_boot, out_csv):
    """Fit on the ORIGINAL trainval, score the ORIGINAL held-out test set."""
    md = Path(cfg["paths"]["models_dir"])
    split = json.load(open(md / "split_info.json"))
    pos = {s: i for i, s in enumerate(ids)}
    tr = np.array([pos[s] for s in split["train"] + split["val"] if s in pos])
    te = np.array([pos[s] for s in split["test"] if s in pos])
    print(f"  trainval={len(tr)}  test={len(te)}")

    rng = np.random.default_rng(SEED)
    boot_idx = rng.integers(0, len(te), size=(n_boot, len(te)))
    rows, preds = [], {}

    for name in feats:
        if name not in cache and name not in ("tangent", "pca64"):
            continue
        Xtr, Xte = build_features(name, tr, te, cache)
        pred, alpha = ridge_predict(Xtr, y[tr], Xte)
        preds[name] = pred
        r = pearsonr(y[te], pred)[0]
        br = np.array([pearsonr(y[te][b], pred[b])[0] for b in boot_idx])
        # Permutation test on the test labels
        perm = np.array([pearsonr(rng.permutation(y[te]), pred)[0] for _ in range(5000)])
        rows.append({
            "features": name, "contaminated": name in CONTAMINATED,
            "r": r, "ci_lo": np.percentile(br, 2.5), "ci_hi": np.percentile(br, 97.5),
            "boot_sd": br.std(ddof=1), "perm_p": float((np.abs(perm) >= abs(r)).mean()),
            "alpha": alpha, "n_test": len(te),
        })
        print(f"  {name:12s} r = {r:+.4f}  95% CI [{np.percentile(br,2.5):+.3f}, "
              f"{np.percentile(br,97.5):+.3f}]  perm p = {rows[-1]['perm_p']:.4f}")

    df = pd.DataFrame(rows).sort_values("r", ascending=False)
    df.to_csv(out_csv, index=False)
    np.savez(OUT_DIR / "holdout_predictions.npz",
             y_true=y[te], **{k: v for k, v in preds.items()})
    return df, preds, y[te]


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cv", "holdout", "both"], default="both")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--features", default="corr,tangent,pca64,vae64_pre,vae64_sft")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg()

    print("Loading data ...", flush=True)
    ts_by_id, fc_by_id, ids, y, sex = load_all(cfg)
    print(f"  {len(ids)} subjects with {TARGET}")

    iu = np.triu_indices(200, k=1)
    cache = {"corr": np.stack([fc_by_id[s][iu] for s in ids]).astype(np.float32)}

    print("Computing shrunk covariances for tangent space ...", flush=True)
    t0 = time.time()
    cache["covs"] = [_shrunk_cov(ts_by_id[s]) for s in ids]
    print(f"  done in {time.time()-t0:.0f}s")

    for nm in ("vae64_pre", "vae64_sft"):
        emb = load_embeddings(nm)
        if emb is None:
            print(f"  [warn] {nm} embeddings not found")
            continue
        missing = [s for s in ids if s not in emb]
        if missing:
            print(f"  [warn] {nm}: {len(missing)} subjects missing -> skipped")
            continue
        cache[nm] = np.stack([emb[s] for s in ids]).astype(np.float32)

    feats = [f for f in args.features.split(",") if f]

    if args.mode in ("cv", "both"):
        print(f"\n=== Repeated nested CV  ({args.repeats} x 5-fold = "
              f"{args.repeats*5} outer folds) ===")
        run_cv(cache, feats, y, sex, args.repeats, OUT_DIR / "honest_cv_results.csv")

    if args.mode in ("holdout", "both"):
        print(f"\n=== Original held-out test set ({args.n_boot} bootstrap) ===")
        run_holdout(cache, feats, ids, y, sex, cfg, args.n_boot,
                    OUT_DIR / "honest_holdout_results.csv")

    print(f"\nResults -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
