"""Reconstruction and prediction metrics for the graph autoencoder."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from joblib import Parallel, delayed
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 1. Brain fingerprinting
# ---------------------------------------------------------------------------

def brain_fingerprinting_accuracy(
    z_db: np.ndarray,       # (N, D) "database" embeddings (e.g. first half)
    z_query: np.ndarray,    # (N, D) "query" embeddings   (e.g. second half)
    metric: str = "cosine",
) -> Dict[str, float]:
    """Compute identification accuracy based on latent embedding similarity."""
    N = len(z_db)
    dist = cdist(z_query, z_db, metric=metric)  # (N, N)

    # Rank of the correct match (0-indexed, lower = better)
    ranks = np.array([
        np.where(np.argsort(dist[i]) == i)[0][0] for i in range(N)
    ])

    acc_top1 = float((ranks == 0).mean())
    acc_top5 = float((ranks < 5).mean())
    mean_rank = float(ranks.mean() + 1)  # 1-indexed

    return {
        "fingerprint_acc_top1": acc_top1,
        "fingerprint_acc_top5": acc_top5,
        "fingerprint_mean_rank": mean_rank,
        "n_subjects": N,
    }


# ---------------------------------------------------------------------------
# 2. Reconstruction quality
# ---------------------------------------------------------------------------

def reconstruction_quality(
    fc_true: np.ndarray,    # (N, n_upper_tri)
    fc_recon: np.ndarray,   # (N, n_upper_tri)
) -> Dict[str, float]:
    """Per-subject Pearson correlation and MSE between true and reconstructed FC."""
    N = len(fc_true)
    pearson_rs = np.array([
        pearsonr(fc_true[i], fc_recon[i])[0] for i in range(N)
    ])
    mses = np.mean((fc_true - fc_recon) ** 2, axis=1)

    return {
        "mean_pearson_r":   float(pearson_rs.mean()),
        "median_pearson_r": float(np.median(pearson_rs)),
        "std_pearson_r":    float(pearson_rs.std()),
        "mean_mse":         float(mses.mean()),
        "median_mse":       float(np.median(mses)),
        "pearson_r_array":  pearson_rs.tolist(),
        "mse_array":        mses.tolist(),
    }


# ---------------------------------------------------------------------------
# 3. Phenotype prediction
# ---------------------------------------------------------------------------

def phenotype_prediction(
    z: np.ndarray,               # (N, D) latent embeddings
    phenotypes: np.ndarray,      # (N, P) phenotype matrix
    pheno_names: List[str],      # length P
    n_folds: int = 5,
    seed: int = 42,
    n_permutations: int = 1000,
) -> pd.DataFrame:
    """Predict each phenotype from latent z using cross-validated Ridge regression.

    Returns DataFrame with columns:
      phenotype, r2_mean, r2_std, r2_cv (list), p_value_permutation,
      r_pearson, p_pearson
    """
    scaler = StandardScaler()
    Z = scaler.fit_transform(z)

    rows = []
    rng = np.random.default_rng(seed)

    for j, name in enumerate(pheno_names):
        y = phenotypes[:, j]
        mask = ~np.isnan(y)
        if mask.sum() < 20:
            continue  # skip if too few valid observations

        Z_m, y_m = Z[mask], y[mask]

        # Cross-validated R²
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        r2_cv = cross_val_score(
            RidgeCV(alphas=np.logspace(-3, 3, 20)),
            Z_m, y_m, cv=cv, scoring="r2"
        )
        r2_mean = float(r2_cv.mean())
        r2_std  = float(r2_cv.std())

        # Pearson r between cross-validated predictions and actual values
        from sklearn.model_selection import cross_val_predict
        y_pred_cv = cross_val_predict(
            RidgeCV(alphas=np.logspace(-3, 3, 20)), Z_m, y_m, cv=cv
        )
        r_p, p_p = pearsonr(y_m, y_pred_cv)
        ridge = RidgeCV(alphas=np.logspace(-3, 3, 20)).fit(Z_m, y_m)
        # Permutation test for R² — parallelized across permutations
        perm_seeds = rng.integers(1_000_000, size=n_permutations).tolist()
        perm_y     = [rng.permutation(y_m) for _ in range(n_permutations)]

        def _one_perm(y_perm, seed_p):
            return cross_val_score(
                RidgeCV(alphas=np.logspace(-2, 2, 10)),
                Z_m, y_perm,
                cv=KFold(n_splits=n_folds, shuffle=True, random_state=seed_p),
                scoring="r2", n_jobs=1,
            ).mean()

        null_r2 = np.array(
            Parallel(n_jobs=-1, prefer="threads")(
                delayed(_one_perm)(py, ps) for py, ps in zip(perm_y, perm_seeds)
            )
        )
        p_perm = float((null_r2 >= r2_mean).mean())

        rows.append({
            "phenotype": name,
            "r2_mean":   r2_mean,
            "r2_std":    r2_std,
            "r_pearson": float(r_p),
            "p_pearson": float(p_p),
            "p_permutation": p_perm,
            "n_valid":   int(mask.sum()),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Latent clustering
# ---------------------------------------------------------------------------

def bootstrap_clustering_stability(
    z: np.ndarray,
    best_k: int,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap estimate of clustering stability with 95% CI."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Z      = scaler.fit_transform(z)
    rng    = np.random.default_rng(seed)
    rows   = []

    for i in range(n_bootstrap):
        idx    = rng.choice(len(Z), size=len(Z), replace=True)
        Z_boot = Z[idx]
        km     = KMeans(n_clusters=best_k, random_state=i, n_init=5, max_iter=100)
        lbl    = km.fit_predict(Z_boot)
        if len(np.unique(lbl)) < 2:
            continue
        rows.append({
            "iteration":      i,
            "silhouette":     float(silhouette_score(Z_boot, lbl)),
            "davies_bouldin": float(davies_bouldin_score(Z_boot, lbl)),
            "calinski":       float(calinski_harabasz_score(Z_boot, lbl)),
        })

    df  = pd.DataFrame(rows)
    sil = df["silhouette"]
    summary = pd.DataFrame([
        {"iteration": "mean",    "silhouette": sil.mean(),
         "davies_bouldin": df["davies_bouldin"].mean(),
         "calinski":       df["calinski"].mean()},
        {"iteration": "ci_low",  "silhouette": sil.quantile(0.025),
         "davies_bouldin": df["davies_bouldin"].quantile(0.025),
         "calinski":       df["calinski"].quantile(0.025)},
        {"iteration": "ci_high", "silhouette": sil.quantile(0.975),
         "davies_bouldin": df["davies_bouldin"].quantile(0.975),
         "calinski":       df["calinski"].quantile(0.975)},
    ])
    return pd.concat([df, summary], ignore_index=True)


def latent_clustering(
    z: np.ndarray,
    k_range: List[int] = None,
    seed: int = 42,
) -> Dict:
    """K-means clustering of latent space across a grid of k values."""
    if k_range is None:
        k_range = [3, 4, 5, 6, 7]

    scaler = StandardScaler()
    Z = scaler.fit_transform(z)

    all_results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=20)
        labels = km.fit_predict(Z)
        sil = float(silhouette_score(Z, labels))
        db  = float(davies_bouldin_score(Z, labels))
        ch  = float(calinski_harabasz_score(Z, labels))
        all_results.append({
            "k": k, "labels": labels,
            "silhouette": sil, "davies_bouldin": db, "calinski": ch,
        })

    best_idx = int(np.argmax([r["silhouette"] for r in all_results]))
    best     = all_results[best_idx]

    return {
        "best_k":       best["k"],
        "labels":       best["labels"],
        "silhouette_scores":     [r["silhouette"]     for r in all_results],
        "davies_bouldin_scores": [r["davies_bouldin"] for r in all_results],
        "calinski_scores":       [r["calinski"]       for r in all_results],
        "k_range":      k_range,
        "all_results":  all_results,
    }
