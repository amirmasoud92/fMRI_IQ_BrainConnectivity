"""Convert connectivity matrices into PyTorch Geometric graphs."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Core graph construction
# ---------------------------------------------------------------------------

def fc_to_log_euclidean(fc: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    """Map a correlation FC matrix to log-Euclidean space."""
    n = fc.shape[0]
    C = fc.copy()
    np.fill_diagonal(C, 1.0)
    # Ledoit-Wolf-style shrinkage toward identity for numerical stability
    C = (1.0 - epsilon) * C + epsilon * np.eye(n, dtype=np.float32)
    eigenvalues, eigenvectors = np.linalg.eigh(C.astype(np.float64))
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    log_C = (eigenvectors * np.log(eigenvalues)) @ eigenvectors.T
    return log_C.astype(np.float32)


def fc_to_pyg_graph(
    fc_matrix: np.ndarray,
    density: float = 0.20,
    self_loops: bool = False,
    use_z: bool = False,
    coords: Optional[np.ndarray] = None,
    signed_edges: bool = False,
    log_euclidean: bool = False,
) -> Data:
    """Build a PyG Data object from a (N, N) FC matrix."""
    n = fc_matrix.shape[0]
    fc = fc_matrix.copy()
    np.fill_diagonal(fc, 0.0)

    # --- Node features: FC profile (or log-Euclidean) + optional MNI coords ---
    if log_euclidean:
        x_np = fc_to_log_euclidean(fc)                             # (N, N) in log space
    else:
        x_np = fc.astype(np.float32)                               # (N, N) Fisher-z
    if coords is not None:
        coords_norm = (np.asarray(coords, dtype=np.float32) / 100.0)  # ≈ [-1, 1]
        x_np = np.concatenate([x_np, coords_norm], axis=1)            # (N, N+3)
    x = torch.tensor(x_np, dtype=torch.float32)

    # --- Build sparse edges from upper triangle, top-k% by |r| -------------
    triu_idx = np.triu_indices(n, k=1)
    weights = np.abs(fc[triu_idx])

    n_edges = int(len(weights) * density)
    top_k = np.argpartition(weights, -n_edges)[-n_edges:]

    src = triu_idx[0][top_k]
    dst = triu_idx[1][top_k]
    w   = weights[top_k]                          # |r|  — used for threshold only
    w_signed = fc[triu_idx][top_k].astype(np.float32)  # signed r — actual edge feature

    # Make undirected (duplicate edges both directions)
    src_ud    = np.concatenate([src, dst])
    dst_ud    = np.concatenate([dst, src])
    w_ud      = np.concatenate([w,        w])       # |r| magnitude (kept for compat)
    w_signed_ud = np.concatenate([w_signed, w_signed])

    if self_loops:
        self_src = np.arange(n)
        src_ud      = np.concatenate([src_ud,      self_src])
        dst_ud      = np.concatenate([dst_ud,      self_src])
        w_ud        = np.concatenate([w_ud,        np.ones(n)])
        w_signed_ud = np.concatenate([w_signed_ud, np.ones(n)])

    edge_index = torch.tensor(
        np.stack([src_ud, dst_ud], axis=0), dtype=torch.long
    )
    if signed_edges:
        # 2-dim: [signed_r, |r|] — GATv2 can distinguish anti-correlated
        # (negative) from co-activated (positive) connections
        edge_attr = torch.tensor(
            np.stack([w_signed_ud, w_ud], axis=1), dtype=torch.float32
        )  # (E, 2)
    else:
        edge_attr = torch.tensor(w_ud[:, None], dtype=torch.float32)  # (E, 1)

    # --- Reconstruction target: upper triangle flattened -------------------
    triu_all = np.triu_indices(n, k=1)
    if log_euclidean:
        # Reconstruction target must match node features — both in log-Euclidean space.
        # Using raw Fisher-z here would create a coordinate mismatch (decoder trained on
        # a space the encoder never sees), causing near-zero recon correlation.
        log_fc = fc_to_log_euclidean(fc)
        fc_vec = torch.tensor(log_fc[triu_all], dtype=torch.float32)
    else:
        fc_vec = torch.tensor(fc[triu_all], dtype=torch.float32)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        fc_full=torch.tensor(fc, dtype=torch.float32),
        fc_vec=fc_vec,
        num_nodes=n,
    )
    # Store normalized coords separately — needed in dynamic forward to rebuild
    # window node features without knowing the full feature layout.
    if coords is not None:
        data.coords_norm = torch.tensor(
            np.asarray(coords, dtype=np.float32) / 100.0
        )  # (N, 3)
    return data


# ---------------------------------------------------------------------------
# Build full graph dataset
# ---------------------------------------------------------------------------

def build_graph_dataset(
    results: List[Dict],
    participants: "pd.DataFrame",
    phenotype_cols: Optional[List[str]] = None,
    density: float = 0.20,
    use_z_matrix: bool = True,
    coords: Optional[np.ndarray] = None,
    signed_edges: bool = False,
    log_euclidean: bool = False,
) -> Tuple[List[Data], List[str]]:
    """Convert list of preprocessed subject dicts into PyG Data objects."""
    import numpy as _np

    graphs, subject_ids = [], []

    # --- First pass: collect raw phenotype values for normalisation ---
    pheno_raw: Dict[str, list] = {col: [] for col in (phenotype_cols or [])}
    for res in results:
        sub = res["subject_id"]
        if phenotype_cols is not None and sub in participants.index:
            row = participants.loc[sub]
            for col in phenotype_cols:
                val = row.get(col, float("nan"))
                try:
                    v = float(val)
                    if not _np.isnan(v):
                        pheno_raw[col].append(v)
                except (ValueError, TypeError):
                    pass

    # Per-column mean and std (fallback to 0/1 if no valid values)
    pheno_mean = {col: float(_np.nanmean(pheno_raw[col])) if pheno_raw[col] else 0.0
                  for col in (phenotype_cols or [])}
    pheno_std  = {col: float(_np.nanstd(pheno_raw[col]))  if len(pheno_raw[col]) > 1 else 1.0
                  for col in (phenotype_cols or [])}
    pheno_std  = {col: max(v, 1e-8) for col, v in pheno_std.items()}  # avoid div/0

    # --- Second pass: build graph objects ---
    for res in results:
        sub = res["subject_id"]
        fc = res["fc_matrix_z"] if use_z_matrix else res["fc_matrix"]

        graph = fc_to_pyg_graph(fc, density=density, coords=coords,
                                signed_edges=signed_edges, log_euclidean=log_euclidean)

        # Attach subject metadata
        graph.subject_id = sub
        graph.n_volumes  = res["n_volumes"]
        graph.mean_fd    = res["mean_fd"]

        # Attach z-scored phenotype tensor if available
        if phenotype_cols is not None and sub in participants.index:
            row = participants.loc[sub]
            pheno_vals = []
            for col in phenotype_cols:
                val = row.get(col, float("nan"))
                try:
                    raw = float(val)
                    # z-score normalise: target ≈ N(0,1) for stable MSE loss
                    pheno_vals.append((raw - pheno_mean[col]) / pheno_std[col])
                except (ValueError, TypeError):
                    pheno_vals.append(float("nan"))
            graph.pheno      = torch.tensor(pheno_vals, dtype=torch.float32)
            graph.pheno_cols = phenotype_cols
            # Store normalisation params for inverse transform in analysis
            graph.pheno_mean = torch.tensor(
                [pheno_mean[c] for c in phenotype_cols], dtype=torch.float32)
            graph.pheno_std  = torch.tensor(
                [pheno_std[c]  for c in phenotype_cols], dtype=torch.float32)

            # Convenience: binary sex label (0=female, 1=male)
            sex_val = str(row.get("sex", "")).lower()
            graph.sex = torch.tensor(
                [1.0 if sex_val == "male" else 0.0], dtype=torch.float32
            )

        graphs.append(graph)
        subject_ids.append(sub)

    return graphs, subject_ids
