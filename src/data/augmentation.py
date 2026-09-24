"""
Graph augmentations for contrastive training of the graph autoencoder.

Augmentation 1 — Edge dropout
-----------------------------
    Randomly removes a fraction of graph edges.  This simulates the
    variability that arises from different FC thresholding choices and
    individual within-session noise in connectivity strength.

Augmentation 2 — Node-feature noise
-----------------------------------
    Adds i.i.d. Gaussian noise to each node's FC-profile feature vector.
    This simulates measurement noise in the BOLD signal and encourages
    the encoder to learn representations that are robust to small
    perturbations of individual ROI connectivity profiles.

Usage
-----
    from src.data.augmentation import augment_batch
    aug_x, aug_ei, aug_ea = augment_batch(batch, edge_drop_rate=0.10,
                                           feat_noise_std=0.02)
    mu, logvar, _ = model.encoder(aug_x, aug_ei, aug_ea, batch.batch)
"""

from __future__ import annotations

import torch
from torch import Tensor


def augment_batch(
    batch: "Batch",
    edge_drop_rate: float = 0.10,
    feat_noise_std: float = 0.02,
    min_edge_frac: float = 0.50,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Create one stochastically augmented view of a batched PyG graph."""
    device = batch.x.device

    # ------------------------------------------------------------------ #
    # Feature noise
    # ------------------------------------------------------------------ #
    aug_x = batch.x + torch.randn_like(batch.x) * feat_noise_std

    # ------------------------------------------------------------------ #
    # Edge dropout
    # ------------------------------------------------------------------ #
    aug_ei = batch.edge_index
    aug_ea = batch.edge_attr

    if edge_drop_rate > 0.0 and aug_ei.size(1) > 0:
        n_edges = aug_ei.size(1)
        keep    = torch.rand(n_edges, device=device) >= edge_drop_rate

        # Safety floor: guarantee ≥ min_edge_frac edges survive
        if keep.float().mean().item() < min_edge_frac:
            n_keep   = max(1, int(n_edges * min_edge_frac))
            perm     = torch.randperm(n_edges, device=device)[:n_keep]
            keep     = torch.zeros(n_edges, dtype=torch.bool, device=device)
            keep[perm] = True

        aug_ei = aug_ei[:, keep]
        if aug_ea is not None:
            aug_ea = aug_ea[keep]

    return aug_x, aug_ei, aug_ea
