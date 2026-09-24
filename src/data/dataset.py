"""PyTorch Geometric dataset of connectivity graphs, with deterministic splits."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Batch, Data

logger = logging.getLogger(__name__)


class BrainConnectivityDataset:
    """In-memory dataset of brain FC graphs with metadata."""

    def __init__(
        self,
        graphs: List[Data],
        subject_ids: List[str],
        transform=None,
    ):
        self._graphs = graphs
        self._subject_ids = subject_ids
        self.transform = transform

    # ------------------------------------------------------------------
    # Required by PyG DataLoader
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._graphs)

    def __getitem__(self, idx: int) -> Data:
        data = self._graphs[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data

    # ------------------------------------------------------------------
    # Metadata properties
    # ------------------------------------------------------------------

    @property
    def subject_ids(self) -> List[str]:
        return self._subject_ids

    @property
    def n_rois(self) -> int:
        return self._graphs[0].num_nodes

    @property
    def n_node_features(self) -> int:
        return self._graphs[0].x.shape[1]

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def split(
        self,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
        seed: int = 42,
    ) -> Tuple["BrainConnectivityDataset", "BrainConnectivityDataset",
               "BrainConnectivityDataset"]:
        """Stratified (by sex) train/val/test split."""
        rng = np.random.default_rng(seed)
        n = len(self._graphs)
        idx = np.arange(n)

        sex_labels = np.array(
            [float(g.sex[0]) if hasattr(g, "sex") else 0.0
             for g in self._graphs]
        )

        train_idx, val_idx, test_idx = [], [], []
        for sex_val in [0.0, 1.0]:
            grp = idx[sex_labels == sex_val]
            rng.shuffle(grp)
            n_grp  = len(grp)
            n_train = int(n_grp * train_frac)
            n_val   = int(n_grp * val_frac)
            train_idx.extend(grp[:n_train].tolist())
            val_idx.extend(grp[n_train:n_train + n_val].tolist())
            test_idx.extend(grp[n_train + n_val:].tolist())

        def _subset(indices):
            return BrainConnectivityDataset(
                [self._graphs[i] for i in indices],
                [self._subject_ids[i] for i in indices],
            )

        return _subset(train_idx), _subset(val_idx), _subset(test_idx)

    def split_by_ids(
        self,
        train_ids: set,
        val_ids: set,
        test_ids: set,
    ) -> Tuple["BrainConnectivityDataset", "BrainConnectivityDataset",
               "BrainConnectivityDataset"]:
        """Reproduce an existing split by subject ID sets (for fair cross-model comparison)."""
        train_idx = [i for i, s in enumerate(self._subject_ids) if s in train_ids]
        val_idx   = [i for i, s in enumerate(self._subject_ids) if s in val_ids]
        test_idx  = [i for i, s in enumerate(self._subject_ids) if s in test_ids]

        def _subset(indices):
            return BrainConnectivityDataset(
                [self._graphs[i] for i in indices],
                [self._subject_ids[i] for i in indices],
            )
        return _subset(train_idx), _subset(val_idx), _subset(test_idx)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_fc_stack(self) -> np.ndarray:
        """Return (N, n_rois, n_rois) array of FC matrices."""
        return np.stack([g.fc_full.numpy() for g in self._graphs], axis=0)

    def get_phenotype_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Return (N, n_phenotypes) phenotype matrix and column names."""
        graphs_with_pheno = [g for g in self._graphs if hasattr(g, "pheno")]
        if not graphs_with_pheno:
            return np.empty((len(self._graphs), 0)), []
        cols = graphs_with_pheno[0].pheno_cols
        mat  = np.stack([g.pheno.numpy() for g in graphs_with_pheno], axis=0)
        return mat, cols

    def __repr__(self) -> str:
        return (f"BrainConnectivityDataset({len(self._graphs)} subjects, "
                f"{self.n_rois} ROIs, {self.n_node_features} node features)")


# ---------------------------------------------------------------------------
# Dynamic dataset: static graph + sliding-window FC + optional rest FC
# ---------------------------------------------------------------------------

class DynamicBrainDataset:
    """Dataset returning (static_graph, dynamic_fc, rest_fc) tuples."""

    def __init__(
        self,
        static_graphs: List[Data],
        subject_ids: List[str],
        dynamic_fc: Optional[np.ndarray] = None,
        rest_fc: Optional[np.ndarray] = None,
    ):
        self._static   = static_graphs
        self._ids      = subject_ids
        self._dfc      = dynamic_fc   # (N, T, N_roi, N_roi) or None
        self._rest     = rest_fc      # (N, N_roi, N_roi) or None

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._static)

    def __getitem__(self, idx: int):
        sg  = self._static[idx]
        dfc = (torch.tensor(self._dfc[idx], dtype=torch.float32)
               if self._dfc is not None else torch.empty(0))
        rfc = (torch.tensor(self._rest[idx], dtype=torch.float32)
               if self._rest is not None else torch.empty(0))
        return sg, dfc, rfc

    # ------------------------------------------------------------------
    # Custom collate: batch static graphs with PyG, stack FC tensors
    # ------------------------------------------------------------------

    @staticmethod
    def collate(batch):
        static_list, dfc_list, rfc_list = zip(*batch)
        static_batch = Batch.from_data_list(list(static_list))
        dfc_batch = torch.stack(list(dfc_list), dim=0)   # (B, T, N, N)
        rfc_batch = torch.stack(list(rfc_list), dim=0)   # (B, N, N)
        return static_batch, dfc_batch, rfc_batch

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def subject_ids(self) -> List[str]:
        return self._ids

    @property
    def has_dynamic(self) -> bool:
        return self._dfc is not None

    @property
    def has_rest(self) -> bool:
        return self._rest is not None

    # ------------------------------------------------------------------
    # Splitting (sex-stratified, identical logic to BrainConnectivityDataset)
    # ------------------------------------------------------------------

    def split(
        self,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
        seed: int = 42,
    ) -> Tuple["DynamicBrainDataset", "DynamicBrainDataset", "DynamicBrainDataset"]:
        rng = np.random.default_rng(seed)
        n = len(self._static)
        idx = np.arange(n)

        sex_labels = np.array(
            [float(g.sex[0]) if hasattr(g, "sex") else 0.0
             for g in self._static]
        )

        train_idx, val_idx, test_idx = [], [], []
        for sex_val in [0.0, 1.0]:
            grp = idx[sex_labels == sex_val]
            rng.shuffle(grp)
            n_grp  = len(grp)
            n_train = int(n_grp * train_frac)
            n_val   = int(n_grp * val_frac)
            train_idx.extend(grp[:n_train].tolist())
            val_idx.extend(grp[n_train:n_train + n_val].tolist())
            test_idx.extend(grp[n_train + n_val:].tolist())

        def _subset(indices):
            dfc_sub  = self._dfc[indices]  if self._dfc  is not None else None
            rest_sub = self._rest[indices] if self._rest is not None else None
            return DynamicBrainDataset(
                [self._static[i] for i in indices],
                [self._ids[i]    for i in indices],
                dynamic_fc=dfc_sub,
                rest_fc=rest_sub,
            )

        return _subset(train_idx), _subset(val_idx), _subset(test_idx)

    def __repr__(self) -> str:
        dyn  = f"T={self._dfc.shape[1]}" if self._dfc is not None else "no-dynamic"
        rest = "rest=yes" if self._rest is not None else "rest=no"
        return (f"DynamicBrainDataset({len(self._static)} subjects, "
                f"{dyn}, {rest})")
