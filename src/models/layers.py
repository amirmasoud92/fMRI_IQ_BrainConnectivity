"""Graph attention, pooling and decoder layers of the graph autoencoder."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATv2Conv, TransformerConv
from torch_geometric.nn.aggr import AttentionalAggregation


class ResGATLayer(nn.Module):
    """GATv2Conv + skip-connection + BatchNorm + ELU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        concat: bool = True,
        dropout: float = 0.2,
        edge_dim: Optional[int] = None,
        residual: bool = True,
    ):
        super().__init__()
        from typing import Optional  # local import to avoid top-level circular
        self.residual = residual
        self.out_dim  = out_channels * heads if concat else out_channels

        self.conv = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            dropout=dropout,
            edge_dim=edge_dim,
            share_weights=False,
        )
        self.norm = nn.BatchNorm1d(self.out_dim)

        # Linear projection for residual if dimensions differ
        if residual and in_channels != self.out_dim:
            self.res_proj = nn.Linear(in_channels, self.out_dim, bias=False)
        else:
            self.res_proj = None

    def forward(self, x: Tensor, edge_index: Tensor,
                edge_attr: Optional[Tensor] = None,
                return_attention: bool = False):
        """
        Returns
        -------
        out  : (N, out_dim) updated node features.
        attn : (E, heads) attention weights (only if return_attention=True).
        """
        if return_attention:
            h, (_, attn) = self.conv(
                x, edge_index, edge_attr=edge_attr, return_attention_weights=True
            )
        else:
            h = self.conv(x, edge_index, edge_attr=edge_attr)

        h = self.norm(h)

        if self.residual:
            skip = self.res_proj(x) if self.res_proj is not None else x
            h = h + skip

        h = F.elu(h)

        if return_attention:
            return h, attn
        return h


# ---------------------------------------------------------------------------

class GlobalAttentionPool(nn.Module):
    """Soft attention pooling: computes graph-level embedding as
    sum_i( a_i * h_i ) where a_i = softmax(gate(h_i)).

    Parameters
    ----------
    in_channels : node embedding dimensionality.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        gate_nn = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.Tanh(),
            nn.Linear(in_channels // 2, 1),
        )
        self.pool = AttentionalAggregation(gate_nn=gate_nn)

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        return self.pool(x, index=batch)


# ---------------------------------------------------------------------------

class InnerProductDecoder(nn.Module):
    """Reconstruct FC matrix via inner product between node embeddings.

    z_i · z_j predicts the connection strength between ROI i and j.
    Operates on per-graph node embeddings of shape (N, D).
    """

    def __init__(self, activation: str = "linear"):
        super().__init__()
        # activation: "linear" | "sigmoid" | "tanh"
        self.activation = activation

    def forward(self, z: Tensor) -> Tensor:
        """
        Parameters
                ----------
                z : (N, D) node embedding matrix for a single graph.
        """
        A_hat = torch.mm(z, z.t())
        if self.activation == "sigmoid":
            A_hat = torch.sigmoid(A_hat)
        elif self.activation == "tanh":
            A_hat = torch.tanh(A_hat)
        return A_hat


# ---------------------------------------------------------------------------

class MLPDecoder(nn.Module):
    """Decode latent code z → full FC matrix via MLP."""

    def __init__(self, latent_dim: int, n_rois: int, hidden_dim: int = 256):
        super().__init__()
        out_dim = n_rois * (n_rois - 1) // 2
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )
        self.n_rois = n_rois

    def forward(self, z: Tensor) -> Tensor:
        """z : (B, latent_dim)  →  out : (B, n_upper_tri)"""
        return self.net(z)


# ---------------------------------------------------------------------------

class GraphTransformerDecoder(nn.Module):
    """Decode latent z → FC matrix via Graph Transformer on the complete ROI graph."""

    def __init__(
        self,
        latent_dim: int,
        n_rois: int,
        node_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if node_dim % n_heads != 0:
            # round up to nearest multiple so head_dim stays integer
            node_dim = ((node_dim + n_heads - 1) // n_heads) * n_heads

        self.n_rois   = n_rois
        self.node_dim = node_dim
        head_dim      = node_dim // n_heads

        # z → same initial embedding broadcast to all N nodes
        self.expand = nn.Sequential(
            nn.Linear(latent_dim, node_dim),
            nn.LayerNorm(node_dim),
        )
        # Unique learnable identity embedding per ROI
        self.roi_pos = nn.Embedding(n_rois, node_dim)

        # Transformer layers: TransformerConv keeps dim node_dim → node_dim
        self.tf_layers = nn.ModuleList([
            TransformerConv(
                in_channels=node_dim,
                out_channels=head_dim,
                heads=n_heads,
                concat=True,   # output: head_dim * n_heads = node_dim
                dropout=dropout,
                beta=True,     # learned gate between self and aggregated neighbours
            )
            for _ in range(n_layers)
        ])
        self.tf_norms = nn.ModuleList([nn.LayerNorm(node_dim) for _ in range(n_layers)])

        # Symmetric edge-weight predictor:  [hᵢ+hⱼ ‖ hᵢ⊙hⱼ] → scalar
        self.edge_pred = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, 1),
        )

        # Pre-compute complete undirected graph edge_index for n_rois nodes
        # Store both directions so each node can aggregate from all neighbours
        r, c = torch.triu_indices(n_rois, n_rois, offset=1)
        src = torch.cat([r, c])
        dst = torch.cat([c, r])
        self.register_buffer("_complete_ei", torch.stack([src, dst]))  # (2, N*(N-1))
        self.register_buffer("_triu_rows", r)
        self.register_buffer("_triu_cols", c)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.roi_pos.weight, std=0.02)

    def _batch_edge_index(self, B: int, device) -> Tensor:
        """Tile the complete-graph edge_index for a batch of B subjects."""
        n_e = self._complete_ei.shape[1]                   # N*(N-1)
        offsets = torch.arange(B, device=device).repeat_interleave(n_e) * self.n_rois
        return self._complete_ei.repeat(1, B) + offsets.unsqueeze(0)

    def forward(self, z: Tensor) -> Tensor:
        """
        Parameters
                ----------
                z : (B, latent_dim)
        """
        B, N = z.size(0), self.n_rois

        # Initial node features: broadcast z + unique ROI position
        h = (
            self.expand(z).unsqueeze(1).expand(B, N, -1)      # (B, N, node_dim)
            + self.roi_pos.weight.unsqueeze(0)                 # (1, N, node_dim)
        ).reshape(B * N, -1)                                   # (B*N, node_dim)

        # Graph Transformer layers on complete graph
        ei = self._batch_edge_index(B, z.device)               # (2, B*N*(N-1))
        for conv, norm in zip(self.tf_layers, self.tf_norms):
            h = norm(conv(h, ei) + h)                          # residual + LayerNorm

        # Predict upper-triangle edge weights
        h = h.view(B, N, -1)                                   # (B, N, node_dim)
        hi = h[:, self._triu_rows, :]                          # (B, n_tri, node_dim)
        hj = h[:, self._triu_cols, :]                          # (B, n_tri, node_dim)
        fc_hat = self.edge_pred(
            torch.cat([hi + hj, hi * hj], dim=-1)             # symmetric features
        ).squeeze(-1)                                          # (B, n_tri)
        return fc_hat

