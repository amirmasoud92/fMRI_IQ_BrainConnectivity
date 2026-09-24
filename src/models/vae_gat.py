"""
Graph variational autoencoder with graph-attention layers (withdrawn model, evaluated in Fig. S1).

Architecture
------------
Encoder
-------
  Input : per-node FC profile  (N × N node features)
           + thresholded FC graph (edge_index, edge_attr)
  →  stack of ResGATLayer  (GATv2Conv + BN + residual)
  →  GlobalAttentionPool   →  graph embedding h  (B × hidden)
  →  Linear heads          →  μ, log-σ²           (B × latent_dim)

Reparameterization
------------------
  z = μ + ε · exp(0.5 · log-σ²),   ε ~ N(0, I)

Decoder (three modes)
---------------------
  A) "inner_product"     : expand z → per-node embeddings → z_i · z_j for all pairs
  B) "mlp"               : z → MLP → upper-triangle FC vector directly
  C) "graph_transformer" : z → broadcast + ROI pos-emb → L × TransformerConv on
                           complete graph → symmetric edge MLP → upper-triangle FC

Optional auxiliary head
-----------------------
  z → MLP → predicted phenotype scores  (multi-task training)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .layers import ResGATLayer, GlobalAttentionPool, InnerProductDecoder, MLPDecoder, GraphTransformerDecoder


class BrainGraphEncoder(nn.Module):
    """Stack of ResGATLayers + GlobalAttentionPool → (μ, log-σ²)."""

    def __init__(
        self,
        node_feat_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        n_heads: int = 8,
        dropout: float = 0.2,
        edge_dim: int = 1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        in_dim = node_feat_dim
        for h_dim in hidden_dims:
            # All but the last layer use multi-head concat
            self.layers.append(
                ResGATLayer(
                    in_channels=in_dim,
                    out_channels=h_dim,
                    heads=n_heads,
                    concat=True,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    residual=True,
                )
            )
            in_dim = h_dim * n_heads

        # Final single-head layer collapses multi-head output
        self.final_layer = ResGATLayer(
            in_channels=in_dim,
            out_channels=hidden_dims[-1],
            heads=1,
            concat=False,
            dropout=dropout,
            edge_dim=edge_dim,
            residual=True,
        )
        self.pool = GlobalAttentionPool(hidden_dims[-1])

        self.fc_mu     = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        batch: Tensor,
        return_attention: bool = False,
    ) -> Tuple[Tensor, Tensor, Optional[List]]:
        """
        Returns
        -------
        mu     : (B, latent_dim)
        logvar : (B, latent_dim)
        attns  : list of attention weight tensors (if return_attention)
        """
        h = x
        attn_weights = [] if return_attention else None

        for layer in self.layers:
            if return_attention:
                h, attn = layer(h, edge_index, edge_attr, return_attention=True)
                attn_weights.append(attn)
            else:
                h = layer(h, edge_index, edge_attr)

        h = self.final_layer(h, edge_index, edge_attr)

        # Graph-level readout
        h_graph = self.pool(h, batch)          # (B, hidden)

        mu     = self.fc_mu(h_graph)           # (B, latent_dim)
        logvar = self.fc_logvar(h_graph)       # (B, latent_dim)

        return mu, logvar, attn_weights

    def encode_to_h(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        batch: Tensor,
    ) -> Tensor:
        """Return pooled graph embedding *before* the μ/σ² linear heads."""
        h = x
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        h = self.final_layer(h, edge_index, edge_attr)
        return self.pool(h, batch)


class BrainGraphDecoder(nn.Module):
    """Reconstruct FC matrix from latent code z."""

    def __init__(
        self,
        latent_dim: int,
        n_rois: int,
        decoder_type: str = "mlp",
        decoder_hidden: int = 256,
        n_heads: int = 8,
        gt_n_layers: int = 2,
    ):
        super().__init__()
        self.decoder_type = decoder_type
        self.n_rois = n_rois
        self.n_edges = n_rois * (n_rois - 1) // 2

        if decoder_type == "mlp":
            self.mlp = MLPDecoder(latent_dim, n_rois, decoder_hidden)

        elif decoder_type == "inner_product":
            # Expand z to per-node embeddings.  A learnable per-ROI embedding is
            # required: without it every node receives the identical vector v,
            # so z_i · z_j = ‖v‖² is constant across all N(N-1)/2 edges and the
            # decoder can only emit one scalar per subject.
            self.node_expand = nn.Sequential(
                nn.Linear(latent_dim, decoder_hidden),
                nn.GELU(),
                nn.Linear(decoder_hidden, decoder_hidden),
            )
            self.roi_pos = nn.Embedding(n_rois, decoder_hidden)
            nn.init.normal_(self.roi_pos.weight, std=0.02)
            self.ip = InnerProductDecoder(activation="linear")
            self._triu_idx = None  # computed on first forward pass

        elif decoder_type == "graph_transformer":
            self.gt = GraphTransformerDecoder(
                latent_dim=latent_dim,
                n_rois=n_rois,
                node_dim=decoder_hidden,
                n_heads=n_heads,
                n_layers=gt_n_layers,
            )

        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type!r}. "
                             f"Choose 'mlp', 'inner_product', or 'graph_transformer'.")

    def _get_triu_idx(self, device):
        if self._triu_idx is None or self._triu_idx[0].device != device:
            rows, cols = torch.triu_indices(self.n_rois, self.n_rois, offset=1)
            self._triu_idx = (rows.to(device), cols.to(device))
        return self._triu_idx

    def forward(self, z: Tensor) -> Tensor:
        """
        Parameters
                ----------
                z : (B, latent_dim)
        """
        B = z.size(0)

        if self.decoder_type == "mlp":
            return self.mlp(z)  # (B, n_upper_tri)

        elif self.decoder_type == "inner_product":
            # z → (B, N, D): broadcast the subject code, then add the learnable
            # per-ROI embedding so that nodes are distinguishable.
            node_emb = self.node_expand(z)                   # (B, D)
            node_emb = (node_emb.unsqueeze(1)                # (B, 1, D)
                        + self.roi_pos.weight.unsqueeze(0))  # (1, N, D)
            # (B, N, N) adjacency via batched inner product
            A = torch.bmm(node_emb, node_emb.transpose(1, 2))  # (B, N, N)
            rows, cols = self._get_triu_idx(z.device)
            fc_hat = A[:, rows, cols]                            # (B, n_upper_tri)
            return fc_hat

        elif self.decoder_type == "graph_transformer":
            return self.gt(z)  # (B, n_upper_tri)


class TemporalEncoder(nn.Module):
    """BiLSTM over a sequence of graph-level embeddings → (μ, log-σ²)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        latent_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.drop      = nn.Dropout(dropout)
        self.fc_mu     = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h_seq: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Parameters
                ----------
                h_seq : (B, T, input_dim) sequence of per-window graph embeddings.
        """
        _, (hn, _) = self.lstm(h_seq)
        # hn : (n_layers*2, B, hidden_dim) — last layer fwd & bwd
        h = self.drop(torch.cat([hn[-2], hn[-1]], dim=1))  # (B, hidden_dim*2)
        return self.fc_mu(h), self.fc_logvar(h)


class FusionMLP(nn.Module):
    """Fuse [z_dynamic ‖ z_rest] → z_fused via a small MLP.

    Parameters
    ----------
    in_dim  : total input dimensionality (latent_dim * n_sources).
    out_dim : output dimensionality (= latent_dim).
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim, out_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, *tensors: Tensor) -> Tensor:
        return self.net(torch.cat(tensors, dim=1))


class DynamicGate(nn.Module):
    """Residual fusion: z_fused = z_static + sigmoid(gate) · proj(z_dyn)."""

    def __init__(self, latent_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Per-dimension gate: sigmoid(-3) ≈ 0.047 → starts near-zero
        self.log_gate = nn.Parameter(torch.full((latent_dim,), -3.0))
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z_static: Tensor, z_dyn: Tensor) -> Tensor:
        gate = torch.sigmoid(self.log_gate)            # (latent_dim,)
        return z_static + gate * self.proj(z_dyn)      # residual addition


class ProjectionHead(nn.Module):
    """2-layer BN-MLP projection head mapping z → contrastive representation space."""

    def __init__(self, latent_dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, proj_dim),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


class PhenotypeHead(nn.Module):
    """Lightweight MLP head to predict continuous phenotypes from z."""

    def __init__(self, latent_dim: int, n_outputs: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_outputs),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class BrainVAEGAT(nn.Module):
    """
    Variational Autoencoder with Graph Attention Networks for
        brain functional connectivity.
    """

    def __init__(
        self,
        n_rois: int = 200,
        node_feat_dim: int = 200,
        hidden_dims: Optional[List[int]] = None,
        latent_dim: int = 64,
        n_heads: int = 8,
        decoder_type: str = "mlp",
        decoder_hidden: int = 256,
        gt_n_layers: int = 2,
        dropout: float = 0.2,
        edge_dim: int = 1,
        n_phenotypes: int = 0,
        proj_dim: int = 0,
        # Dynamic FC / temporal encoder
        use_dynamic: bool = False,
        temporal_hidden: int = 128,
        temporal_layers: int = 2,
        # Resting-state fusion
        use_rest_fusion: bool = False,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.n_rois          = n_rois
        self.latent_dim      = latent_dim
        self.n_phenotypes    = n_phenotypes
        self._n_upper        = n_rois * (n_rois - 1) // 2
        self.use_dynamic     = use_dynamic
        self.use_rest_fusion = use_rest_fusion

        self.encoder = BrainGraphEncoder(
            node_feat_dim=node_feat_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            n_heads=n_heads,
            dropout=dropout,
            edge_dim=edge_dim,
        )

        self.decoder = BrainGraphDecoder(
            latent_dim=latent_dim,
            n_rois=n_rois,
            decoder_type=decoder_type,
            decoder_hidden=decoder_hidden,
            n_heads=n_heads,
            gt_n_layers=gt_n_layers,
        )

        # Temporal encoder: lightweight linear window encoder + BiLSTM
        # The static GATv2 encoder is ALWAYS the primary reconstruction path.
        # The temporal encoder captures sliding-window dynamics and contributes
        # via a learned residual gate (DynamicGate) initialised near-zero so
        # the model starts identical to the static VAE-GAT and gradually opens
        # the gate as training progresses.
        if use_dynamic:
            # Full upper-triangle per window (19900 values for 200 ROIs) preserves
            # all pairwise FC information. Mean+std (400 features) degraded r.
            n_upper = n_rois * (n_rois - 1) // 2
            self.window_proj = nn.Sequential(
                nn.Linear(n_upper, hidden_dims[-1]),
                nn.LayerNorm(hidden_dims[-1]),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.temporal_encoder = TemporalEncoder(
                input_dim=hidden_dims[-1],
                hidden_dim=temporal_hidden,
                n_layers=temporal_layers,
                latent_dim=latent_dim,
                dropout=dropout,
            )
            # Residual gate: z_fused = z_static + sigmoid(gate) * proj(mu_dyn)
            self.dyn_gate = DynamicGate(latent_dim, dropout=dropout)
        else:
            self.window_proj      = None
            self.temporal_encoder = None
            self.dyn_gate         = None

        # Fusion MLP: [z_dynamic || z_rest] → z_fused  (rest currently disabled)
        if use_rest_fusion:
            self.fusion_mlp = FusionMLP(latent_dim * 2, latent_dim)
        else:
            self.fusion_mlp = None

        if n_phenotypes > 0:
            self.pheno_head = PhenotypeHead(latent_dim, n_phenotypes)
        else:
            self.pheno_head = None

        if proj_dim > 0:
            self.proj_head = ProjectionHead(latent_dim, proj_dim)
        else:
            self.proj_head = None

    # -----------------------------------------------------------------------
    # Core VAE operations
    # -----------------------------------------------------------------------

    def _reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Sample z ~ N(μ, σ²) via the reparameterization trick.

        Deterministic (returns μ) in eval mode.
        """
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(
        self,
        data: "Batch",
        return_attention: bool = False,
    ) -> Tuple[Tensor, Tensor, Optional[List]]:
        """Run encoder on a PyG batch.

        Returns
        -------
        mu, logvar : (B, latent_dim) each
        attn_list  : list of attention tensors (if return_attention)
        """
        return self.encoder(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            batch=data.batch,
            return_attention=return_attention,
        )

    def decode(self, z: Tensor) -> Tensor:
        """Reconstruct FC upper triangle from z. Returns (B, n_upper_tri)."""
        return self.decoder(z)

    def forward(
        self,
        data: "Batch",
        return_attention: bool = False,
    ) -> Dict[str, Tensor]:
        """Full forward pass."""
        mu, logvar, attn = self.encode(data, return_attention=return_attention)
        z      = self._reparameterize(mu, logvar)
        fc_hat = self.decode(z)

        out = dict(fc_hat=fc_hat, mu=mu, logvar=logvar, z=z)

        if self.pheno_head is not None:
            out["pheno_hat"] = self.pheno_head(z)

        if self.proj_head is not None:
            out["z_proj"] = self.proj_head(z)

        if return_attention and attn is not None:
            out["attn"] = attn

        return out

    # -----------------------------------------------------------------------
    # Dynamic FC forward pass (BiLSTM temporal encoder + optional rest fusion)
    # -----------------------------------------------------------------------

    def forward_dynamic(
        self,
        static_batch: "Batch",
        dfc_batch: Optional[Tensor] = None,
        rest_batch: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, Tensor]:
        """Forward pass with static GATv2 encoder as primary path + optional dynamic fusion."""
        if dfc_batch is None or self.temporal_encoder is None or dfc_batch.numel() == 0:
            return self.forward(static_batch, return_attention=return_attention)

        # ------------------------------------------------------------------
        # Primary path: static GATv2 encoder (KL and reconstruction anchor)
        # ------------------------------------------------------------------
        mu, logvar, attn = self.encode(static_batch, return_attention=return_attention)
        z_static = self._reparameterize(mu, logvar)

        # ------------------------------------------------------------------
        # Secondary path: temporal encoder on sliding-window FC
        # Uses the mean (μ_dyn) deterministically — no reparameterization —
        # so the dynamic branch acts as a feature extractor, not a VAE branch.
        # ------------------------------------------------------------------
        B, T, N, _ = dfc_batch.shape
        device     = dfc_batch.device
        triu_r, triu_c = torch.triu_indices(N, N, offset=1, device=device)

        h_windows = []
        for t in range(T):
            fc_vec = dfc_batch[:, t, triu_r, triu_c]   # (B, n_upper_tri)
            h_t    = self.window_proj(fc_vec)            # (B, hidden_dims[-1])
            h_windows.append(h_t)

        h_seq     = torch.stack(h_windows, dim=1)        # (B, T, hidden_dims[-1])
        mu_dyn, _ = self.temporal_encoder(h_seq)         # (B, latent_dim) — mean only

        # Residual gate: z_fused = z_static + sigmoid(gate) * proj(mu_dyn)
        z_fused = self.dyn_gate(z_static, mu_dyn)

        # ------------------------------------------------------------------
        # Optional resting-state branch (currently disabled)
        # ------------------------------------------------------------------
        mu_rest = logvar_rest = None
        if rest_batch is not None and self.fusion_mlp is not None and rest_batch.numel() > 0:
            batch_vec  = static_batch.batch
            edge_index = static_batch.edge_index
            sub_of_edge = batch_vec[edge_index[0]]
            local_row   = edge_index[0] - sub_of_edge * N
            local_col   = edge_index[1] - sub_of_edge * N
            has_coords  = static_batch.x.shape[1] > N
            coords_all  = static_batch.x[:, N:] if has_coords else None
            x_rest = rest_batch.reshape(B * N, N)
            if coords_all is not None:
                x_rest = torch.cat([x_rest, coords_all], dim=1)
            ew_rest = rest_batch[sub_of_edge, local_row, local_col].abs().unsqueeze(1)
            mu_rest, logvar_rest, _ = self.encoder(x_rest, edge_index, ew_rest, batch_vec)
            z_rest  = self._reparameterize(mu_rest, logvar_rest)
            z_fused = self.fusion_mlp(z_fused, z_rest)

        fc_hat = self.decode(z_fused)

        out = {
            "fc_hat": fc_hat,
            "mu":     mu,       # static encoder μ → drives KL in ELBOLoss
            "logvar": logvar,   # static encoder log-σ²
            "z":      z_fused,  # fused representation (deterministic in eval)
        }
        if mu_rest is not None:
            out["mu_rest"]     = mu_rest
            out["logvar_rest"] = logvar_rest

        if self.pheno_head is not None:
            out["pheno_hat"] = self.pheno_head(z_fused)
        if self.proj_head is not None:
            out["z_proj"] = self.proj_head(z_fused)
        if return_attention and attn is not None:
            out["attn"] = attn

        return out

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def embed(self, data: "Batch") -> Tensor:
        """Return posterior mean μ (deterministic latent embedding)."""
        self.eval()
        mu, _, _ = self.encode(data)
        return mu

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        proj = f", proj_dim={self.proj_head.net[-1].out_features}" if self.proj_head else ""
        return (
            f"BrainVAEGAT(\n"
            f"  n_rois={self.n_rois}, latent_dim={self.latent_dim}{proj},\n"
            f"  encoder={self.encoder.__class__.__name__},\n"
            f"  decoder={self.decoder.decoder_type},\n"
            f"  params={self.parameter_count():,}\n)"
        )
