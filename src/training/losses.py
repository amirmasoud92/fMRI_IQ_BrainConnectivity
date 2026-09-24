"""Evidence lower bound, contrastive and disentanglement losses of the graph autoencoder."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Individual loss components
# ---------------------------------------------------------------------------

def compute_reconstruction_loss(
    fc_hat: Tensor,
    fc_target: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """Frobenius-norm-based MSE between predicted and true FC vectors."""
    loss = F.mse_loss(fc_hat, fc_target, reduction=reduction)
    return loss


def compute_kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
    """Analytical KL[ N(μ, σ²) ‖ N(0, I) ]."""
    kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    return kl_per_dim.sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# Combined ELBO loss
# ---------------------------------------------------------------------------

class ELBOLoss(nn.Module):
    """Evidence Lower BOund loss for BrainVAEGAT."""

    def __init__(
        self,
        beta: float = 1.0,
        pheno_weight: float = 0.1,
        free_bits: float = 0.0,
    ):
        super().__init__()
        self.beta         = beta
        self.pheno_weight = pheno_weight
        self.free_bits    = free_bits

    def forward(
        self,
        model_out: Dict[str, Tensor],
        batch: "Batch",
        beta_override: Optional[float] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Parameters
                ----------
                model_out    : dict from BrainVAEGAT.forward().
                batch        : PyG batch (contains fc_vec, pheno).
                beta_override: override self.beta (used during KL annealing).
        """
        beta = beta_override if beta_override is not None else self.beta

        # --- Reconstruction ---
        fc_target = batch.fc_vec          # (B·E,) concatenated across graphs
        # fc_vec in batch is stored as flat for all graphs; need to reshape
        # torch_geometric concatenates along dim-0; fc_vec is (total_edges,)
        # We stored the full upper-tri vector per graph so fc_vec has shape
        # (n_graphs * n_upper_tri) after collation – need to split by graph.
        # Simpler: fc_hat and batch.fc_vec are already aligned in this flat form
        # because DataLoader collates them as (B, n_upper_tri) when batch_size
        # is handled by regular DataLoader rather than PyG DataLoader.
        # Using PyG DataLoader the .fc_vec is (total, ) concatenated.
        # We'll re-stack using ptr:
        n_upper = model_out["fc_hat"].shape[1]  # inferred from decoder output
        # model_out["fc_hat"] : (B, n_upper) already correct from batched forward
        # batch.fc_vec was set in graph_builder as 1D; after PyG collation it is
        # (B * n_upper,). Reshape:
        B = model_out["fc_hat"].shape[0]
        fc_target_2d = batch.fc_vec.view(B, n_upper)

        recon_loss = compute_reconstruction_loss(model_out["fc_hat"], fc_target_2d)

        # --- KL divergence ---
        mu     = model_out["mu"]
        logvar = model_out["logvar"]

        if self.free_bits > 0.0:
            # Free-bits: clamp per-dim KL from below
            kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
            kl_per_dim = torch.clamp(kl_per_dim, min=self.free_bits)
            kl_loss    = kl_per_dim.sum(dim=-1).mean()
        else:
            kl_loss = compute_kl_divergence(mu, logvar)

        total = recon_loss + beta * kl_loss

        # --- Optional rest-branch KL (auxiliary, weighted 0.5× to avoid dominating) ---
        kl_rest_loss = torch.tensor(0.0, device=total.device)
        if "mu_rest" in model_out and "logvar_rest" in model_out:
            mu_r = model_out["mu_rest"]
            lv_r = model_out["logvar_rest"]
            if self.free_bits > 0.0:
                kl_r_per_dim = -0.5 * (1.0 + lv_r - mu_r.pow(2) - lv_r.exp())
                kl_r_per_dim = torch.clamp(kl_r_per_dim, min=self.free_bits)
                kl_rest_loss = kl_r_per_dim.sum(dim=-1).mean()
            else:
                kl_rest_loss = compute_kl_divergence(mu_r, lv_r)
            total = total + beta * 0.5 * kl_rest_loss

        # --- Optional phenotype prediction ---
        pheno_loss = torch.tensor(0.0, device=total.device)
        if (self.pheno_weight > 0.0
                and "pheno_hat" in model_out
                and hasattr(batch, "pheno")):
            pheno_target = batch.pheno.view(B, -1)
            # Mask out NaN phenotype entries
            mask = ~torch.isnan(pheno_target)
            if mask.any():
                pheno_loss = F.mse_loss(
                    model_out["pheno_hat"][mask], pheno_target[mask]
                )
                total = total + self.pheno_weight * pheno_loss

        metrics = {
            "loss_total":   total.item(),
            "loss_recon":   recon_loss.item(),
            "loss_kl":      kl_loss.item(),
            "loss_kl_rest": kl_rest_loss.item(),
            "loss_pheno":   pheno_loss.item(),
            "beta":         beta,
        }
        return total, metrics


# ---------------------------------------------------------------------------
# β-annealing scheduler
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# NT-Xent contrastive loss  (SimCLR, Chen et al. 2020)
# ---------------------------------------------------------------------------

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross-Entropy for multi-view contrastive
        learning on brain graphs.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        """
        Parameters
                ----------
                z1, z2 : (B, D) projected embeddings from two augmented views.
        """
        B  = z1.size(0)
        # Concatenate and L2-normalise
        z  = F.normalize(torch.cat([z1, z2], dim=0), dim=-1)  # (2B, D)

        # Full cosine-similarity matrix, scaled by temperature
        sim = torch.mm(z, z.T) / self.temperature              # (2B, 2B)

        # Zero out the diagonal (self-similarity is not a valid pair)
        sim.masked_fill_(
            torch.eye(2 * B, dtype=torch.bool, device=z.device), float("-inf")
        )

        # Positive pair for row i ∈ [0, B)  → column i+B, and vice-versa
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(B,        device=z.device),
        ])  # (2B,)

        return F.cross_entropy(sim, labels)


# ---------------------------------------------------------------------------
# DIP-VAE-II disentanglement regularizer  (Kumar et al. 2017)
# ---------------------------------------------------------------------------

class DIPLoss(nn.Module):
    """DIP-VAE-II covariance regularizer for disentangled latent representations."""

    def __init__(self, lambda_od: float = 10.0, lambda_d: float = 0.1):
        super().__init__()
        self.lambda_od = lambda_od
        self.lambda_d  = lambda_d

    def forward(self, mu: Tensor, logvar: Tensor) -> Tensor:
        B  = mu.size(0)
        mu_c = mu - mu.mean(dim=0, keepdim=True)             # centre the means

        # Covariance of the posterior mean across the batch
        cov_mu = (mu_c.T @ mu_c) / B                        # (D, D)

        # Expected posterior variance per dimension
        sigma2 = logvar.exp().mean(dim=0)                    # (D,)

        # Full approximate marginal covariance of q(z)
        cov_q  = cov_mu + torch.diag(sigma2)                 # (D, D)

        eye    = torch.eye(mu.size(1), device=mu.device)
        off_d  = cov_q * (1.0 - eye)
        on_d   = cov_q.diag() - 1.0

        return self.lambda_od * off_d.pow(2).sum() + self.lambda_d * on_d.pow(2).sum()


# ---------------------------------------------------------------------------
# β-annealing scheduler
# ---------------------------------------------------------------------------

class BetaScheduler:
    """Linear warm-up of β from beta_start to beta_end over warmup_epochs."""

    def __init__(
        self,
        beta_start: float = 0.0,
        beta_end: float = 4.0,
        warmup_epochs: int = 50,
    ):
        self.beta_start    = beta_start
        self.beta_end      = beta_end
        self.warmup_epochs = warmup_epochs

    def get_beta(self, epoch: int) -> float:
        if self.warmup_epochs <= 0:
            return self.beta_end
        frac = min(epoch / self.warmup_epochs, 1.0)
        return self.beta_start + frac * (self.beta_end - self.beta_start)
