"""Training loop for the graph autoencoder, with mixed precision and early stopping."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import contextlib
import warnings

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from loguru import logger

from ..models.vae_gat import BrainVAEGAT
from .losses import ELBOLoss, BetaScheduler, NTXentLoss, DIPLoss
from ..data.augmentation import augment_batch


class Trainer:
    """Full training loop for BrainVAEGAT."""

    def __init__(
        self,
        model: BrainVAEGAT,
        loss_fn: ELBOLoss,
        beta_sched: BetaScheduler,
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        clip_grad: Optional[float] = 1.0,
        patience: int = 30,
        amp: bool = True,
        device: Optional[torch.device] = None,
        output_dir: str = "outputs/models",
        lr_scheduler: str = "cosine",
        n_epochs: int = 300,
        contrastive_fn: Optional[NTXentLoss] = None,
        contrastive_weight: float = 0.5,
        dip_fn: Optional[DIPLoss] = None,
        dip_weight: float = 0.01,
        aug_edge_drop: float = 0.10,
        aug_feat_noise: float = 0.02,
    ):
        self.model       = model
        self.loss_fn     = loss_fn
        self.beta_sched  = beta_sched
        self.clip_grad   = clip_grad
        self.patience    = patience
        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_epochs    = n_epochs

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        logger.info(f"Training on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        self.optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        if lr_scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=n_epochs, eta_min=lr * 1e-3
            )
        elif lr_scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=10
            )
        else:
            self.scheduler = None

        self.use_amp = amp and self.device.type == "cuda"
        self.scaler  = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.contrastive_fn     = contrastive_fn
        self.contrastive_weight = contrastive_weight
        self.dip_fn             = dip_fn
        self.dip_weight         = dip_weight
        self.aug_edge_drop      = aug_edge_drop
        self.aug_feat_noise     = aug_feat_noise

        self._history: List[Dict] = []
        self._best_val_loss = float("inf")
        self._patience_ctr  = 0

    # -----------------------------------------------------------------------
    # Single epoch
    # -----------------------------------------------------------------------

    def _run_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        train: bool,
    ) -> Dict[str, float]:
        self.model.train(train)
        beta = self.beta_sched.get_beta(epoch)

        epoch_metrics: Dict[str, List[float]] = {
            "loss_total": [], "loss_recon": [], "loss_kl": [],
            "loss_kl_rest": [], "loss_pheno": [], "loss_contrastive": [], "loss_dip": [],
        }

        use_contrastive = (
            train
            and self.contrastive_fn is not None
            and hasattr(self.model, "proj_head")
            and self.model.proj_head is not None
        )
        use_dip = train and self.dip_fn is not None

        for batch in loader:
            # Unpack tuple batches from DynamicBrainDataset, or handle plain PyG batch
            if isinstance(batch, (tuple, list)):
                static_batch, dfc_batch, rest_batch = batch
                static_batch = static_batch.to(self.device)
                dfc_batch  = (dfc_batch.to(self.device)
                              if isinstance(dfc_batch, torch.Tensor) and dfc_batch.numel() > 0
                              else None)
                rest_batch = (rest_batch.to(self.device)
                              if isinstance(rest_batch, torch.Tensor) and rest_batch.numel() > 0
                              else None)
                graph_batch = static_batch
            else:
                graph_batch = batch.to(self.device)
                dfc_batch   = None
                rest_batch  = None

            if train:
                self.optimizer.zero_grad(set_to_none=True)

            # Validation runs without gradients — prevents holding 30 window activation
            # graphs in GPU memory simultaneously (which caused OOM during eval).
            _grad_ctx = contextlib.nullcontext() if train else torch.no_grad()
            with _grad_ctx, torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                if dfc_batch is not None:
                    out = self.model.forward_dynamic(graph_batch, dfc_batch, rest_batch)
                else:
                    out = self.model(graph_batch)

                loss, metrics = self.loss_fn(out, graph_batch, beta_override=beta)

                c_loss_val = dip_loss_val = 0.0

                # --- NT-Xent contrastive loss via graph augmentation ---
                if use_contrastive:
                    aug_x1, aug_ei1, aug_ea1 = augment_batch(
                        graph_batch, edge_drop_rate=self.aug_edge_drop,
                        feat_noise_std=self.aug_feat_noise,
                    )
                    aug_x2, aug_ei2, aug_ea2 = augment_batch(
                        graph_batch, edge_drop_rate=self.aug_edge_drop,
                        feat_noise_std=self.aug_feat_noise,
                    )
                    mu1, _, _ = self.model.encoder(aug_x1, aug_ei1, aug_ea1, graph_batch.batch)
                    mu2, _, _ = self.model.encoder(aug_x2, aug_ei2, aug_ea2, graph_batch.batch)
                    z1_proj   = self.model.proj_head(mu1)
                    z2_proj   = self.model.proj_head(mu2)
                    c_loss    = self.contrastive_fn(z1_proj, z2_proj)
                    loss      = loss + self.contrastive_weight * c_loss
                    c_loss_val = c_loss.item()

                # --- DIP-VAE-II disentanglement regularizer ---
                if use_dip:
                    dip_loss     = self.dip_fn(out["mu"], out["logvar"])
                    loss         = loss + self.dip_weight * dip_loss
                    dip_loss_val = dip_loss.item()

            if train:
                self.scaler.scale(loss).backward()
                if self.clip_grad:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()

            for k in ["loss_total", "loss_recon", "loss_kl", "loss_kl_rest", "loss_pheno"]:
                epoch_metrics[k].append(metrics[k])
            epoch_metrics["loss_contrastive"].append(c_loss_val)
            epoch_metrics["loss_dip"].append(dip_loss_val)

        return {k: float(np.mean(v)) for k, v in epoch_metrics.items()}

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> List[Dict]:
        """Train for self.n_epochs epochs with early stopping.

        Returns
        -------
        history : list of per-epoch metric dicts.
        """
        logger.info(f"Starting training for {self.n_epochs} epochs.")
        t0 = time.time()

        # Suppress PyTorch's spurious "lr_scheduler before optimizer" warning -
        # our ordering is correct (optimizer.step inside _run_epoch, scheduler.step after).
        warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step\\(\\)`")

        for epoch in range(1, self.n_epochs + 1):
            train_m = self._run_epoch(train_loader, epoch, train=True)
            val_m   = self._run_epoch(val_loader,   epoch, train=False)

            if isinstance(self.scheduler, CosineAnnealingLR):
                self.scheduler.step()
            elif isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_m["loss_total"])

            lr_now = self.optimizer.param_groups[0]["lr"]
            row = {
                "epoch":     epoch,
                "lr":        lr_now,
                "beta":      self.beta_sched.get_beta(epoch),
                **{f"train_{k}": v for k, v in train_m.items()},
                **{f"val_{k}":   v for k, v in val_m.items()},
            }
            self._history.append(row)

            if epoch % 10 == 0 or epoch == 1:
                c_str = (f" | c={train_m['loss_contrastive']:.3f}"
                         if train_m["loss_contrastive"] > 0 else "")
                d_str = (f" | dip={train_m['loss_dip']:.3f}"
                         if train_m["loss_dip"] > 0 else "")
                logger.info(
                    f"Epoch {epoch:3d} | train={train_m['loss_total']:.4f} | "
                    f"val={val_m['loss_total']:.4f} | recon={val_m['loss_recon']:.4f} | "
                    f"kl={val_m['loss_kl']:.4f} | beta={row['beta']:.3f} | "
                    f"lr={lr_now:.2e}{c_str}{d_str}"
                )

            # Early stopping & checkpointing
            # Use recon+pheno only - total loss grows during beta warmup, which
            # would otherwise fire early stopping before the model has trained.
            val_loss = val_m["loss_recon"] + val_m["loss_pheno"]
            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._patience_ctr  = 0
                self._save_checkpoint(epoch, val_loss, tag="best")
            else:
                self._patience_ctr += 1
                if self._patience_ctr >= self.patience:
                    logger.info(
                        f"Early stopping at epoch {epoch} (no improvement for {self.patience} epochs)."
                    )
                    break

        elapsed = time.time() - t0
        logger.info(f"Training complete. Time: {elapsed:.1f} s")
        self._save_history()
        return self._history

    # -----------------------------------------------------------------------
    # Checkpoint helpers
    # -----------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, val_loss: float, tag: str = "best"):
        ckpt = {
            "epoch":          epoch,
            "val_loss":       val_loss,
            "model_state":    self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }
        path = self.output_dir / f"checkpoint_{tag}.pt"
        torch.save(ckpt, path)

    def load_best(self):
        """Load weights from the best checkpoint saved during training."""
        path = self.output_dir / "checkpoint_best.pt"
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state"])
        logger.info(f"Loaded best checkpoint from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f}).")

    def _save_history(self):
        path = self.output_dir / "training_history.json"
        with open(path, "w") as f:
            json.dump(self._history, f, indent=2)

    @property
    def history(self) -> List[Dict]:
        return self._history

    # -----------------------------------------------------------------------
    # Inference helpers
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def get_latent_embeddings(
        self,
        loader: DataLoader,
    ) -> Tuple[np.ndarray, List[str]]:
        """Encode all subjects -> return (N, latent_dim) array and IDs."""
        self.model.eval()
        all_z, all_ids = [], []
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                static_batch, dfc_batch, rest_batch = batch
                static_batch = static_batch.to(self.device)
                dfc_batch  = (dfc_batch.to(self.device)
                              if isinstance(dfc_batch, torch.Tensor) and dfc_batch.numel() > 0
                              else None)
                rest_batch = (rest_batch.to(self.device)
                              if isinstance(rest_batch, torch.Tensor) and rest_batch.numel() > 0
                              else None)
                out = self.model.forward_dynamic(static_batch, dfc_batch, rest_batch)
                # out["z"] in eval mode = z_static + gate*proj(mu_dyn): deterministic fused embedding
                mu = out["z"]
                all_ids.extend(static_batch.subject_id)
            else:
                graph_batch = batch.to(self.device)
                mu, _, _ = self.model.encode(graph_batch)
                all_ids.extend(graph_batch.subject_id)
            all_z.append(mu.cpu().numpy())
        return np.vstack(all_z), all_ids

    @torch.no_grad()
    def get_reconstructions(
        self,
        loader: DataLoader,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (fc_true, fc_reconstructed) arrays of shape (N, n_upper_tri)."""
        self.model.eval()
        fc_true, fc_recon = [], []
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                static_batch, dfc_batch, rest_batch = batch
                static_batch = static_batch.to(self.device)
                dfc_batch  = (dfc_batch.to(self.device)
                              if isinstance(dfc_batch, torch.Tensor) and dfc_batch.numel() > 0
                              else None)
                rest_batch = (rest_batch.to(self.device)
                              if isinstance(rest_batch, torch.Tensor) and rest_batch.numel() > 0
                              else None)
                out         = self.model.forward_dynamic(static_batch, dfc_batch, rest_batch)
                graph_batch = static_batch
            else:
                graph_batch = batch.to(self.device)
                out         = self.model(graph_batch)
            B    = out["fc_hat"].shape[0]
            n_up = out["fc_hat"].shape[1]
            fc_recon.append(out["fc_hat"].cpu().numpy())
            fc_true.append(graph_batch.fc_vec.view(B, n_up).cpu().numpy())
        return np.vstack(fc_true), np.vstack(fc_recon)
