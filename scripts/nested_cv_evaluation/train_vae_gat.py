#!/usr/bin/env python
"""
Train the graph variational autoencoder on ID1000 connectivity.

The model is withdrawn from the paper; the code is kept because Fig. S1 evaluates its reconstructions and its
cross-validated accuracy.

Usage
-----
    python scripts/nested_cv_evaluation/train_vae_gat.py
    python scripts/nested_cv_evaluation/train_vae_gat.py --config config/config.yaml --seed 42
"""

import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore", message="An issue occurred while importing")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from loguru import logger
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader

from src.data.preprocessing import load_participants
from src.data.graph_builder import build_graph_dataset
from src.data.dataset import BrainConnectivityDataset, DynamicBrainDataset
from src.models.vae_gat import BrainVAEGAT
from src.training.losses import ELBOLoss, BetaScheduler, NTXentLoss, DIPLoss
from src.training.trainer import Trainer


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--debug", action="store_true",
                   help="Use 50 subjects for a quick smoke test")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Load pre-computed FC matrices from HDF5
# ---------------------------------------------------------------------------

def load_fc_from_h5(h5_path: str) -> list:
    """Load all subjects' FC data from HDF5 into list of dicts."""
    results = []
    with h5py.File(h5_path, "r") as f:
        grp = f["subjects"]
        for sub_id in sorted(grp.keys()):
            sg = grp[sub_id]
            results.append({
                "subject_id":  sub_id,
                "fc_matrix":   sg["fc_matrix"][:],
                "fc_matrix_z": sg["fc_matrix_z"][:],
                "time_series": sg["time_series"][:],
                "n_volumes":   int(sg.attrs["n_volumes"]),
                "mean_fd":     float(sg.attrs["mean_fd"]),
            })
    logger.info(f"Loaded FC matrices for {len(results)} subjects from {h5_path}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg  = load_config(args.config)
    seed = args.seed or cfg["project"]["seed"]
    set_seed(seed)

    logger.remove()
    logger.add(sys.stdout, level="INFO")

    out_dir    = Path(cfg["paths"]["output_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    conn_dir   = Path(cfg["paths"]["connectivity_dir"])

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    h5_path = conn_dir / "fc_matrices.h5"
    if not h5_path.exists():
        logger.error(f"FC matrices not found at {h5_path}. Run script 01 first.")
        sys.exit(1)

    results      = load_fc_from_h5(h5_path)
    participants = load_participants(cfg["paths"]["bids_root"])

    if args.debug:
        results = results[:50]
        logger.warning(f"DEBUG mode: using only {len(results)} subjects.")

    # Load atlas metadata (includes MNI centroid coordinates)
    atlas_meta = np.load(conn_dir / "atlas_meta.npz", allow_pickle=True)
    networks   = atlas_meta["networks"].tolist()
    coords     = atlas_meta["coords"]   # (n_rois, 3) MNI coordinates

    # -----------------------------------------------------------------------
    # Determine whether dynamic FC and rest fusion are active
    # -----------------------------------------------------------------------
    m_cfg        = cfg["model"]
    dyn_cfg      = cfg.get("dynamic_fc", {})
    rest_cfg     = cfg.get("resting_state", {})

    use_coords   = cfg["graph"].get("use_coords", False)
    use_dynamic  = dyn_cfg.get("enabled", False) and m_cfg.get("use_dynamic", False)
    use_rest     = rest_cfg.get("enabled", False) and m_cfg.get("use_rest_fusion", False)

    logger.info(
        f"Features: coords={use_coords} | dynamic_fc={use_dynamic} | rest_fusion={use_rest}"
    )

    # -----------------------------------------------------------------------
    # Build static graph dataset (coords-augmented when use_coords=true)
    # -----------------------------------------------------------------------
    pheno_cols = cfg["phenotypes"]["continuous"]
    graphs, subject_ids = build_graph_dataset(
        results=results,
        participants=participants,
        phenotype_cols=pheno_cols,
        density=cfg["graph"]["density"],
        use_z_matrix=True,
        coords=coords if use_coords else None,
        signed_edges=int(m_cfg.get("edge_dim", 1)) == 2,
        log_euclidean=cfg["graph"].get("log_euclidean", False),
    )
    logger.info(f"Graph dataset: {len(graphs)} subjects, {graphs[0].num_nodes} ROIs, "
                f"node_feat_dim={graphs[0].x.shape[1]}")

    # -----------------------------------------------------------------------
    # Load dynamic FC windows from HDF5 (if enabled)
    # -----------------------------------------------------------------------
    dynamic_fc_all = None
    if use_dynamic:
        n_rois    = cfg["model"]["n_rois"]
        n_windows = int(dyn_cfg["n_windows"])
        logger.info("Loading dynamic FC windows from HDF5...")
        dfc_list = []
        with h5py.File(h5_path, "r") as f:
            for sub_id in subject_ids:
                sg = f["subjects"][sub_id]
                if "dynamic_fc_z" in sg:
                    dfc_list.append(sg["dynamic_fc_z"][:n_windows])
                else:
                    logger.warning(f"{sub_id}: dynamic_fc_z missing — using zeros. "
                                   f"Re-run script 01 with dynamic_fc.enabled=true.")
                    dfc_list.append(np.zeros((n_windows, n_rois, n_rois), dtype=np.float32))
        dynamic_fc_all = np.stack(dfc_list, axis=0)   # (N, T, n_rois, n_rois)
        logger.info(f"Dynamic FC shape: {dynamic_fc_all.shape}")

    # -----------------------------------------------------------------------
    # Load resting-state FC from HDF5 (if enabled)
    # -----------------------------------------------------------------------
    rest_fc_all = None
    if use_rest:
        n_rois = cfg["model"]["n_rois"]
        logger.info("Loading resting-state FC from HDF5...")
        rest_list = []
        with h5py.File(h5_path, "r") as f:
            rest_grp = f.get("subjects_rest")
            for sub_id in subject_ids:
                if rest_grp is not None and sub_id in rest_grp:
                    rest_list.append(rest_grp[sub_id]["fc_matrix_z"][:])
                else:
                    rest_list.append(np.zeros((n_rois, n_rois), dtype=np.float32))
        rest_fc_all = np.stack(rest_list, axis=0)   # (N, n_rois, n_rois)
        logger.info(f"Resting-state FC shape: {rest_fc_all.shape}")

    # -----------------------------------------------------------------------
    # Build dataset and dataloaders
    # -----------------------------------------------------------------------
    batch_size = cfg["training"]["batch_size"]

    if use_dynamic:
        dataset = DynamicBrainDataset(
            static_graphs=graphs,
            subject_ids=subject_ids,
            dynamic_fc=dynamic_fc_all,
            rest_fc=rest_fc_all,
        )
        train_ds, val_ds, test_ds = dataset.split(
            train_frac=cfg["training"]["train_frac"],
            val_frac=cfg["training"]["val_frac"],
            seed=seed,
        )
        logger.info(f"Split: train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}")
        logger.info(f"{dataset}")

        collate = DynamicBrainDataset.collate
        train_loader = TorchDataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                       num_workers=0, pin_memory=False,
                                       collate_fn=collate)
        val_loader   = TorchDataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                                       num_workers=0, collate_fn=collate)
        test_loader  = TorchDataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                                       num_workers=0, collate_fn=collate)
        all_loader   = TorchDataLoader(dataset,  batch_size=batch_size, shuffle=False,
                                       num_workers=0, collate_fn=collate)
    else:
        dataset = BrainConnectivityDataset(graphs, subject_ids)
        logger.info(f"{dataset}")
        train_ds, val_ds, test_ds = dataset.split(
            train_frac=cfg["training"]["train_frac"],
            val_frac=cfg["training"]["val_frac"],
            seed=seed,
        )
        logger.info(f"Split: train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=0, pin_memory=False)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                                  num_workers=0)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                                  num_workers=0)
        all_loader   = DataLoader(dataset,  batch_size=batch_size, shuffle=False,
                                  num_workers=0)

    # -----------------------------------------------------------------------
    # Build model
    # -----------------------------------------------------------------------
    n_rois        = m_cfg["n_rois"]
    node_feat_dim = m_cfg["node_feat_dim"]   # 203 when use_coords=true, else 200

    c_cfg    = cfg.get("contrastive", {})
    proj_dim = int(c_cfg.get("proj_dim", 0)) if c_cfg.get("enabled", False) else 0

    model = BrainVAEGAT(
        n_rois=n_rois,
        node_feat_dim=node_feat_dim,
        hidden_dims=m_cfg["hidden_dims"],
        latent_dim=m_cfg["latent_dim"],
        n_heads=m_cfg["n_heads"],
        decoder_type=m_cfg["decoder_type"],
        decoder_hidden=m_cfg["decoder_hidden"],
        gt_n_layers=int(m_cfg.get("gt_decoder_layers", 2)),
        dropout=m_cfg["dropout"],
        edge_dim=int(m_cfg.get("edge_dim", 1)),
        n_phenotypes=len(pheno_cols),
        proj_dim=proj_dim,
        use_dynamic=use_dynamic,
        temporal_hidden=int(m_cfg.get("temporal_hidden", 128)),
        temporal_layers=int(m_cfg.get("temporal_layers", 2)),
        use_rest_fusion=use_rest,
    )
    logger.info(f"Model: {model}")

    # -----------------------------------------------------------------------
    # Loss + scheduler
    # -----------------------------------------------------------------------
    t_cfg = cfg["training"]
    loss_fn = ELBOLoss(
        beta=t_cfg["beta_end"],
        pheno_weight=t_cfg.get("pheno_weight", 0.10),
        free_bits=t_cfg.get("free_bits", 0.5),
    )
    beta_sched = BetaScheduler(
        beta_start=t_cfg["beta_start"],
        beta_end=t_cfg["beta_end"],
        warmup_epochs=t_cfg["beta_warmup_epochs"],
    )

    # Optional contrastive loss
    contrastive_fn = None
    if c_cfg.get("enabled", False) and proj_dim > 0:
        contrastive_fn = NTXentLoss(temperature=float(c_cfg.get("temperature", 0.5)))
        logger.info(f"NT-Xent contrastive loss ENABLED (tau={c_cfg['temperature']}, "
                    f"lam={c_cfg.get('weight', 0.5)}, proj_dim={proj_dim})")

    # Optional DIP-VAE-II disentanglement regularizer
    dip_cfg = cfg.get("disentanglement", {})
    dip_fn  = None
    if dip_cfg.get("dip_enabled", False):
        dip_fn = DIPLoss(
            lambda_od=float(dip_cfg.get("dip_lambda_od", 10.0)),
            lambda_d =float(dip_cfg.get("dip_lambda_d",  0.1)),
        )
        logger.info(f"DIP-VAE-II regularizer ENABLED (lam_od={dip_cfg['dip_lambda_od']}, "
                    f"lam_d={dip_cfg['dip_lambda_d']}, weight={dip_cfg.get('dip_weight', 0.01)})")

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        beta_sched=beta_sched,
        lr=t_cfg["learning_rate"],
        weight_decay=t_cfg["weight_decay"],
        clip_grad=t_cfg["clip_grad_norm"],
        patience=t_cfg["patience"],
        amp=t_cfg["amp"],
        output_dir=models_dir,
        lr_scheduler=t_cfg["lr_scheduler"],
        n_epochs=t_cfg["n_epochs"],
        contrastive_fn=contrastive_fn,
        contrastive_weight=float(c_cfg.get("weight", 0.5)),
        dip_fn=dip_fn,
        dip_weight=float(dip_cfg.get("dip_weight", 0.01)),
        aug_edge_drop=float(c_cfg.get("aug_edge_drop", 0.10)),
        aug_feat_noise=float(c_cfg.get("aug_feat_noise", 0.02)),
    )

    # Delete stale attention cache so script 03 recomputes with new model
    attn_cache = out_dir / "attention_weights.npz"
    if attn_cache.exists():
        attn_cache.unlink()

    history = trainer.fit(train_loader, val_loader)

    # -----------------------------------------------------------------------
    # Evaluate on test set
    # -----------------------------------------------------------------------
    trainer.load_best()
    fc_true, fc_recon = trainer.get_reconstructions(test_loader)

    from src.evaluation.metrics import reconstruction_quality
    rq = reconstruction_quality(fc_true, fc_recon)
    logger.info(f"Test reconstruction quality: r={rq['mean_pearson_r']:.3f} ± {rq['std_pearson_r']:.3f}")

    # Save reconstructions for Figure 8 in analysis script
    out_dir.mkdir(parents=True, exist_ok=True)
    recon_path = out_dir / "fc_reconstructions.npz"
    np.savez(recon_path, fc_true=fc_true, fc_recon=fc_recon)
    logger.info(f"Reconstructions saved -> {recon_path}")

    # -----------------------------------------------------------------------
    # Extract latent embeddings for all subjects
    # -----------------------------------------------------------------------
    z_all, ids_all = trainer.get_latent_embeddings(all_loader)
    logger.info(f"Latent embeddings: {z_all.shape}")

    # Save
    emb_path = out_dir / "latent_embeddings.npz"
    np.savez(
        emb_path,
        z=z_all,
        subject_ids=np.array(ids_all),
        pheno_cols=np.array(pheno_cols),
    )
    logger.info(f"Latent embeddings saved -> {emb_path}")

    # Save test split IDs
    split_info = {
        "train": train_ds.subject_ids,
        "val":   val_ds.subject_ids,
        "test":  test_ds.subject_ids,
    }
    with open(models_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    # Final report
    logger.info("=" * 60)
    logger.info("Training complete.")
    logger.info(f"  Mode:               {'dynamic+rest' if use_rest else 'dynamic' if use_dynamic else 'static'}")
    logger.info(f"  Best val loss:      {trainer._best_val_loss:.4f}")
    logger.info(f"  Test recon r:       {rq['mean_pearson_r']:.3f}")
    logger.info(f"  Checkpoints:        {models_dir}")
    logger.info(f"  Embeddings:         {emb_path}")


if __name__ == "__main__":
    main()
