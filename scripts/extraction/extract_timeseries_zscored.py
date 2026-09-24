#!/usr/bin/env python
"""
Extract z-scored parcel time series and connectivity matrices from the fMRIPrep derivatives.

High-motion frames are removed after cleaning. Used for the decoder, embedding and co-skewness
comparisons; the primary analyses use the percent-signal-change extraction.

Usage
-----
    python scripts/extraction/extract_timeseries_zscored.py

Output
------
    outputs/connectivity/fc_matrices.h5
      /subjects/<sub-id>/fc_matrix       (200, 200) Pearson-r
      /subjects/<sub-id>/fc_matrix_z     (200, 200) Fisher-z
      /subjects/<sub-id>/time_series     (T, 200)
      /subjects/<sub-id>/dynamic_fc_z    (n_windows, 200, 200) sliding-window Fisher-z [if enabled]
      /subjects/<sub-id>/attrs           n_volumes, mean_fd
      /subjects_rest/<sub-id>/fc_matrix_z (200, 200) Fisher-z resting-state [if enabled]
    outputs/connectivity/atlas_meta.npz
      labels, networks, coords (MNI centroids)
    outputs/connectivity/processing_log.csv
"""

import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore", message="An issue occurred while importing")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from tqdm import tqdm

from src.data.preprocessing import FMRIPreprocessor, load_participants, load_schaefer_atlas


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()

    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(
        Path(cfg["paths"]["output_dir"]) / "logs" / "extract_connectivity.log",
        level="DEBUG", rotation="10 MB"
    )
    Path(cfg["paths"]["output_dir"], "logs").mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load atlas
    # -----------------------------------------------------------------------
    logger.info("Loading Schaefer-200 atlas...")
    atlas_img, labels, networks, coords = load_schaefer_atlas(
        n_rois=cfg["atlas"]["n_rois"],
        resolution_mm=cfg["atlas"]["resolution_mm"]
    )
    logger.info(f"Atlas loaded: {len(labels)} ROIs across {len(set(networks))} networks")

    # Save atlas metadata
    out_dir = Path(cfg["paths"]["connectivity_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "atlas_meta.npz",
        labels=np.array(labels),
        networks=np.array(networks),
        coords=coords
    )
    logger.info(f"Atlas metadata saved to {out_dir / 'atlas_meta.npz'}")

    # -----------------------------------------------------------------------
    # Load participants
    # -----------------------------------------------------------------------
    participants = load_participants(cfg["paths"]["bids_root"])
    logger.info(f"Participants loaded: {len(participants)} subjects")

    # Discover subject IDs from fmriprep directory
    fmriprep_dir = Path(cfg["paths"]["fmriprep_dir"])
    sub_dirs = sorted([
        d.name for d in fmriprep_dir.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    ])
    logger.info(f"Found {len(sub_dirs)} subject directories in fmriprep")

    n_max = cfg["dataset"].get("n_subjects_max")
    if n_max:
        sub_dirs = sub_dirs[:n_max]
        logger.info(f"Limiting to {n_max} subjects (n_subjects_max set)")

    # -----------------------------------------------------------------------
    # Dynamic FC config
    # -----------------------------------------------------------------------
    dyn_cfg     = cfg.get("dynamic_fc", {})
    dyn_enabled = dyn_cfg.get("enabled", False)
    if dyn_enabled:
        tr          = cfg["dataset"]["tr"]
        window_size = int(dyn_cfg["window_size_s"] / tr)   # TRs
        stride      = int(dyn_cfg["stride_s"] / tr)        # TRs
        n_windows   = int(dyn_cfg["n_windows"])
        logger.info(
            f"Dynamic FC ENABLED: window={window_size} TRs, "
            f"stride={stride} TRs, n_windows={n_windows}"
        )

    # Resting-state config
    rest_cfg     = cfg.get("resting_state", {})
    rest_enabled = rest_cfg.get("enabled", False)
    rest_task    = rest_cfg.get("task", "restingstate")
    if rest_enabled:
        logger.info(f"Resting-state FC ENABLED: task={rest_task}")

    # -----------------------------------------------------------------------
    # Initialise preprocessor
    # -----------------------------------------------------------------------
    preprocessor = FMRIPreprocessor(
        fmriprep_dir=fmriprep_dir,
        atlas_img=atlas_img,
        labels=labels,
        tr=cfg["dataset"]["tr"],
        fd_threshold=cfg["preprocessing"]["fd_threshold"],
        high_pass=cfg["preprocessing"]["high_pass"],
        low_pass=cfg["preprocessing"]["low_pass"],
        smoothing_fwhm=cfg["preprocessing"]["smoothing_fwhm"],
        min_volumes=cfg["preprocessing"]["min_volumes_after_scrub"],
        task=cfg["dataset"]["task"],
        space=cfg["dataset"]["space"],
    )

    # -----------------------------------------------------------------------
    # Process subjects (movie-watching)
    # -----------------------------------------------------------------------
    logger.info(f"Processing {len(sub_dirs)} subjects (task={cfg['dataset']['task']})...")
    results, excluded = preprocessor.process_all(
        subject_ids=sub_dirs, show_progress=True
    )

    # -----------------------------------------------------------------------
    # Save to HDF5
    # -----------------------------------------------------------------------
    h5_path  = out_dir / "fc_matrices.h5"
    log_rows = []

    with h5py.File(h5_path, "w") as f:
        grp = f.create_group("subjects")

        for res in tqdm(results, desc="Writing HDF5", unit="sub"):
            sub = res["subject_id"]
            sg  = grp.create_group(sub)
            sg.create_dataset("fc_matrix",   data=res["fc_matrix"],
                              compression="gzip", compression_opts=4)
            sg.create_dataset("fc_matrix_z", data=res["fc_matrix_z"],
                              compression="gzip", compression_opts=4)
            sg.create_dataset("time_series", data=res["time_series"],
                              compression="gzip", compression_opts=4)
            sg.attrs["n_volumes"] = res["n_volumes"]
            sg.attrs["mean_fd"]   = res["mean_fd"]

            # --- Dynamic sliding-window FC ---
            if dyn_enabled:
                dfc = preprocessor.compute_dynamic_fc(
                    res["time_series"],
                    window_size=window_size,
                    stride=stride,
                    n_windows=n_windows,
                )  # (n_windows, N, N)
                sg.create_dataset("dynamic_fc_z", data=dfc,
                                  compression="gzip", compression_opts=4)

            log_rows.append({
                "subject_id": sub,
                "status":     "included",
                "n_volumes":  res["n_volumes"],
                "mean_fd":    res["mean_fd"],
            })

        # --- Resting-state FC (separate group, only subjects with rest data) ---
        if rest_enabled:
            rest_grp    = f.create_group("subjects_rest")
            n_rest_ok   = 0
            n_rest_miss = 0
            logger.info(f"Extracting resting-state FC for {len(results)} subjects...")
            for res in tqdm(results, desc="Resting-state FC", unit="sub"):
                sub      = res["subject_id"]
                rest_res = preprocessor.process_subject_rest(sub, task=rest_task)
                if rest_res is not None:
                    rsg = rest_grp.create_group(sub)
                    rsg.create_dataset("fc_matrix_z", data=rest_res["fc_matrix_z"],
                                       compression="gzip", compression_opts=4)
                    rsg.attrs["n_volumes"] = rest_res["n_volumes"]
                    rsg.attrs["mean_fd"]   = rest_res["mean_fd"]
                    n_rest_ok += 1
                else:
                    n_rest_miss += 1
            logger.info(
                f"Resting-state: {n_rest_ok} extracted, {n_rest_miss} missing/excluded."
            )

    for sub in excluded:
        log_rows.append({"subject_id": sub, "status": "excluded",
                         "n_volumes": np.nan, "mean_fd": np.nan})

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(out_dir / "processing_log.csv", index=False)

    logger.info("=" * 60)
    logger.info(f"Saved FC matrices for {len(results)} subjects → {h5_path}")
    logger.info(f"Excluded: {len(excluded)} subjects")
    logger.info(f"Processing log: {out_dir / 'processing_log.csv'}")

    # -----------------------------------------------------------------------
    # Quick QC summary
    # -----------------------------------------------------------------------
    included_df = log_df[log_df["status"] == "included"]
    fd_mean, fd_std   = included_df["mean_fd"].mean(), included_df["mean_fd"].std()
    vol_mean, vol_std = included_df["n_volumes"].mean(), included_df["n_volumes"].std()
    logger.info(
        f"QC summary — mean FD: {fd_mean:.3f} ± {fd_std:.3f} mm | "
        f"volumes: {vol_mean:.0f} ± {vol_std:.0f}"
    )
    logger.info("Extraction complete.")


if __name__ == "__main__":
    main()
