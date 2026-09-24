"""fMRIPrep confound handling and parcel time-series extraction for AOMIC-ID1000."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from nilearn import datasets, image, maskers
from nilearn.connectome import ConnectivityMeasure
import nibabel as nib
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFOUND_36P = [
    "trans_x", "trans_x_derivative1", "trans_x_power2", "trans_x_derivative1_power2",
    "trans_y", "trans_y_derivative1", "trans_y_power2", "trans_y_derivative1_power2",
    "trans_z", "trans_z_derivative1", "trans_z_power2", "trans_z_derivative1_power2",
    "rot_x",   "rot_x_derivative1",   "rot_x_power2",   "rot_x_derivative1_power2",
    "rot_y",   "rot_y_derivative1",   "rot_y_power2",   "rot_y_derivative1_power2",
    "rot_z",   "rot_z_derivative1",   "rot_z_power2",   "rot_z_derivative1_power2",
    "csf",     "csf_derivative1",     "csf_power2",     "csf_derivative1_power2",
    "white_matter",            "white_matter_derivative1",
    "white_matter_power2",     "white_matter_derivative1_power2",
    "global_signal",           "global_signal_derivative1",
    "global_signal_power2",    "global_signal_derivative1_power2",
]


# ---------------------------------------------------------------------------
# Participants table
# ---------------------------------------------------------------------------

def load_participants(bids_root: str | Path, dropna_cols: Optional[List[str]] = None
                      ) -> pd.DataFrame:
    """Load participants.tsv and return a tidy DataFrame."""
    tsv = Path(bids_root) / "participants.tsv"
    df = pd.read_csv(tsv, sep="\t", na_values="n/a")
    df = df.set_index("participant_id")
    if dropna_cols:
        df = df.dropna(subset=dropna_cols)
    logger.info("Loaded %d participants from %s", len(df), tsv)
    return df


# ---------------------------------------------------------------------------
# Atlas
# ---------------------------------------------------------------------------

def load_schaefer_atlas(n_rois: int = 200, resolution_mm: int = 2) -> Tuple:
    """Fetch Schaefer 2018 atlas with 7-network labels."""
    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=n_rois, yeo_networks=7, resolution_mm=resolution_mm
    )
    # nilearn ≥ 0.10 returns atlas.maps as a file path string; load it explicitly
    atlas_img = nib.load(atlas.maps) if isinstance(atlas.maps, str) else atlas.maps
    labels = [lbl.decode() if isinstance(lbl, bytes) else lbl
              for lbl in atlas.labels]
    # Network name is the 3rd underscore-separated token: e.g.
    # "7Networks_LH_Vis_1" → "Vis"
    networks = []
    for lbl in labels:
        parts = lbl.split("_")
        networks.append(parts[2] if len(parts) > 2 else "Unknown")

    # Compute centroids in MNI space
    atlas_data = np.asarray(atlas_img.dataobj)
    affine = atlas_img.affine
    roi_ids = np.arange(1, n_rois + 1)
    coords = []
    for rid in roi_ids:
        vox = np.argwhere(atlas_data == rid)
        if len(vox) == 0:
            coords.append([0.0, 0.0, 0.0])
        else:
            vox_mean = vox.mean(axis=0)
            mni = nib.affines.apply_affine(affine, vox_mean)
            coords.append(mni.tolist())
    coords = np.array(coords)

    return atlas_img, labels, networks, coords


# ---------------------------------------------------------------------------
# Per-subject confound loading and scrubbing
# ---------------------------------------------------------------------------

def load_confounds(confound_file: str | Path,
                   fd_threshold: float = 0.5,
                   columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load fMRIPrep confound regressors and compute a scrubbing mask."""
    if columns is None:
        columns = CONFOUND_36P

    df = pd.read_csv(confound_file, sep="\t", na_values="n/a")

    # Build scrubbing mask before selecting columns
    if "framewise_displacement" in df.columns:
        fd = df["framewise_displacement"].fillna(0.0)
        # .to_numpy(copy=True) avoids the read-only view that pandas 2.x returns from .values
        keep = (fd <= fd_threshold).to_numpy(dtype=bool, copy=True)
        keep[0] = False  # always scrub first volume (no FD defined)
    else:
        keep = np.ones(len(df), dtype=bool)
        keep[0] = False

    # Select only requested columns (skip missing ones with a warning)
    avail = [c for c in columns if c in df.columns]
    missing = set(columns) - set(avail)
    if missing:
        logger.warning("Confound columns not found, skipping: %s", missing)

    conf_df = df[avail].fillna(0.0)
    return conf_df, keep


# ---------------------------------------------------------------------------
# Core extractor
# ---------------------------------------------------------------------------

class FMRIPreprocessor:
    """Extract ROI time series and FC matrices from fMRIPrep BOLD outputs."""

    def __init__(
        self,
        fmriprep_dir: str | Path,
        atlas_img,
        labels: List[str],
        tr: float = 2.0,
        fd_threshold: float = 0.5,
        high_pass: float = 0.01,
        low_pass: float = 0.10,
        smoothing_fwhm: Optional[float] = 6.0,
        min_volumes: int = 100,
        task: str = "moviewatching",
        space: str = "MNI152NLin2009cAsym",
    ):
        self.fmriprep_dir = Path(fmriprep_dir)
        self.atlas_img = atlas_img
        self.labels = labels
        self.tr = tr
        self.fd_threshold = fd_threshold
        self.high_pass = high_pass
        self.low_pass = low_pass
        self.smoothing_fwhm = smoothing_fwhm
        self.min_volumes = min_volumes
        self.task = task
        self.space = space

        self._masker = maskers.NiftiLabelsMasker(
            labels_img=atlas_img,
            standardize="zscore_sample",
            detrend=True,
            t_r=tr,
            high_pass=high_pass,
            low_pass=low_pass,
            smoothing_fwhm=smoothing_fwhm,
            verbose=0,
        )
        self._fc_measure = ConnectivityMeasure(kind="correlation", vectorize=False)

    def _bold_path(self, sub_id: str) -> Optional[Path]:
        fname = (f"{sub_id}_task-{self.task}_space-{self.space}"
                 f"_desc-preproc_bold.nii.gz")
        p = self.fmriprep_dir / sub_id / "func" / fname
        return p if p.exists() else None

    def _confound_path(self, sub_id: str) -> Optional[Path]:
        fname = (f"{sub_id}_task-{self.task}"
                 f"_desc-confounds_regressors.tsv")
        p = self.fmriprep_dir / sub_id / "func" / fname
        return p if p.exists() else None

    def process_subject(self, sub_id: str) -> Optional[Dict]:
        """Process a single subject. Returns dict or None if excluded."""
        bold_path = self._bold_path(sub_id)
        conf_path = self._confound_path(sub_id)

        if bold_path is None or conf_path is None:
            logger.warning("%s: BOLD or confound file not found – skipping.", sub_id)
            return None

        try:
            conf_df, keep = load_confounds(conf_path, self.fd_threshold)
            n_clean = int(keep.sum())
            if n_clean < self.min_volumes:
                logger.warning(
                    "%s: only %d clean volumes (< %d) – skipping.",
                    sub_id, n_clean, self.min_volumes
                )
                return None

            # Extract ROI time series with masker (applies detrend + filter)
            bold_img = image.load_img(str(bold_path))
            # Pass confounds to masker for simultaneous regression + filter
            conf_array = conf_df.values.astype(np.float32)
            time_series = self._masker.fit_transform(
                bold_img, confounds=conf_array
            )  # shape: (T, n_rois)

            # Apply scrubbing after masker (remove high-motion volumes)
            time_series = time_series[keep]

            # Fisher-z transform after computing Pearson correlation
            fc_matrix = np.array(np.corrcoef(time_series.T))  # ensure writable copy
            np.fill_diagonal(fc_matrix, 0.0)
            fc_matrix = np.clip(fc_matrix, -0.9999, 0.9999)
            fc_matrix_z = np.arctanh(fc_matrix)
            np.fill_diagonal(fc_matrix_z, 0.0)

            mean_fd = float(
                pd.read_csv(conf_path, sep="\t", na_values="n/a")
                  ["framewise_displacement"].fillna(0.0).mean()
            )

            return {
                "subject_id": sub_id,
                "time_series": time_series.astype(np.float32),
                "fc_matrix": fc_matrix.astype(np.float32),
                "fc_matrix_z": fc_matrix_z.astype(np.float32),
                "n_volumes": n_clean,
                "mean_fd": mean_fd,
            }

        except Exception as exc:
            logger.error("%s: preprocessing failed – %s", sub_id, exc)
            return None

    def compute_dynamic_fc(
        self,
        time_series: np.ndarray,
        window_size: int = 30,
        stride: int = 10,
        n_windows: int = 30,
    ) -> np.ndarray:
        """Sliding-window functional connectivity matrices (Fisher-z)."""
        T, N = time_series.shape
        starts = np.arange(0, T - window_size + 1, stride)
        windows = []
        for s in starts:
            seg = time_series[s : s + window_size]
            fc = np.corrcoef(seg.T).astype(np.float32)
            np.fill_diagonal(fc, 0.0)
            fc = np.clip(fc, -0.9999, 0.9999)
            fc_z = np.arctanh(fc)
            np.fill_diagonal(fc_z, 0.0)
            windows.append(fc_z)
            if len(windows) == n_windows:
                break

        # Zero-pad if fewer windows than requested
        while len(windows) < n_windows:
            windows.append(np.zeros((N, N), dtype=np.float32))

        return np.stack(windows, axis=0)  # (n_windows, N, N)

    def process_subject_rest(self, sub_id: str, task: str = "restingstate") -> Optional[Dict]:
        """Extract static FC from the resting-state run of a subject.

        Uses the same pipeline as process_subject() but with a different task
        identifier. Returns None if resting-state data is not available.
        """
        old_task = self.task
        self.task = task
        try:
            result = self.process_subject(sub_id)
        finally:
            self.task = old_task
        if result is not None:
            result["modality"] = "rest"
        return result

    def process_all(
        self,
        subject_ids: List[str],
        n_jobs: int = 1,
        show_progress: bool = True,
    ) -> Tuple[List[Dict], List[str]]:
        """Process all subjects, return (results_list, excluded_list)."""
        results, excluded = [], []

        it = tqdm(subject_ids, desc="Extracting FC", unit="sub") if show_progress else subject_ids
        for sub_id in it:
            res = self.process_subject(sub_id)
            if res is not None:
                results.append(res)
            else:
                excluded.append(sub_id)

        logger.info(
            "Processed %d subjects (%d excluded).", len(results), len(excluded)
        )
        return results, excluded
