"""
Loader for the two region time-series extractions: percent signal change with every frame retained,
and the earlier z-scored extraction with high-motion frames removed.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

VALID = ("legacy", "psc")


def load_series(conn, series: str = "legacy"):
    """Return (ctx, sub) dicts mapping subject id -> (T, R) float array."""
    if series not in VALID:
        raise ValueError(f"series must be one of {VALID}, got {series!r}")
    conn = Path(conn)
    ctx, sub = {}, {}
    if series == "psc":
        path = conn / "unscrubbed_ts.h5"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing -- run scripts/extraction/extract_timeseries_psc.py first")
        with h5py.File(path, "r") as f:
            for s in f["subjects"]:
                x = f["subjects"][s]["time_series"][:]
                ctx[s] = x
                sub[s] = np.zeros((x.shape[0], 0), dtype=x.dtype)
    else:
        with h5py.File(conn / "fc_matrices.h5", "r") as f:
            for s in f["subjects"]:
                ctx[s] = f["subjects"][s]["time_series"][:]
        with h5py.File(conn / "subcortical_ts.h5", "r") as f:
            for s in f["subjects"]:
                sub[s] = f["subjects"][s]["time_series"][:]
    return ctx, sub


def roi_networks(conn, n_roi=215):
    """Yeo-7 network label for each extracted ROI, correctly aligned."""
    import numpy as np
    meta = np.load(Path(conn) / "atlas_meta.npz", allow_pickle=True)
    raw = [str(x) for x in meta["networks"]]
    n_parcel = len(meta["coords"])                 # 200 real parcels
    nets = []
    for n in raw[1:n_parcel + 1]:                  # skip background at index 0
        parts = n.split("_")
        nets.append(parts[2] if len(parts) > 2 else n)
    return (nets + ["Subcortex"] * (n_roi - len(nets)))[:n_roi]
