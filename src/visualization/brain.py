"""Render Schaefer-200 parcel values on the inflated fsaverage5 surface."""
from __future__ import annotations

import io
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from src.visualization.journal import FIG_ROOT, NETWORK

N_CORTICAL = 200
VIEWS = (("left", "lateral"), ("left", "medial"), ("right", "medial"), ("right", "lateral"))
CACHE = FIG_ROOT / ".cache" / "schaefer200_fsaverage5_labels.npz"
SULC_DARKNESS = 0.55


def _fill_label_gaps(labels: np.ndarray, faces: np.ndarray, rings: int = 2) -> np.ndarray:
    """Unlabelled vertices take the most frequent label of their labelled neighbours, `rings` times over."""
    from scipy import sparse
    n = labels.size
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    adj = sparse.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    adj = ((adj + adj.T) > 0).tocsr()
    out = labels.copy()
    for _ in range(rings):
        new = out.copy()
        for v in np.flatnonzero(out == 0):
            nb = out[adj.indices[adj.indptr[v]:adj.indptr[v + 1]]]
            nb = nb[nb > 0]
            if nb.size:
                new[v] = np.bincount(nb).argmax()
        out = new
    return out


_STATE: dict = {}


def fsaverage():
    if "fs" not in _STATE:
        from nilearn import datasets
        _STATE["fs"] = datasets.fetch_surf_fsaverage("fsaverage5")
    return _STATE["fs"]


def vertex_labels() -> dict[str, np.ndarray]:
    """Schaefer label (1-200, 0 = none) per fsaverage5 vertex, for each hemisphere."""
    if "labels" in _STATE:
        return _STATE["labels"]
    if CACHE.exists():
        z = np.load(CACHE)
        _STATE["labels"] = {"left": z["left"], "right": z["right"]}
        return _STATE["labels"]
    from nilearn import datasets, surface
    sch = datasets.fetch_atlas_schaefer_2018(n_rois=N_CORTICAL, yeo_networks=7)
    fs = fsaverage()
    labels = {}
    for hemi in ("left", "right"):
        tex = surface.vol_to_surf(sch.maps, fs[f"pial_{hemi}"], inner_mesh=fs[f"white_{hemi}"],
                                  interpolation="nearest_most_frequent", n_samples=10)
        faces = np.asarray(surface.load_surf_mesh(fs[f"pial_{hemi}"]).faces)
        labels[hemi] = _fill_label_gaps(np.rint(tex).astype(int), faces)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **labels)
    _STATE["labels"] = labels
    return labels


def parcel_networks() -> list[str]:
    """Yeo-7 network of each of the 200 Schaefer parcels, in atlas order."""
    from nilearn import datasets
    names = datasets.fetch_atlas_schaefer_2018(n_rois=N_CORTICAL, yeo_networks=7).labels
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    names = [n for n in names if n != "Background"]
    assert len(names) == N_CORTICAL, len(names)
    return [n.split("_")[2] for n in names]


def textures(values: np.ndarray) -> dict[str, np.ndarray]:
    """Parcel values (first 200 entries used) -> per-vertex textures, NaN where no parcel."""
    values = np.asarray(values, float)[:N_CORTICAL]
    out = {}
    for hemi, lab in vertex_labels().items():
        t = np.full(lab.shape, np.nan)
        ok = lab > 0
        t[ok] = values[lab[ok] - 1]
        out[hemi] = t
    return out


def _crop(fig) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_STATE.get("dpi", 300), facecolor="white")
    plt.close(fig)
    buf.seek(0)
    img = plt.imread(buf)
    mask = (img[..., :3] < 0.985).any(-1)
    ys, xs = np.where(mask)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def render(texture: np.ndarray, hemi: str, view: str, cmap, vmin: float, vmax: float,
           dpi: int = 300) -> np.ndarray:
    """One inflated view of a continuous parcel map, cropped RGBA."""
    from nilearn import plotting
    fs = fsaverage()
    fig = plt.figure(figsize=(3.2, 2.6))
    ax = fig.add_subplot(111, projection="3d")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plotting.plot_surf_stat_map(fs[f"infl_{hemi}"], texture, hemi=hemi, view=view, bg_map=fs[f"sulc_{hemi}"],
                                    axes=ax, colorbar=False, cmap=cmap, vmin=vmin, vmax=vmax,
                                    symmetric_cbar=False, bg_on_data=True)
    ax.set_box_aspect(None, zoom=1.45)
    _STATE["dpi"] = dpi
    return _crop(fig)


def render_networks(hemi: str, view: str, dpi: int = 300) -> np.ndarray:
    """One inflated view of the Schaefer-200 parcellation coloured by Yeo-7 network, cropped RGBA."""
    from nilearn import plotting
    order = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
    code = np.array([order.index(n) + 1 for n in parcel_networks()], float)
    lab = vertex_labels()[hemi]
    roi = np.zeros(lab.shape)
    roi[lab > 0] = code[lab[lab > 0] - 1]
    fs = fsaverage()
    fig = plt.figure(figsize=(3.2, 2.6))
    ax = fig.add_subplot(111, projection="3d")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plotting.plot_surf_roi(fs[f"infl_{hemi}"], roi, hemi=hemi, view=view, bg_map=fs[f"sulc_{hemi}"], axes=ax,
                               cmap=ListedColormap([NETWORK[n] for n in order]), vmin=1, vmax=len(order),
                               colorbar=False, bg_on_data=True)
    ax.set_box_aspect(None, zoom=1.45)
    _STATE["dpi"] = dpi
    return _crop(fig)


def place(ax, image: np.ndarray) -> None:
    """Draw a rendered brain in ordinary axes without frame or ticks."""
    ax.imshow(image, interpolation="lanczos")
    ax.set_axis_off()
