#!/usr/bin/env python
"""
Build one feature matrix spanning the three collections: tangent-space connectivity, regional amplitude,
confounds and within-collection normal scores of the intelligence measure.

Representation
--------------
Every subject is reduced to one tangent vector, using the same log-Euclidean
Target Harmonisation
--------------------
ID1000 measures the IST (590 items, total score); PIOP measures Raven's APM (36
items). They cannot be put on a common scale by assumption, so each dataset's
target is converted to within-dataset normal scores via the rank-based inverse
normal transform. This makes the pooled target distribution homogeneous without
assuming the two instruments are interchangeable in raw units, and it is
invariant to the marginal shape of either test. Prediction is always scored
WITHIN a target dataset, where scale is irrelevant to a correlation anyway.

Outputs outputs/piop/pooled_features.npz with, for every subject: the tangent
vector, the log amplitude, the normal-scored target, the raw target, the
confound block, and a dataset label.

Usage
-----
  python scripts/extraction/build_pooled_features.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import norm, rankdata
from sklearn.covariance import ledoit_wolf

from src.data.paths import collection_root
from src.data.preprocessing import load_participants

warnings.filterwarnings("ignore")

CFG = "config/pipeline.yaml"
OUT = Path("outputs/piop")
VARIANT = "base"
ICV = "EstimatedTotalIntraCranialVol"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOTS = {name: collection_root(name) for name in ("PIOP1", "PIOP2")}


def sym_funcm(X, fn):
    X = 0.5 * (X + X.transpose(-1, -2)).double()
    w, V = torch.linalg.eigh(X)
    return V @ torch.diag_embed(fn(w.clamp_min(1e-10))) @ V.transpose(-1, -2)


def upper(L):
    n = L.shape[-1]
    iu = torch.triu_indices(n, n, offset=1, device=L.device)
    return torch.cat([torch.diagonal(L, dim1=-2, dim2=-1),
                      L[..., iu[0], iu[1]] * np.sqrt(2.0)], dim=-1).float()


def run_tangent(x):
    """One run -> (tangent vector of the covariance, log amplitude)."""
    C = torch.from_numpy(ledoit_wolf(x, assume_centered=False)[0]).to(DEV).double()
    sd = torch.sqrt(torch.diagonal(C))
    return (upper(sym_funcm(C.unsqueeze(0), torch.log))[0].cpu().numpy(),
            torch.log(sd).float().cpu().numpy())


def rank_normal(y):
    """Rank-based inverse normal transform, within dataset."""
    return norm.ppf((rankdata(y) - 0.375) / (len(y) + 0.25))


def motion(fd):
    fd = np.asarray(fd, float)[1:]
    return [fd.mean(), fd.max(), (fd > 0.5).mean()] if fd.size else [np.nan] * 3


def build_id1000():
    cfg = yaml.safe_load(open(CFG))
    ids0 = json.loads(Path("outputs/honest/holdout_manifest.json").read_text())["discovery"]
    parts = load_participants(cfg["paths"]["bids_root"])
    frames = []
    for f in sorted(Path("derivatives/fs_stats").glob("data-*.tsv")):
        d = pd.read_csv(f, sep="\t")
        d = d.rename(columns={d.columns[0]: "sid"})
        d["sid"] = d["sid"].astype(str)
        d = d.set_index("sid"); d = d.loc[:, ~d.columns.duplicated()]
        d.columns = [f"{f.stem}::{c}" for c in d.columns]; frames.append(d)
    M = pd.concat(frames, axis=1).apply(pd.to_numeric, errors="coerce")
    vcol = [c for c in M.columns if c.endswith("::" + ICV)][0]
    ids, T, A, mot = [], [], [], []
    id_path = (Path(cfg["paths"]["connectivity_dir"]) / "unscrubbed_ts.h5"
               if VARIANT == "base" else OUT / "ID1000_broad_ts.h5")
    # FD comes from the fmriprep confound files, not from the HDF5: the
    # broadband re-extraction stores only the timeseries, and reading the
    # source keeps both variants on identical motion summaries anyway
    fp = Path(cfg["paths"]["fmriprep_dir"])
    task = cfg["dataset"]["task"]

    def fd_of(sub):
        q = fp / sub / "func" / f"{sub}_task-{task}_desc-confounds_regressors.tsv"
        v = pd.read_csv(q, sep="\t", usecols=["framewise_displacement"],
                        na_values="n/a")["framewise_displacement"]
        return v.fillna(0.0).to_numpy()

    with h5py.File(id_path, "r") as f:
        for i, s in enumerate(ids0, 1):
            if s not in f["subjects"]:
                continue
            g = f["subjects"][s]
            t, a = run_tangent(g["time_series"][:].astype(np.float64))
            ids.append(s); T.append(t); A.append(a); mot.append(motion(fd_of(s)))
            if i % 100 == 0:
                print(f"    ID1000 {i}/{len(ids0)}", flush=True)
    y = np.array([float(parts.loc[s, "IST_intelligence_total"]) for s in ids])
    sex = np.array([1 if str(parts.loc[s, "sex"]).lower().startswith("m") else 0
                    for s in ids])
    v = M.loc[ids, vcol].to_numpy(float)
    return ids, np.stack(T), np.stack(A), y, sex, v, np.array(mot, float)


def build_piop(ds):
    root = ROOTS[ds]
    parts = pd.read_csv(root / "participants.tsv", sep="\t", na_values="n/a") \
        .set_index("participant_id").dropna(subset=["raven_score"])
    aseg = pd.read_csv(root / "derivatives" / "fs_stats" /
                       "data-subcortical_type-aseg_measure-volume_hemi-both.tsv", sep="\t")
    aseg = aseg.rename(columns={aseg.columns[0]: "sid"}).set_index("sid")
    icv = pd.to_numeric(aseg[ICV], errors="coerce")
    ids, T, A, mot = [], [], [], []
    with h5py.File(OUT / (f"{ds}_ts.h5" if VARIANT == "base"
                          else f"{ds}_broad_ts.h5"), "r") as f:
        for i, s in enumerate(parts.index, 1):
            if s not in f["subjects"] or len(f["subjects"][s]) == 0:
                continue
            if not np.isfinite(icv.get(s, np.nan)):
                continue
            ts, ams, w, fds = [], [], [], []
            for task in sorted(f["subjects"][s]):
                g = f["subjects"][s][task]
                x = g["time_series"][:].astype(np.float64)
                t, a = run_tangent(x)
                ts.append(t); ams.append(a)
                w.append(x.shape[0] * float(g.attrs["tr"]) / 60.0)
                fds.append(motion(g["fd"][:]))
            wn = np.asarray(w, float); wn = wn / wn.sum()
            ids.append(s)
            T.append(np.average(np.stack(ts), axis=0, weights=wn))
            A.append(np.average(np.stack(ams), axis=0, weights=wn))
            mot.append(np.average(np.asarray(fds, float), axis=0, weights=wn))
            if i % 50 == 0:
                print(f"    {ds} {i}/{len(parts)}", flush=True)
    y = parts.loc[ids, "raven_score"].to_numpy(float)
    sex = np.array([1 if str(parts.loc[s, "sex"]).upper().startswith("M") else 0
                    for s in ids])
    return ids, np.stack(T), np.stack(A), y, sex, icv.loc[ids].to_numpy(float), \
        np.array(mot, float)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["base", "broad"], default="base",
                    help="'broad' uses the no-low-pass re-extraction of all "
                         "three collections")
    args = ap.parse_args()
    global VARIANT
    VARIANT = args.variant
    parts = {}
    print("building features (this is the expensive step, cached afterwards)")
    for name, fn in (("ID1000", build_id1000), ("PIOP1", lambda: build_piop("PIOP1")),
                     ("PIOP2", lambda: build_piop("PIOP2"))):
        print(f"  {name} ...", flush=True)
        parts[name] = fn()
        print(f"  {name}: n = {len(parts[name][0])}, "
              f"tangent dim {parts[name][1].shape[1]}")

    ids, T, A, y_raw, yz, ds, Z = [], [], [], [], [], [], []
    for name, (i, t, a, y, sex, v, mot) in parts.items():
        for j in range(mot.shape[1]):
            mot[:, j] = np.where(np.isfinite(mot[:, j]), mot[:, j],
                                 np.nanmedian(mot[:, j]))
        vz = (v - v.mean()) / v.std()
        ids += i; T.append(t); A.append(a)
        y_raw.append(y); yz.append(rank_normal(y))
        ds += [name] * len(i)
        Z.append(np.column_stack([sex, vz, vz ** 2, mot]))

    sfx = "" if VARIANT == "base" else "_broad"
    np.savez_compressed(
        OUT / f"pooled_features{sfx}.npz",
        ids=np.array(ids), tangent=np.concatenate(T), amplitude=np.concatenate(A),
        y_raw=np.concatenate(y_raw), y=np.concatenate(yz),
        dataset=np.array(ds), Z=np.concatenate(Z))
    print(f"\nSaved -> {OUT}/pooled_features.npz")
    print(f"  total n = {len(ids)}  "
          + "  ".join(f"{k}={sum(1 for d in ds if d == k)}" for k in parts))


if __name__ == "__main__":
    main()
