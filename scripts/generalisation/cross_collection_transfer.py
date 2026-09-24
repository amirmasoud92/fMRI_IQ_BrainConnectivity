#!/usr/bin/env python
"""
Transfer of the regional amplitude model between collections, and similarity of its forward maps.

Usage
-----
  python scripts/generalisation/cross_collection_transfer.py
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
from scipy.stats import pearsonr
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler

from src.data.paths import collection_root
from src.data.preprocessing import load_participants

warnings.filterwarnings("ignore")

CFG = "config/pipeline.yaml"
OUT = Path("outputs/piop")
ALPHAS = np.logspace(-2, 6, 25)
ICV = "EstimatedTotalIntraCranialVol"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOTS = {name: collection_root(name) for name in ("PIOP1", "PIOP2")}
SHARED = ["workingmemory", "emomatching", "restingstate"]
ARMS = ["amplitude", "corr_tangent", "fc_tangent"]


def sym_funcm(X, fn):
    X = 0.5 * (X + X.transpose(-1, -2)).double()
    w, V = torch.linalg.eigh(X)
    return V @ torch.diag_embed(fn(w.clamp_min(1e-10))) @ V.transpose(-1, -2)


def upper(L):
    n = L.shape[-1]
    iu = torch.triu_indices(n, n, offset=1, device=L.device)
    return torch.cat([torch.diagonal(L, dim1=-2, dim2=-1),
                      L[..., iu[0], iu[1]] * np.sqrt(2.0)], dim=-1).float()


def features(series):
    covs = np.stack([ledoit_wolf(x, assume_centered=False)[0] for x in series])
    C = torch.from_numpy(covs).to(DEV).double()
    sd = torch.sqrt(torch.diagonal(C, dim1=-2, dim2=-1))
    R = C / (sd.unsqueeze(-1) * sd.unsqueeze(-2))
    return {"amplitude": torch.log(sd).float().cpu().numpy(),
            "corr_tangent": upper(sym_funcm(R, torch.log)).cpu().numpy(),
            "fc_tangent": upper(sym_funcm(C, torch.log)).cpu().numpy()}


def cell(dataset, task):
    """Features, target, confounds for one dataset-by-task cell."""
    if dataset == "ID1000":
        cfg = yaml.safe_load(open(CFG))
        ids0 = json.loads(Path("outputs/honest/holdout_manifest.json")
                          .read_text())["discovery"]
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
        ids, series, mot = [], [], []
        with h5py.File(Path(cfg["paths"]["connectivity_dir"]) / "unscrubbed_ts.h5", "r") as f:
            for s in ids0:
                if s not in f["subjects"]:
                    continue
                g = f["subjects"][s]
                ids.append(s); series.append(g["time_series"][:].astype(np.float64))
                fd = np.asarray(g["fd"][:], float)[1:]
                mot.append([fd.mean(), fd.max(), (fd > 0.5).mean()])
        y = np.array([float(parts.loc[s, "IST_intelligence_total"]) for s in ids])
        sex = np.array([1 if str(parts.loc[s, "sex"]).lower().startswith("m") else 0
                        for s in ids])
        v = M.loc[ids, vcol].to_numpy(float)
    else:
        root = ROOTS[dataset]
        parts = pd.read_csv(root / "participants.tsv", sep="\t", na_values="n/a") \
            .set_index("participant_id").dropna(subset=["raven_score"])
        aseg = pd.read_csv(root / "derivatives" / "fs_stats" /
                           "data-subcortical_type-aseg_measure-volume_hemi-both.tsv",
                           sep="\t")
        aseg = aseg.rename(columns={aseg.columns[0]: "sid"}).set_index("sid")
        icv = pd.to_numeric(aseg[ICV], errors="coerce")
        ids, series, mot = [], [], []
        with h5py.File(OUT / f"{dataset}_ts.h5", "r") as f:
            for s in parts.index:
                if s not in f["subjects"] or task not in f["subjects"][s]:
                    continue
                if not np.isfinite(icv.get(s, np.nan)):
                    continue
                g = f["subjects"][s][task]
                ids.append(s); series.append(g["time_series"][:].astype(np.float64))
                fd = np.asarray(g["fd"][:], float)[1:]
                mot.append([fd.mean(), fd.max(), (fd > 0.5).mean()])
        y = parts.loc[ids, "raven_score"].to_numpy(float)
        sex = np.array([1 if str(parts.loc[s, "sex"]).upper().startswith("M") else 0
                        for s in ids])
        v = icv.loc[ids].to_numpy(float)
    vz = (v - v.mean()) / v.std()
    Z = np.column_stack([sex, vz, vz ** 2, np.array(mot)])
    return dict(ids=ids, X=features(series), y=y, Z=Z, n=len(ids))


def haufe(X, y):
    """Forward map A = Cov(X) w for a ridge model fitted on standardised X."""
    Xs = StandardScaler().fit_transform(X)
    w = RidgeCV(alphas=ALPHAS, cv=5).fit(Xs, y).coef_
    return (Xs - Xs.mean(0)).T @ ((Xs - Xs.mean(0)) @ w) / (len(y) - 1)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps-only", action="store_true",
                    help="skip prediction transfer; recompute the forward maps only (Step 7)")
    ap.add_argument("--maps-out", default=None,
                    help="save the 11 amplitude forward maps (rows = regions, cols = cells)")
    ap.add_argument("--similarity-out", default=str(OUT / "amplitude_map_similarity.csv"))
    args = ap.parse_args()
    bar = "=" * 80
    print(bar); print("CROSS-DATASET TRANSFER"); print(bar)

    cells = {("ID1000", "moviewatching"): None}
    for ds, tasks in (("PIOP1", ["anticipation", "emomatching", "faces", "gstroop",
                                 "restingstate", "workingmemory"]),
                      ("PIOP2", ["emomatching", "restingstate", "stopsignal",
                                 "workingmemory"])):
        for t in tasks:
            cells[(ds, t)] = None
    for k in list(cells):
        print(f"  loading {k[0]}/{k[1]} ...", flush=True)
        cells[k] = cell(*k)

    rows = []
    for task in ([] if args.maps_only else SHARED):
        if task == SHARED[0]:
            print(f"\n{bar}\n1. PREDICTION TRANSFER, tasks present in both PIOP datasets\n{bar}")
            print(f"  {'task':16s} {'direction':18s} {'arm':14s} {'n tr':>5s} {'n te':>5s} "
                  f"{'r raw':>8s} {'r dec':>8s}")
        for a, b in (("PIOP1", "PIOP2"), ("PIOP2", "PIOP1")):
            A, B = cells[(a, task)], cells[(b, task)]
            lr = LinearRegression().fit(A["Z"], A["y"])
            ytr_d = A["y"] - lr.predict(A["Z"])
            yte_d = B["y"] - lr.predict(B["Z"])   # train-dataset coefficients only
            for arm in ARMS:
                # standardise within each dataset separately -- see module docstring
                Xtr = StandardScaler().fit_transform(A["X"][arm])
                Xte = StandardScaler().fit_transform(B["X"][arm])
                p_raw = RidgeCV(alphas=ALPHAS, cv=5).fit(Xtr, A["y"]).predict(Xte)
                p_dec = RidgeCV(alphas=ALPHAS, cv=5).fit(Xtr, ytr_d).predict(Xte)
                r_raw = pearsonr(B["y"], p_raw)[0]
                r_dec = pearsonr(yte_d, p_dec)[0]
                print(f"  {task:16s} {a+' -> '+b:18s} {arm:14s} {A['n']:5d} "
                      f"{B['n']:5d} {r_raw:+8.4f} {r_dec:+8.4f}")
                rows.append(dict(task=task, train=a, test=b, arm=arm,
                                 n_train=A["n"], n_test=B["n"],
                                 r_raw=r_raw, r_dec=r_dec))
        print()
    if not args.maps_only:
        pd.DataFrame(rows).to_csv(OUT / "cross_dataset_transfer.csv", index=False)

    print(f"{bar}\n2. AMPLITUDE FORWARD-MAP SIMILARITY across all 11 cells\n{bar}")
    keys = list(cells)
    maps = {}
    for k in keys:
        c = cells[k]
        lr = LinearRegression().fit(c["Z"], c["y"])
        maps[k] = haufe(c["X"]["amplitude"], c["y"] - lr.predict(c["Z"]))
    lab = [f"{d[:4]}/{t[:9]}" for d, t in keys]
    Mx = np.corrcoef(np.stack([maps[k] for k in keys]))
    print(f"\n  {'':16s}" + "".join(f"{l[:8]:>9s}" for l in lab))
    for i, l in enumerate(lab):
        print(f"  {l:16s}" + "".join(f"{Mx[i, j]:+9.2f}" for j in range(len(lab))))
    pd.DataFrame(Mx, index=lab, columns=lab).to_csv(args.similarity_out)
    if args.maps_out:
        pd.DataFrame(np.stack([maps[k] for k in keys]).T,
                     columns=[f"{d}/{t}" for d, t in keys]).rename_axis("roi").to_csv(args.maps_out)
        print(f"  forward maps ({len(keys)} cells x {len(maps[keys[0]])} regions) -> {args.maps_out}")

    iu = np.triu_indices(len(keys), 1)
    same = np.array([keys[i][0] == keys[j][0] for i, j in zip(*iu)])
    print(f"\n  mean map correlation WITHIN a dataset  : {Mx[iu][same].mean():+.3f} "
          f"(n = {same.sum()} pairs)")
    print(f"  mean map correlation ACROSS datasets   : {Mx[iu][~same].mean():+.3f} "
          f"(n = {(~same).sum()} pairs)")
    p12 = [(i, j) for i, j in zip(*iu)
           if {keys[i][0], keys[j][0]} == {"PIOP1", "PIOP2"} and keys[i][1] == keys[j][1]]
    if p12:
        v = [Mx[i, j] for i, j in p12]
        print(f"  SAME TASK, PIOP1 vs PIOP2              : "
              + ", ".join(f"{keys[i][1]} {Mx[i, j]:+.2f}" for i, j in p12)
              + f"   (mean {np.mean(v):+.3f})")
    print("\n  A high across-dataset map correlation means the same regions carry the")
    print("  signal everywhere and the mechanism is shared. A near-zero one means the")
    print("  amplitude finding is local and must be written as a bounded result.")
    print(f"\nSaved -> {OUT}/cross_dataset_transfer.csv, amplitude_map_similarity.csv")


if __name__ == "__main__":
    main()
