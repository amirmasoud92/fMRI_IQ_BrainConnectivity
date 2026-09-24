#!/usr/bin/env python
"""
Centroid-rotation spin test for the network maps, and size-matched nulls for the network lesions.

Usage
-----
  python scripts/brain_maps/spatial_nulls_centroid.py --spins 10000 --draws 150
"""
from __future__ import annotations

import argparse, json, sys, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.data.preprocessing import load_participants
from src.data.series import roi_networks

warnings.filterwarnings("ignore")

CFG_PATH = "config/pipeline.yaml"
TARGET   = "IST_intelligence_total"
ALPHAS   = np.logspace(-2, 6, 25)
SEED     = 42
SHRINK   = 0.55
OUT_DIR  = Path("outputs/honest")
FS_DIR   = Path("derivatives/fs_stats")
MANIFEST = OUT_DIR / "holdout_manifest.json"
VOL_COL  = "EstimatedTotalIntraCranialVol"


def motion_stats(sub_id, fmriprep, task):
    p = Path(fmriprep) / sub_id / "func" / f"{sub_id}_task-{task}_desc-confounds_regressors.tsv"
    if not p.exists():
        return np.nan, np.nan, np.nan
    try:
        fd = pd.read_csv(p, sep="\t", usecols=["framewise_displacement"],
                         na_values="n/a")["framewise_displacement"].astype(float)
        fd = fd.fillna(0.0).to_numpy()[1:]
        if fd.size == 0:
            return np.nan, np.nan, np.nan
        return float(fd.mean()), float(fd.max()), float((fd > 0.5).mean())
    except Exception:
        return np.nan, np.nan, np.nan


def rand_rotation(rng):
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    Q *= np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def spin_perm(coords, rng):
    """Permutation of cortical parcels preserving spatial structure.

    Rotates each hemisphere's centroids independently and matches rotated to
    original parcels by optimal assignment (Hungarian), which guarantees a true
    permutation rather than the many-to-one map a greedy nearest-neighbour
    assignment would give.
    """
    perm = np.empty(len(coords), int)
    for side in (coords[:, 0] < 0, coords[:, 0] >= 0):
        idx = np.where(side)[0]
        C = coords[idx] - coords[idx].mean(0)
        Cr = C @ rand_rotation(rng)
        r, c = linear_sum_assignment(cdist(Cr, C))
        perm[idx[r]] = idx[c]
    return perm


def sym_funcm(X, fn):
    X = 0.5 * (X + X.transpose(-1, -2)).double()
    w, V = torch.linalg.eigh(X)
    return V @ torch.diag_embed(fn(w.clamp_min(1e-10))) @ V.transpose(-1, -2)


def upper(L):
    n = L.shape[-1]
    iu = torch.triu_indices(n, n, offset=1, device=L.device)
    return torch.cat([torch.diagonal(L, dim1=-2, dim2=-1),
                      L[..., iu[0], iu[1]] * np.sqrt(2.0)], dim=-1).float()


def embed_from_DR(D, R, dev):
    Dt = torch.from_numpy(np.ascontiguousarray(D)).to(dev).double()
    C = Dt.unsqueeze(-1) * R * Dt.unsqueeze(-2)
    n = D.shape[1]
    tr = torch.diagonal(C, dim1=-2, dim2=-1).sum(-1) / n
    eye = torch.eye(n, dtype=C.dtype, device=dev)
    C = (1 - SHRINK) * C + SHRINK * tr.view(-1, 1, 1) * eye
    return upper(sym_funcm(C, torch.log)).cpu().numpy()


def gcv_pred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    m = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xtr), ytr)
    return m.predict(sc.transform(Xte))


def cv5_pred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    return RidgeCV(alphas=ALPHAS, cv=5).fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spins", type=int, default=10000)
    ap.add_argument("--draws", type=int, default=100)
    # Part B needs one refit per (draw x fold), so the fold count drives the
    # runtime. 1 repeat (5 folds) is used for BOTH the observed statistic and
    # the null, so the comparison stays internally valid -- the observed loss
    # is simply estimated less precisely than in the main analysis.
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(CFG_PATH))
    conn = Path(cfg["paths"]["connectivity_dir"])
    meta = np.load(conn / "atlas_meta.npz", allow_pickle=True)
    coords = np.asarray(meta["coords"], float)
    nets = np.array(roi_networks(conn, 215))
    n_ctx = len(coords)

    # ================= PART A: spin test on the Haufe maps =================
    print("=" * 78); print("PART A  spin test on Haufe network enrichment"); print("=" * 78)
    ed = pd.read_csv(OUT_DIR / "psc" / "haufe_edges.csv")
    ei, ej = ed["edge_i"].to_numpy(), ed["edge_j"].to_numpy()
    rows_a = []
    for wcol, lab in [("haufe_raw", "raw"), ("haufe_deconfounded", "deconfounded")]:
        w = np.abs(ed[wcol].to_numpy(float))
        uniq = sorted(set(nets))
        code = {g: i for i, g in enumerate(uniq)}
        G = len(uniq)

        def pair_means(labels):
            a = np.array([code[x] for x in labels])
            lo = np.minimum(a[ei], a[ej]); hi = np.maximum(a[ei], a[ej])
            pid = lo * G + hi
            s = np.bincount(pid, weights=w, minlength=G * G)
            c = np.bincount(pid, minlength=G * G)
            return s / np.maximum(c, 1), c

        obs, cnt = pair_means(nets)
        rng = np.random.default_rng(SEED)
        sub_idx = np.arange(n_ctx, len(nets))
        null = np.empty((args.spins, G * G), float)
        for t in range(args.spins):
            p = spin_perm(coords, rng)
            lab_s = np.empty(len(nets), dtype=object)
            lab_s[:n_ctx] = nets[:n_ctx][p]
            lab_s[sub_idx] = nets[rng.permutation(sub_idx)]
            null[t], _ = pair_means(lab_s)
        mu, sd = null.mean(0), null.std(0)
        for i in range(G):
            for j in range(i, G):
                k = i * G + j
                if cnt[k] < 20:
                    continue
                z = (obs[k] - mu[k]) / max(sd[k], 1e-12)
                p_spin = (np.sum(null[:, k] >= obs[k]) + 1) / (args.spins + 1)
                rows_a.append({"target": lab, "net_i": uniq[i], "net_j": uniq[j],
                               "n_edges": int(cnt[k]), "mean_abs_weight": obs[k],
                               "z_spin": z, "p_spin": p_spin})
    A = pd.DataFrame(rows_a)
    A.to_csv(OUT_DIR / "spatial_null_haufe.csv", index=False)
    for lab in A["target"].unique():
        sub = A[A["target"] == lab].sort_values("z_spin", ascending=False)
        sig = (sub["p_spin"] < 0.05).sum()
        print(f"\n  {lab}: {sig}/{len(sub)} network pairs survive the SPIN null "
              f"at p<0.05")
        print(f"  {'pair':26s} {'mean|w|':>9s} {'z_spin':>8s} {'p_spin':>8s}")
        for _, r in sub.head(6).iterrows():
            print(f"  {r['net_i']+'-'+r['net_j']:26s} {r['mean_abs_weight']:9.5f} "
                  f"{r['z_spin']:+8.2f} {r['p_spin']:8.4f}")

    # ================= PART B: lesion null ==================================
    print("\n" + "=" * 78); print("PART B  size-matched lesion null for the amplitude effect")
    print("=" * 78)
    ids = json.loads(MANIFEST.read_text())["discovery"]
    raw = {}
    with h5py.File(conn / "unscrubbed_ts.h5", "r") as f:
        for s in ids:
            raw[s] = f["subjects"][s]["time_series"][:].astype(np.float64)
    frames = []
    for f in sorted(FS_DIR.glob("data-*.tsv")):
        d = pd.read_csv(f, sep="\t"); d = d.rename(columns={d.columns[0]: "sid"})
        d["sid"] = d["sid"].astype(str); d = d.set_index("sid")
        d = d.loc[:, ~d.columns.duplicated()]
        d.columns = [f"{f.stem}::{c}" for c in d.columns]; frames.append(d)
    M = pd.concat(frames, axis=1)
    M = M.loc[:, ~M.columns.duplicated()].apply(pd.to_numeric, errors="coerce")
    vol_col = [c for c in M.columns if c.endswith("::" + VOL_COL)][0]
    parts = load_participants(cfg["paths"]["bids_root"])
    y   = np.array([float(parts.loc[s, TARGET]) for s in ids])
    sex = np.array([1 if str(parts.loc[s, "sex"]).lower().startswith("m") else 0 for s in ids])
    vol = M.loc[ids, vol_col].values.astype(float)
    mot = np.array([motion_stats(s, cfg["paths"]["fmriprep_dir"],
                                 cfg["dataset"]["task"]) for s in ids])
    for j in range(mot.shape[1]):
        mot[:, j] = np.where(np.isfinite(mot[:, j]), mot[:, j], np.nanmedian(mot[:, j]))
    vzs = (vol - vol.mean()) / vol.std()
    Zm = np.column_stack([sex, vzs, vzs ** 2, mot])

    Dobs, Rl = [], []
    for s in ids:
        x = raw[s] - raw[s].mean(0, keepdims=True)
        S = (x.T @ x) / (len(x) - 1)
        d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
        Dobs.append(d); Rl.append(S / np.outer(d, d))
    Dobs = np.stack(Dobs)
    Rt = torch.from_numpy(np.stack(Rl)).to(dev).double()
    N, n_roi = Dobs.shape
    q = pd.qcut(y, 4, labels=False, duplicates="drop")
    strat = np.asarray(q) * 2 + sex
    folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats,
                                         random_state=SEED).split(np.zeros(N), strat))

    def score(mask, pred_fn):
        """Mean deconfounded r with the masked regions' amplitude flattened."""
        out = []
        for tr, te in folds:
            D = Dobs.copy()
            D[:, mask] = Dobs[tr].mean(0)[mask]
            X = embed_from_DR(D, Rt, dev)
            lr = LinearRegression().fit(Zm[tr], y[tr])
            out.append(float(pearsonr(y[te] - lr.predict(Zm[te]),
                                      pred_fn(X[tr], y[tr] - lr.predict(Zm[tr]),
                                              X[te]))[0]))
        return float(np.mean(out))

    full = score(np.zeros(n_roi, bool), cv5_pred)
    full_g = score(np.zeros(n_roi, bool), gcv_pred)
    print(f"  intact model  cv5 {full:+.4f}   gcv {full_g:+.4f}   "
          f"(gap {abs(full-full_g):.4f} -- gcv used for the null)")

    ctx_coords = coords
    rng = np.random.default_rng(SEED)
    rows_b = []
    for net in ["Vis", "SomMot"]:
        idx = np.where(nets == net)[0]
        k = len(idx)
        obs_loss = full_g - score(np.isin(np.arange(n_roi), idx), gcv_pred)
        rand_losses, cont_losses = [], []
        for t in range(args.draws):
            m1 = np.zeros(n_roi, bool)
            m1[rng.choice(n_roi, k, replace=False)] = True
            rand_losses.append(full_g - score(m1, gcv_pred))
            seed_i = rng.integers(n_ctx)
            order = np.argsort(cdist(ctx_coords[seed_i:seed_i + 1], ctx_coords)[0])
            m2 = np.zeros(n_roi, bool); m2[order[:k]] = True
            cont_losses.append(full_g - score(m2, gcv_pred))
            if (t + 1) % 25 == 0:
                print(f"    {net}: {t+1}/{args.draws} draws", flush=True)
        for nm, L in [("random", rand_losses), ("contiguous", cont_losses)]:
            L = np.asarray(L)
            p = (np.sum(L >= obs_loss) + 1) / (len(L) + 1)
            rows_b.append({"network": net, "k": k, "null": nm,
                           "observed_loss": obs_loss, "null_mean": L.mean(),
                           "null_sd": L.std(), "z": (obs_loss - L.mean()) / max(L.std(), 1e-12),
                           "p_empirical": p, "n_draws": len(L)})
            print(f"  {net:7s} k={k:3d}  vs {nm:11s} null: observed {obs_loss:+.4f}, "
                  f"null {L.mean():+.4f} +/- {L.std():.4f}, "
                  f"z={(obs_loss-L.mean())/max(L.std(),1e-12):+.2f}, p={p:.4f}")
    pd.DataFrame(rows_b).to_csv(OUT_DIR / "spatial_null_lesion.csv", index=False)
    print(f"\nSaved -> {OUT_DIR}/spatial_null_haufe.csv, spatial_null_lesion.csv")


if __name__ == "__main__":
    main()
