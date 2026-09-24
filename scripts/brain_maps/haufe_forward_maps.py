#!/usr/bin/env python
"""
Haufe forward weights of the connectivity model, summarised by region and by network pair, with split-half
reliability and a label-permutation null.

Three targets are mapped
------------------------
  raw            IST
  deconfounded   IST residualised on sex + brain volume + volume^2
  deconf+motion  additionally residualised on mean FD, max FD and spike fraction
Residualisation is refit inside each training fold, never on the full sample.

The deconfounded maps are the scientifically interesting ones: they localise the
part of the signal that is NOT head size, which is the dissociation this project
has established (FC retains ~91 % after control, structural blocks 53-60 %).

Outputs
-------
  outputs/honest/haufe_edges.csv      per-edge forward weights (all three targets)
  outputs/honest/haufe_networks.csv   Yeo network x network summary + null z/p
  outputs/honest/haufe_regions.csv    per-region total |weight|

Usage
-----
  python scripts/brain_maps/haufe_forward_maps.py --repeats 5
"""
from __future__ import annotations

import argparse, sys, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.data.preprocessing import load_participants
from src.data.series import load_series
from src.data.series import roi_networks

warnings.filterwarnings("ignore")

CFG_PATH = "config/pipeline.yaml"
TARGET   = "IST_intelligence_total"
ALPHAS   = np.logspace(-2, 6, 25)
SEED     = 42
SHRINK   = 0.55
N_PERM   = 10000
OUT_DIR  = Path("outputs/honest")
SERIES   = "legacy"          # set from --series in main()
FS_DIR   = Path("derivatives/fs_stats")
VOL_COL  = "EstimatedTotalIntraCranialVol"


def sym_funcm(X, fn):
    X = 0.5 * (X + X.transpose(-1, -2)).double()
    w, V = torch.linalg.eigh(X)
    return V @ torch.diag_embed(fn(w.clamp_min(1e-10))) @ V.transpose(-1, -2)


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


def haufe_pattern(Xtr, ytr):
    """Fit ridge, return the Haufe forward pattern and the fitted model."""
    sc = StandardScaler().fit(Xtr)
    Z = sc.transform(Xtr)
    m = RidgeCV(alphas=ALPHAS, cv=5).fit(Z, ytr)
    yhat = m.predict(Z)
    Zc = Z - Z.mean(0, keepdims=True)
    a = (Zc.T @ (yhat - yhat.mean())) / (len(Z) - 1)
    a = a / max(np.var(yhat, ddof=1), 1e-12)
    return a, m, sc


def edge_matrix(edge_w, n_roi, iu):
    """Symmetric |forward weight| adjacency with a zero diagonal."""
    W = np.zeros((n_roi, n_roi))
    W[iu[0], iu[1]] = np.abs(edge_w)
    return W + W.T


def net_from_W(W, code, n_net):
    """Mean |weight| per network pair, as H^T W H with a one-hot label matrix.

    `code` is the integer network index per ROI. Only this vector is permuted in
    the null, so the null costs two small matmuls per draw rather than a
    23,005-edge aggregation loop.
    """
    H = np.zeros((len(code), n_net))
    H[np.arange(len(code)), code] = 1.0
    n = H.sum(0)
    C = np.outer(n, n) - np.diag(n)      # exclude self-pairs (W has zero diagonal)
    return (H.T @ W @ H) / np.maximum(C, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--series", choices=["legacy", "psc"], default="legacy",
                    help="region timeseries source (see src/data/series.py)")
    args = ap.parse_args()
    global OUT_DIR, SERIES
    SERIES = args.series
    if SERIES != "legacy":
        OUT_DIR = OUT_DIR / SERIES
    print(f"  series = {SERIES}   -> {OUT_DIR}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(CFG_PATH))
    conn = Path(cfg["paths"]["connectivity_dir"])

    ctx, sub = {}, {}
    ctx, sub = load_series(conn, SERIES)
    meta = np.load(conn / "atlas_meta.npz", allow_pickle=True)
    nets_raw = [str(x) for x in meta["networks"]]

    frames = []
    for f in sorted(FS_DIR.glob("data-*.tsv")):
        d = pd.read_csv(f, sep="\t"); d = d.rename(columns={d.columns[0]: "sid"})
        d["sid"] = d["sid"].astype(str); d = d.set_index("sid")
        d = d.loc[:, ~d.columns.duplicated()]
        d.columns = [f"{f.stem}::{c}" for c in d.columns]
        frames.append(d)
    M = pd.concat(frames, axis=1)
    M = M.loc[:, ~M.columns.duplicated()].apply(pd.to_numeric, errors="coerce")
    M = M.dropna(axis=1, how="all"); M = M.loc[:, M.std(skipna=True) > 0]
    vol_col = [c for c in M.columns if c.endswith("::" + VOL_COL)][0]
    parts = load_participants(cfg["paths"]["bids_root"])

    ids, cov = [], []
    for s in sorted(set(ctx) & set(sub) & set(M.index)):
        if ctx[s].shape[0] != sub[s].shape[0]:
            continue
        if s not in parts.index or pd.isna(parts.loc[s, TARGET]) or pd.isna(M.loc[s, vol_col]):
            continue
        x = np.hstack([ctx[s], sub[s]]).astype(np.float32)
        x = x - x.mean(0, keepdims=True)
        S = (x.T @ x) / (len(x) - 1); d = S.shape[0]
        cov.append((1 - SHRINK) * S + SHRINK * (np.trace(S) / d) * np.eye(d, dtype=np.float32))
        ids.append(s)
    C = torch.from_numpy(np.stack(cov)).to(dev).double()
    n_roi = C.shape[-1]
    L = sym_funcm(C, torch.log)
    iu_t = torch.triu_indices(n_roi, n_roi, offset=1, device=L.device)
    X = torch.cat([torch.diagonal(L, dim1=-2, dim2=-1),
                   L[..., iu_t[0], iu_t[1]] * np.sqrt(2.0)], dim=-1).float().cpu().numpy()
    iu = iu_t.cpu().numpy()

    y = np.array([float(parts.loc[s, TARGET]) for s in ids])
    sex = np.array([1 if str(parts.loc[s, "sex"]).lower().startswith("m") else 0 for s in ids])
    vol = M.loc[ids, vol_col].values.astype(float)
    mot = np.array([motion_stats(s, cfg["paths"]["fmriprep_dir"], cfg["dataset"]["task"])
                    for s in ids])
    for j in range(mot.shape[1]):
        mot[:, j] = np.where(np.isfinite(mot[:, j]), mot[:, j], np.nanmedian(mot[:, j]))
    Zc = np.column_stack([sex, vol, vol ** 2])
    Zm = np.column_stack([sex, vol, vol ** 2, mot])
    print(f"{len(ids)} subjects, {X.shape[1]} features ({n_roi} diag + {iu.shape[1]} edges)")

    # atlas_meta index 0 is the atlas BACKGROUND, not a parcel; the previous code
    # used networks[j] for parcel j and so shifted every cortical ROI onto its
    # neighbour's network (14/200 actually changed) and invented an "Unknown"
    # network. See src/data/series.roi_networks.
    labels = roi_networks(conn, n_roi)
    uniq = sorted(set(labels))
    print(f"  networks: {uniq}")

    q = pd.qcut(y, 4, labels=False, duplicates="drop")
    strat = np.asarray(q) * 2 + sex

    TARGETS = [("raw", None), ("deconfounded", Zc), ("deconf_motion", Zm)]
    acc = {t: [] for t, _ in TARGETS}
    rs = {t: [] for t, _ in TARGETS}

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats, random_state=SEED)
    for k, (tr, te) in enumerate(rskf.split(np.zeros(len(y)), strat), 1):
        for tag, Zconf in TARGETS:
            if Zconf is None:
                ytr, yte = y[tr], y[te]
            else:
                lr = LinearRegression().fit(Zconf[tr], y[tr])
                ytr, yte = y[tr] - lr.predict(Zconf[tr]), y[te] - lr.predict(Zconf[te])
            a, m, sc = haufe_pattern(X[tr], ytr)
            acc[tag].append(a)
            rs[tag].append(pearsonr(yte, m.predict(sc.transform(X[te])))[0])
        if k % 5 == 0:
            print(f"  fold {k}/{args.repeats*5}  raw r={np.mean(rs['raw']):+.3f} "
                  f"dec r={np.mean(rs['deconfounded']):+.3f} "
                  f"dec+mot r={np.mean(rs['deconf_motion']):+.3f}", flush=True)

    # ---- split-half reliability on DISJOINT subject halves ------------------
    print(f"\n  estimating split-half reliability on disjoint halves ...", flush=True)
    half_rel = {t: [] for t, _ in TARGETS}
    for rep in range(args.repeats):
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED + rep)
        h1, h2 = list(skf.split(np.zeros(len(y)), strat))[0]
        for tag, Zconf in TARGETS:
            maps = []
            for h in (h1, h2):
                if Zconf is None:
                    yh = y[h]
                else:
                    lr = LinearRegression().fit(Zconf[h], y[h])
                    yh = y[h] - lr.predict(Zconf[h])
                maps.append(haufe_pattern(X[h], yh)[0])
            half_rel[tag].append(pearsonr(maps[0], maps[1])[0])

    res = {}
    rng = np.random.default_rng(SEED)
    for tag, _ in TARGETS:
        A = np.stack(acc[tag])
        mean = A.mean(0)
        cv_rel = float(np.mean([pearsonr(A[i], A[j])[0]
                                for i in range(len(A)) for j in range(i + 1, len(A))]))
        sh_rel = float(np.mean(half_rel[tag]))
        edge = mean[n_roi:]
        Wr = np.zeros(n_roi)
        np.add.at(Wr, iu[0], np.abs(edge))
        np.add.at(Wr, iu[1], np.abs(edge))
        W = edge_matrix(edge, n_roi, iu)
        code = np.array([uniq.index(l) for l in labels])
        net = net_from_W(W, code, len(uniq))

        # ROI-label permutation null: shuffle network assignment, keep weights
        null = np.empty((N_PERM, len(uniq), len(uniq)))
        for b in range(N_PERM):
            null[b] = net_from_W(W, code[rng.permutation(n_roi)], len(uniq))
        mu, sd = null.mean(0), null.std(0)
        Z = (net - mu) / np.maximum(sd, 1e-12)
        P = (np.abs(null - mu) >= np.abs(net - mu)[None]).mean(0)
        res[tag] = (mean, Wr, net, Z, P, cv_rel, sh_rel)

        print(f"\n{'='*74}\nHaufe forward map: {tag}\n{'='*74}")
        print(f"  test r = {np.mean(rs[tag]):+.4f}")
        print(f"  reliability  cross-validation folds {cv_rel:.3f}  "
              f"(inflated: folds share ~60% of subjects)")
        print(f"  reliability  DISJOINT split-halves  {sh_rel:.3f}  <- the honest number")
        seen, top = set(), []
        order = np.dstack(np.unravel_index(np.argsort(-net, axis=None), net.shape))[0]
        for a_, b_ in order:
            key = tuple(sorted((a_, b_)))
            if key in seen:
                continue
            seen.add(key); top.append((a_, b_))
            if len(top) >= 8:
                break
        print(f"  strongest network pairs (mean |weight| per edge, vs label-permuted null):")
        for a_, b_ in top:
            star = "*" if P[a_, b_] < 0.05 else " "
            print(f"    {uniq[a_]:14s} - {uniq[b_]:14s} {net[a_, b_]:.5f}  "
                  f"z = {Z[a_, b_]:+6.2f}  p = {P[a_, b_]:.4f} {star}")
        nsig = int((P < 0.05).sum() // 2)
        print(f"  network pairs beating the ROI-permutation null at p<0.05: "
              f"{nsig}/{len(uniq)*(len(uniq)+1)//2}")
        top_roi = np.argsort(-Wr)[:10]
        print(f"  regions carrying most total |weight|:")
        for i in top_roi:
            print(f"    ROI {i:3d} ({labels[i]:12s}) {Wr[i]:.4f}")

    pd.DataFrame({"edge_i": iu[0], "edge_j": iu[1],
                  "net_i": [labels[i] for i in iu[0]],
                  "net_j": [labels[j] for j in iu[1]],
                  **{f"haufe_{t}": res[t][0][n_roi:] for t, _ in TARGETS}}
                 ).to_csv(OUT_DIR / "haufe_edges.csv", index=False)
    rows = []
    for t, _ in TARGETS:
        _, _, net, Z, P, cv_rel, sh_rel = res[t]
        for i, a_ in enumerate(uniq):
            for j, b_ in enumerate(uniq):
                if j < i:
                    continue
                rows.append({"target": t, "net_i": a_, "net_j": b_,
                             "mean_abs_weight": net[i, j], "z_vs_null": Z[i, j],
                             "p_perm": P[i, j], "cv_fold_reliability": cv_rel,
                             "splithalf_reliability": sh_rel})
    pd.DataFrame(rows).to_csv(OUT_DIR / "haufe_networks.csv", index=False)
    pd.DataFrame({"roi": np.arange(n_roi), "network": labels,
                  **{f"abs_weight_{t}": res[t][1] for t, _ in TARGETS}}
                 ).to_csv(OUT_DIR / "haufe_regions.csv", index=False)
    print(f"\nSaved -> {OUT_DIR}/haufe_edges.csv, haufe_networks.csv, haufe_regions.csv")


if __name__ == "__main__":
    main()
