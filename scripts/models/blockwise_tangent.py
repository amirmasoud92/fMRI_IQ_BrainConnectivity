#!/usr/bin/env python
"""
Blockwise tangent embeddings, each block linearised at its own reference, against one global tangent chart.

Region order is permuted as a control for the block structure.

Arms
----
  global          full 215x215 log-Euclidean tangent            [current best]
  sliding         multi-scale diagonal blocks, w in {10,20,40,80}, stride w/2,
                  each tangent-projected at its OWN train-fold Frechet mean
  yeo             Yeo-7 network blocks + subcortex, same treatment
  global+sliding  concatenation
  global+yeo      concatenation

Usage
-----
  python scripts/models/blockwise_tangent.py --repeats 3
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.data.preprocessing import load_participants
from src.data.series import load_series
from src.data.series import roi_networks
from src.stats.cv_inference import corrected_ttest_rel

warnings.filterwarnings("ignore")

CFG_PATH = "config/pipeline.yaml"
TARGET   = "IST_intelligence_total"
ALPHAS   = np.logspace(-2, 6, 25)
SEED     = 42
SHRINK   = 0.55
SCALES   = [(10, 5), (20, 10), (40, 20), (80, 40)]   # (window, stride)
OUT_DIR  = Path("outputs/honest")
SERIES   = "legacy"          # set from --series in main()


def sym_funcm(X, fn):
    X = 0.5 * (X + X.transpose(-1, -2)).double()
    w, V = torch.linalg.eigh(X)
    return V @ torch.diag_embed(fn(w.clamp_min(1e-10))) @ V.transpose(-1, -2)


def frechet_mean(S, n_iter=10, tol=1e-7):
    G = S.mean(0)
    for _ in range(n_iter):
        Gs  = sym_funcm(G.unsqueeze(0), torch.sqrt)[0]
        Gis = sym_funcm(G.unsqueeze(0), lambda w: w.rsqrt())[0]
        T = sym_funcm(Gis @ S @ Gis, torch.log).mean(0)
        if torch.linalg.norm(T) < tol:
            break
        G = Gs @ sym_funcm(T.unsqueeze(0), torch.exp)[0] @ Gs
        G = 0.5 * (G + G.t())
    return G


def upper(L):
    n = L.shape[-1]
    iu = torch.triu_indices(n, n, offset=1, device=L.device)
    return torch.cat([torch.diagonal(L, dim1=-2, dim2=-1),
                      L[..., iu[0], iu[1]] * np.sqrt(2.0)], dim=-1).float()


def tangent_at(S, G):
    Gis = sym_funcm(G.unsqueeze(0), lambda w: w.rsqrt())[0]
    return upper(sym_funcm(Gis @ S @ Gis, torch.log))


def block_slices(n, scales=SCALES):
    """Overlapping diagonal windows at several scales."""
    out = []
    for w, stride in scales:
        for i in range(0, n - w + 1, stride):
            out.append((i, i + w))
    return out


def yeo_slices(networks, n_sub):
    """Contiguous Yeo-network runs over the cortical ROIs, plus a subcortex block."""
    out, start = [], 0
    for i in range(1, len(networks) + 1):
        if i == len(networks) or networks[i] != networks[start]:
            if i - start >= 4:                      # skip degenerate runs
                out.append((start, i))
            start = i
    n_ctx = len(networks)
    out.append((n_ctx, n_ctx + n_sub))
    return out


def block_features(C, slices, tr_idx, dev):
    """Tangent-project each diagonal block at its OWN train-fold Frechet mean."""
    tr = torch.as_tensor(tr_idx, device=dev)
    feats = []
    for a, b in slices:
        Sb = C[:, a:b, a:b]
        G = frechet_mean(Sb[tr])
        feats.append(tangent_at(Sb, G).cpu().numpy())
    return np.hstack(feats)


def ridge_r(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    m = RidgeCV(alphas=ALPHAS, cv=5).fit(sc.transform(Xtr), ytr)
    return float(pearsonr(yte, m.predict(sc.transform(Xte)))[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
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
    parts = load_participants(cfg["paths"]["bids_root"])

    ids, cov = [], []
    for s in sorted(set(ctx) & set(sub)):
        if ctx[s].shape[0] != sub[s].shape[0]:
            continue
        if s not in parts.index or pd.isna(parts.loc[s, TARGET]):
            continue
        x = np.hstack([ctx[s], sub[s]]).astype(np.float32)
        x = x - x.mean(0, keepdims=True)
        S = (x.T @ x) / (len(x) - 1)
        d = S.shape[0]
        cov.append((1 - SHRINK) * S + SHRINK * (np.trace(S) / d)
                   * np.eye(d, dtype=np.float32))
        ids.append(s)
    C = torch.from_numpy(np.stack(cov)).to(dev).double()
    y = np.array([float(parts.loc[s, TARGET]) for s in ids])
    sex = np.array([1 if str(parts.loc[s, "sex"]).lower().startswith("m") else 0
                    for s in ids])
    n_roi = C.shape[-1]
    # see src/data/series.roi_networks -- atlas_meta index 0 is BACKGROUND, so the
    # old slice treated 201 entries as 201 cortical parcels (there are 200).
    _all = roi_networks(conn, n_roi)
    n_ctx = sum(1 for x in _all if x != "Subcortex")
    nets = _all[:n_ctx]
    print(f"{len(ids)} subjects, SPD({n_roi}), cortical ROIs {n_ctx}")

    sl_slide = block_slices(n_roi)
    sl_yeo = yeo_slices(nets, n_roi - n_ctx)
    print(f"  sliding blocks: {len(sl_slide)}  (scales {[w for w,_ in SCALES]})")
    print(f"  yeo blocks    : {len(sl_yeo)}  sizes {[b-a for a,b in sl_yeo]}")

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_roi)
    C_perm = C[:, perm][:, :, perm]

    q = pd.qcut(y, 4, labels=False, duplicates="drop")
    strat = np.asarray(q) * 2 + sex
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats,
                                   random_state=SEED)

    rows = []
    for k, (tr, te) in enumerate(rskf.split(np.zeros(len(y)), strat), 1):
        t0 = time.time()
        rec = {"fold": k}
        G = frechet_mean(C[torch.as_tensor(tr, device=dev)])
        Xg = tangent_at(C, G).cpu().numpy()
        Xs = block_features(C, sl_slide, tr, dev)
        Xy = block_features(C, sl_yeo, tr, dev)
        Xsp = block_features(C_perm, sl_slide, tr, dev)
        Xyp = block_features(C_perm, sl_yeo, tr, dev)

        arms = {
            "global":         Xg,
            "sliding":        Xs,
            "yeo":            Xy,
            "global+sliding": np.hstack([Xg, Xs]),
            "global+yeo":     np.hstack([Xg, Xy]),
            "perm_sliding":   Xsp,
            "perm_yeo":       Xyp,
        }
        for nm, X in arms.items():
            rec[nm] = ridge_r(X[tr], y[tr], X[te], y[te])
        rec["secs"] = time.time() - t0
        rows.append(rec)
        print(f"  fold {k:2d}/{args.repeats*5}  global={rec['global']:+.3f} "
              f"slide={rec['sliding']:+.3f} g+s={rec['global+sliding']:+.3f} "
              f"perm={rec['perm_sliding']:+.3f}  ({rec['secs']:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "blockwise_tangent_folds.csv", index=False)

    cols = [c for c in df.columns if c not in ("fold", "secs")]
    print(f"\n{'='*74}\nBlockwise tangent, {args.repeats}x5-fold CV (n={len(ids)})\n{'='*74}")
    for c in sorted(cols, key=lambda c: -df[c].mean()):
        print(f"  {c:16s} r = {df[c].mean():+.4f} +/- {df[c].std(ddof=1):.4f}")

    print(f"\n  paired tests vs global:")
    for c in cols:
        if c == "global":
            continue
        t, p = corrected_ttest_rel(df[c], df["global"])
        print(f"    {c:16s} delta = {(df[c]-df['global']).mean():+.4f}  "
              f"t = {t:+5.2f}  p = {p:.3g}  "
              f"wins {int((df[c]>df['global']).sum())}/{len(df)}")

    print(f"\n  ORDER-DEPENDENCE CONTROL (real vs ROI-permuted):")
    for real, pm in [("sliding", "perm_sliding"), ("yeo", "perm_yeo")]:
        t, p = corrected_ttest_rel(df[real], df[pm])
        d = (df[real] - df[pm]).mean()
        verdict = ("block structure IS meaningful" if p < 0.05 and d > 0
                   else "NO evidence block structure matters")
        print(f"    {real:8s} - {pm:13s} delta = {d:+.4f}  t = {t:+5.2f}  "
              f"p = {p:.3g}   -> {verdict}")

    print(f"\nSaved -> {OUT_DIR}/blockwise_tangent_folds.csv")


if __name__ == "__main__":
    main()
