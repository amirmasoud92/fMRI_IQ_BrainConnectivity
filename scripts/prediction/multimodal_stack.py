#!/usr/bin/env python
"""
Stacked model over connectivity, morphometry and diffusion blocks, with the decoder of each block chosen
by inner cross-validation and a ridge meta-learner.

Blocks
------
  fc     log-Euclidean tangent, Schaefer-200 + 15 HO subcortical, shrink 0.55
  morph  FreeSurfer thickness/area/volume/meancurv + aseg/wmparc
  fa     global white-matter FA distribution summaries

VBM is excluded: it added +0.006 (n.s.) and actually hurt FC+morph.

Usage
-----
  python scripts/prediction/multimodal_stack.py --repeats 5
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.data.preprocessing import load_participants
from src.stats.cv_inference import corrected_ttest_rel

warnings.filterwarnings("ignore")

CFG_PATH   = "config/pipeline.yaml"
TARGET     = "IST_intelligence_total"
ALPHAS     = np.logspace(-2, 6, 25)
SEED       = 42
SHRINK_NEW = 0.55
SHRINK_OLD = 0.15
OUT_DIR    = Path("outputs/honest")

KRR_GAMMAS = [0.25, 0.5, 1.0, 2.0]      # multipliers on the median heuristic
KRR_ALPHAS = [0.1, 1.0, 10.0, 100.0]


# ---------------------------------------------------------------------------
# SPD helpers
# ---------------------------------------------------------------------------

def sym_funcm(X, fn):
    X = 0.5 * (X + X.transpose(-1, -2)).double()
    w, V = torch.linalg.eigh(X)
    return V @ torch.diag_embed(fn(w.clamp_min(1e-10))) @ V.transpose(-1, -2)


def frechet_mean(S, n_iter=12, tol=1e-7):
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


def vec_sym(L):
    n = L.shape[-1]
    iu = torch.triu_indices(n, n, offset=1, device=L.device)
    return torch.cat([torch.diagonal(L, dim1=-2, dim2=-1),
                      L[..., iu[0], iu[1]] * np.sqrt(2.0)], dim=-1).float()


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------

def ridge_predict(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    return RidgeCV(alphas=ALPHAS, cv=5).fit(sc.transform(Xtr), ytr).predict(
        sc.transform(Xte))


def _sq_dists(A, B, dev):
    """Squared Euclidean distances, computed on the GPU."""
    a = torch.from_numpy(A).to(dev).float()
    b = torch.from_numpy(B).to(dev).float()
    return torch.cdist(a, b).pow(2).cpu().numpy()


def krr_predict(Xtr, ytr, Xte, dev):
    """Kernel ridge with an RBF kernel; gamma/alpha chosen by inner 3-fold.

    Distances are computed once and reused for every candidate, so the grid
    search costs a handful of n x n solves rather than re-kernelising 23 000
    features 48 times.
    """
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    D_tt = _sq_dists(A, A, dev)
    D_et = _sq_dists(B, A, dev)
    med = np.median(np.sqrt(D_tt[np.triu_indices_from(D_tt, k=1)]))
    ym = ytr.mean()

    best, best_s = None, -np.inf
    for gm in KRR_GAMMAS:
        g = 1.0 / (2 * (gm * med) ** 2)
        K = np.exp(-g * D_tt)
        for al in KRR_ALPHAS:
            inner = []
            for a, b in KFold(3, shuffle=True, random_state=SEED).split(A):
                Kaa = K[np.ix_(a, a)] + al * np.eye(len(a))
                try:
                    coef = np.linalg.solve(Kaa, ytr[a] - ym)
                except np.linalg.LinAlgError:
                    inner.append(-1.0); continue
                pv = K[np.ix_(b, a)] @ coef + ym
                inner.append(pearsonr(ytr[b], pv)[0] if np.std(pv) > 1e-9 else -1.0)
            s = float(np.mean(inner))
            if s > best_s:
                best_s, best = s, (g, al)

    g, al = best
    K = np.exp(-g * D_tt) + al * np.eye(len(A))
    coef = np.linalg.solve(K, ytr - ym)
    return np.exp(-g * D_et) @ coef + ym


def oof(X, y, tr, decoder, dev, n_inner=5):
    o = np.zeros(len(tr))
    for a, b in KFold(n_inner, shuffle=True, random_state=SEED).split(tr):
        o[b] = (krr_predict(X[tr[a]], y[tr[a]], X[tr[b]], dev) if decoder == "krr"
                else ridge_predict(X[tr[a]], y[tr[a]], X[tr[b]]))
    return o


# ---------------------------------------------------------------------------

def load_all(cfg, dev, series="legacy"):
    """
    series="legacy": scrubbed, zscore_sample (fc_matrices + subcortical_ts).
           series="psc":    unscrubbed percent-signal-change, all 290 TRs, 215 regions.
    """
    conn = Path(cfg["paths"]["connectivity_dir"])
    ctx, sub = {}, {}
    if series == "psc":
        with h5py.File(conn / "unscrubbed_ts.h5", "r") as f:
            for s in f["subjects"]:
                x = f["subjects"][s]["time_series"][:]
                ctx[s] = x                              # already 200 ctx + 15 sub
                sub[s] = np.zeros((x.shape[0], 0), x.dtype)
    else:
        with h5py.File(conn / "fc_matrices.h5", "r") as f:
            for s in f["subjects"]:
                ctx[s] = f["subjects"][s]["time_series"][:]
        with h5py.File(conn / "subcortical_ts.h5", "r") as f:
            for s in f["subjects"]:
                sub[s] = f["subjects"][s]["time_series"][:]

    frames = []
    for f in sorted(Path("derivatives/fs_stats").glob("data-*.tsv")):
        d = pd.read_csv(f, sep="\t")
        d = d.rename(columns={d.columns[0]: "sid"})
        d["sid"] = d["sid"].astype(str)
        d = d.set_index("sid")
        d = d.loc[:, ~d.columns.duplicated()]
        d.columns = [f"{f.stem}::{c}" for c in d.columns]
        frames.append(d)
    M = pd.concat(frames, axis=1)
    M = M.loc[:, ~M.columns.duplicated()].apply(pd.to_numeric, errors="coerce")
    M = M.dropna(axis=1, how="all")
    M = M.loc[:, M.std(skipna=True) > 0]

    z = np.load(conn / "dwi_fa_features.npz", allow_pickle=True)
    FA = pd.DataFrame(z["features"], index=[str(s) for s in z["subject_ids"]])

    parts = load_participants(cfg["paths"]["bids_root"])
    ids = [s for s in sorted(set(ctx) & set(M.index) & set(FA.index))
           if ctx[s].shape[0] == sub[s].shape[0]
           and s in parts.index and not pd.isna(parts.loc[s, TARGET])]

    def cov(v):
        out = []
        for s in ids:
            x = np.hstack([ctx[s], sub[s]]).astype(np.float32)
            x = x - x.mean(0, keepdims=True)
            S = (x.T @ x) / (len(x) - 1)
            d = S.shape[0]
            out.append((1 - v) * S + v * (np.trace(S) / d) * np.eye(d, dtype=np.float32))
        return torch.from_numpy(np.stack(out)).to(dev).double()

    y = np.array([float(parts.loc[s, TARGET]) for s in ids])
    sex = np.array([1 if str(parts.loc[s, "sex"]).lower().startswith("m") else 0
                    for s in ids])
    Mo = M.loc[ids]; Mo = Mo.fillna(Mo.median()).values.astype(np.float32)
    Fa = FA.loc[ids].values.astype(np.float32)
    return ids, cov(SHRINK_NEW), cov(SHRINK_OLD), Mo, Fa, y, sex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--series", choices=["legacy", "psc"], default="legacy",
                    help="timeseries for the FC block (see load_all)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(CFG_PATH))

    print("Loading ...", flush=True)
    ids, C_new, C_old, Mo, Fa, y, sex = load_all(cfg, dev, args.series)
    print(f"  FC series: {args.series}", flush=True)
    print(f"  {len(ids)} subjects | morph {Mo.shape} | fa {Fa.shape}")

    # Log-Euclidean has no train-fold dependence -> compute once, outside the loop
    X_fc_le = vec_sym(sym_funcm(C_new, torch.log)).cpu().numpy()
    print(f"  log-Euclidean FC features: {X_fc_le.shape}")

    q = pd.qcut(y, 4, labels=False, duplicates="drop")
    strat = np.asarray(q) * 2 + sex
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats,
                                   random_state=SEED)

    rows = []
    for k, (tr, te) in enumerate(rskf.split(np.zeros(len(y)), strat), 1):
        t0 = time.time()
        rec = {"fold": k}

        # ---- legacy arm: AIRM + shrinkage 0.15 + Ridge --------------------
        G = frechet_mean(C_old[torch.as_tensor(tr, device=dev)])
        Gis = sym_funcm(G.unsqueeze(0), lambda w: w.rsqrt())[0]
        X_old = vec_sym(sym_funcm(Gis @ C_old @ Gis, torch.log)).cpu().numpy()
        p_leg = ridge_predict(X_old[tr], y[tr], X_old[te])
        rec["legacy_fc"] = float(pearsonr(y[te], p_leg)[0])

        # ---- tuned arm ----------------------------------------------------
        blocks = {"fc": X_fc_le, "morph": Mo, "fa": Fa}
        preds, oofs = {}, {}
        for nm, X in blocks.items():
            for dec in ("ridge", "krr"):
                p = (krr_predict(X[tr], y[tr], X[te], dev) if dec == "krr"
                     else ridge_predict(X[tr], y[tr], X[te]))
                rec[f"{nm}_{dec}"] = float(pearsonr(y[te], p)[0])
                preds[(nm, dec)] = p
            # pick each block's decoder by inner CV, not by test performance
            best_dec = max(("ridge", "krr"),
                           key=lambda d: pearsonr(
                               y[tr], oof(blocks[nm], y, tr, d, dev))[0])
            rec[f"{nm}_chosen"] = best_dec
            preds[nm] = preds[(nm, best_dec)]
            oofs[nm] = oof(blocks[nm], y, tr, best_dec, dev)

        names = list(blocks)
        Ztr = np.column_stack([oofs[n] for n in names])
        Zte = np.column_stack([preds[n] for n in names])
        rec["stack_tuned"] = float(pearsonr(
            y[te], ridge_predict(Ztr, y[tr], Zte))[0])

        # FC+FA only (was statistically tied with the 3-block stack)
        Z2t = np.column_stack([oofs[n] for n in ("fc", "fa")])
        Z2e = np.column_stack([preds[n] for n in ("fc", "fa")])
        rec["stack_fc_fa"] = float(pearsonr(
            y[te], ridge_predict(Z2t, y[tr], Z2e))[0])

        rec["secs"] = time.time() - t0
        rows.append(rec)
        print(f"  fold {k:2d}/{args.repeats*5}  legacy_fc={rec['legacy_fc']:+.3f}  "
              f"fc_krr={rec['fc_krr']:+.3f}  stack={rec['stack_tuned']:+.3f}  "
              f"({rec['secs']:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    _sfx = "" if args.series == "legacy" else f"_{args.series}"
    df.to_csv(OUT_DIR / f"multimodal_tuned{_sfx}_folds.csv", index=False)

    num = [c for c in df.columns if c not in ("fold", "secs")
           and not c.endswith("_chosen")]
    print(f"\n{'='*72}\nTuned multimodal, {args.repeats}x5-fold CV (n={len(ids)})\n{'='*72}")
    for c in sorted(num, key=lambda c: -df[c].mean()):
        print(f"  {c:16s} r = {df[c].mean():+.4f} +/- {df[c].std(ddof=1):.4f}")

    print(f"\n  decoder chosen per block (by inner CV):")
    for nm in ("fc", "morph", "fa"):
        print(f"    {nm:6s} {df[f'{nm}_chosen'].value_counts().to_dict()}")

    print(f"\n  paired tests vs legacy_fc (AIRM + Ridge + shrink 0.15):")
    for c in num:
        if c == "legacy_fc":
            continue
        t, p = corrected_ttest_rel(df[c], df["legacy_fc"])
        print(f"    {c:16s} delta = {(df[c]-df['legacy_fc']).mean():+.4f}  "
              f"t = {t:+5.2f}  p = {p:.3g}  wins {int((df[c]>df['legacy_fc']).sum())}/{len(df)}")

    print(f"\nSaved -> {OUT_DIR}/multimodal_tuned{_sfx}_folds.csv")


if __name__ == "__main__":
    main()
