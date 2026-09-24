#!/usr/bin/env python
"""
Single-use evaluation of the four frozen models on the sealed hold-out set.

Every train-dependent step is fitted on the discovery participants alone, including the reliability
exponent, which was selected earlier on the discovery split and is not re-tuned here.

Usage
-----
  python scripts/prediction/evaluate_holdout.py
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
from scipy.stats import pearsonr
from sklearn.covariance import ledoit_wolf
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from src.data.preprocessing import load_participants

warnings.filterwarnings("ignore")

CFG_PATH = "config/pipeline.yaml"
TARGET   = "IST_intelligence_total"
ALPHAS   = np.logspace(-2, 6, 25)
SEED     = 42
SHRINK   = 0.55
GAMMA    = 0.5
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


def sym_funcm(X, fn):
    X = 0.5 * (X + X.transpose(-1, -2)).double()
    w, V = torch.linalg.eigh(X)
    return V @ torch.diag_embed(fn(w.clamp_min(1e-10))) @ V.transpose(-1, -2)


def upper(L):
    n = L.shape[-1]
    iu = torch.triu_indices(n, n, offset=1, device=L.device)
    return torch.cat([torch.diagonal(L, dim1=-2, dim2=-1),
                      L[..., iu[0], iu[1]] * np.sqrt(2.0)], dim=-1).float()


def tangent(series, dev, lw=False):
    if lw:
        covs = [ledoit_wolf(x, assume_centered=False)[0] for x in series]
    else:
        covs = []
        for x in series:
            x = x - x.mean(0, keepdims=True)
            S = (x.T @ x) / (len(x) - 1)
            d = S.shape[0]
            covs.append((1 - SHRINK) * S + SHRINK * (np.trace(S) / d) * np.eye(d))
    C = torch.from_numpy(np.stack(covs)).to(dev).double()
    return upper(sym_funcm(C, torch.log)).cpu().numpy()


def colwise_corr(A, B):
    A = A - A.mean(0, keepdims=True); B = B - B.mean(0, keepdims=True)
    return (A * B).sum(0) / np.clip(
        np.sqrt((A ** 2).sum(0) * (B ** 2).sum(0)), 1e-12, None)


def ridge_fit_pred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    m = RidgeCV(alphas=ALPHAS, cv=5).fit(sc.transform(Xtr), ytr)
    return m.predict(sc.transform(Xte))


def krr_fit_pred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    best, bs = None, -np.inf
    for g in [1e-4, 1e-3, 1e-2]:
        for a in [0.1, 1.0, 10.0]:
            sc_ = []
            for tr_, te_ in KFold(3, shuffle=True, random_state=SEED).split(A):
                m = KernelRidge(kernel="rbf", gamma=g, alpha=a).fit(A[tr_], ytr[tr_])
                sc_.append(pearsonr(ytr[te_], m.predict(A[te_]))[0])
            if np.mean(sc_) > bs:
                bs, best = np.mean(sc_), (g, a)
    g, a = best
    return KernelRidge(kernel="rbf", gamma=g, alpha=a).fit(A, ytr).predict(B)


def ci95(r, n):
    z = np.arctanh(r); se = 1 / np.sqrt(n - 3)
    return np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluation-out", default=str(OUT_DIR / "holdout_evaluation.csv"))
    ap.add_argument("--save-predictions", default=None)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(CFG_PATH))
    conn = Path(cfg["paths"]["connectivity_dir"])
    man = json.loads(MANIFEST.read_text())
    disc_all, hold_all = man["discovery"], man["holdout"]
    print(f"manifest sealed {man['sealed_utc']}  sha256 {man['holdout_sha256'][:16]}...")

    raw = {}
    with h5py.File(conn / "unscrubbed_ts.h5", "r") as f:
        for s in disc_all + hold_all:
            if s in f["subjects"]:
                raw[s] = f["subjects"][s]["time_series"][:].astype(np.float64)

    frames = []
    for f in sorted(FS_DIR.glob("data-*.tsv")):
        d = pd.read_csv(f, sep="\t"); d = d.rename(columns={d.columns[0]: "sid"})
        d["sid"] = d["sid"].astype(str); d = d.set_index("sid")
        d = d.loc[:, ~d.columns.duplicated()]
        d.columns = [f"{f.stem}::{c}" for c in d.columns]; frames.append(d)
    M = pd.concat(frames, axis=1)
    M = M.loc[:, ~M.columns.duplicated()].apply(pd.to_numeric, errors="coerce")
    M = M.dropna(axis=1, how="all"); M = M.loc[:, M.std(skipna=True) > 0]
    vol_col = [c for c in M.columns if c.endswith("::" + VOL_COL)][0]
    parts = load_participants(cfg["paths"]["bids_root"])
    z = np.load(conn / "dwi_fa_features.npz", allow_pickle=True)
    FA = pd.DataFrame(z["features"], index=[str(s) for s in z["subject_ids"]])

    def usable(lst):
        return [s for s in lst if s in raw and s in M.index and s in FA.index
                and not pd.isna(M.loc[s, vol_col])
                and not pd.isna(parts.loc[s, TARGET])]
    disc, hold = usable(disc_all), usable(hold_all)
    ids = disc + hold
    ntr, nte = len(disc), len(hold)
    print(f"  discovery {ntr} of {len(disc_all)};  hold-out {nte} of {len(hold_all)} "
          f"(subjects lacking FreeSurfer or DWI are dropped from both)")

    y = np.array([float(parts.loc[s, TARGET]) for s in ids])
    sex = np.array([1 if str(parts.loc[s, "sex"]).lower().startswith("m") else 0 for s in ids])
    vol = M.loc[ids, vol_col].values.astype(float)
    age = pd.to_numeric(parts.loc[ids, "age"], errors="coerce").values.astype(float)
    age = np.where(np.isfinite(age), age, np.nanmedian(age))
    mot = np.array([motion_stats(s, cfg["paths"]["fmriprep_dir"],
                                 cfg["dataset"]["task"]) for s in ids])
    for j in range(mot.shape[1]):
        mot[:, j] = np.where(np.isfinite(mot[:, j]), mot[:, j], np.nanmedian(mot[:, j]))
    vzs = (vol - vol.mean()) / vol.std()
    CONF = np.column_stack([sex, age, vzs, vzs ** 2, mot])

    tr = np.arange(ntr); te = np.arange(ntr, ntr + nte)
    T = raw[ids[0]].shape[0]; h = T // 2
    X_fc = tangent([raw[s] for s in ids], dev)
    XA = tangent([raw[s][:h] for s in ids], dev, lw=True)
    XB = tangent([raw[s][h:2 * h] for s in ids], dev, lw=True)
    Mo = M.loc[ids]; Mo = Mo.fillna(Mo.median()).values.astype(np.float32)
    Fa = FA.loc[ids].values.astype(np.float32)
    print(f"  features: FC {X_fc.shape}, morphometry {Mo.shape}, FA {Fa.shape}",
          flush=True)

    lr = LinearRegression().fit(CONF[tr], y[tr])
    ytr_d = y[tr] - lr.predict(CONF[tr])
    yte_d = y[te] - lr.predict(CONF[te])

    results = []
    predictions = {}

    def record(name, pred, pred_d):
        predictions[name] = (np.asarray(pred, float), np.asarray(pred_d, float))
        r = float(pearsonr(y[te], pred)[0])
        rd = float(pearsonr(yte_d, pred_d)[0])
        lo, hi = ci95(r, nte)
        results.append({"arm": name, "r": r, "ci_lo": lo, "ci_hi": hi,
                        "r_deconfounded": rd, "n_holdout": nte})
        print(f"  {name:18s} r = {r:+.4f}  [95% CI {lo:+.3f}, {hi:+.3f}]"
              f"   deconfounded {rd:+.4f}", flush=True)

    record("confounds", ridge_fit_pred(CONF[tr], y[tr], CONF[te]),
           ridge_fit_pred(CONF[tr], ytr_d, CONF[te]))
    record("fc_ridge", ridge_fit_pred(X_fc[tr], y[tr], X_fc[te]),
           ridge_fit_pred(X_fc[tr], ytr_d, X_fc[te]))

    rel = np.clip(colwise_corr(XA[tr], XB[tr]), 0.0, 1.0) ** GAMMA
    sc = StandardScaler().fit(X_fc[tr])
    Ztr, Zte = sc.transform(X_fc[tr]) * rel, sc.transform(X_fc[te]) * rel
    record("fc_relweighted",
           RidgeCV(alphas=ALPHAS, cv=5).fit(Ztr, y[tr]).predict(Zte),
           RidgeCV(alphas=ALPHAS, cv=5).fit(Ztr, ytr_d).predict(Zte))

    # ---- tuned multimodal stack, all choices made on the discovery data ----
    blocks = {"fc": X_fc, "morph": Mo, "fa": Fa}
    for tag, yy_tr, yy_te in [("", y[tr], y[te]), ("_dec", ytr_d, yte_d)]:
        oof, test_pred, picks = {}, {}, {}
        for nm, Xb in blocks.items():
            scores = {"ridge": [], "krr": []}
            for a, b in KFold(5, shuffle=True, random_state=SEED).split(tr):
                for dec, fn in (("ridge", ridge_fit_pred), ("krr", krr_fit_pred)):
                    p_ = fn(Xb[tr][a], yy_tr[a], Xb[tr][b])
                    scores[dec].append(pearsonr(yy_tr[b], p_)[0])
            dec = max(scores, key=lambda d: np.mean(scores[d]))
            picks[nm] = dec
            fn = ridge_fit_pred if dec == "ridge" else krr_fit_pred
            o = np.zeros(ntr)
            for a, b in KFold(5, shuffle=True, random_state=SEED).split(tr):
                o[b] = fn(Xb[tr][a], yy_tr[a], Xb[tr][b])
            oof[nm] = o
            test_pred[nm] = fn(Xb[tr], yy_tr, Xb[te])
        Zt = np.column_stack([oof[n] for n in blocks])
        Ze = np.column_stack([test_pred[n] for n in blocks])
        pred = ridge_fit_pred(Zt, yy_tr, Ze)
        if tag == "":
            stack_raw, stack_picks = pred, picks
        else:
            stack_dec = pred
    print(f"  per-block decoder chosen on discovery: {stack_picks}")
    record("stack_tuned", stack_raw, stack_dec)

    df = pd.DataFrame(results)
    df.to_csv(args.evaluation_out, index=False)
    if args.save_predictions:
        pred_df = pd.DataFrame({
            "subject_id": [ids[i] for i in te],
            "sex_male": sex[te],
            "y": y[te],
            "y_deconfounded": yte_d,
            "y_discovery_mean": float(y[tr].mean()),
            "ytr_d_discovery_mean": float(ytr_d.mean()),
        })
        for name, (p_raw, p_dec) in predictions.items():
            pred_df[f"pred_{name}"] = p_raw
            pred_df[f"pred_{name}_dec"] = p_dec
        pred_df.to_csv(args.save_predictions, index=False)
        print(f"  per-subject predictions -> {args.save_predictions}")
    bar = "=" * 76
    print(f"\n{bar}\nSEALED HOLD-OUT EVALUATION (n = {nte}) -- single use, now spent"
          f"\n{bar}")
    print(df.to_string(index=False))
    print(f"\n  For comparison, the discovery-phase cross-validated figures were")
    print(f"  stack_tuned +0.4344 and fc_ridge +0.4032 on the full sample.")
    print(f"  A lower number here is expected: those were measured after ~50")
    print(f"  variants had been scored on the same folds.")
    print(f"\nSaved -> {args.evaluation_out}")


if __name__ == "__main__":
    main()
