#!/usr/bin/env python
"""
Split-half reliability of connectivity and amplitude features, and prediction across disjoint halves of the run.

Usage
-----
  python scripts/models/crosstime_reliability.py --repeats 3
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
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.data.preprocessing import load_participants
from src.data.series import roi_networks
from src.stats.cv_inference import corrected_ttest_rel

warnings.filterwarnings("ignore")

CFG_PATH = "config/pipeline.yaml"
TARGET   = "IST_intelligence_total"
ALPHAS   = np.logspace(-2, 6, 25)
SEED     = 42
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


def features(series, dev):
    """Ledoit-Wolf covariance -> log-Euclidean tangent, plus log amplitude."""
    covs = [ledoit_wolf(x, assume_centered=False)[0] for x in series]
    C = torch.from_numpy(np.stack(covs)).to(dev).double()
    fc = upper(sym_funcm(C, torch.log)).cpu().numpy()
    amp = np.stack([np.log(np.maximum(x.var(0, ddof=1), 1e-12)) for x in series])
    return fc, amp.astype(np.float32)


def ridge_pred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    m = RidgeCV(alphas=ALPHAS, cv=5).fit(sc.transform(Xtr), ytr)
    return m.predict(sc.transform(Xte))


def colwise_corr(A, B):
    """Correlation between matching columns of A and B, across rows."""
    A = A - A.mean(0, keepdims=True); B = B - B.mean(0, keepdims=True)
    num = (A * B).sum(0)
    den = np.sqrt((A ** 2).sum(0) * (B ** 2).sum(0))
    return num / np.clip(den, 1e-12, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(CFG_PATH))
    conn = Path(cfg["paths"]["connectivity_dir"])
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

    T = raw[ids[0]].shape[0]
    h = T // 2
    N, R = len(ids), raw[ids[0]].shape[1]
    print(f"n = {N}, {T} volumes split into halves of {h}, {R} regions")

    fc_full, amp_full = features([raw[s] for s in ids], dev)
    fc_A, amp_A = features([raw[s][:h] for s in ids], dev)
    fc_B, amp_B = features([raw[s][h:2 * h] for s in ids], dev)
    print(f"  features built: FC {fc_full.shape}, amplitude {amp_full.shape}",
          flush=True)

    # ---------------- PART A: reliability -----------------------------------
    print(f"\n{'='*78}\nPART A  split-half reliability of the features\n{'='*78}")
    n_diag = R
    rel_fc = colwise_corr(fc_A, fc_B)
    rel_amp = colwise_corr(amp_A, amp_B)
    rel_diag, rel_edge = rel_fc[:n_diag], rel_fc[n_diag:]
    rows = []
    for nm, v in [("amplitude (215)", rel_amp), ("tangent diagonal (215)", rel_diag),
                  ("tangent edges (23005)", rel_edge), ("tangent all", rel_fc)]:
        rows.append({"feature_set": nm, "n": len(v), "mean": v.mean(),
                     "median": np.median(v), "q25": np.percentile(v, 25),
                     "q75": np.percentile(v, 75), "frac_above_0.5": (v > 0.5).mean()})
        print(f"  {nm:24s} mean {v.mean():+.3f}  median {np.median(v):+.3f}  "
              f"IQR [{np.percentile(v,25):+.3f}, {np.percentile(v,75):+.3f}]  "
              f"frac>0.5 {100*(v>0.5).mean():.0f}%")
    pd.DataFrame(rows).to_csv(OUT_DIR / "crosstime_reliability_features.csv", index=False)

    prof_amp = np.array([pearsonr(amp_A[i], amp_B[i])[0] for i in range(N)])
    prof_fc = np.array([pearsonr(fc_A[i], fc_B[i])[0] for i in range(N)])
    print(f"\n  WITHIN-subject profile stability across halves (a different, easier "
          f"quantity):")
    print(f"    amplitude profile  mean r = {prof_amp.mean():+.3f}")
    print(f"    tangent FC profile mean r = {prof_fc.mean():+.3f}")
    print(f"  Between-subject feature reliability above is the number that bounds "
          f"prediction;\n  the within-subject figure is inflated by the common "
          f"spatial pattern everyone shares.", flush=True)

    nets = np.array(roi_networks(conn, R))
    pd.DataFrame({"roi": np.arange(R), "network": nets,
                  "amp_reliability": rel_amp, "diag_reliability": rel_diag}
                 ).to_csv(OUT_DIR / "crosstime_reliability_regions.csv", index=False)
    print("\n  amplitude reliability by network:")
    for g in sorted(set(nets)):
        m = nets == g
        print(f"    {g:12s} {rel_amp[m].mean():+.3f}")

    # ---------------- PART B: cross-time prediction -------------------------
    print(f"\n{'='*78}\nPART B  cross-time prediction\n{'='*78}", flush=True)
    q = pd.qcut(y, 4, labels=False, duplicates="drop")
    strat = np.asarray(q) * 2 + sex
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats, random_state=SEED)
    arms = {"full": (fc_full, fc_full, amp_full, amp_full),
            "A_A": (fc_A, fc_A, amp_A, amp_A),
            "B_B": (fc_B, fc_B, amp_B, amp_B),
            "A_B": (fc_A, fc_B, amp_A, amp_B),
            "B_A": (fc_B, fc_A, amp_B, amp_A)}
    rows = []
    for k, (tr, te) in enumerate(rskf.split(np.zeros(N), strat), 1):
        rec = {"fold": k}
        lr = LinearRegression().fit(Zm[tr], y[tr])
        ytr, yte = y[tr] - lr.predict(Zm[tr]), y[te] - lr.predict(Zm[te])
        for nm, (Ftr, Fte, Atr, Ate) in arms.items():
            rec["fc_" + nm] = float(pearsonr(yte, ridge_pred(Ftr[tr], ytr, Fte[te]))[0])
            rec["amp_" + nm] = float(pearsonr(yte, ridge_pred(Atr[tr], ytr, Ate[te]))[0])
        rows.append(rec)
        print(f"  fold {k:2d}/{args.repeats*5}  fc full={rec['fc_full']:+.3f} "
              f"A_A={rec['fc_A_A']:+.3f} A_B={rec['fc_A_B']:+.3f}", flush=True)

    df = pd.DataFrame(rows); df.to_csv(OUT_DIR / "crosstime_folds.csv", index=False)
    print(f"\n  {'arm':10s} {'FC':>18s} {'amplitude':>18s}")
    for nm in arms:
        print(f"  {nm:10s} {df['fc_'+nm].mean():+9.4f} +/- {df['fc_'+nm].std(ddof=1):.4f}"
              f" {df['amp_'+nm].mean():+9.4f} +/- {df['amp_'+nm].std(ddof=1):.4f}")
    same_fc = (df["fc_A_A"] + df["fc_B_B"]) / 2
    cross_fc = (df["fc_A_B"] + df["fc_B_A"]) / 2
    same_a = (df["amp_A_A"] + df["amp_B_B"]) / 2
    cross_a = (df["amp_A_B"] + df["amp_B_A"]) / 2
    for nm, s_, c_ in [("FC", same_fc, cross_fc), ("amplitude", same_a, cross_a)]:
        t, p = corrected_ttest_rel(c_, s_)
        print(f"\n  {nm}: same-segment {s_.mean():+.4f}  cross-segment {c_.mean():+.4f}"
              f"   delta {c_.mean()-s_.mean():+.4f}  t={t:+.2f}  p={p:.3g}")
        print(f"    cross-segment retains {100*c_.mean()/s_.mean():.0f}% of same-segment")
    print(f"\nSaved -> {OUT_DIR}/crosstime_folds.csv, "
          f"crosstime_reliability_features.csv, crosstime_reliability_regions.csv")


if __name__ == "__main__":
    main()
