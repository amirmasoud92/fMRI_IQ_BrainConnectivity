#!/usr/bin/env python
"""
Evaluate the frozen pipeline in each PIOP collection against the accuracy specified in advance.

Runs differ in repetition time, so they are combined by averaging their tangent vectors weighted by
duration rather than by concatenating time series.

Arms
----
  confounds     sex, ICV, ICV^2, three motion summaries -- the floor
  longest_run   the single longest run alone
  pooled        all runs, minute-weighted        <-- PRIMARY, pre-registered
  pooled_amp    log amplitude, pooled the same way
  pooled_corr   correlation only, amplitude removed

The longest_run vs pooled contrast is the sharp part of the prediction: the
ceiling law says tripling the data buys almost nothing.

Usage
-----
  python scripts/generalisation/piop_frozen_evaluation.py --dataset PIOP1 --repeats 3
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from src.data.paths import collection_root
from src.stats.cv_inference import corrected_ttest_rel

warnings.filterwarnings("ignore")

ROOTS = {name: collection_root(name) for name in ("PIOP1", "PIOP2")}
TS_DIR = OUT_DIR = Path("outputs/piop")
PREREG = Path("outputs/honest/piop_prereg.json")
TARGET = "raven_score"
ALPHAS = np.logspace(-2, 6, 25)
SEED = 42
ICV = "EstimatedTotalIntraCranialVol"


def sym_funcm(X, fn):
    X = 0.5 * (X + X.transpose(-1, -2)).double()
    w, V = torch.linalg.eigh(X)
    return V @ torch.diag_embed(fn(w.clamp_min(1e-10))) @ V.transpose(-1, -2)


def upper(L):
    n = L.shape[-1]
    iu = torch.triu_indices(n, n, offset=1, device=L.device)
    return torch.cat([torch.diagonal(L, dim1=-2, dim2=-1),
                      L[..., iu[0], iu[1]] * np.sqrt(2.0)], dim=-1).float()


def run_features(x, dev):
    """One run -> (tangent of C, tangent of R, log amplitude)."""
    C = torch.from_numpy(ledoit_wolf(x, assume_centered=False)[0]).to(dev).double()
    sd = torch.sqrt(torch.diagonal(C))
    R = C / (sd.unsqueeze(-1) * sd.unsqueeze(-2))
    return (upper(sym_funcm(C.unsqueeze(0), torch.log))[0].cpu().numpy(),
            upper(sym_funcm(R.unsqueeze(0), torch.log))[0].cpu().numpy(),
            torch.log(sd).float().cpu().numpy())


def motion_from_fd(fd):
    fd = np.asarray(fd, float)[1:]
    if fd.size == 0:
        return [np.nan] * 3
    return [fd.mean(), fd.max(), (fd > 0.5).mean()]


def build(h5, ids, dev):
    """Minute-weighted pooled features, plus the single longest run, per subject."""
    keep, pooled, p_corr, p_amp, longest, mot, minutes, nruns = ([] for _ in range(8))
    with h5py.File(h5, "r") as f:
        for s in ids:
            if s not in f["subjects"] or len(f["subjects"][s]) == 0:
                continue
            fcs, cors, amps, w, fds, best = [], [], [], [], [], (-1.0, None)
            for task in sorted(f["subjects"][s]):
                g = f["subjects"][s][task]
                x = g["time_series"][:].astype(np.float64)
                mins = x.shape[0] * float(g.attrs["tr"]) / 60.0
                fc, co, am = run_features(x, dev)
                fcs.append(fc); cors.append(co); amps.append(am); w.append(mins)
                fds.append(motion_from_fd(g["fd"][:]))
                if mins > best[0]:
                    best = (mins, fc)
            w = np.asarray(w, float)
            wn = w / w.sum()
            keep.append(s)
            pooled.append(np.average(np.stack(fcs), axis=0, weights=wn))
            p_corr.append(np.average(np.stack(cors), axis=0, weights=wn))
            p_amp.append(np.average(np.stack(amps), axis=0, weights=wn))
            longest.append(best[1])
            mot.append(np.average(np.asarray(fds, float), axis=0, weights=wn))
            minutes.append(w.sum()); nruns.append(len(w))
    return (keep, np.stack(pooled), np.stack(p_corr), np.stack(p_amp),
            np.stack(longest), np.asarray(mot, float),
            np.asarray(minutes), np.asarray(nruns))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(ROOTS), required=True)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root, h5 = ROOTS[args.dataset], TS_DIR / f"{args.dataset}_ts.h5"
    if not h5.exists():
        sys.exit(f"extraction not finished: {h5} missing")

    parts = pd.read_csv(root / "participants.tsv", sep="\t",
                        na_values="n/a").set_index("participant_id").dropna(subset=[TARGET])
    aseg = pd.read_csv(root / "derivatives" / "fs_stats" /
                       "data-subcortical_type-aseg_measure-volume_hemi-both.tsv", sep="\t")
    aseg = aseg.rename(columns={aseg.columns[0]: "sid"}).set_index("sid")
    icv = pd.to_numeric(aseg[ICV], errors="coerce")
    ids0 = [s for s in parts.index if s in icv.index and np.isfinite(icv.get(s, np.nan))]

    ids, pooled, p_corr, p_amp, longest, mot, minutes, nruns = build(h5, ids0, dev)
    y = parts.loc[ids, TARGET].to_numpy(float)
    sex = np.array([1 if str(parts.loc[s, "sex"]).upper().startswith("M") else 0
                    for s in ids])
    for j in range(mot.shape[1]):
        mot[:, j] = np.where(np.isfinite(mot[:, j]), mot[:, j], np.nanmedian(mot[:, j]))
    v = icv.loc[ids].to_numpy(float)
    vz = (v - v.mean()) / v.std()
    Zc = np.column_stack([sex, vz, vz ** 2, mot])

    pre = json.load(open(PREREG))["datasets"][args.dataset]
    bar = "=" * 78
    print(bar)
    print(f"{args.dataset}: PRIMARY PRE-REGISTERED TEST")
    print(bar)
    print(f"  n = {len(ids)} subjects with Raven, ICV and at least one run")
    print(f"  runs per subject: median {np.median(nruns):.0f} "
          f"[{nruns.min()}-{nruns.max()}]")
    print(f"  minutes per subject: median {np.median(minutes):.1f} "
          f"[{minutes.min():.1f}-{minutes.max():.1f}]   "
          f"(pre-registration assumed {pre['total_minutes']:.1f})")
    print(f"\n  PREDICTED (locked {json.load(open(PREREG))['registered']}): "
          f"r in [{pre['predicted_r_concat_lower']:+.3f}, "
          f"{pre['predicted_r_concat_upper']:+.3f}]")
    print(f"  and pooling every run should beat the longest single run by ~+0.01\n")

    X = {"pooled": pooled, "longest_run": longest,
         "pooled_corr": p_corr, "pooled_amp": p_amp}
    q = pd.qcut(y, 4, labels=False, duplicates="drop")
    strat = np.asarray(q) * 2 + sex
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats, random_state=SEED)

    rows = []
    for k, (tr, te) in enumerate(rskf.split(np.zeros(len(ids)), strat), 1):
        rec = {"fold": k}
        lr = LinearRegression().fit(Zc[tr], y[tr])
        ytr_d, yte_d = y[tr] - lr.predict(Zc[tr]), y[te] - lr.predict(Zc[te])
        rec["confounds"] = float(pearsonr(y[te], lr.predict(Zc[te]))[0])
        for a, M in X.items():
            sc = StandardScaler().fit(M[tr])
            A, B = sc.transform(M[tr]), sc.transform(M[te])
            rec[a] = float(pearsonr(
                y[te], RidgeCV(alphas=ALPHAS, cv=5).fit(A, y[tr]).predict(B))[0])
            rec[a + "_dec"] = float(pearsonr(
                yte_d, RidgeCV(alphas=ALPHAS, cv=5).fit(A, ytr_d).predict(B))[0])
        rows.append(rec)
        print(f"    fold {k:2d}/{args.repeats*5}  pooled={rec['pooled']:+.3f}  "
              f"longest={rec['longest_run']:+.3f}", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(OUT_DIR / f"{args.dataset}_frozen_folds.csv", index=False)

    print(f"\n{bar}\nRESULT ({args.repeats}x5 CV, n = {len(ids)})\n{bar}")
    print(f"  {'arm':14s} {'raw r':>18s} {'deconfounded':>18s}")
    print(f"  {'confounds':14s} {d.confounds.mean():+9.4f} "
          f"+/-{d.confounds.std(ddof=1):.3f}")
    for a in ["pooled", "longest_run", "pooled_corr", "pooled_amp"]:
        star = "  <-- PRIMARY" if a == "pooled" else ""
        print(f"  {a:14s} {d[a].mean():+9.4f} +/-{d[a].std(ddof=1):.3f} "
              f"{d[a+'_dec'].mean():+9.4f} +/-{d[a+'_dec'].std(ddof=1):.3f}{star}")

    obs = d["pooled"].mean()
    lo, hi = pre["predicted_r_concat_lower"], pre["predicted_r_concat_upper"]
    inside = lo <= obs <= hi
    print(f"\n  observed  {obs:+.4f}")
    print(f"  predicted [{lo:+.4f}, {hi:+.4f}]")
    print(f"  -> {'INSIDE the pre-registered interval' if inside else 'OUTSIDE'}"
          f"   (miss = {0.0 if inside else min(abs(obs-lo), abs(obs-hi)):+.4f})")

    gain = d["pooled"] - d["longest_run"]
    t, p = corrected_ttest_rel(d["pooled"], d["longest_run"])
    print(f"\n  pooling every run vs the single longest run: {gain.mean():+.4f}  "
          f"t={t:+.2f}  p={p:.3g}")
    print(f"  the ceiling law predicted about +0.01 for this gain.")
    print(f"\n  ID1000 sealed hold-out for reference: +0.4026 at n = 928, 10.6 min.")
    print(f"  {args.dataset} has {np.median(minutes)/10.6:.1f}x the scan time per "
          f"subject and {len(ids)/928:.2f}x the subjects.")
    print(f"\nSaved -> {OUT_DIR}/{args.dataset}_frozen_folds.csv")


if __name__ == "__main__":
    main()
