#!/usr/bin/env python
"""
Regional amplitude prediction in every PIOP task, from rest to demanding tasks.

The Decomposition, Identical To Id1000
--------------------------------------
Each run's covariance is factorised as C = D R D with D = diag(sigma) the
regional amplitudes and R the correlation matrix, and three feature sets are
Usage
-----
  python scripts/generalisation/piop_state_amplitude.py --dataset PIOP1 --repeats 3
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
TS_DIR = Path("outputs/piop")
OUT_DIR = Path("outputs/piop")
PREREG = Path("outputs/honest/piop_prereg.json")
TARGET = "raven_score"
ALPHAS = np.logspace(-2, 6, 25)
SEED = 42
ICV = "EstimatedTotalIntraCranialVol"
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


def features(series, dev):
    """Return (log amplitude, tangent of correlation, tangent of covariance)."""
    covs = np.stack([ledoit_wolf(x, assume_centered=False)[0] for x in series])
    C = torch.from_numpy(covs).to(dev).double()
    sd = torch.sqrt(torch.diagonal(C, dim1=-2, dim2=-1))
    R = C / (sd.unsqueeze(-1) * sd.unsqueeze(-2))
    return (torch.log(sd).float().cpu().numpy(),
            upper(sym_funcm(R, torch.log)).cpu().numpy(),
            upper(sym_funcm(C, torch.log)).cpu().numpy())


def motion_from_fd(fd):
    fd = np.asarray(fd, float)[1:]
    if fd.size == 0:
        return [np.nan] * 3
    return [fd.mean(), fd.max(), (fd > 0.5).mean()]


def load_task(h5, task, ids):
    """Timeseries and per-run motion for every subject in ids that has this task."""
    keep, series, mot = [], [], []
    with h5py.File(h5, "r") as f:
        for s in ids:
            if s not in f["subjects"] or task not in f["subjects"][s]:
                continue
            g = f["subjects"][s][task]
            keep.append(s)
            series.append(g["time_series"][:].astype(np.float64))
            mot.append(motion_from_fd(g["fd"][:]))
    return keep, series, np.array(mot, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(ROOTS), required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--min-n", type=int, default=60,
                    help="skip a task with fewer subjects than this")
    ap.add_argument("--h5", default=None,
                    help="override the timeseries file (used to smoke-test on a "
                         "partial extraction; never for reported results)")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = ROOTS[args.dataset]
    h5 = Path(args.h5) if args.h5 else TS_DIR / f"{args.dataset}_ts.h5"
    if not h5.exists():
        sys.exit(f"extraction not finished: {h5} missing")

    parts = pd.read_csv(root / "participants.tsv", sep="\t",
                        na_values="n/a").set_index("participant_id")
    parts = parts.dropna(subset=[TARGET])
    aseg = pd.read_csv(
        root / "derivatives" / "fs_stats" /
        "data-subcortical_type-aseg_measure-volume_hemi-both.tsv", sep="\t")
    aseg = aseg.rename(columns={aseg.columns[0]: "sid"}).set_index("sid")
    icv = pd.to_numeric(aseg[ICV], errors="coerce")

    with h5py.File(h5, "r") as f:
        tasks = sorted({t for s in f["subjects"] for t in f["subjects"][s]})
    prereg = json.load(open(PREREG))["datasets"][args.dataset] if PREREG.exists() else None

    print("=" * 78)
    print(f"{args.dataset}: state dependence of the amplitude signature")
    print(f"target {TARGET}, {args.repeats}x5 CV, ID1000 pipeline frozen")
    print("=" * 78)

    rows = []
    for task in tasks:
        ids0 = [s for s in parts.index if s in icv.index and np.isfinite(icv.get(s, np.nan))]
        ids, series, mot = load_task(h5, task, ids0)
        if len(ids) < args.min_n:
            print(f"\n{task}: only {len(ids)} subjects, skipped")
            continue
        y = parts.loc[ids, TARGET].to_numpy(float)
        sex = np.array([1 if str(parts.loc[s, "sex"]).upper().startswith("M") else 0
                        for s in ids])
        for j in range(mot.shape[1]):
            mot[:, j] = np.where(np.isfinite(mot[:, j]), mot[:, j],
                                 np.nanmedian(mot[:, j]))
        v = icv.loc[ids].to_numpy(float)
        vz = (v - v.mean()) / v.std()
        Zc = np.column_stack([sex, vz, vz ** 2, mot])

        amp, cor, fc = features(series, dev)
        X = {"amplitude": amp, "corr_tangent": cor, "fc_tangent": fc}

        q = pd.qcut(y, 4, labels=False, duplicates="drop")
        strat = np.asarray(q) * 2 + sex
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats,
                                       random_state=SEED)
        fold = []
        for k, (tr, te) in enumerate(rskf.split(np.zeros(len(ids)), strat), 1):
            rec = {"fold": k}
            lr = LinearRegression().fit(Zc[tr], y[tr])
            ytr_d, yte_d = y[tr] - lr.predict(Zc[tr]), y[te] - lr.predict(Zc[te])
            rec["confounds"] = float(pearsonr(y[te], lr.predict(Zc[te]))[0])
            for a in ARMS:
                sc = StandardScaler().fit(X[a][tr])
                A, B = sc.transform(X[a][tr]), sc.transform(X[a][te])
                rec[a] = float(pearsonr(
                    y[te], RidgeCV(alphas=ALPHAS, cv=5).fit(A, y[tr]).predict(B))[0])
                rec[a + "_dec"] = float(pearsonr(
                    yte_d, RidgeCV(alphas=ALPHAS, cv=5).fit(A, ytr_d).predict(B))[0])
            fold.append(rec)
        d = pd.DataFrame(fold)
        d.insert(0, "task", task)
        rows.append(d)

        pred = (prereg["predicted_r_per_task"].get(task) if prereg else None)
        head = (f"\n{task}  (n = {len(ids)}, T = {series[0].shape[0]} volumes)"
                + (f"   pre-registered r = {pred:+.3f}" if pred else ""))
        print(head)
        print(f"    {'arm':14s} {'raw r':>16s} {'deconfounded':>16s}")
        print(f"    {'confounds':14s} {d.confounds.mean():+8.4f} "
              f"+/-{d.confounds.std(ddof=1):.3f}{'':>16s}")
        for a in ARMS:
            print(f"    {a:14s} {d[a].mean():+8.4f} +/-{d[a].std(ddof=1):.3f} "
                  f"{d[a + '_dec'].mean():+8.4f} +/-{d[a + '_dec'].std(ddof=1):.3f}")
        t, p = corrected_ttest_rel(d["fc_tangent"], d["corr_tangent"])
        print(f"    amplitude's contribution (fc - corr): "
              f"{(d['fc_tangent'] - d['corr_tangent']).mean():+.4f}  "
              f"t={t:+.2f}  p={p:.3g}")

    if not rows:
        sys.exit("no task had enough subjects")
    df = pd.concat(rows, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / f"{args.dataset}_state_folds.csv", index=False)

    print("\n" + "=" * 78)
    print("SUMMARY: does amplitude predict without a stimulus?")
    print("=" * 78)
    g = df.groupby("task")
    print(f"  {'task':16s} {'amplitude':>10s} {'corr only':>10s} {'full FC':>10s} "
          f"{'amp share':>10s}")
    for task, sub in g:
        share = (sub.amplitude.mean() / sub.fc_tangent.mean()
                 if sub.fc_tangent.mean() > 0 else np.nan)
        print(f"  {task:16s} {sub.amplitude.mean():+10.4f} "
              f"{sub.corr_tangent.mean():+10.4f} {sub.fc_tangent.mean():+10.4f} "
              f"{share:>9.0%}")
    rest = g.get_group("restingstate") if "restingstate" in g.groups else None
    if rest is not None:
        t, p = corrected_ttest_rel(rest.amplitude, rest.confounds)
        print(f"\n  AT REST, with no stimulus at all: amplitude r = "
              f"{rest.amplitude.mean():+.4f} vs confound floor "
              f"{rest.confounds.mean():+.4f}   t={t:+.2f}  p={p:.3g}")
        print("  If this is clearly above the floor, the signature is intrinsic and")
        print("  the ID1000 visual-cortex result is about the brain, not the film.")
    print(f"\nSaved -> {OUT_DIR}/{args.dataset}_state_folds.csv")


if __name__ == "__main__":
    main()
