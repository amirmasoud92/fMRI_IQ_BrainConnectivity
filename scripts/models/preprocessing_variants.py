#!/usr/bin/env python
"""
Score preprocessing variants that each change one factor: smoothing, filter band or global signal regression.

Usage
-----
  python scripts/models/preprocessing_variants.py --repeats 3
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
from nilearn import signal as nsignal
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.data.preprocessing import CONFOUND_36P, load_participants
from src.stats.cv_inference import corrected_ttest_rel

warnings.filterwarnings("ignore")

CFG_PATH = "config/pipeline.yaml"
TARGET   = "IST_intelligence_total"
ALPHAS   = np.logspace(-2, 6, 25)
SEED     = 42
SHRINK   = 0.55
OUT_DIR  = Path("outputs/honest")
FS_DIR   = Path("derivatives/fs_stats")
VOL_COL  = "EstimatedTotalIntraCranialVol"

GSR_COLS = [c for c in CONFOUND_36P if c.startswith("global_signal")]

#  name      -> (smoothing key, low_pass, high_pass, drop confound columns)
ARMS = {
    "base":     ("smooth6", 0.10, 0.01, [], False),
    "wide":     ("smooth6", 0.25, 0.01, [], False),
    "hponly":   ("smooth6", None, 0.01, [], False),
    "nosmooth": ("smooth0", 0.10, 0.01, [], False),
    "nogsr":    ("smooth6", 0.10, 0.01, GSR_COLS, False),
    # At TR 2.2 s Nyquist is 0.227 Hz, so the band the `wide` arm opens up
    # (0.1-0.227 Hz) is exactly where cardiac (~1 Hz) and respiratory (~0.3 Hz)
    # signal ALIASES. That is the original reason for low-passing at 0.1. These
    # two arms add the RETROICOR regressors to the confound set: if the wide-band
    # gain is aliased physiology, it should collapse here; if it survives, the
    # extra spectrum is carrying real information.
    "base_phys": ("smooth6", 0.10, 0.01, [], True),
    "wide_phys": ("smooth6", 0.25, 0.01, [], True),
}
PHYS_DIR = Path("derivatives/physiology")


def load_physio(sub_id, task, n_tr):
    p = (PHYS_DIR / sub_id / "physio" /
         f"{sub_id}_task-{task}_recording-respcardiac_desc-retroicor_regressors.tsv")
    if not p.exists():
        return None
    try:
        d = pd.read_csv(p, sep="\t").apply(pd.to_numeric, errors="coerce")
    except Exception:
        return None
    if len(d) != n_tr:
        return None
    P = d.to_numpy(float)
    P = P[:, np.isfinite(P).all(0)]
    if P.size == 0:
        return None
    P = P[:, P.std(0) > 1e-12]
    return P if P.shape[1] else None


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


def embed(series, dev):
    covs = []
    for x in series:
        x = x - x.mean(0, keepdims=True)
        S = (x.T @ x) / max(len(x) - 1, 1)
        d = S.shape[0]
        covs.append((1 - SHRINK) * S + SHRINK * (np.trace(S) / d) * np.eye(d))
    C = torch.from_numpy(np.stack(covs)).to(dev).double()
    return upper(sym_funcm(C, torch.log)).cpu().numpy()


def ridge_pred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    return RidgeCV(alphas=ALPHAS, cv=5).fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))


def isc_stats(TS):
    """Mean leave-one-out ISC per region and the stimulus-locked variance share."""
    X = np.stack(TS)
    Xz = (X - X.mean(1, keepdims=True)) / np.clip(X.std(1, keepdims=True), 1e-8, None)
    N = len(Xz)
    Ssum = Xz.sum(0)
    out = np.empty((N, X.shape[2]))
    for i in range(N):
        tmpl = (Ssum - Xz[i]) / (N - 1)
        a = Xz[i] - Xz[i].mean(0, keepdims=True)
        b = tmpl - tmpl.mean(0, keepdims=True)
        den = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
        out[i] = (a * b).sum(0) / np.clip(den, 1e-12, None)
    return float(out.mean()), float(np.mean(out ** 2)), float(out.mean(0).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(CFG_PATH))
    conn = Path(cfg["paths"]["connectivity_dir"])
    tr_s = cfg["dataset"]["tr"]

    src = conn / "preproc_variants_raw.h5"
    if not src.exists():
        sys.exit(f"Missing {src} -- run scripts/extraction/extract_preprocessing_variants.py first")
    raw6, raw0, confs = {}, {}, {}
    with h5py.File(src, "r") as f:
        cols = json.loads(f.attrs["confound_columns"])
        for s in f["subjects"]:
            raw6[s] = f["subjects"][s]["smooth6"][:].astype(np.float64)
            raw0[s] = f["subjects"][s]["smooth0"][:].astype(np.float64)
            confs[s] = f["subjects"][s]["confounds"][:].astype(np.float64)
    ids = sorted(raw6)
    print(f"{len(ids)} subjects, {len(cols)} confound columns")

    task_name = cfg["dataset"]["task"]
    phys = {s: load_physio(s, task_name, len(raw6[s])) for s in ids}
    n_phys = sum(v is not None for v in phys.values())
    print(f"  RETROICOR available for {n_phys}/{len(ids)} subjects")

    def clean_arm(name):
        key, lp, hp, drop, use_phys = ARMS[name]
        srcd = raw6 if key == "smooth6" else raw0
        keep = [i for i, c in enumerate(cols) if c not in drop]
        out = []
        for s in ids:
            C = confs[s][:, keep]
            if use_phys and phys[s] is not None:
                C = np.hstack([C, phys[s]])
            out.append(nsignal.clean(srcd[s], detrend=True, standardize="psc",
                                     confounds=C, low_pass=lp, high_pass=hp,
                                     t_r=tr_s))
        return out

    # ---- validation: does timeseries-level cleaning reproduce the pipeline? --
    print("\nVALIDATION  base vs the stored image-level extraction")
    base_ts = clean_arm("base")
    with h5py.File(conn / "unscrubbed_ts.h5", "r") as f:
        ref = {s: f["subjects"][s]["time_series"][:].astype(np.float64)
               for s in ids if s in f["subjects"]}
    diffs, cors = [], []
    for i, s in enumerate(ids):
        if s not in ref or ref[s].shape != base_ts[i].shape:
            continue
        a, b = base_ts[i].ravel(), ref[s].ravel()
        diffs.append(np.abs(a - b).max())
        cors.append(np.corrcoef(a, b)[0, 1])
    if diffs:
        print(f"  {len(diffs)} subjects compared")
        print(f"  max |difference|  median {np.median(diffs):.2e}   worst {max(diffs):.2e}")
        print(f"  correlation       median {np.median(cors):.6f}   worst {min(cors):.6f}")
        ok = np.median(cors) > 0.999
        print(f"  -> commutation {'VALIDATED' if ok else 'FAILED -- results below are suspect'}")
    else:
        print("  [no overlapping subjects to validate against]")

    # ---- build every arm ---------------------------------------------------
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
    vz = (vol - vol.mean()) / vol.std()
    Zm = np.column_stack([sex, vz, vz ** 2, mot])

    print("\nARMS  (each differs from base in exactly one factor)")
    X, isc_rows = {}, []
    for nm in ARMS:
        ts = base_ts if nm == "base" else clean_arm(nm)
        X[nm] = embed(ts, dev)
        m_isc, m_isc2, mx = isc_stats(ts)
        isc_rows.append({"arm": nm, "mean_isc": m_isc,
                         "stim_frac": m_isc2, "max_region_isc": mx})
        key, lp, hp, drop, use_phys = ARMS[nm]
        print(f"  {nm:9s} {key} band {hp}-{lp if lp else 'inf'} Hz, "
              f"{len(cols)-len(drop)}{'+phys' if use_phys else ''} confounds | "
              f"mean ISC {m_isc:+.4f}, "
              f"stimulus-locked share {100*m_isc2:.1f}%", flush=True)
    pd.DataFrame(isc_rows).to_csv(OUT_DIR / "preproc_variants_isc.csv", index=False)

    q = pd.qcut(y, 4, labels=False, duplicates="drop")
    strat = np.asarray(q) * 2 + sex
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats, random_state=SEED)

    rows = []
    for k, (tr, te) in enumerate(rskf.split(np.zeros(len(y)), strat), 1):
        rec = {"fold": k}
        lr = LinearRegression().fit(Zm[tr], y[tr])
        ytr_d, yte_d = y[tr] - lr.predict(Zm[tr]), y[te] - lr.predict(Zm[te])
        for nm, Xb in X.items():
            rec[nm] = float(pearsonr(y[te], ridge_pred(Xb[tr], y[tr], Xb[te]))[0])
            rec[nm + "_dec"] = float(pearsonr(yte_d,
                                              ridge_pred(Xb[tr], ytr_d, Xb[te]))[0])
        rows.append(rec)
        print(f"  fold {k:2d}/{args.repeats*5}  " +
              "  ".join(f"{n}={rec[n]:+.3f}" for n in ARMS), flush=True)

    df = pd.DataFrame(rows); df.to_csv(OUT_DIR / "preproc_variants_folds.csv", index=False)
    bar = "=" * 76
    print(f"\n{bar}\nPreprocessing ablation, {args.repeats}x5 CV (n={len(ids)})\n{bar}")
    for nm in sorted(ARMS, key=lambda n: -df[n].mean()):
        print(f"  {nm:10s} r = {df[nm].mean():+.4f} +/- {df[nm].std(ddof=1):.4f}"
              f"   deconf {df[nm+'_dec'].mean():+.4f}")
    print("\n  paired vs base (identical folds):")
    for nm in ARMS:
        if nm == "base":
            continue
        d = df[nm] - df["base"]; t, p = corrected_ttest_rel(df[nm], df["base"])
        dd = df[nm + "_dec"] - df["base_dec"]
        print(f"    {nm:10s} {d.mean():+.4f}  t={t:+5.2f}  p={p:.3g}  "
              f"wins {int((d>0).sum())}/{len(df)}   | deconf {dd.mean():+.4f}")
    print(f"\nSaved -> {OUT_DIR}/preproc_variants_folds.csv, preproc_variants_isc.csv")


if __name__ == "__main__":
    main()
