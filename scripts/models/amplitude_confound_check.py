#!/usr/bin/env python
"""
Whether the regional amplitude signal survives control for brain volume and head motion.

Usage
-----
  python scripts/models/amplitude_confound_check.py --repeats 3
"""
from __future__ import annotations

import argparse, sys, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.data.preprocessing import load_participants
from src.stats.cv_inference import corrected_ttest_rel

warnings.filterwarnings("ignore")

CFG_PATH = "config/pipeline.yaml"
TARGET   = "IST_intelligence_total"
ALPHAS   = np.logspace(-2, 6, 25)
SEED     = 42
OUT_DIR  = Path("outputs/honest")
FS_DIR   = Path("derivatives/fs_stats")
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


def ridge_pred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    return RidgeCV(alphas=ALPHAS, cv=5).fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(CFG_PATH))
    conn = Path(cfg["paths"]["connectivity_dir"])

    up = conn / "unscrubbed_ts.h5"
    if not up.exists():
        sys.exit(f"Missing {up} -- run scripts/extraction/extract_timeseries_psc.py first")
    amp = {}
    with h5py.File(up, "r") as f:
        for s in f["subjects"]:
            x = f["subjects"][s]["time_series"][:]
            km = f["subjects"][s]["keep"][:].astype(bool)
            if km.sum() < 20:
                km = np.ones(len(x), bool)
            amp[s] = np.log(np.maximum(x[km].var(0, ddof=1), 1e-12))

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

    ids = [s for s in sorted(set(amp) & set(M.index))
           if s in parts.index and not pd.isna(parts.loc[s, TARGET])
           and not pd.isna(M.loc[s, vol_col])]
    AMP = np.stack([amp[s] for s in ids]).astype(np.float32)
    y   = np.array([float(parts.loc[s, TARGET]) for s in ids])
    sex = np.array([1 if str(parts.loc[s, "sex"]).lower().startswith("m") else 0 for s in ids])
    vol = M.loc[ids, vol_col].values.astype(float)
    age = pd.to_numeric(parts.loc[ids, "age"], errors="coerce").values.astype(float)
    age = np.where(np.isfinite(age), age, np.nanmedian(age))
    mot = np.array([motion_stats(s, cfg["paths"]["fmriprep_dir"],
                                 cfg["dataset"]["task"]) for s in ids])
    for j in range(mot.shape[1]):
        mot[:, j] = np.where(np.isfinite(mot[:, j]), mot[:, j], np.nanmedian(mot[:, j]))

    vz = (vol - vol.mean()) / vol.std()
    Z_sexmot = np.column_stack([sex, mot])                      # what the old script used
    Z_full   = np.column_stack([sex, age, vz, vz ** 2, mot])    # + size, age
    a_mean = AMP.mean(1)
    print(f"n = {len(ids)},  amplitude block = {AMP.shape[1]} features")
    print(f"  corr(mean log-amplitude, volume)    = {pearsonr(a_mean, vol)[0]:+.3f}")
    print(f"  corr(mean log-amplitude, mean FD)   = {pearsonr(a_mean, mot[:, 0])[0]:+.3f}")
    print(f"  corr(mean log-amplitude, spike frac)= {pearsonr(a_mean, mot[:, 2])[0]:+.3f}")
    print(f"  corr(volume, IST)                   = {pearsonr(vol, y)[0]:+.3f}", flush=True)

    q = pd.qcut(y, 4, labels=False, duplicates="drop")
    strat = np.asarray(q) * 2 + sex
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats, random_state=SEED)

    rows = []
    for k, (tr, te) in enumerate(rskf.split(np.zeros(len(y)), strat), 1):
        rec = {"fold": k}
        rec["amp_raw"] = float(pearsonr(y[te], ridge_pred(AMP[tr], y[tr], AMP[te]))[0])
        for tag, Z in [("sexmot", Z_sexmot), ("full", Z_full)]:
            lr = LinearRegression().fit(Z[tr], y[tr])
            rec[f"amp_dec_{tag}"] = float(pearsonr(
                y[te] - lr.predict(Z[te]),
                ridge_pred(AMP[tr], y[tr] - lr.predict(Z[tr]), AMP[te]))[0])
        rec["conf_only"] = float(pearsonr(y[te], ridge_pred(Z_full[tr], y[tr], Z_full[te]))[0])
        XA = np.hstack([AMP, Z_full])
        rec["amp_plus_conf"] = float(pearsonr(y[te], ridge_pred(XA[tr], y[tr], XA[te]))[0])
        rows.append(rec)
        print(f"  fold {k:2d}/{args.repeats*5}  raw={rec['amp_raw']:+.3f} "
              f"dec_full={rec['amp_dec_full']:+.3f} conf={rec['conf_only']:+.3f} "
              f"amp+conf={rec['amp_plus_conf']:+.3f}", flush=True)

    df = pd.DataFrame(rows); df.to_csv(OUT_DIR / "amplitude_confound_folds.csv", index=False)
    print(f"\n{'='*72}\nAmplitude vs confounds, {args.repeats}x5-fold CV (n={len(ids)})\n{'='*72}")
    for c in [c for c in df.columns if c != "fold"]:
        print(f"  {c:18s} r = {df[c].mean():+.4f} +/- {df[c].std(ddof=1):.4f}")
    d = df["amp_plus_conf"] - df["conf_only"]
    t, p = corrected_ttest_rel(df["amp_plus_conf"], df["conf_only"])
    print(f"\n  INCREMENT of amplitude over confounds: {d.mean():+.4f}  "
          f"t = {t:+.2f}  p = {p:.3g}  wins {int((d>0).sum())}/{len(df)}")
    print(f"  retained after sex+motion only : "
          f"{100*df['amp_dec_sexmot'].mean()/df['amp_raw'].mean():.0f}%")
    print(f"  retained after +size+age       : "
          f"{100*df['amp_dec_full'].mean()/df['amp_raw'].mean():.0f}%")
    print(f"\nSaved -> {OUT_DIR}/amplitude_confound_folds.csv")


if __name__ == "__main__":
    main()
