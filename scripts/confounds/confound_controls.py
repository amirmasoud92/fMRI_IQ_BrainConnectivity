#!/usr/bin/env python
"""
Confound controls for the connectivity model: non-linear confound terms, models within each sex, and test
sets in which the target is empirically independent of a confound.

Data and scope
--------------
Discovery subjects only (the sealed hold-out is never read). PSC series,
log-Euclidean tangent FC (shrinkage 0.55), FreeSurfer morphometry and global DWI-FA,
exactly as in the headline pipeline. Confounds are restricted to biological and
acquisition variables: sex, total intracranial volume
(eTIV), mean framewise displacement, max FD, fraction of spike frames (FD > 0.5).
Age, education and SES are deliberately NOT used: education proxies the target
(over-control). Morphometry imputation is fitted inside training folds.

Models: fc_ridge (FC -> ridge) and stack_ridge (FC, morphometry, FA -> ridge per block,
ridge meta-learner over 5-fold out-of-fold predictions). The stack uses ridge only
for every block, not the tuned per-block KRR selection of the headline model. That is
a documented simplification: KRR tied ridge on FC (+0.0026) and hurt morphometry.

A. Non-linear confounds (3x5 repeated CV, stratified IST quartile x sex)
   target raw | linear-deconfounded | spline-deconfounded, where
   linear = sex + eTIV + eTIV^2 + meanFD + maxFD + spike
   spline = sex + natural cubic splines (4 df, knots from the training fold) for eTIV,
            meanFD, maxFD; spike fraction stays linear if too few distinct values
   also: confound-only models (linear vs spline) and the imaging increment over each
   confound floor, r(stack[imaging, confounds]) - r(confounds).

B. Within sex (same folds)
   pooled model scored within each sex; sex-specific models (train and test within one
   sex); and an n-MATCHED pooled control, trained on a mixed-sex subsample of the
   training fold equal in size to that sex's training set. Specific vs n-matched
   isolates sex-specificity from the loss of training data.

C. Confound-isolating test sets (Chyzhyk, Varoquaux, Milham & Thirion 2022, GigaScience)
   Starting from a random half of the discovery subjects, subjects are removed in
   small steps by their density ratio p(y)p(z)/p(y,z) (Gaussian KDE; within-sex KDE when
   sex is involved) until the test set shows no dependence between IST and the
   confound. The model is trained on the remaining subjects. Each isolated set is
   paired with two RANDOM test sets drawn from the same starting pool: one of identical
   size, and one of identical size AND IST distribution (importance-sampled with
   weights p_isolated(IST) / p_start(IST)). Confounds: eTIV, sex, meanFD, and all
   three jointly.

   AMENDMENT (made during synthetic validation, BEFORE any real data were run):
   density-ratio removal reaches independence partly by trimming IST's extremes. In
   synthetic data IST's SD fell to 65% of the starting pool. A correlation on a
   range-restricted sample is mechanically lower, so comparing against a
   size-matched random set alone would bias C2 toward "signal lost" even with no
   confounding. The IST-distribution-matched set removes that artefact and is the
   PRIMARY comparison for C2; the size-matched comparison is still reported.

Pre-specified criteria (fixed before running; corrected inference, alpha .05)
-----------------------------------------------------------------------------
C1 isolation valid        every isolated set has |r(IST, confound)| <= 0.05 and >= 150
                          subjects; median mutual-information reduction >= 80%
C2 signal survives        fc_ridge r on isolated sets > 0 (corrected 95% CI excludes 0)
   isolation              AND retains >= 50% of the r on size- and IST-distribution-
                          matched random sets (amended, see C above)
C3 non-linear confounding spline-deconfounded fc r lower than linear-deconfounded by
   material                > 0.02 with corrected p < .05 (otherwise: not material)
C4 within-sex signal      sex-specific fc r > 0 with corrected CI excluding 0, in BOTH sexes
C5 sex-specificity        sex-specific vs n-matched pooled differs with corrected p < .05
C6 imaging increment      stack increment over the SPLINE confound floor > 0, corrected
   survives                CI excluding 0

Outputs (outputs/honest/)
-------------------------
  confound_control_folds.csv        A, one row per fold
  confound_control_withinsex.csv    B, one row per fold
  confound_control_isolation.csv    C, one row per isolated set (with matched random set)
  confound_control_summary.csv      every test and the C1-C6 verdicts

Usage
-----
  python scripts/confounds/confound_controls.py                    # full
  python scripts/confounds/confound_controls.py --repeats 1 --isolated 2 --jobs 4 --tag _smoke
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import numpy as np
import pandas as pd
import patsy
import yaml
from joblib import Parallel, delayed
from scipy.stats import gaussian_kde, pearsonr
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.data.preprocessing import load_participants

warnings.filterwarnings("ignore")

CFG_PATH = "config/pipeline.yaml"
TARGET = "IST_intelligence_total"
ALPHAS = np.logspace(-2, 6, 25)
SEED = 42
SHRINK = 0.55
SPLINE_DF = 4
MIN_TEST = 150
START_FRAC = 0.5
N_REMOVE = 3
ISO_TOL = 0.05
OUT_DIR = Path("outputs/honest")
FS_DIR = Path("derivatives/fs_stats")
MANIFEST = OUT_DIR / "holdout_manifest.json"
VOL_COL = "EstimatedTotalIntraCranialVol"
CONFOUND_COLS = ["sex", "etiv", "fd_mean", "fd_max", "spike"]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def motion_stats(sub_id, fmriprep, task):
    p = Path(fmriprep) / sub_id / "func" / f"{sub_id}_task-{task}_desc-confounds_regressors.tsv"
    fd = pd.read_csv(p, sep="\t", usecols=["framewise_displacement"],
                     na_values="n/a")["framewise_displacement"].astype(float)
    fd = fd.fillna(0.0).to_numpy()[1:]
    return float(fd.mean()), float(fd.max()), float((fd > 0.5).mean())


def load_discovery():
    import torch

    cfg = yaml.safe_load(open(CFG_PATH))
    conn = Path(cfg["paths"]["connectivity_dir"])
    man = json.loads(MANIFEST.read_text())
    disc_all = man["discovery"]                          # the hold-out list is never read

    raw = {}
    with h5py.File(conn / "unscrubbed_ts.h5", "r") as f:
        for s in disc_all:
            if s in f["subjects"]:
                raw[s] = f["subjects"][s]["time_series"][:].astype(np.float64)
    frames = []
    for fp in sorted(FS_DIR.glob("data-*.tsv")):
        d = pd.read_csv(fp, sep="\t"); d = d.rename(columns={d.columns[0]: "sid"})
        d["sid"] = d["sid"].astype(str); d = d.set_index("sid")
        d = d.loc[:, ~d.columns.duplicated()]
        d.columns = [f"{fp.stem}::{c}" for c in d.columns]; frames.append(d)
    M = pd.concat(frames, axis=1)
    M = M.loc[:, ~M.columns.duplicated()].apply(pd.to_numeric, errors="coerce")
    M = M.dropna(axis=1, how="all"); M = M.loc[:, M.std(skipna=True) > 0]
    vol_col = [c for c in M.columns if c.endswith("::" + VOL_COL)][0]
    parts = load_participants(cfg["paths"]["bids_root"])
    z = np.load(conn / "dwi_fa_features.npz", allow_pickle=True)
    FA = pd.DataFrame(z["features"], index=[str(s) for s in z["subject_ids"]])

    ids = [s for s in disc_all if s in raw and s in M.index and s in FA.index
           and not pd.isna(M.loc[s, vol_col]) and not pd.isna(parts.loc[s, TARGET])]
    y = np.array([float(parts.loc[s, TARGET]) for s in ids])
    sex = np.array([1.0 if str(parts.loc[s, "sex"]).lower().startswith("m") else 0.0 for s in ids])
    mot = np.array([motion_stats(s, cfg["paths"]["fmriprep_dir"], cfg["dataset"]["task"])
                    for s in ids])
    C = pd.DataFrame({"sex": sex, "etiv": M.loc[ids, vol_col].to_numpy(float),
                      "fd_mean": mot[:, 0], "fd_max": mot[:, 1], "spike": mot[:, 2]})

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    covs = []
    for s in ids:
        x = raw[s] - raw[s].mean(0, keepdims=True)
        S = (x.T @ x) / (len(x) - 1)
        d = S.shape[0]
        covs.append((1 - SHRINK) * S + SHRINK * (np.trace(S) / d) * np.eye(d))
    Ct = torch.from_numpy(np.stack(covs)).to(dev).double()
    Ct = 0.5 * (Ct + Ct.transpose(-1, -2))
    w, V = torch.linalg.eigh(Ct)
    L = V @ torch.diag_embed(torch.log(w.clamp_min(1e-10))) @ V.transpose(-1, -2)
    n = L.shape[-1]
    iu = torch.triu_indices(n, n, offset=1, device=L.device)
    X_fc = torch.cat([torch.diagonal(L, dim1=-2, dim2=-1), L[..., iu[0], iu[1]] * np.sqrt(2.0)],
                     dim=-1).float().cpu().numpy()
    Mo = M.loc[ids].to_numpy(np.float32)                 # imputed inside folds
    Fa = FA.loc[ids].to_numpy(np.float32)
    return ids, y, C, X_fc, Mo, Fa


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
def ridge_pred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    return RidgeCV(alphas=ALPHAS, cv=5).fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))


def impute_fold(Xtr, Xte):
    med = np.nanmedian(Xtr, axis=0)
    return np.where(np.isnan(Xtr), med, Xtr), np.where(np.isnan(Xte), med, Xte)


def stack_pred(blocks, ytr, tr_rows, te_rows):
    """Ridge per block, ridge meta-learner over inner 5-fold out-of-fold predictions."""
    oof, test = [], []
    for Xb in blocks:
        Xtr, Xte = impute_fold(Xb[tr_rows], Xb[te_rows])
        o = np.zeros(len(tr_rows))
        for a, b in KFold(5, shuffle=True, random_state=SEED).split(Xtr):
            o[b] = ridge_pred(Xtr[a], ytr[a], Xtr[b])
        oof.append(o)
        test.append(ridge_pred(Xtr, ytr, Xte))
    return ridge_pred(np.column_stack(oof), ytr, np.column_stack(test)), np.column_stack(oof)


def linear_design(Ctr: pd.DataFrame, Cte: pd.DataFrame, with_sex=True):
    cols = (["sex"] if with_sex else []) + ["etiv", "fd_mean", "fd_max", "spike"]
    A, B = Ctr[cols].to_numpy(float), Cte[cols].to_numpy(float)
    mu, sd = A.mean(0), A.std(0) + 1e-12
    A, B = (A - mu) / sd, (B - mu) / sd
    e = cols.index("etiv")
    return np.column_stack([A, A[:, e] ** 2]), np.column_stack([B, B[:, e] ** 2])


def spline_design(Ctr: pd.DataFrame, Cte: pd.DataFrame, with_sex=True):
    """Natural cubic regression splines; knots from the TRAINING data only."""
    terms = ["sex"] if with_sex else []
    for c in ["etiv", "fd_mean", "fd_max", "spike"]:
        if Ctr[c].nunique() >= 3 * SPLINE_DF and np.unique(np.quantile(Ctr[c], np.linspace(0, 1, SPLINE_DF + 1))).size == SPLINE_DF + 1:
            terms.append(f"cr({c}, df={SPLINE_DF}, constraints='center')")
        else:
            terms.append(c)                              # too few distinct values: linear
    formula = "0 + " + " + ".join(terms)
    dtr = patsy.dmatrix(formula, Ctr, return_type="matrix")
    dte = patsy.build_design_matrices([dtr.design_info], Cte, return_type="matrix")[0]
    return np.asarray(dtr), np.asarray(dte), terms


def residualise(ytr, yte, Dtr, Dte):
    lr = LinearRegression().fit(Dtr, ytr)
    return ytr - lr.predict(Dtr), yte - lr.predict(Dte)


def r(a, b):
    return float(pearsonr(a, b)[0]) if np.std(a) > 0 and np.std(b) > 0 else np.nan


# ---------------------------------------------------------------------------
# A + B: one CV fold
# ---------------------------------------------------------------------------
def run_fold(k, tr, te, y, C, X_fc, Mo, Fa, tag_spline_terms):
    t0 = time.time()
    Ctr, Cte = C.iloc[tr].reset_index(drop=True), C.iloc[te].reset_index(drop=True)
    Ltr, Lte = linear_design(Ctr, Cte)
    Str, Ste, terms = spline_design(Ctr, Cte)
    rec = {"fold": k, "n_train": len(tr), "n_test": len(te),
           "spline_terms": " + ".join(terms)}

    targets = {"raw": (y[tr], y[te]),
               "lin": residualise(y[tr], y[te], Ltr, Lte),
               "spline": residualise(y[tr], y[te], Str, Ste)}
    for name, (ytr, yte) in targets.items():
        p_fc = ridge_pred(X_fc[tr], ytr, X_fc[te])
        p_st, oof_b = stack_pred([X_fc, Mo, Fa], ytr, tr, te)
        rec[f"fc_{name}"] = r(yte, p_fc)
        rec[f"stack_{name}"] = r(yte, p_st)
        if name == "raw":
            p_fc_raw, p_st_raw, oof_blocks = p_fc, p_st, oof_b    # reused below, not refit

    # confound floors and imaging increments (raw target)
    for dname, (Dtr, Dte) in {"lin": (Ltr, Lte), "spline": (Str, Ste)}.items():
        lr = LinearRegression().fit(Dtr, y[tr])
        p_conf = lr.predict(Dte)
        rec[f"conf_{dname}"] = r(y[te], p_conf)
        o_conf = np.zeros(len(tr))
        for a, b in KFold(5, shuffle=True, random_state=SEED).split(Dtr):
            o_conf[b] = LinearRegression().fit(Dtr[a], y[tr][a]).predict(Dtr[b])
        p_st = p_st_raw
        oof_img = np.zeros(len(tr))
        for a, b in KFold(5, shuffle=True, random_state=SEED + 1).split(oof_blocks):
            oof_img[b] = ridge_pred(oof_blocks[a], y[tr][a], oof_blocks[b])
        joint = ridge_pred(np.column_stack([oof_img, o_conf]), y[tr],
                           np.column_stack([p_st, p_conf]))
        rec[f"joint_{dname}"] = r(y[te], joint)
        rec[f"increment_{dname}"] = rec[f"joint_{dname}"] - rec[f"conf_{dname}"]

    # B: within sex
    wb = {"fold": k}
    rng = np.random.default_rng(SEED + k)
    sex = C["sex"].to_numpy()
    for sname, sval in (("male", 1.0), ("female", 0.0)):
        te_s = te[sex[te] == sval]
        tr_s = tr[sex[tr] == sval]
        wb[f"n_test_{sname}"], wb[f"n_train_{sname}"] = len(te_s), len(tr_s)
        pos = np.searchsorted(te, te_s)
        wb[f"pooled_{sname}"] = r(y[te_s], p_fc_raw[pos])
        wb[f"specific_{sname}"] = r(y[te_s], ridge_pred(X_fc[tr_s], y[tr_s], X_fc[te_s]))
        sub = rng.choice(tr, size=len(tr_s), replace=False)
        wb[f"nmatched_{sname}"] = r(y[te_s], ridge_pred(X_fc[sub], y[sub], X_fc[te_s]))
        Ctr_s, Cte_s = C.iloc[tr_s].reset_index(drop=True), C.iloc[te_s].reset_index(drop=True)
        Dtr, Dte = linear_design(Ctr_s, Cte_s, with_sex=False)
        ytr_d, yte_d = residualise(y[tr_s], y[te_s], Dtr, Dte)
        wb[f"specific_dec_{sname}"] = r(yte_d, ridge_pred(X_fc[tr_s], ytr_d, X_fc[te_s]))
    rec["secs"] = wb["secs"] = round(time.time() - t0, 1)
    return rec, wb


# ---------------------------------------------------------------------------
# C: confound-isolating test sets
# ---------------------------------------------------------------------------
def _std(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-12)


def density_ratio(y, Z, sex):
    """w = p(y) p(z) / p(y, z). Continuous z via Gaussian KDE; sex handled by
    conditioning (KDEs within each sex), binary-only z via p(y) / p(y | sex)."""
    ys = _std(y)
    w = np.empty(len(y))
    groups = [np.ones(len(y), bool)] if sex is None else [sex == 1, sex == 0]
    py = gaussian_kde(ys)(ys)
    for g in groups:
        if Z is None or Z.shape[1] == 0:
            w[g] = py[g] / gaussian_kde(ys[g])(ys[g])
            continue
        Zg = np.column_stack([_std(c) for c in Z[g].T])
        pz = gaussian_kde(Zg.T)(Zg.T)
        pyz = gaussian_kde(np.vstack([ys[g], Zg.T]))(np.vstack([ys[g], Zg.T]))
        w[g] = py[g] * pz / np.maximum(pyz, 1e-300)
    return w


def dependence(y, Zc, sex, seed=0):
    """max |r| and mutual information between IST and each confound."""
    rs, mis = [], []
    if Zc is not None:
        for c in Zc.T:
            rs.append(abs(r(y, c)))
            mis.append(float(mutual_info_regression(c.reshape(-1, 1), y, random_state=seed)[0]))
    if sex is not None:
        rs.append(abs(r(y, sex)))
        mis.append(float(mutual_info_classif(y.reshape(-1, 1), sex.astype(int),
                                             random_state=seed)[0]))
    return max(rs), float(np.sum(mis))


def isolate(idx_pool, y, C, confound, min_test=MIN_TEST):
    """Remove subjects by lowest density ratio until IST is independent of the confound."""
    cont = {"etiv": ["etiv"], "fd_mean": ["fd_mean"], "sex": [], "joint": ["etiv", "fd_mean"]}[confound]
    use_sex = confound in ("sex", "joint")
    pool = np.array(idx_pool)
    history = 0
    while True:
        yy = y[pool]
        Z = C.iloc[pool][cont].to_numpy(float) if cont else None
        sx = C.iloc[pool]["sex"].to_numpy() if use_sex else None
        cols = ([] if Z is None else list(Z.T)) + ([] if sx is None else [sx])
        dep_r = max(abs(r(yy, c)) for c in cols)        # stopping rule: |r| only (MI is slow)
        if dep_r <= ISO_TOL:
            return pool, True, history
        if len(pool) - N_REMOVE < min_test:
            return pool, False, history
        w = density_ratio(yy, Z, sx)
        pool = np.delete(pool, np.argsort(w)[:N_REMOVE])
        history += 1


CONFOUND_SEED = {"etiv": 1, "sex": 2, "fd_mean": 3, "joint": 4}   # NOT hash(): randomised per process


def run_isolated(j, confound, y, C, X_fc, n_all):
    rng = np.random.default_rng(SEED * 1000 + j * 17 + CONFOUND_SEED[confound])
    start = rng.choice(n_all, size=int(START_FRAC * n_all), replace=False)
    cont = {"etiv": ["etiv"], "fd_mean": ["fd_mean"], "sex": [], "joint": ["etiv", "fd_mean"]}[confound]
    use_sex = confound in ("sex", "joint")
    zs = lambda idx: (C.iloc[idx][cont].to_numpy(float) if cont else None,
                      C.iloc[idx]["sex"].to_numpy() if use_sex else None)
    Z0, s0 = zs(start)
    r_before, mi_before = dependence(y[start], Z0, s0)
    test_iso, ok, steps = isolate(start, y, C, confound)
    Z1, s1 = zs(test_iso)
    r_after, mi_after = dependence(y[test_iso], Z1, s1)
    test_rand = rng.choice(start, size=len(test_iso), replace=False)
    # same size AND same IST distribution as the isolated set (controls range restriction)
    ys = _std(y[start])
    iso_std = (y[test_iso] - y[start].mean()) / (y[start].std() + 1e-12)
    wy = gaussian_kde(iso_std)(ys) / np.maximum(gaussian_kde(ys)(ys), 1e-300)
    test_ymatch = rng.choice(start, size=len(test_iso), replace=False, p=wy / wy.sum())
    out = {"confound": confound, "set": j, "n_start": len(start), "n_test": len(test_iso),
           "isolated_ok": ok, "removal_steps": steps,
           "dep_r_before": r_before, "dep_r_after": r_after,
           "mi_before": mi_before, "mi_after": mi_after,
           "y_sd_start": float(y[start].std()), "y_sd_isolated": float(y[test_iso].std()),
           "y_sd_ymatch": float(y[test_ymatch].std())}
    Lconf = ["sex", "etiv", "fd_mean", "fd_max", "spike"]
    for label, test in (("iso", test_iso), ("rand", test_rand), ("ymatch", test_ymatch)):
        train = np.setdiff1d(np.arange(n_all), test)
        out[f"n_train_{label}"] = len(train)
        out[f"fc_{label}"] = r(y[test], ridge_pred(X_fc[train], y[train], X_fc[test]))
        Ctr = C.iloc[train][Lconf].to_numpy(float); Cte = C.iloc[test][Lconf].to_numpy(float)
        out[f"conf_{label}"] = r(y[test], LinearRegression().fit(Ctr, y[train]).predict(Cte))
    return out


# ---------------------------------------------------------------------------
def summarise(A, B, Ciso):
    rows = []

    def add(section, label, a, b, ratio=None, **extra):
        from src.stats.cv_inference import bayesian_correlated_ttest, corrected_resampled_ttest
        a, b = np.asarray(a, float), np.asarray(b, float)
        ct = corrected_resampled_ttest(a, b, ratio=ratio) if ratio is not None else corrected_resampled_ttest(a, b)
        bt = bayesian_correlated_ttest(a, b, ratio=ratio) if ratio is not None else bayesian_correlated_ttest(a, b)
        rows.append({"section": section, "comparison": label, "mean_a": float(np.nanmean(a)),
                     "mean_b": float(np.nanmean(b)), "delta": ct.delta, "ci_low": ct.ci_low,
                     "ci_high": ct.ci_high, "p_corrected": ct.p, "p_naive": ct.naive_p,
                     "p_rope": bt.p_rope, "bayes": bt.decision, "folds": ct.n_folds, **extra})

    zero = lambda v: np.zeros(len(v))
    for m in ("fc", "stack"):
        add("A", f"{m}: raw vs 0", A[f"{m}_raw"], zero(A))
        add("A", f"{m}: linear-deconfounded vs 0", A[f"{m}_lin"], zero(A))
        add("A", f"{m}: spline-deconfounded vs 0", A[f"{m}_spline"], zero(A))
        add("A", f"{m}: spline - linear (deconfounded)", A[f"{m}_spline"], A[f"{m}_lin"])
    add("A", "confound floor: spline - linear", A["conf_spline"], A["conf_lin"])
    add("A", "increment over linear floor vs 0", A["increment_lin"], zero(A))
    add("A", "increment over spline floor vs 0", A["increment_spline"], zero(A))
    for s in ("male", "female"):
        add("B", f"{s}: sex-specific vs 0", B[f"specific_{s}"], zero(B))
        add("B", f"{s}: sex-specific deconfounded vs 0", B[f"specific_dec_{s}"], zero(B))
        add("B", f"{s}: sex-specific - n-matched pooled", B[f"specific_{s}"], B[f"nmatched_{s}"])
        add("B", f"{s}: pooled - n-matched pooled (effect of n)", B[f"pooled_{s}"], B[f"nmatched_{s}"])
    for conf, g in Ciso.groupby("confound"):
        ratio = float((g["n_test"] / g["n_train_iso"]).mean())
        add("C", f"{conf}: fc on isolated sets vs 0", g["fc_iso"], np.zeros(len(g)), ratio=ratio)
        add("C", f"{conf}: fc isolated - fc size-matched random", g["fc_iso"], g["fc_rand"], ratio=ratio)
        add("C", f"{conf}: fc isolated - fc IST-matched random (primary)", g["fc_iso"], g["fc_ymatch"],
            ratio=ratio)
        add("C", f"{conf}: confounds-only isolated - random", g["conf_iso"], g["conf_rand"], ratio=ratio)
    S = pd.DataFrame(rows)

    def get(label):
        return S[S.comparison == label].iloc[0]

    verdicts = []
    c1 = {c: bool(g["isolated_ok"].all() and (g["dep_r_after"] <= ISO_TOL).all() and (g["n_test"] >= MIN_TEST).all()
                  and float(np.median(1 - g["mi_after"] / np.maximum(g["mi_before"], 1e-12))) >= 0.80)
          for c, g in Ciso.groupby("confound")}
    verdicts.append(("C1 isolation valid", all(c1.values()), json.dumps(c1)))
    c2 = {}
    for c, g in Ciso.groupby("confound"):
        z = get(f"{c}: fc on isolated sets vs 0")
        retained = float(g["fc_iso"].mean() / g["fc_ymatch"].mean()) if g["fc_ymatch"].mean() > 0 else np.nan
        retained_size = float(g["fc_iso"].mean() / g["fc_rand"].mean()) if g["fc_rand"].mean() > 0 else np.nan
        c2[c] = {"ci_excludes_0": bool(z.ci_low > 0), "retained_vs_IST_matched": round(retained, 3),
                 "retained_vs_size_matched": round(retained_size, 3),
                 "pass": bool(z.ci_low > 0 and retained >= 0.5)}
    verdicts.append(("C2 signal survives isolation", all(v["pass"] for v in c2.values()), json.dumps(c2)))
    d = get("fc: spline - linear (deconfounded)")
    verdicts.append(("C3 non-linear confounding material", bool(d.delta < -0.02 and d.p_corrected < 0.05),
                     f"delta {d.delta:+.4f}, p {d.p_corrected:.3g}"))
    c4 = {s: bool(get(f"{s}: sex-specific vs 0").ci_low > 0) for s in ("male", "female")}
    verdicts.append(("C4 within-sex signal in both sexes", all(c4.values()), json.dumps(c4)))
    c5 = {s: round(float(get(f"{s}: sex-specific - n-matched pooled").p_corrected), 4) for s in ("male", "female")}
    verdicts.append(("C5 sex-specificity beyond n", any(v < 0.05 for v in c5.values()), json.dumps(c5)))
    inc = get("increment over spline floor vs 0")
    verdicts.append(("C6 imaging increment over spline floor", bool(inc.ci_low > 0),
                     f"increment {inc.delta:+.4f} [{inc.ci_low:+.4f}, {inc.ci_high:+.4f}]"))
    V = pd.DataFrame(verdicts, columns=["criterion", "met", "detail"])
    return S, V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--isolated", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    t0 = time.time()
    ids, y, C, X_fc, Mo, Fa = load_discovery()
    n = len(ids)
    print(f"discovery subjects {n} | FC {X_fc.shape} | morphometry {Mo.shape} | FA {Fa.shape} "
          f"| load {time.time()-t0:.0f}s", flush=True)
    print(f"  corr(IST, eTIV) {r(y, C.etiv):+.3f}  corr(IST, sex) {r(y, C.sex):+.3f}  "
          f"corr(IST, meanFD) {r(y, C.fd_mean):+.3f}  corr(sex, eTIV) {r(C.sex, C.etiv):+.3f}",
          flush=True)

    strat = np.asarray(pd.qcut(y, 4, labels=False, duplicates="drop")) * 2 + C["sex"].to_numpy().astype(int)
    folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats,
                                         random_state=SEED).split(np.zeros(n), strat))
    t1 = time.time()
    res = Parallel(n_jobs=args.jobs, backend="loky")(
        delayed(run_fold)(k, tr, te, y, C, X_fc, Mo, Fa, None)
        for k, (tr, te) in enumerate(folds, 1))
    A = pd.DataFrame([a for a, _ in res]).sort_values("fold")
    B = pd.DataFrame([b for _, b in res]).sort_values("fold")
    print(f"  A+B: {len(folds)} folds in {time.time()-t1:.0f}s", flush=True)

    t2 = time.time()
    jobs = [(j, c) for c in ("etiv", "sex", "fd_mean", "joint") for j in range(args.isolated)]
    iso = Parallel(n_jobs=args.jobs, backend="loky")(
        delayed(run_isolated)(j, c, y, C, X_fc, n) for j, c in jobs)
    Ciso = pd.DataFrame(iso)
    print(f"  C: {len(jobs)} isolated sets in {time.time()-t2:.0f}s", flush=True)

    S, V = summarise(A, B, Ciso)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    A.to_csv(OUT_DIR / f"confound_control_folds{args.tag}.csv", index=False)
    B.to_csv(OUT_DIR / f"confound_control_withinsex{args.tag}.csv", index=False)
    Ciso.to_csv(OUT_DIR / f"confound_control_isolation{args.tag}.csv", index=False)
    pd.concat([S.assign(kind="test"), V.assign(kind="criterion")], ignore_index=True).to_csv(
        OUT_DIR / f"confound_control_summary{args.tag}.csv", index=False)

    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 80)
    print("\nTESTS (corrected inference)")
    print(S[["section", "comparison", "mean_a", "mean_b", "delta", "ci_low", "ci_high",
             "p_corrected", "bayes"]].round(4).to_string(index=False))
    print("\nISOLATION DIAGNOSTICS (mean per confound)")
    print(Ciso.groupby("confound")[["n_test", "isolated_ok", "dep_r_before", "dep_r_after", "mi_before",
                                    "mi_after", "y_sd_start", "y_sd_isolated", "y_sd_ymatch", "fc_iso",
                                    "fc_rand", "fc_ymatch", "conf_iso", "conf_rand",
                                    "conf_ymatch"]].mean().round(4).to_string())
    print("\nPRE-SPECIFIED CRITERIA")
    print(V.to_string(index=False))
    print(f"\nSaved -> {OUT_DIR}/confound_control_*{args.tag}.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
