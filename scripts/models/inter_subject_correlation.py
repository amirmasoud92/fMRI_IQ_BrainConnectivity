#!/usr/bin/env python
"""
Inter-subject correlation, inter-subject functional connectivity and representational similarity on the movie.

Motion
------
High-motion frames ARE censored here. For subject i the correlation with the
template runs only over frames where that subject's FD <= threshold, while the
template still spans the full run; because the same timepoints are indexed on
both sides, temporal alignment is preserved and no volume is deleted. Both
series are re-standardised on the retained frames. An UNCENSORED arm is computed
alongside so the effect of the choice is visible rather than assumed, and mean
FD / max FD / spike fraction enter the deconfounding model.

Is-Rsa
------
The subject x subject ISC-similarity matrix is compared with an IST-similarity
Usage
-----
  python scripts/models/inter_subject_correlation.py --repeats 5
"""
from __future__ import annotations

import argparse, sys, time, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import pearsonr, rankdata
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.data.preprocessing import load_participants
from src.stats.cv_inference import corrected_ttest_rel

warnings.filterwarnings("ignore")

CFG_PATH = "config/pipeline.yaml"
TARGET   = "IST_intelligence_total"
ALPHAS   = np.logspace(-2, 6, 25)
SEED     = 42
SHRINK   = 0.55
N_WIN    = 8
N_PERM   = 10000
MIN_KEEP = 20          # below this many retained frames, fall back to all frames
OUT_DIR  = Path("outputs/honest")
FS_DIR   = Path("derivatives/fs_stats")
VOL_COL  = "EstimatedTotalIntraCranialVol"


def sym_funcm(X, fn):
    X = 0.5 * (X + X.transpose(-1, -2)).double()
    w, V = torch.linalg.eigh(X)
    return V @ torch.diag_embed(fn(w.clamp_min(1e-10))) @ V.transpose(-1, -2)


def upper(L):
    n = L.shape[-1]
    iu = torch.triu_indices(n, n, offset=1, device=L.device)
    return torch.cat([torch.diagonal(L, dim1=-2, dim2=-1),
                      L[..., iu[0], iu[1]] * np.sqrt(2.0)], dim=-1).float()


def zsc(A, axis=0):
    m = A.mean(axis=axis, keepdims=True)
    s = A.std(axis=axis, keepdims=True)
    return (A - m) / np.clip(s, 1e-8, None)


def loo_isc(TS, keep, tr_idx, sl=None, censor=True):
    """Leave-one-out ISC per region. Template = TRAINING subjects only.

    TS : (N, T, R), keep : (N, T) bool.  Returns (N, R).
    With censor=True, subject i is correlated with the template over only those
    frames where subject i's FD is below threshold. The same timepoints index
    both sides, so alignment to the film is preserved.
    """
    X = TS[:, sl, :] if sl is not None else TS
    K = keep[:, sl] if sl is not None else keep
    Xz = zsc(X, axis=1)
    N, T, R = Xz.shape
    tr_mask = np.zeros(N, bool); tr_mask[tr_idx] = True
    Ssum = Xz[tr_mask].sum(0)
    n_tr = int(tr_mask.sum())
    out = np.empty((N, R), np.float32)
    for i in range(N):
        tmpl = (Ssum - Xz[i]) / (n_tr - 1) if tr_mask[i] else Ssum / n_tr
        k = K[i] if censor else np.ones(T, bool)
        if k.sum() < MIN_KEEP:
            k = np.ones(T, bool)
        # Pearson r computed from centred sums of squares. Standardising with a
        # population SD and then dividing by (n-1) would inflate every value by
        # n/(n-1); because n is the number of frames each subject retains after
        # FD censoring, that factor varies BY SUBJECT and tracks motion, which is
        # exactly the leak the censoring is meant to remove.
        a = Xz[i][k] - Xz[i][k].mean(0, keepdims=True)
        b = tmpl[k] - tmpl[k].mean(0, keepdims=True)
        den = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
        out[i] = (a * b).sum(0) / np.clip(den, 1e-12, None)
    return out


def loo_isfc(TS, keep, tr_idx, dev, censor=True):
    """Leave-one-out ISFC: corr(region a in subject i, region b in the others)."""
    Xz = torch.from_numpy(zsc(TS, axis=1)).to(dev).float()   # (N, T, R)
    N, T, R = Xz.shape
    Kt = torch.from_numpy(keep).to(dev)
    tr = torch.zeros(N, dtype=torch.bool, device=dev)
    tr[torch.as_tensor(np.asarray(tr_idx), device=dev)] = True
    Ssum = Xz[tr].sum(0)
    n_tr = int(tr.sum())
    iu = torch.triu_indices(R, R, offset=1, device=dev)
    out = torch.empty(N, R + iu.shape[1], device=dev)
    for i in range(N):
        tmpl = (Ssum - Xz[i]) / (n_tr - 1) if tr[i] else Ssum / n_tr
        k = Kt[i] if censor else torch.ones(T, dtype=torch.bool, device=dev)
        if int(k.sum()) < MIN_KEEP:
            k = torch.ones(T, dtype=torch.bool, device=dev)
        a, b = Xz[i][k], tmpl[k]
        a = a - a.mean(0)
        b = b - b.mean(0)
        na = a.norm(dim=0).clamp_min(1e-12)
        nb = b.norm(dim=0).clamp_min(1e-12)
        M = (a.t() @ b) / (na[:, None] * nb[None, :])   # exact Pearson r
        M = 0.5 * (M + M.t())
        out[i] = torch.cat([torch.diagonal(M), M[iu[0], iu[1]]])
    return out.cpu().numpy()


def ridge_pred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    return RidgeCV(alphas=ALPHAS, cv=5).fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))


def oof(X, y, tr, n_inner=5):
    o = np.zeros(len(tr))
    for a, b in KFold(n_inner, shuffle=True, random_state=SEED).split(tr):
        o[b] = ridge_pred(X[tr[a]], y[tr[a]], X[tr[b]])
    return o


def mantel(Dn, Db, n_perm=N_PERM, seed=SEED):
    """Mantel test with exact permutation, computed without redoing the ranking.

    Permuting rows and columns of a symmetric matrix permutes the off-diagonal
    pairs but not their multiset of values, so the rank matrix can be built once
    and reindexed. The permuted rank vector then has fixed mean and SD and only
    its dot product with the (fixed) behavioural ranks varies.
    """
    N = len(Dn)
    iu = np.triu_indices(N, 1)
    ra = rankdata(Dn[iu]); rb = rankdata(Db[iu])
    Rn = np.zeros((N, N)); Rn[iu] = ra; Rn = Rn + Rn.T
    a_c = ra - ra.mean(); b_c = rb - rb.mean()
    denom = np.sqrt((a_c ** 2).sum() * (b_c ** 2).sum())
    r0 = float((a_c * b_c).sum() / denom)
    rng = np.random.default_rng(seed)
    cnt_two = cnt_one = 0
    for _ in range(n_perm):
        p = rng.permutation(N)
        v = Rn[np.ix_(p, p)][iu]
        rp = float(((v - v.mean()) * b_c).sum() / denom)
        if abs(rp) >= abs(r0):
            cnt_two += 1
        if rp >= r0:
            cnt_one += 1
    return r0, (cnt_two + 1) / (n_perm + 1), (cnt_one + 1) / (n_perm + 1)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(CFG_PATH))
    conn = Path(cfg["paths"]["connectivity_dir"])
    up = conn / "unscrubbed_ts.h5"
    if not up.exists():
        sys.exit(f"Missing {up} -- run scripts/extraction/extract_timeseries_psc.py first")

    raw, kp = {}, {}
    with h5py.File(up, "r") as f:
        for s in f["subjects"]:
            raw[s] = f["subjects"][s]["time_series"][:]
            kp[s] = f["subjects"][s]["keep"][:].astype(bool)
    lens = np.array([v.shape[0] for v in raw.values()])
    modal = int(np.bincount(lens).argmax())
    n_off = int((lens != modal).sum())
    print(f"{len(raw)} subjects; lengths {lens.min()}-{lens.max()}, modal {modal} "
          f"({len(raw)-n_off}/{len(raw)} at modal length)")
    if n_off:
        print(f"  dropping {n_off} subjects not at modal length rather than "
              f"truncating the cohort to the shortest run")

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

    ids = [s for s in sorted(set(raw) & set(M.index))
           if s in parts.index and not pd.isna(parts.loc[s, TARGET])
           and not pd.isna(M.loc[s, vol_col]) and raw[s].shape[0] == modal]
    TS = np.stack([raw[s] for s in ids]).astype(np.float32)      # (N, T, R)
    KEEP = np.stack([kp[s] for s in ids])
    y = np.array([float(parts.loc[s, TARGET]) for s in ids])
    sex = np.array([1 if str(parts.loc[s, "sex"]).lower().startswith("m") else 0 for s in ids])
    vol = M.loc[ids, vol_col].values.astype(float)
    mot = np.array([motion_stats(s, cfg["paths"]["fmriprep_dir"], cfg["dataset"]["task"])
                    for s in ids])
    for j in range(mot.shape[1]):
        mot[:, j] = np.where(np.isfinite(mot[:, j]), mot[:, j], np.nanmedian(mot[:, j]))
    Zc = np.column_stack([sex, vol, vol ** 2])
    Zm = np.column_stack([sex, vol, vol ** 2, mot])
    N, T, R = TS.shape
    print(f"  aligned array: {TS.shape}   frames retained after FD censoring: "
          f"{100*KEEP.mean():.1f}%  (min subject {100*KEEP.mean(1).min():.1f}%)")

    # static FC baseline on the SAME unscrubbed percent-signal-change series
    covs = []
    for i in range(N):
        x = TS[i] - TS[i].mean(0, keepdims=True)
        S = (x.T @ x) / (T - 1)
        covs.append((1 - SHRINK) * S + SHRINK * (np.trace(S) / R) * np.eye(R, dtype=np.float32))
    X_fc = upper(sym_funcm(torch.from_numpy(np.stack(covs)).to(dev).double(),
                           torch.log)).cpu().numpy()

    wins = np.array_split(np.arange(T), N_WIN)
    q = pd.qcut(y, 4, labels=False, duplicates="drop")
    strat = np.asarray(q) * 2 + sex
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats, random_state=SEED)

    rows = []
    for k, (tr, te) in enumerate(rskf.split(np.zeros(N), strat), 1):
        t0 = time.time(); rec = {"fold": k}

        ISC = loo_isc(TS, KEEP, tr)
        ISC_NC = loo_isc(TS, KEEP, tr, censor=False)
        W = np.stack([loo_isc(TS, KEEP, tr, sl=w) for w in wins])       # (W, N, R)
        ISCW = np.hstack([W.mean(0), W.std(0)])
        ISFC = loo_isfc(TS, KEEP, tr, dev)

        blocks = {"isc": ISC, "isc_nocensor": ISC_NC, "isc_win": ISCW,
                  "isfc": ISFC, "fc": X_fc}
        preds = {}
        for nm, Xb in blocks.items():
            p = ridge_pred(Xb[tr], y[tr], Xb[te]); preds[nm] = p
            rec[nm] = float(pearsonr(y[te], p)[0])
            for tag, Z_ in [("dec", Zc), ("decm", Zm)]:
                lr = LinearRegression().fit(Z_[tr], y[tr])
                rec[f"{nm}_{tag}"] = float(pearsonr(
                    y[te] - lr.predict(Z_[te]),
                    ridge_pred(Xb[tr], y[tr] - lr.predict(Z_[tr]), Xb[te]))[0])
        # oof() is deterministic given (block, tr) -- the inner KFold is seeded --
        # so the three combos below were refitting the same level-1 models. fc was
        # being rebuilt 3x and isfc 2x, at ~23,220 features each. Memoised per
        # fold: bit-identical predictions, roughly half the fold time.
        oof_cache = {}
        for combo in [("fc", "isc"), ("fc", "isfc"), ("fc", "isc", "isfc")]:
            for n in combo:
                if n not in oof_cache:
                    oof_cache[n] = oof(blocks[n], y, tr)
            Zt = np.column_stack([oof_cache[n] for n in combo])
            Ze = np.column_stack([preds[n] for n in combo])
            rec["stack_" + "+".join(combo)] = float(
                pearsonr(y[te], ridge_pred(Zt, y[tr], Ze))[0])
        rec["secs"] = time.time() - t0
        rows.append(rec)
        print(f"  fold {k:2d}/{args.repeats*5}  fc={rec['fc']:+.3f} isc={rec['isc']:+.3f} "
              f"isfc={rec['isfc']:+.3f} fc+isc={rec['stack_fc+isc']:+.3f}  "
              f"({rec['secs']:.0f}s)", flush=True)

    df = pd.DataFrame(rows); df.to_csv(OUT_DIR / "isc_folds.csv", index=False)
    cols = [c for c in df.columns if c not in ("fold", "secs")]
    main_cols = [c for c in cols if not c.endswith(("_dec", "_decm"))]
    print(f"\n{'='*76}\nISC / ISFC, {args.repeats}x5-fold CV (n={N}, T={T})\n{'='*76}")
    for c in sorted(main_cols, key=lambda c: -df[c].mean()):
        print(f"  {c:24s} r = {df[c].mean():+.4f} +/- {df[c].std(ddof=1):.4f}")

    print(f"\n  paired tests vs static FC (same folds):")
    for c in main_cols:
        if c == "fc":
            continue
        t, p = corrected_ttest_rel(df[c], df["fc"])
        print(f"    {c:24s} delta = {(df[c]-df['fc']).mean():+.4f}  t = {t:+5.2f}  "
              f"p = {p:.3g}  wins {int((df[c]>df['fc']).sum())}/{len(df)}")

    print(f"\n  effect of FD censoring on ISC: "
          f"{(df['isc']-df['isc_nocensor']).mean():+.4f} "
          f"(p = {corrected_ttest_rel(df['isc'], df['isc_nocensor'])[1]:.3g})")

    print(f"\n  survival of deconfounding  (dec = sex+size;  decm = + motion):")
    print(f"    {'block':14s} {'raw':>9s} {'dec':>9s} {'ret%':>6s} {'decm':>9s} {'ret%':>6s}")
    for nm in ["fc", "isc", "isc_win", "isfc"]:
        r_, d1, d2 = df[nm].mean(), df[f"{nm}_dec"].mean(), df[f"{nm}_decm"].mean()
        print(f"    {nm:14s} {r_:+9.4f} {d1:+9.4f} {100*d1/r_:5.0f}% "
              f"{d2:+9.4f} {100*d2/r_:5.0f}%")

    # ---- IS-RSA (descriptive, full sample) ---------------------------------
    print(f"\n  IS-RSA (Mantel, {N_PERM} permutations):")
    ISC_all = loo_isc(TS, KEEP, np.arange(N))
    Dn = np.corrcoef(zsc(ISC_all, axis=0))
    lr = LinearRegression().fit(Zm, y)
    y_res = y - lr.predict(Zm)
    isrsa = []
    for ylab, yy in [("raw IST", y), ("deconfounded IST", y_res)]:
        for name, Db in [("nearest-neighbour", -np.abs(yy[:, None] - yy[None, :])),
                         ("Anna Karenina", np.minimum(yy[:, None], yy[None, :]))]:
            r0, p2, p1 = mantel(Dn, Db)
            print(f"    {ylab:18s} {name:18s} rho = {r0:+.4f}  "
                  f"p(two-tailed) = {p2:.4f}  p(one-tailed) = {p1:.4f}")
            isrsa.append({"target": ylab, "model": name, "rho": r0,
                          "p_two_tailed": p2, "p_one_tailed": p1})
    pd.DataFrame(isrsa).to_csv(OUT_DIR / "isc_isrsa.csv", index=False)
    pd.DataFrame({"roi": np.arange(R), "mean_isc": ISC_all.mean(0),
                  "isc_ist_r": [pearsonr(ISC_all[:, j], y)[0] for j in range(R)],
                  "isc_ist_r_dec": [pearsonr(ISC_all[:, j], y_res)[0] for j in range(R)]}
                 ).to_csv(OUT_DIR / "isc_regions.csv", index=False)
    print(f"\nSaved -> {OUT_DIR}/isc_folds.csv, isc_regions.csv, isc_isrsa.csv")


if __name__ == "__main__":
    main()
