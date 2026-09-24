#!/usr/bin/env python
"""
Predict the accuracy of the frozen pipeline in each PIOP collection from the ID1000 saturation curves,
before any PIOP time series are extracted.

Model
-----
The two curves were measured one at a time: duration was swept at fixed n = 560
training subjects, and n was swept at fixed full length T = 10.6 min. They are
therefore both conditional on the other axis sitting at its reference value, and
combining them requires an assumption, which is stated rather than hidden:
attenuations from the two axes act multiplicatively relative to the reference
point,

    r(n, T) = r_ref * [f_n(n) / f_n(n_ref)] * [f_T(T) / f_T(T_ref)]

This is the natural form if each axis independently attenuates a common true
signal, and it reproduces the reference cell exactly by construction. Any
interaction between the two axes is unmodelled, and is one way the prediction
can fail.

THREE ATTENUATIONS BEYOND n AND T
---------------------------------
1. INSTRUMENT. ID1000 measures intelligence with the IST (590 items); PIOP uses
   Raven's APM set II (36 items). A shorter test is less reliable, and observed
   prediction scales with the square root of criterion reliability. Central
   assumption: rel(IST) = 0.95, rel(Raven) = 0.85. These are literature values,
   not measured here, so the resulting factor is reported explicitly.

2. RANGE RESTRICTION. ID1000 was sampled to be representative of Dutch
   educational level; PIOP is university students only. This is estimated from
   the data rather than assumed, using the education-intelligence correlation as
   a common yardstick: it is 0.46 in ID1000 (three-level education) and is
   computed here for PIOP (binary applied/academic), with the standard
   dichotomisation correction applied so the two are comparable.

3. CONCATENATION. The ID1000 duration curve was measured on contiguous windows
   of ONE continuous condition. PIOP's minutes come from four to six DIFFERENT
   tasks. Treating 35 minutes of six paradigms as equivalent to 35 minutes of
   one is optimistic, so both are reported: a per-run prediction and a fully
   concatenated one. The gap between them is small, and that is itself the
   point -- the model says duration barely matters either way.

Usage
-----
  python scripts/generalisation/piop_accuracy_prediction.py
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.paths import collection_root

OUT_DIR = Path("outputs/honest")
FITS = OUT_DIR / "ceiling_fits.csv"
ROOTS = {name: collection_root(name) for name in ("PIOP1", "PIOP2")}
TR = {"mb3": 0.75, "seq": 2.0}
N_FOLDS = 5                      # 4/5 of the cohort trains, as in ID1000

# reference cell of the ID1000 ceiling grid: full length, full training set
N_REF, T_REF, R_REF = 560, 10.6, 0.4146
REL_IST, REL_RAVEN = 0.95, 0.85           # criterion reliabilities (literature)
R_EDU_ID1000 = 0.46                       # IST vs 3-level education, AOMIC Table 6
DICHOT = 0.80                             # cost of collapsing 3 levels to 2


def sat(x, x0):
    """Classical saturation factor sqrt(x / (x + x0))."""
    return np.sqrt(x / (x + x0))


def run_minutes(root):
    """Median minutes per run per task, measured from the confound files."""
    fp = root / "derivatives" / "fmriprep"
    rows = []
    for s in sorted(p.name for p in fp.glob("sub-*") if p.is_dir()):
        for f in sorted((fp / s / "func").glob("*_desc-confounds_regressors.tsv")):
            acq = f.name.split("acq-")[1].split("_")[0]
            rows.append({"sub": s,
                         "task": f.name.split("task-")[1].split("_acq")[0],
                         "minutes": (sum(1 for _ in open(f)) - 1) * TR[acq] / 60})
    d = pd.DataFrame(rows)
    return d.groupby("task").minutes.median(), float(d.groupby("sub").minutes.sum().median())


def range_restriction(root):
    """Estimate range restriction from the education-intelligence correlation."""
    d = pd.read_csv(root / "participants.tsv", sep="\t", na_values="n/a")
    d = d.dropna(subset=["raven_score", "education_category"])
    g = d.groupby("education_category").raven_score
    m, p = g.mean(), g.size() / len(d)
    lo, hi = sorted(m.index)
    rpb = (m[hi] - m[lo]) / d.raven_score.std(ddof=1) * np.sqrt(p[lo] * p[hi])
    return abs(rpb), abs(rpb) / DICHOT, abs(rpb) / DICHOT / R_EDU_ID1000


def main():
    fits = pd.read_csv(FITS).set_index("label")
    r_inf_T, T0 = fits.loc["scan length", ["r_inf", "T0"]]
    r_inf_n, n0 = fits.loc["sample size", ["r_inf", "T0"]]
    fn_ref, fT_ref = sat(N_REF, n0), sat(T_REF, T0)
    inst = np.sqrt(REL_RAVEN / REL_IST)

    bar = "=" * 78
    L = [bar, "PRE-REGISTERED PREDICTION FOR AOMIC-PIOP1 AND PIOP2",
         f"registered {date.today().isoformat()}, before any PIOP timeseries was extracted",
         bar, "",
         "ID1000 ceiling fits used as the generating model:",
         f"  duration     r_inf = {r_inf_T:.4f}   T0 = {T0:.2f} min",
         f"  sample size  r_inf = {r_inf_n:.4f}   n0 = {n0:.1f} subjects",
         f"  reference cell: n_train = {N_REF}, T = {T_REF} min, r = {R_REF:+.4f}", "",
         "Instrument attenuation (Raven 36 items vs IST 590 items):",
         f"  sqrt(rel_Raven / rel_IST) = sqrt({REL_RAVEN}/{REL_IST}) = {inst:.3f}", ""]

    rec = {"registered": date.today().isoformat(),
           "model": "r(n,T) = r_ref * [f_n(n)/f_n(n_ref)] * [f_T(T)/f_T(T_ref)]",
           "fits": {"r_inf_T": float(r_inf_T), "T0": float(T0),
                    "r_inf_n": float(r_inf_n), "n0": float(n0),
                    "n_ref": N_REF, "T_ref": T_REF, "r_ref": R_REF},
           "datasets": {}}

    for name, root in ROOTS.items():
        per_task, total = run_minutes(root)
        n_sub = int(pd.read_csv(root / "participants.tsv", sep="\t",
                                na_values="n/a").raven_score.notna().sum())
        n_tr = int(round(n_sub * (N_FOLDS - 1) / N_FOLDS))
        rpb, r_corr, u = range_restriction(root)

        def pred(T, atten):
            return R_REF * (sat(n_tr, n0) / fn_ref) * (sat(T, T0) / fT_ref) * atten

        L += [bar, f"{name}:  N = {n_sub} with Raven, n_train = {n_tr} at {N_FOLDS}-fold CV",
              f"  scan time per subject: {total:.1f} min across {len(per_task)} runs",
              f"  cohort factor   f_n({n_tr})/f_n({N_REF}) = "
              f"{sat(n_tr, n0) / fn_ref:.3f}   <-- the dominant loss",
              f"  duration factor f_T({total:.1f})/f_T({T_REF}) = "
              f"{sat(total, T0) / fT_ref:.3f}   <-- barely helps",
              f"  range restriction: r(edu,IQ) = {rpb:.3f} point-biserial, "
              f"{r_corr:.3f} corrected, vs {R_EDU_ID1000:.2f} in ID1000  ->  u = {u:.3f}",
              "", "  predicted r (raw, not deconfounded), per single run:"]
        for t, mn in per_task.sort_values().items():
            L.append(f"      {t:16s} {mn:5.2f} min   r = {pred(mn, inst * u):+.3f}"
                     f"   (without range correction: {pred(mn, inst):+.3f})")
        best = per_task.max()
        L += ["",
              f"      {'ALL CONCATENATED':16s} {total:5.2f} min   "
              f"r = {pred(total, inst * u):+.3f}"
              f"   (without range correction: {pred(total, inst):+.3f})", "",
              f"  >> HEADLINE PREDICTION for {name}: r in "
              f"[{pred(total, inst * u):+.3f}, {pred(total, inst):+.3f}], "
              f"centre {0.5 * (pred(total, inst * u) + pred(total, inst)):+.3f}",
              f"     Concatenating every run buys only "
              f"{pred(total, inst * u) - pred(best, inst * u):+.4f} over the single",
              f"     longest run ({best:.2f} min). That near-zero gain is the "
              f"sharpest part of", "     the prediction and the easiest to falsify.", ""]

        rec["datasets"][name] = {
            "n_subjects": n_sub, "n_train": n_tr, "total_minutes": round(total, 2),
            "per_task_minutes": {k: round(v, 2) for k, v in per_task.items()},
            "cohort_factor": round(float(sat(n_tr, n0) / fn_ref), 4),
            "duration_factor": round(float(sat(total, T0) / fT_ref), 4),
            "instrument_factor": round(float(inst), 4),
            "range_restriction_u": round(float(u), 4),
            "predicted_r_concat_lower": round(float(pred(total, inst * u)), 4),
            "predicted_r_concat_upper": round(float(pred(total, inst)), 4),
            "predicted_r_per_task": {k: round(float(pred(v, inst * u)), 4)
                                     for k, v in per_task.items()}}

    L += [bar, "THE CLAIM UNDER TEST", bar,
          "ID1000 reached r = +0.403 on a sealed hold-out with 11 minutes per subject.",
          "PIOP1 has three times the scan time and a quarter of the subjects. If the",
          "ceiling analysis is right, that is a bad trade and PIOP must land near 0.3,",
          "not near 0.4. A PIOP result at or above the ID1000 level falsifies the",
          "cohort-limited claim outright.", "",
          "Fixed in advance, so none of it can be chosen after seeing the answer:",
          "  * pipeline    identical to ID1000 -- Schaefer-200 + 15 subcortical, 36P",
          "                confounds, PSC standardisation, 0.01-0.1 Hz, 6 mm FWHM",
          "  * estimator   tangent-space ridge, Ledoit-Wolf shrinkage, RidgeCV inner loop",
          "  * CV          5-fold x 3 repeats, stratified on Raven quartile x sex",
          "  * confounds   sex, ICV, ICV^2, 3 motion summaries -- as in ID1000; education",
          "                is reported separately and NOT added, to keep the confound",
          "                floor comparable across collections",
          "  * primary     concatenated-run tangent ridge, against the interval above",
          "  * no model selection on PIOP; the ID1000 pipeline is applied frozen", bar]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    txt = "\n".join(L)
    print(txt)
    (OUT_DIR / "piop_prereg.txt").write_text(txt, encoding="utf-8")
    json.dump(rec, open(OUT_DIR / "piop_prereg.json", "w"), indent=2)
    print(f"\nLocked -> {OUT_DIR}/piop_prereg.txt and piop_prereg.json")


if __name__ == "__main__":
    main()
