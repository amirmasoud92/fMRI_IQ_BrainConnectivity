#!/usr/bin/env python
"""
Reporting metrics for the frozen models on the sealed hold-out: variance explained, absolute error,
calibration, within-sex accuracy and paired differences between models.

GATE (enforced before anything is computed)
-------------------------------------------
1. The re-run (holdout_evaluation_rerun.csv) must reproduce every r and
   r_deconfounded of the original evaluation (holdout_evaluation.csv) to
   |diff| <= 1e-6.
2. The saved per-subject predictions must reproduce the re-run's own r values
   (internal consistency, <= 1e-10).
If either fails the script stops and reports; no metric is written.

Metrics per arm, raw target and deconfounded target
---------------------------------------------------
  r (Fisher 95% CI)
  R2_holdout        1 - SSE / sum((y - mean_holdout)^2)
  R2_vs_discovery   1 - SSE / sum((y - mean_discovery)^2): can be negative, and
                    is what a deployed model faces, since it cannot know the
                    hold-out mean
  MAE, RMSE         IST points (raw); residual points (deconfounded)
  MAE_mean_model    MAE of always predicting the discovery mean (context)
  calibration       OLS y = intercept + slope * prediction; bootstrap 95% CI
                    (slope 1 = calibrated; < 1 = predictions too spread,
                    > 1 = shrunk toward the mean, the usual case for ridge)
  within-sex r      males and females separately, Fisher CIs, and a test of the
                    difference between the two independent correlations

Outputs
-------
  outputs/honest/holdout_extended_metrics.csv
  outputs/honest/holdout_paired_comparisons.csv
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / "outputs" / "honest"
ORIGINAL = H / "holdout_evaluation.csv"
RERUN = H / "holdout_evaluation_rerun.csv"
PREDS = H / "holdout_predictions_sealed.csv"
OUT_METRICS = H / "holdout_extended_metrics.csv"
OUT_PAIRED = H / "holdout_paired_comparisons.csv"
ACCESS_LOG = H / "holdout_access_log.json"
ARMS = ["confounds", "fc_ridge", "fc_relweighted", "stack_tuned"]
PAIRS = [("stack_tuned", "fc_ridge"), ("fc_relweighted", "fc_ridge"),
         ("fc_ridge", "confounds"), ("stack_tuned", "confounds")]
N_BOOT = 10_000
SEED = 20260914
GATE_TOL = 1e-6


def fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    z, se = np.arctanh(r), 1.0 / np.sqrt(n - 3)
    q = stats.norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - q * se)), float(np.tanh(z + q * se))


def zou_ci(r1: float, r2: float, r12: float, n: int, alpha: float = 0.05):
    """Zou (2007) CI for r1 - r2, two correlations sharing one variable (y)."""
    l1, u1 = fisher_ci(r1, n, alpha)
    l2, u2 = fisher_ci(r2, n, alpha)
    c = ((r12 - 0.5 * r1 * r2) * (1 - r1 ** 2 - r2 ** 2 - r12 ** 2) + r12 ** 3) / \
        ((1 - r1 ** 2) * (1 - r2 ** 2))
    d = r1 - r2
    lo = d - np.sqrt((r1 - l1) ** 2 + (u2 - r2) ** 2 - 2 * c * (r1 - l1) * (u2 - r2))
    hi = d + np.sqrt((u1 - r1) ** 2 + (r2 - l2) ** 2 - 2 * c * (u1 - r1) * (r2 - l2))
    return float(lo), float(hi)


def corr(a, b) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def gate(preds: pd.DataFrame) -> dict:
    orig = pd.read_csv(ORIGINAL).set_index("arm")
    rerun = pd.read_csv(RERUN).set_index("arm")
    diffs = {}
    for arm in ARMS:
        for col in ("r", "r_deconfounded"):
            diffs[f"{arm}.{col}"] = abs(float(orig.loc[arm, col]) - float(rerun.loc[arm, col]))
    worst = max(diffs.values())
    internal = {}
    for arm in ARMS:
        internal[f"{arm}.r"] = abs(corr(preds["y"], preds[f"pred_{arm}"]) - float(rerun.loc[arm, "r"]))
        internal[f"{arm}.r_deconfounded"] = abs(
            corr(preds["y_deconfounded"], preds[f"pred_{arm}_dec"]) - float(rerun.loc[arm, "r_deconfounded"]))
    worst_internal = max(internal.values())
    return {"max_abs_diff_vs_original": worst, "max_abs_diff_internal": worst_internal,
            "passed": worst <= GATE_TOL and worst_internal <= 1e-10, "per_value": diffs,
            "n_holdout": int(len(preds))}


def calibration(y, p, rng, n_boot=N_BOOT):
    X = np.column_stack([np.ones_like(p), p])
    b0, b1 = np.linalg.lstsq(X, y, rcond=None)[0]
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    slopes, ints = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        yy, pp = y[idx[i]], p[idx[i]]
        bb = np.linalg.lstsq(np.column_stack([np.ones_like(pp), pp]), yy, rcond=None)[0]
        ints[i], slopes[i] = bb
    return (float(b1), float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5)),
            float(b0), float(np.percentile(ints, 2.5)), float(np.percentile(ints, 97.5)))


def main():
    preds = pd.read_csv(PREDS)
    g = gate(preds)
    print(f"GATE  max |r diff| vs original = {g['max_abs_diff_vs_original']:.3g} (tol {GATE_TOL:g});"
          f"  internal = {g['max_abs_diff_internal']:.3g}  ->  {'PASSED' if g['passed'] else 'FAILED'}")
    log = json.loads(ACCESS_LOG.read_text(encoding="utf-8"))
    entry = log["accesses"][-1]
    entry["gate"] = {k: v for k, v in g.items() if k != "per_value"}
    entry["gate_checked_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if not g["passed"]:
        entry["status"] = "STOPPED: gate failed, no metric computed"
        ACCESS_LOG.write_text(json.dumps(log, indent=2), encoding="utf-8")
        for k, v in sorted(g["per_value"].items(), key=lambda kv: -kv[1])[:8]:
            print(f"  {k:28s} |diff| = {v:.3g}")
        sys.exit(2)

    rng = np.random.default_rng(SEED)
    male = preds["sex_male"].to_numpy() == 1
    rows = []
    for arm in ARMS:
        for target, ycol, pcol, mean_col in [("raw", "y", f"pred_{arm}", "y_discovery_mean"),
                                             ("deconfounded", "y_deconfounded", f"pred_{arm}_dec",
                                              "ytr_d_discovery_mean")]:
            y, p = preds[ycol].to_numpy(float), preds[pcol].to_numpy(float)
            n = len(y)
            r = corr(y, p)
            lo, hi = fisher_ci(r, n)
            sse = float(((y - p) ** 2).sum())
            mu_disc = float(preds[mean_col].iloc[0])
            slope, s_lo, s_hi, icpt, i_lo, i_hi = calibration(y, p, rng)
            rm, rf = corr(y[male], p[male]), corr(y[~male], p[~male])
            nm, nf = int(male.sum()), int((~male).sum())
            zdiff = (np.arctanh(rm) - np.arctanh(rf)) / np.sqrt(1 / (nm - 3) + 1 / (nf - 3))
            rows.append({
                "arm": arm, "target": target, "n": n, "r": r, "r_ci_lo": lo, "r_ci_hi": hi,
                "R2_holdout": 1 - sse / float(((y - y.mean()) ** 2).sum()),
                "R2_vs_discovery_mean": 1 - sse / float(((y - mu_disc) ** 2).sum()),
                "MAE": float(np.abs(y - p).mean()), "RMSE": float(np.sqrt(sse / n)),
                "MAE_mean_model": float(np.abs(y - mu_disc).mean()),
                "calibration_slope": slope, "slope_ci_lo": s_lo, "slope_ci_hi": s_hi,
                "calibration_intercept": icpt, "intercept_ci_lo": i_lo, "intercept_ci_hi": i_hi,
                "r_male": rm, "r_male_ci_lo": fisher_ci(rm, nm)[0], "r_male_ci_hi": fisher_ci(rm, nm)[1],
                "n_male": nm,
                "r_female": rf, "r_female_ci_lo": fisher_ci(rf, nf)[0],
                "r_female_ci_hi": fisher_ci(rf, nf)[1], "n_female": nf,
                "p_sex_difference": float(2 * stats.norm.sf(abs(zdiff)))})
    met = pd.DataFrame(rows)
    met.to_csv(OUT_METRICS, index=False)

    prow = []
    idx = rng.integers(0, len(preds), size=(N_BOOT, len(preds)))
    for target, ycol, suf in [("raw", "y", ""), ("deconfounded", "y_deconfounded", "_dec")]:
        y = preds[ycol].to_numpy(float)
        for a, b in PAIRS:
            pa, pb = preds[f"pred_{a}{suf}"].to_numpy(float), preds[f"pred_{b}{suf}"].to_numpy(float)
            ra, rb, rab = corr(y, pa), corr(y, pb), corr(pa, pb)
            boot = np.empty(N_BOOT)
            for i in range(N_BOOT):
                j = idx[i]
                boot[i] = corr(y[j], pa[j]) - corr(y[j], pb[j])
            p_two = float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))
            zl, zh = zou_ci(ra, rb, rab, len(y))
            prow.append({"target": target, "arm_a": a, "arm_b": b, "r_a": ra, "r_b": rb,
                         "r_between_predictions": rab, "delta_r": ra - rb,
                         "boot_ci_lo": float(np.percentile(boot, 2.5)),
                         "boot_ci_hi": float(np.percentile(boot, 97.5)),
                         "boot_p_two_sided": max(p_two, 1.0 / N_BOOT),
                         "zou_ci_lo": zl, "zou_ci_hi": zh, "n": len(y), "n_boot": N_BOOT})
    pair = pd.DataFrame(prow)
    pair.to_csv(OUT_PAIRED, index=False)

    entry["status"] = "completed: gate passed, reporting metrics computed"
    ACCESS_LOG.write_text(json.dumps(log, indent=2), encoding="utf-8")

    pd.set_option("display.width", 200)
    print("\nEXTENDED METRICS (sealed hold-out)")
    print(met[["arm", "target", "r", "R2_holdout", "R2_vs_discovery_mean", "MAE", "MAE_mean_model",
               "calibration_slope", "slope_ci_lo", "slope_ci_hi", "r_male", "r_female",
               "p_sex_difference"]].round(4).to_string(index=False))
    print("\nPAIRED COMPARISONS (bootstrap over hold-out subjects)")
    print(pair[["target", "arm_a", "arm_b", "delta_r", "boot_ci_lo", "boot_ci_hi",
                "boot_p_two_sided", "zou_ci_lo", "zou_ci_hi"]].round(4).to_string(index=False))
    print(f"\nSaved -> {OUT_METRICS.relative_to(ROOT).as_posix()}, "
          f"{OUT_PAIRED.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
