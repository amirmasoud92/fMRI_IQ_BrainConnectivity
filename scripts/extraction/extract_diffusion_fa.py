#!/usr/bin/env python
"""
Summarise each participant's fractional anisotropy map within the released diffusion brain mask.

The summaries are invariant to head position, so no registration is needed.

Features per subject (~20)
--------------------------
  mean/median/SD/skew/kurtosis of FA over the brain mask
  the same over the white-matter compartment (FA > 0.2)
  FA percentiles 5..95
  volume fractions above FA thresholds 0.2/0.3/0.4/0.5
  total mask volume in voxels

Output
------
  outputs/connectivity/dwi_fa_features.npz  (features, subject_ids, feature_names)

Usage
-----
  python scripts/extraction/extract_diffusion_fa.py [--limit N]
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import nibabel as nib
import numpy as np
import yaml
from scipy import stats

warnings.filterwarnings("ignore")

CFG_PATH = "config/pipeline.yaml"
DWI_DIR  = Path("derivatives/dwipreproc")
PCTS     = [5, 10, 25, 50, 75, 90, 95]
THRESHES = [0.2, 0.3, 0.4, 0.5]


def feature_names():
    n = ["fa_mean", "fa_median", "fa_sd", "fa_skew", "fa_kurt",
         "wm_fa_mean", "wm_fa_median", "wm_fa_sd", "wm_frac"]
    n += [f"fa_p{p}" for p in PCTS]
    n += [f"fa_frac_gt{t}" for t in THRESHES]
    n += ["mask_voxels"]
    return n


def subject_features(fa_path, mask_path):
    fa = np.asarray(nib.load(str(fa_path)).dataobj, dtype=np.float32)
    if mask_path.exists():
        m = np.asarray(nib.load(str(mask_path)).dataobj) > 0
    else:
        m = fa > 0
    v = fa[m & np.isfinite(fa)]
    v = v[(v > 0) & (v <= 1.0)]                     # FA is bounded in [0, 1]
    if v.size < 1000:
        return None
    wm = v[v > 0.2]
    out = [v.mean(), np.median(v), v.std(), stats.skew(v), stats.kurtosis(v),
           wm.mean() if wm.size else np.nan,
           np.median(wm) if wm.size else np.nan,
           wm.std() if wm.size else np.nan,
           wm.size / v.size]
    out += list(np.percentile(v, PCTS))
    out += [float((v > t).mean()) for t in THRESHES]
    out += [float(v.size)]
    return np.array(out, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(CFG_PATH))
    out_dir = Path(cfg["paths"]["connectivity_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    subs = sorted(p.name for p in DWI_DIR.glob("sub-*") if p.is_dir())
    if args.limit:
        subs = subs[: args.limit]
    print(f"{len(subs)} DWI subjects")

    names = feature_names()
    feats, kept, failed = [], [], []
    t0 = time.time()

    for i, s in enumerate(subs, 1):
        fa = DWI_DIR / s / "dwi" / f"{s}_model-DTI_desc-WLS_FA.nii.gz"
        mk = DWI_DIR / s / "dwi" / f"{s}_desc-brain_mask.nii.gz"
        if not fa.exists():
            failed.append((s, "no FA"))
            continue
        try:
            f = subject_features(fa, mk)
            if f is None:
                failed.append((s, "too few voxels"))
                continue
            feats.append(f)
            kept.append(s)
        except Exception as exc:
            failed.append((s, str(exc)[:50]))
        if i % 150 == 0 or i == len(subs):
            el = time.time() - t0
            print(f"  {i}/{len(subs)}  ok={len(kept)}  elapsed={el/60:.1f}m  "
                  f"eta={(el/i*(len(subs)-i))/60:.1f}m", flush=True)

    X = np.stack(feats).astype(np.float32)
    np.savez_compressed(out_dir / "dwi_fa_features.npz",
                        features=X, subject_ids=np.array(kept),
                        feature_names=np.array(names))
    print(f"\nDone: {len(kept)} ok, {len(failed)} failed -> {X.shape}")
    if failed:
        print("  first failures:", failed[:5])
    print(f"  -> {out_dir / 'dwi_fa_features.npz'}")


if __name__ == "__main__":
    main()
