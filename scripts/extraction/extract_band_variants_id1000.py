#!/usr/bin/env python
"""
Re-extract ID1000 time series with and without global signal regression and the low-pass filter.

The Four Variants
-----------------
  base          36P including global signal, 0.01-0.10 Hz   (current pipeline)
  noGSR         32P, global signal and its expansions dropped, 0.01-0.10 Hz
  broad         36P, 0.01 Hz high-pass only, no low-pass
  noGSR_broad   32P, high-pass only                          (maximal signal)

Usage
-----
  python scripts/extraction/extract_band_variants_id1000.py --limit 5
  python scripts/extraction/extract_band_variants_id1000.py
"""
from __future__ import annotations

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
import yaml
from nilearn import datasets as nlds
from nilearn import image, maskers
from nilearn.signal import clean

from src.data.preprocessing import CONFOUND_36P

warnings.filterwarnings("ignore")

CFG = "config/pipeline.yaml"
OUT = Path("outputs/piop")
TR = 2.2
DROP_SUB = {
    "Background",
    "Left Cerebral White Matter", "Right Cerebral White Matter",
    "Left Cerebral Cortex", "Right Cerebral Cortex",
    "Left Lateral Ventricle", "Right Lateral Ventricle",
}
GSR_COLS = [c for c in CONFOUND_36P if c.startswith("global_signal")]
NO_GSR = [c for c in CONFOUND_36P if not c.startswith("global_signal")]

VARIANTS = {
    "base":        dict(cols=CONFOUND_36P, low=0.10),
    "noGSR":       dict(cols=NO_GSR,       low=0.10),
    "broad":       dict(cols=CONFOUND_36P, low=None),
    "noGSR_broad": dict(cols=NO_GSR,       low=None),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(CFG))
    pp = cfg["preprocessing"]
    fmriprep = Path(cfg["paths"]["fmriprep_dir"])
    task, space = cfg["dataset"]["task"], cfg["dataset"]["space"]
    ids = json.loads(Path("outputs/honest/holdout_manifest.json")
                     .read_text())["discovery"]
    if args.limit:
        ids = ids[: args.limit]

    sch = nlds.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7)
    ho = nlds.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    hol = [l.decode() if isinstance(l, bytes) else l for l in ho.labels]
    col_idx = [i - 1 for i, l in enumerate(hol) if l not in DROP_SUB and i > 0]

    # no cleaning in the masker: smoothing and parcel averaging only
    mk = dict(standardize=False, detrend=False, smoothing_fwhm=pp["smoothing_fwhm"],
              resampling_target="data", verbose=0)
    m_ctx = maskers.NiftiLabelsMasker(labels_img=sch.maps, **mk)
    m_sub = maskers.NiftiLabelsMasker(labels_img=ho.maps, **mk)
    print(f"Nyquist at TR = {TR} s is {1/(2*TR):.3f} Hz; broadband variants use a "
          f"high-pass only")
    print(f"{len(ids)} subjects, {len(VARIANTS)} variants from one masking pass\n")

    paths = {k: OUT / f"ID1000_{k}_ts.h5" for k in VARIANTS}
    files = {k: h5py.File(p, "w") for k, p in paths.items()}
    grps = {k: f.create_group("subjects") for k, f in files.items()}
    t0, done, failed = time.time(), 0, []
    for i, s in enumerate(ids, 1):
        bold = fmriprep / s / "func" / f"{s}_task-{task}_space-{space}_desc-preproc_bold.nii.gz"
        confp = fmriprep / s / "func" / f"{s}_task-{task}_desc-confounds_regressors.tsv"
        if not bold.exists() or not confp.exists():
            failed.append(s); continue
        try:
            img = image.load_img(str(bold))
            raw = np.hstack([m_ctx.fit_transform(img),
                             m_sub.fit_transform(img)[:, col_idx]]).astype(np.float64)
            conf_all = pd.read_csv(confp, sep="\t", na_values="n/a")
            for k, v in VARIANTS.items():
                cols = [c for c in v["cols"] if c in conf_all.columns]
                C = conf_all[cols].fillna(0.0).to_numpy(np.float64)
                ts = clean(raw.copy(), confounds=C, standardize="psc",
                           detrend=True, t_r=TR, high_pass=pp["high_pass"],
                           low_pass=v["low"])
                grps[k].create_dataset(f"{s}/time_series",
                                       data=ts.astype(np.float32), compression="gzip")
            done += 1
        except Exception as exc:
            failed.append(s)
            if len(failed) < 4:
                print(f"    {s}: {str(exc)[:70]}")
        if i % 25 == 0 or i == len(ids):
            el = time.time() - t0
            print(f"  {i}/{len(ids)}  ok={done}  fail={len(failed)}  "
                  f"{el/60:.1f}m  eta={(el/i*(len(ids)-i))/60:.1f}m", flush=True)
    for f in files.values():
        f.close()

    print(f"\nDone: {done} subjects, {len(failed)} failed")
    # confirm the baseline variant reproduces the existing extraction
    conn = Path(cfg["paths"]["connectivity_dir"])
    with h5py.File(paths["base"], "r") as a, h5py.File(conn / "unscrubbed_ts.h5", "r") as b:
        chk = [s for s in list(a["subjects"])[:5] if s in b["subjects"]]
        for s in chk:
            x = a["subjects"][s]["time_series"][:]
            z = b["subjects"][s]["time_series"][:]
            n = min(len(x), len(z))
            r = np.median([np.corrcoef(x[:n, j], z[:n, j])[0, 1]
                           for j in range(x.shape[1])])
            print(f"  base vs existing extraction, {s}: median region "
                  f"correlation {r:.4f}")
    print("\n  a correlation near 1.0 confirms mask-once-then-clean is equivalent "
          "to the original per-variant extraction")
    for k, p in paths.items():
        print(f"    {p}  ({p.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
