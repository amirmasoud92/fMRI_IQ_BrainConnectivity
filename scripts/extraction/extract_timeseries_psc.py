#!/usr/bin/env python
"""
Extract percent-signal-change parcel time series with every frame retained.

Percent signal change preserves regional signal amplitude, which z-scoring over time removes, and keeping
all frames preserves the alignment between participants that inter-subject analyses need.

Output
------
  outputs/connectivity/unscrubbed_ts.h5
      subjects/<sub>/time_series  (T, 215)  -- percent signal change, T aligned
      subjects/<sub>/fd           (T,)      -- framewise displacement
      subjects/<sub>/keep         (T,) bool -- FD <= threshold

Usage
-----
  python scripts/extraction/extract_timeseries_psc.py --limit 5
  python scripts/extraction/extract_timeseries_psc.py
"""
from __future__ import annotations

import argparse, sys, time, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import numpy as np
import pandas as pd
import yaml
from nilearn import datasets as nlds
from nilearn import image, maskers

from src.data.preprocessing import load_confounds

warnings.filterwarnings("ignore")

CFG_PATH = "config/pipeline.yaml"
DROP_SUB = {
    "Background",
    "Left Cerebral White Matter", "Right Cerebral White Matter",
    "Left Cerebral Cortex", "Right Cerebral Cortex",
    "Left Lateral Ventricle", "Right Lateral Ventricle",
}


def build_masker(cfg, atlas_img):
    """Masker in PERCENT SIGNAL CHANGE, not z-score."""
    pp = cfg["preprocessing"]
    return maskers.NiftiLabelsMasker(
        labels_img=atlas_img, standardize="psc", detrend=True,
        t_r=cfg["dataset"]["tr"], high_pass=pp["high_pass"], low_pass=pp["low_pass"],
        smoothing_fwhm=pp["smoothing_fwhm"], resampling_target="data", verbose=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(CFG_PATH))
    pp = cfg["preprocessing"]
    fmriprep = Path(cfg["paths"]["fmriprep_dir"])
    conn = Path(cfg["paths"]["connectivity_dir"])
    task, space = cfg["dataset"]["task"], cfg["dataset"]["space"]

    sch = nlds.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7)
    ho = nlds.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    ho_labels = [l.decode() if isinstance(l, bytes) else l for l in ho.labels]
    keep_idx = [i for i, l in enumerate(ho_labels) if l not in DROP_SUB and i > 0]
    col_idx = [i - 1 for i in keep_idx]
    print(f"Schaefer-200 + {len(col_idx)} subcortical = {200 + len(col_idx)} regions")

    m_ctx = build_masker(cfg, sch.maps)
    m_sub = build_masker(cfg, ho.maps)

    with h5py.File(conn / "fc_matrices.h5", "r") as f:
        subs = sorted(f["subjects"].keys())
    if args.limit:
        subs = subs[: args.limit]
    print(f"Processing {len(subs)} subjects (NO scrubbing -- alignment preserved)")

    out_path = conn / "unscrubbed_ts.h5"
    done, failed, lengths = 0, [], []
    t0 = time.time()

    with h5py.File(out_path, "w") as out:
        grp = out.create_group("subjects")
        for i, s in enumerate(subs, 1):
            bold = fmriprep / s / "func" / f"{s}_task-{task}_space-{space}_desc-preproc_bold.nii.gz"
            confp = fmriprep / s / "func" / f"{s}_task-{task}_desc-confounds_regressors.tsv"
            if not bold.exists() or not confp.exists():
                failed.append((s, "missing file")); continue
            try:
                conf_df, keep = load_confounds(confp, pp["fd_threshold"])
                img = image.load_img(str(bold))
                ca = conf_df.values.astype(np.float32)
                ts_c = m_ctx.fit_transform(img, confounds=ca)
                ts_s = m_sub.fit_transform(img, confounds=ca)[:, col_idx]
                ts = np.hstack([ts_c, ts_s]).astype(np.float32)   # NO scrubbing
                fd = pd.read_csv(confp, sep="\t", usecols=["framewise_displacement"],
                                 na_values="n/a")["framewise_displacement"]
                fd = fd.fillna(0.0).to_numpy().astype(np.float32)
                g = grp.create_group(s)
                g.create_dataset("time_series", data=ts)
                g.create_dataset("fd", data=fd[: len(ts)])
                g.create_dataset("keep", data=keep[: len(ts)])
                lengths.append(ts.shape[0]); done += 1
            except Exception as exc:
                failed.append((s, str(exc)[:70]))
            if i % 25 == 0 or i == len(subs):
                el = time.time() - t0
                print(f"  {i}/{len(subs)}  ok={done}  fail={len(failed)}  "
                      f"elapsed={el/60:.1f}m  eta={(el/i*(len(subs)-i))/60:.1f}m", flush=True)

    L = np.array(lengths)
    print(f"\nDone: {done} ok, {len(failed)} failed -> {out_path}")
    print(f"  lengths: min={L.min()} max={L.max()} distinct={len(set(L.tolist()))}")
    print(f"  subjects at modal length: {(L == np.bincount(L).argmax()).sum()}/{done}")
    if len(set(L.tolist())) == 1:
        print("  -> fully aligned; ISC is valid")
    else:
        print("  -> lengths differ (variable run length); ISC must be computed on the "
              "common prefix, which the analysis script handles")
    if failed:
        print("  first failures:", failed[:4])


if __name__ == "__main__":
    main()
