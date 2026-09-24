#!/usr/bin/env python
"""
Extract Harvard-Oxford subcortical time series for the z-scored extraction.

Output
------
  outputs/connectivity/subcortical_ts.h5
      subjects/<sub>/time_series   (T, n_sub_rois)
      subjects/<sub>/fc_matrix     (n_sub_rois, n_sub_rois)
  outputs/connectivity/subcortical_meta.npz   label names

Usage
-----
  python scripts/extraction/extract_timeseries_subcortical.py                # all subjects
  python scripts/extraction/extract_timeseries_subcortical.py --limit 20     # smoke test
  python scripts/extraction/extract_timeseries_subcortical.py --workers 4    # parallel
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import numpy as np
import yaml
from nilearn import datasets as nlds
from nilearn import image, maskers

from src.data.preprocessing import load_confounds

warnings.filterwarnings("ignore")

CFG_PATH = "config/pipeline.yaml"

# Harvard-Oxford subcortical labels to DROP (not discrete grey-matter ROIs).
DROP_LABELS = {
    "Background",
    "Left Cerebral White Matter", "Right Cerebral White Matter",
    "Left Cerebral Cortex", "Right Cerebral Cortex",
    "Left Lateral Ventricle", "Right Lateral Ventricle",
}


def build_masker(cfg, atlas_img):
    pp = cfg["preprocessing"]
    return maskers.NiftiLabelsMasker(
        labels_img=atlas_img,
        standardize="zscore_sample",
        detrend=True,
        t_r=cfg["dataset"]["tr"],
        high_pass=pp["high_pass"],
        low_pass=pp["low_pass"],
        smoothing_fwhm=pp["smoothing_fwhm"],
        resampling_target="data",
        verbose=0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--config", default=CFG_PATH)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    pp = cfg["preprocessing"]
    fmriprep = Path(cfg["paths"]["fmriprep_dir"])
    conn = Path(cfg["paths"]["connectivity_dir"])
    conn.mkdir(parents=True, exist_ok=True)
    task = cfg["dataset"]["task"]
    space = cfg["dataset"]["space"]

    # ── Atlas ────────────────────────────────────────────────────────────────
    ho = nlds.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    labels = [l.decode() if isinstance(l, bytes) else l for l in ho.labels]
    keep_idx = [i for i, l in enumerate(labels) if l not in DROP_LABELS]
    keep_names = [labels[i] for i in keep_idx]
    print(f"Harvard-Oxford subcortical: {len(labels)} labels -> keeping {len(keep_names)}")
    for n in keep_names:
        print(f"    {n}")

    # NiftiLabelsMasker returns one column per NON-ZERO label value, in
    # ascending label order.  Label value i corresponds to labels[i], so the
    # returned column for label value i sits at position i-1.
    col_idx = [i - 1 for i in keep_idx if i > 0]

    masker = build_masker(cfg, ho.maps)

    # ── Subject list: mirror the cortical h5 so the two align exactly ───────
    with h5py.File(conn / "fc_matrices.h5", "r") as f:
        subs = sorted(f["subjects"].keys())
    if args.limit:
        subs = subs[: args.limit]
    print(f"\nProcessing {len(subs)} subjects ...")

    out_path = conn / "subcortical_ts.h5"
    done, failed = 0, []
    t_start = time.time()

    with h5py.File(out_path, "w") as out:
        grp = out.create_group("subjects")
        for i, sub in enumerate(subs, 1):
            bold = fmriprep / sub / "func" / (
                f"{sub}_task-{task}_space-{space}_desc-preproc_bold.nii.gz")
            confp = fmriprep / sub / "func" / (
                f"{sub}_task-{task}_desc-confounds_regressors.tsv")
            if not bold.exists() or not confp.exists():
                failed.append((sub, "missing file"))
                continue
            try:
                conf_df, keep = load_confounds(confp, pp["fd_threshold"])
                if int(keep.sum()) < pp["min_volumes_after_scrub"]:
                    failed.append((sub, "too few clean volumes"))
                    continue
                ts = masker.fit_transform(
                    image.load_img(str(bold)),
                    confounds=conf_df.values.astype(np.float32),
                )
                ts = ts[keep][:, col_idx]                 # scrub, then select ROIs
                fc = np.corrcoef(ts.T)
                np.fill_diagonal(fc, 0.0)

                g = grp.create_group(sub)
                g.create_dataset("time_series", data=ts.astype(np.float32))
                g.create_dataset("fc_matrix", data=fc.astype(np.float32))
                done += 1
            except Exception as exc:
                failed.append((sub, str(exc)[:80]))

            if i % 25 == 0 or i == len(subs):
                el = time.time() - t_start
                eta = el / i * (len(subs) - i)
                print(f"  {i}/{len(subs)}  ok={done}  fail={len(failed)}  "
                      f"elapsed={el/60:.1f}m  eta={eta/60:.1f}m", flush=True)

    np.savez(conn / "subcortical_meta.npz", labels=np.array(keep_names))
    print(f"\nDone: {done} ok, {len(failed)} failed -> {out_path}")
    if failed:
        print("First failures:", failed[:5])


if __name__ == "__main__":
    main()
