#!/usr/bin/env python
"""
Extract uncleaned parcel time series once per smoothing level, so that filter and confound variants can be
applied to the region signals rather than re-running the masker for each variant.

Output
------
  outputs/connectivity/preproc_variants_raw.h5
      subjects/<sub>/smooth6  (T, 215)  uncleaned, 6 mm smoothing
      subjects/<sub>/smooth0  (T, 215)  uncleaned, no smoothing
      subjects/<sub>/confounds (T, 36)  the 36P matrix, unfiltered
      confound_columns attr

Usage
-----
  python scripts/extraction/extract_preprocessing_variants.py --n 250
"""
from __future__ import annotations

import argparse, json, sys, time, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import numpy as np
import pandas as pd
import yaml
from nilearn import datasets as nlds
from nilearn import image, maskers

from src.data.preprocessing import CONFOUND_36P, load_confounds, load_participants

warnings.filterwarnings("ignore")

CFG_PATH = "config/pipeline.yaml"
MANIFEST = Path("outputs/honest/holdout_manifest.json")
SEED = 20260825
DROP_SUB = {
    "Background",
    "Left Cerebral White Matter", "Right Cerebral White Matter",
    "Left Cerebral Cortex", "Right Cerebral Cortex",
    "Left Lateral Ventricle", "Right Lateral Ventricle",
}


def raw_masker(atlas_img):
    """No cleaning at all -- cleaning is applied later at the timeseries level."""
    return maskers.NiftiLabelsMasker(
        labels_img=atlas_img, standardize=False, detrend=False,
        low_pass=None, high_pass=None, smoothing_fwhm=None,
        resampling_target="data", verbose=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(CFG_PATH))
    pp = cfg["preprocessing"]
    fmriprep = Path(cfg["paths"]["fmriprep_dir"])
    conn = Path(cfg["paths"]["connectivity_dir"])
    task, space = cfg["dataset"]["task"], cfg["dataset"]["space"]

    ids_all = json.loads(MANIFEST.read_text())["discovery"]
    parts = load_participants(cfg["paths"]["bids_root"])
    y = np.array([float(parts.loc[s, "IST_intelligence_total"]) for s in ids_all])
    sex = np.array([1 if str(parts.loc[s, "sex"]).lower().startswith("m") else 0
                    for s in ids_all])
    strat = pd.qcut(y, 4, labels=False, duplicates="drop") * 2 + sex
    rng = np.random.default_rng(SEED)
    picked = []
    for lev in np.unique(strat):                     # stratified subsample
        idx = np.where(strat == lev)[0]
        k = int(round(args.n * len(idx) / len(ids_all)))
        picked += list(rng.choice(idx, size=min(k, len(idx)), replace=False))
    subs = sorted(ids_all[i] for i in picked)
    if args.limit:
        subs = subs[: args.limit]
    print(f"{len(subs)} subjects, stratified from the {len(ids_all)} discovery split")

    sch = nlds.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7)
    ho = nlds.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    ho_labels = [l.decode() if isinstance(l, bytes) else l for l in ho.labels]
    keep_idx = [i for i, l in enumerate(ho_labels) if l not in DROP_SUB and i > 0]
    col_idx = [i - 1 for i in keep_idx]
    m_ctx, m_sub = raw_masker(sch.maps), raw_masker(ho.maps)
    print(f"  {200 + len(col_idx)} regions; smoothing levels: 6 mm and 0 mm")

    out_path = conn / "preproc_variants_raw.h5"
    done, failed = 0, []
    t0 = time.time()
    with h5py.File(out_path, "w") as out:
        out.attrs["confound_columns"] = json.dumps(CONFOUND_36P)
        out.attrs["smoothing_fwhm"] = pp["smoothing_fwhm"]
        grp = out.create_group("subjects")
        for i, s in enumerate(subs, 1):
            bold = fmriprep / s / "func" / f"{s}_task-{task}_space-{space}_desc-preproc_bold.nii.gz"
            confp = fmriprep / s / "func" / f"{s}_task-{task}_desc-confounds_regressors.tsv"
            if not bold.exists() or not confp.exists():
                failed.append((s, "missing")); continue
            try:
                conf_df, _ = load_confounds(confp, pp["fd_threshold"])
                img = image.load_img(str(bold))
                img_s = image.smooth_img(img, pp["smoothing_fwhm"])
                g = grp.create_group(s)
                for tag, im in (("smooth6", img_s), ("smooth0", img)):
                    ts = np.hstack([m_ctx.fit_transform(im),
                                    m_sub.fit_transform(im)[:, col_idx]]).astype(np.float32)
                    g.create_dataset(tag, data=ts)
                g.create_dataset("confounds", data=conf_df.values.astype(np.float32))
                done += 1
            except Exception as exc:
                failed.append((s, str(exc)[:70]))
            if i % 10 == 0 or i == len(subs):
                el = time.time() - t0
                print(f"  {i}/{len(subs)}  ok={done}  fail={len(failed)}  "
                      f"{el/60:.1f}m elapsed, eta {(el/i*(len(subs)-i))/60:.1f}m",
                      flush=True)

    print(f"\nDone: {done} ok, {len(failed)} failed -> {out_path}")
    if failed:
        print("  first failures:", failed[:4])


if __name__ == "__main__":
    main()
