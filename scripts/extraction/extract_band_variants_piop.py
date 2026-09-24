#!/usr/bin/env python
"""
Re-extract PIOP time series with and without the low-pass filter.

Usage
-----
  python scripts/extraction/extract_band_variants_piop.py --dataset PIOP1 --shard 0 --nshards 6
  python scripts/extraction/extract_band_variants_piop.py --dataset PIOP1 --merge 6
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
import pandas as pd
import yaml
from nilearn import datasets as nlds
from nilearn import image, maskers
from nilearn.signal import clean

from src.data.paths import collection_root
from src.data.preprocessing import CONFOUND_36P

warnings.filterwarnings("ignore")

CFG = "config/pipeline.yaml"
OUT = Path("outputs/piop")
ROOTS = {name: collection_root(name) for name in ("PIOP1", "PIOP2")}
TR_OF = {"mb3": 0.75, "seq": 2.0}
SPACE = "MNI152NLin2009cAsym"
DROP_SUB = {
    "Background",
    "Left Cerebral White Matter", "Right Cerebral White Matter",
    "Left Cerebral Cortex", "Right Cerebral Cortex",
    "Left Lateral Ventricle", "Right Lateral Ventricle",
}
VARIANTS = {"base": 0.10, "broad": None}


def merge(dataset, nshards):
    for v in VARIANTS:
        out = OUT / f"{dataset}_{v}_ts.h5"
        parts = [OUT / f"{dataset}_{v}_shard{i}of{nshards}.h5" for i in range(nshards)]
        miss = [p.name for p in parts if not p.exists()]
        if miss:
            sys.exit(f"cannot merge {v}: {miss}")
        n_s = n_r = 0
        with h5py.File(out, "w") as o:
            g = o.create_group("subjects")
            for p in parts:
                with h5py.File(p, "r") as s:
                    for sub in s["subjects"]:
                        if sub in g:
                            sys.exit(f"{sub} in two shards")
                        s.copy(s["subjects"][sub], g, name=sub)
                        n_s += 1; n_r += len(s["subjects"][sub])
        print(f"  {out.name}: {n_s} subjects, {n_r} runs")
        for p in parts:
            p.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(ROOTS), required=True)
    ap.add_argument("--shard", type=int)
    ap.add_argument("--nshards", type=int)
    ap.add_argument("--merge", type=int)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    if args.merge:
        merge(args.dataset, args.merge); return

    cfg = yaml.safe_load(open(CFG))
    pp = cfg["preprocessing"]
    fmriprep = ROOTS[args.dataset] / "derivatives" / "fmriprep"
    sch = nlds.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7)
    ho = nlds.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    hol = [l.decode() if isinstance(l, bytes) else l for l in ho.labels]
    col_idx = [i - 1 for i, l in enumerate(hol) if l not in DROP_SUB and i > 0]
    mk = dict(standardize=False, detrend=False, smoothing_fwhm=pp["smoothing_fwhm"],
              resampling_target="data", verbose=0)
    m_ctx = maskers.NiftiLabelsMasker(labels_img=sch.maps, **mk)
    m_sub = maskers.NiftiLabelsMasker(labels_img=ho.maps, **mk)

    subs = sorted(p.name for p in fmriprep.glob("sub-*") if p.is_dir())
    if args.limit:
        subs = subs[: args.limit]
    sfx = ""
    if args.shard is not None:
        subs = subs[args.shard::args.nshards]
        sfx = f"_shard{args.shard}of{args.nshards}"
    print(f"{args.dataset}{sfx}: {len(subs)} subjects, variants {list(VARIANTS)}")

    files = {v: h5py.File(OUT / f"{args.dataset}_{v}{sfx}.h5", "w") for v in VARIANTS}
    grps = {v: f.create_group("subjects") for v, f in files.items()}
    t0, nrun = time.time(), 0
    for i, s in enumerate(subs, 1):
        for bold in sorted((fmriprep / s / "func").glob(
                f"*_space-{SPACE}_desc-preproc_bold.nii.gz")):
            stem = bold.name.split("_space-")[0]
            conf = bold.parent / f"{stem}_desc-confounds_regressors.tsv"
            if not conf.exists():
                continue
            task = stem.split("task-")[1].split("_acq")[0]
            tr = TR_OF[stem.split("acq-")[1]]
            try:
                img = image.load_img(str(bold))
                raw = np.hstack([m_ctx.fit_transform(img),
                                 m_sub.fit_transform(img)[:, col_idx]]).astype(np.float64)
                cdf = pd.read_csv(conf, sep="\t", na_values="n/a")
                cols = [c for c in CONFOUND_36P if c in cdf.columns]
                C = cdf[cols].fillna(0.0).to_numpy(np.float64)
                fd = cdf["framewise_displacement"].fillna(0.0).to_numpy()
                for v, low in VARIANTS.items():
                    ts = clean(raw.copy(), confounds=C, standardize="psc",
                               detrend=True, t_r=tr, high_pass=pp["high_pass"],
                               low_pass=low)
                    g = grps[v].require_group(s).create_group(task)
                    g.attrs["tr"] = tr
                    g.create_dataset("time_series", data=ts.astype(np.float32),
                                     compression="gzip")
                    g.create_dataset("fd", data=fd[: len(ts)].astype(np.float32))
                nrun += 1
            except Exception as exc:
                print(f"    {s}/{task}: {str(exc)[:60]}")
        if i % 10 == 0 or i == len(subs):
            el = time.time() - t0
            print(f"  {i}/{len(subs)}  runs={nrun}  {el/60:.1f}m  "
                  f"eta={(el/i*(len(subs)-i))/60:.1f}m", flush=True)
    for f in files.values():
        f.close()
    print(f"\nDone: {nrun} runs from {len(subs)} subjects")


if __name__ == "__main__":
    main()
