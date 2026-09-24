#!/usr/bin/env python
"""
Extract PIOP1 and PIOP2 parcel time series with the ID1000 pipeline applied unchanged.

Nothing is tuned on PIOP. The filter band is held fixed across collections, so the masker is given each
run's repetition time.

Storage
-------
Sharding
--------
A single process takes about 16 s per run, which is roughly six hours for
PIOP1's 1296 runs. --shard i --nshards k processes only the subjects with
index i mod k and writes its own HDF5, so k processes can run concurrently and
merge afterwards. Subjects are assigned by stride rather than by contiguous
block so that every shard sees a similar mix and the timing estimates from one
shard generalise. HDF5 is not safe for concurrent writers to one file, which is
why each shard gets its own; merge_shards() concatenates them at the end.

Usage
-----
  python scripts/extraction/extract_timeseries_piop.py --dataset PIOP1 --limit 3     # benchmark
  python scripts/extraction/extract_timeseries_piop.py --dataset PIOP1 --shard 0 --nshards 6
  python scripts/extraction/extract_timeseries_piop.py --dataset PIOP1 --merge 6
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

from src.data.paths import collection_root
from src.data.preprocessing import load_confounds

warnings.filterwarnings("ignore")

CFG_PATH = "config/pipeline.yaml"
ROOTS = {name: collection_root(name) for name in ("PIOP1", "PIOP2")}
TR = {"mb3": 0.75, "seq": 2.0}
SPACE = "MNI152NLin2009cAsym"
OUT_DIR = Path("outputs/piop")

# identical to extract_unscrubbed.py -- non-grey and duplicate-of-cortex labels
DROP_SUB = {
    "Background",
    "Left Cerebral White Matter", "Right Cerebral White Matter",
    "Left Cerebral Cortex", "Right Cerebral Cortex",
    "Left Lateral Ventricle", "Right Lateral Ventricle",
}


def build_masker(cfg, atlas_img, tr):
    """Masker in percent signal change at this run's TR.

    standardize="psc" rather than "zscore_sample" is the single choice the whole
    amplitude line of the paper rests on: z-scoring divides each region by its own
    temporal SD and destroys regional BOLD amplitude. See extract_unscrubbed.py
    for the measurement showing the z-scored diagonal is a motion proxy.
    """
    pp = cfg["preprocessing"]
    return maskers.NiftiLabelsMasker(
        labels_img=atlas_img, standardize="psc", detrend=True, t_r=tr,
        high_pass=pp["high_pass"], low_pass=pp["low_pass"],
        smoothing_fwhm=pp["smoothing_fwhm"], resampling_target="data", verbose=0)


def find_runs(fmriprep, sub):
    """Every (task, acq) this subject has a preprocessed MNI BOLD and confounds for."""
    runs = []
    for bold in sorted((fmriprep / sub / "func").glob(
            f"*_space-{SPACE}_desc-preproc_bold.nii.gz")):
        stem = bold.name.split("_space-")[0]
        conf = bold.parent / f"{stem}_desc-confounds_regressors.tsv"
        if not conf.exists():
            continue
        task = stem.split("task-")[1].split("_acq")[0]
        acq = stem.split("acq-")[1]
        runs.append((task, acq, bold, conf))
    return runs


def merge_shards(dataset, nshards):
    """Concatenate per-shard HDF5 files into one, verifying no subject collides."""
    out_path = OUT_DIR / f"{dataset}_ts.h5"
    parts = [OUT_DIR / f"{dataset}_ts_shard{i}of{nshards}.h5" for i in range(nshards)]
    missing = [p.name for p in parts if not p.exists()]
    if missing:
        sys.exit(f"cannot merge, shards not finished: {missing}")
    n_sub = n_run = 0
    with h5py.File(out_path, "w") as out:
        grp = out.create_group("subjects")
        for p in parts:
            with h5py.File(p, "r") as src:
                for s in src["subjects"]:
                    if s in grp:
                        sys.exit(f"subject {s} appears in more than one shard")
                    src.copy(src["subjects"][s], grp, name=s)
                    n_sub += 1
                    n_run += len(src["subjects"][s])
    print(f"merged {nshards} shards -> {out_path}")
    print(f"  {n_sub} subjects, {n_run} runs")
    for p in parts:
        p.unlink()
    print(f"  removed {nshards} shard files")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(ROOTS), required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N subjects (benchmark mode)")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--nshards", type=int, default=None)
    ap.add_argument("--merge", type=int, default=None, metavar="NSHARDS",
                    help="merge NSHARDS shard files and exit")
    args = ap.parse_args()

    if args.merge:
        merge_shards(args.dataset, args.merge)
        return
    if (args.shard is None) != (args.nshards is None):
        sys.exit("--shard and --nshards must be given together")

    cfg = yaml.safe_load(open(CFG_PATH))
    pp = cfg["preprocessing"]
    root = ROOTS[args.dataset]
    fmriprep = root / "derivatives" / "fmriprep"

    sch = nlds.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7)
    ho = nlds.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    ho_labels = [l.decode() if isinstance(l, bytes) else l for l in ho.labels]
    keep_idx = [i for i, l in enumerate(ho_labels) if l not in DROP_SUB and i > 0]
    col_idx = [i - 1 for i in keep_idx]
    n_roi = 200 + len(col_idx)
    print(f"{args.dataset}: Schaefer-200 + {len(col_idx)} subcortical = {n_roi} regions")
    print(f"  PSC standardisation, {pp['high_pass']}-{pp['low_pass']} Hz, "
          f"{pp['smoothing_fwhm']} mm FWHM, 36P confounds  [frozen from ID1000]")

    maskers_by_tr = {}                      # atlas resampling is the slow part; cache it

    def get_maskers(tr):
        if tr not in maskers_by_tr:
            maskers_by_tr[tr] = (build_masker(cfg, sch.maps, tr),
                                 build_masker(cfg, ho.maps, tr))
        return maskers_by_tr[tr]

    subs = sorted(p.name for p in fmriprep.glob("sub-*") if p.is_dir())
    if args.limit:
        subs = subs[: args.limit]
    suffix = f"_bench{args.limit}" if args.limit else ""
    if args.shard is not None:
        subs = subs[args.shard::args.nshards]      # stride, so shards are comparable
        suffix = f"_shard{args.shard}of{args.nshards}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.dataset}_ts{suffix}.h5"
    print(f"Processing {len(subs)} subjects -> {out_path}\n")

    t0 = time.time()
    n_runs = n_fail = 0
    per_task_time, failures, shapes = {}, [], []

    with h5py.File(out_path, "w") as out:
        grp = out.create_group("subjects")
        for i, s in enumerate(subs, 1):
            runs = find_runs(fmriprep, s)
            if args.tasks:
                runs = [r for r in runs if r[0] in args.tasks]
            sg = grp.create_group(s)
            for task, acq, bold, conf in runs:
                tr = TR[acq]
                t_run = time.time()
                try:
                    conf_df, keep = load_confounds(conf, pp["fd_threshold"])
                    m_ctx, m_sub = get_maskers(tr)
                    img = image.load_img(str(bold))
                    ca = conf_df.values.astype(np.float32)
                    ts = np.hstack([m_ctx.fit_transform(img, confounds=ca),
                                    m_sub.fit_transform(img, confounds=ca)[:, col_idx]]
                                   ).astype(np.float32)
                    fd = pd.read_csv(conf, sep="\t", usecols=["framewise_displacement"],
                                     na_values="n/a")["framewise_displacement"]
                    fd = fd.fillna(0.0).to_numpy().astype(np.float32)
                    g = sg.create_group(task)
                    g.attrs["tr"], g.attrs["acq"] = tr, acq
                    g.create_dataset("time_series", data=ts, compression="gzip")
                    g.create_dataset("fd", data=fd[: len(ts)])
                    g.create_dataset("keep", data=keep[: len(ts)])
                    n_runs += 1
                    shapes.append((task, ts.shape[0]))
                    per_task_time.setdefault(task, []).append(time.time() - t_run)
                except Exception as exc:
                    n_fail += 1
                    failures.append((s, task, str(exc)[:70]))
            el = time.time() - t0
            print(f"  [{i:3d}/{len(subs)}] {s}  {len(runs)} runs  "
                  f"elapsed={el/60:.2f}m  eta={(el/i*(len(subs)-i))/60:.1f}m", flush=True)

    el = time.time() - t0
    print(f"\nDone: {n_runs} runs from {len(subs)} subjects, {n_fail} failed "
          f"in {el/60:.2f} min -> {out_path}")
    if per_task_time:
        print("\n  seconds per run by task:")
        for t, v in sorted(per_task_time.items(), key=lambda kv: -np.mean(kv[1])):
            print(f"    {t:16s} {np.mean(v):6.1f} s   (n={len(v)})")
        sec = np.mean([x for v in per_task_time.values() for x in v])
        full = {"PIOP1": 216 * 6, "PIOP2": 226 * 4}[args.dataset]
        print(f"\n  mean {sec:.1f} s/run  ->  full {args.dataset} "
              f"({full} runs) projected at {sec*full/3600:.1f} h")
    if shapes:
        d = pd.DataFrame(shapes, columns=["task", "T"])
        print("\n  volumes retained per task (should equal the raw run length):")
        for t, g in d.groupby("task"):
            u = sorted(set(g["T"]))
            print(f"    {t:16s} {u if len(u) < 4 else f'{min(u)}-{max(u)}'}")
    if failures:
        print("\n  failures:", failures[:6])


if __name__ == "__main__":
    main()
