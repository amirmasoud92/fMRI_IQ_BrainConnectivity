#!/usr/bin/env python
"""
Seal a stratified evaluation set and record a digest of it, so that it cannot be silently redrawn.

Usage
-----
  python scripts/prediction/seal_holdout.py            # seal, or verify an existing seal
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.data.preprocessing import load_participants

CFG_PATH = "config/pipeline.yaml"
TARGET   = "IST_intelligence_total"
SEED     = 20260825
FRAC     = 0.20
OUT      = Path("outputs/honest/holdout_manifest.json")
FS_DIR   = Path("derivatives/fs_stats")
VOL_COL  = "EstimatedTotalIntraCranialVol"


def digest(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def main():
    cfg = yaml.safe_load(open(CFG_PATH))
    conn = Path(cfg["paths"]["connectivity_dir"])
    with h5py.File(conn / "unscrubbed_ts.h5", "r") as f:
        subs = {s: f["subjects"][s]["time_series"].shape[0] for s in f["subjects"]}

    frames = []
    for f in sorted(FS_DIR.glob("data-*.tsv")):
        d = pd.read_csv(f, sep="\t"); d = d.rename(columns={d.columns[0]: "sid"})
        d["sid"] = d["sid"].astype(str); d = d.set_index("sid")
        d = d.loc[:, ~d.columns.duplicated()]
        d.columns = [f"{f.stem}::{c}" for c in d.columns]; frames.append(d)
    M = pd.concat(frames, axis=1)
    M = M.loc[:, ~M.columns.duplicated()].apply(pd.to_numeric, errors="coerce")
    vol_col = [c for c in M.columns if c.endswith("::" + VOL_COL)][0]
    parts = load_participants(cfg["paths"]["bids_root"])

    ids = [s for s in sorted(set(subs) & set(M.index))
           if s in parts.index and not pd.isna(parts.loc[s, TARGET])
           and not pd.isna(M.loc[s, vol_col]) and subs[s] == 290]
    y = np.array([float(parts.loc[s, TARGET]) for s in ids])
    sex = np.array([1 if str(parts.loc[s, "sex"]).lower().startswith("m") else 0
                    for s in ids])
    strat = pd.qcut(y, 4, labels=False, duplicates="drop") * 2 + sex

    disc, hold = train_test_split(np.arange(len(ids)), test_size=FRAC,
                                  random_state=SEED, stratify=strat)
    disc_ids = [ids[i] for i in sorted(disc)]
    hold_ids = [ids[i] for i in sorted(hold)]

    if OUT.exists():
        man = json.loads(OUT.read_text())
        ok = man["holdout_sha256"] == digest(man["holdout"])
        same = set(man["holdout"]) == set(hold_ids)
        print(f"manifest already exists: {OUT}")
        print(f"  sealed   : {man['sealed_utc']}")
        print(f"  discovery: {len(man['discovery'])}   holdout: {len(man['holdout'])}")
        print(f"  checksum verifies      : {ok}")
        print(f"  reproduces from seed   : {same}")
        print("  NOT resealing. Delete the manifest deliberately if you truly "
              "intend to redraw.")
        return 0 if ok else 1

    man = {
        "sealed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED, "frac": FRAC, "target": TARGET,
        "n_total": len(ids), "n_discovery": len(disc_ids), "n_holdout": len(hold_ids),
        "stratified_by": "IST quartile x sex",
        "discovery": disc_ids, "holdout": hold_ids,
        "holdout_sha256": digest(hold_ids),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(man, indent=1))
    print(f"sealed {len(hold_ids)} of {len(ids)} subjects -> {OUT}")
    print(f"  discovery n = {len(disc_ids)}")
    print(f"  holdout   n = {len(hold_ids)}   sha256 {man['holdout_sha256'][:16]}...")
    for nm, idx in [("discovery", disc), ("holdout", hold)]:
        print(f"  {nm:9s} IST {y[idx].mean():7.2f} +/- {y[idx].std():5.2f}   "
              f"male {100*sex[idx].mean():.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
