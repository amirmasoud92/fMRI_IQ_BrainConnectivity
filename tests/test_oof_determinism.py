"""Test that cached out-of-fold predictions equal recomputed ones."""
import os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "models"))
os.chdir(ROOT)

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import RepeatedStratifiedKFold

import coskewness as C
from src.data.preprocessing import load_participants
from src.data.series import load_series

cfg = yaml.safe_load(open(C.CFG_PATH))
conn = cfg["paths"]["connectivity_dir"]
ctx, sub = load_series(conn, "psc")
parts = load_participants(cfg["paths"]["bids_root"])
ids = [s for s in sorted(ctx) if s in parts.index and not pd.isna(parts.loc[s, C.TARGET])]
y = np.array([float(parts.loc[s, C.TARGET]) for s in ids])
sex = np.array([1 if str(parts.loc[s, "sex"]).lower().startswith("m") else 0 for s in ids])

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
S = []
for s in ids:
    x = torch.from_numpy(np.hstack([ctx[s], sub[s]])).to(dev).double()
    xc = x - x.mean(0, keepdim=True)
    S.append((xc.t() @ xc) / (len(xc) - 1))
X = C.upper(C.sym_funcm(C.shrink_batch(torch.stack(S)), torch.log)).cpu().numpy()
print(f"{len(ids)} subjects, {X.shape[1]} features")

strat = np.asarray(pd.qcut(y, 4, labels=False, duplicates="drop")) * 2 + sex
tr, te = next(RepeatedStratifiedKFold(n_splits=5, n_repeats=5,
                                      random_state=C.SEED).split(np.zeros(len(y)), strat))

t0 = time.time(); a = C.oof(X, y, tr); t1 = time.time()
b = C.oof(X, y, tr); t2 = time.time()
print(f"oof call 1: {t1-t0:.0f}s   call 2: {t2-t1:.0f}s")
print(f"bitwise identical      : {np.array_equal(a, b)}")
print(f"max |difference|       : {np.abs(a - b).max():.3e}")
print(f"identical bytes (hash) : {hash(a.tobytes()) == hash(b.tobytes())}")
