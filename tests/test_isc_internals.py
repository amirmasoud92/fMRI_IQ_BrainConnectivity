"""Tests for the leave-one-out inter-subject correlation bookkeeping and the fast Mantel permutation."""
import os, sys
sys.path.insert(0, os.path.abspath("scripts/models"))
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr, rankdata

import inter_subject_correlation as M

rng = np.random.default_rng(0)

# ---------------------------------------------------------------- loo_isc ---
N, T, R = 12, 60, 4
TS = rng.normal(size=(N, T, R)).astype(np.float32)
TS += rng.normal(size=(1, T, R)).astype(np.float32)      # shared stimulus signal
KEEP = rng.random((N, T)) > 0.15
tr = np.array([0, 1, 2, 3, 4, 5, 6, 7])                  # training subjects

got = M.loo_isc(TS, KEEP, tr, censor=True)

# naive reference, written straight from the intended semantics
Xz = M.zsc(TS, axis=1)
ref = np.empty((N, R))
for i in range(N):
    if i in tr:
        tmpl = Xz[[j for j in tr if j != i]].mean(0)
    else:
        tmpl = Xz[tr].mean(0)
    k = KEEP[i]
    for j in range(R):
        ref[i, j] = pearsonr(Xz[i][k, j], tmpl[k, j])[0]
print(f"loo_isc  max abs err vs naive reference : {np.abs(got - ref).max():.3e}")

got_nc = M.loo_isc(TS, KEEP, tr, censor=False)
ref_nc = np.empty((N, R))
for i in range(N):
    tmpl = (Xz[[j for j in tr if j != i]].mean(0) if i in tr else Xz[tr].mean(0))
    for j in range(R):
        ref_nc[i, j] = pearsonr(Xz[i][:, j], tmpl[:, j])[0]
print(f"loo_isc  uncensored max abs err          : {np.abs(got_nc - ref_nc).max():.3e}")

# a test subject must not contribute to its own template
te = 11
tmpl_te = Xz[tr].mean(0)
chk = np.array([pearsonr(Xz[te][KEEP[te], j], tmpl_te[KEEP[te], j])[0] for j in range(R)])
print(f"loo_isc  test subject excluded from tmpl : {np.abs(got[te] - chk).max():.3e}")

# --------------------------------------------------------------- loo_isfc ---
dev = torch.device("cpu")
gi = M.loo_isfc(TS, KEEP, tr, dev, censor=True)
i = 3
tmpl = Xz[[j for j in tr if j != i]].mean(0)
k = KEEP[i]
a = Xz[i][k] - Xz[i][k].mean(0); b = tmpl[k] - tmpl[k].mean(0)
Mref = (a.T @ b) / np.outer(np.linalg.norm(a, axis=0), np.linalg.norm(b, axis=0))
Mref = 0.5 * (Mref + Mref.T)
iu = np.triu_indices(R, 1)
ref_vec = np.concatenate([np.diag(Mref), Mref[iu]])
print(f"loo_isfc max abs err vs naive reference  : {np.abs(gi[i] - ref_vec).max():.3e}")

# ----------------------------------------------------------------- mantel ---
Nm = 40
A = rng.normal(size=(Nm, 8)); Dn = np.corrcoef(A)
yv = rng.normal(size=Nm); Db = -np.abs(yv[:, None] - yv[None, :])
iu = np.triu_indices(Nm, 1)

r0_ref = spearmanr(Dn[iu], Db[iu])[0]
r0_got = M.mantel(Dn, Db, n_perm=1)[0]
print(f"mantel   observed rho err vs scipy       : {abs(r0_got - r0_ref):.3e}")

# permuted statistics must match scipy exactly
ra = rankdata(Dn[iu]); rb = rankdata(Db[iu])
Rn = np.zeros((Nm, Nm)); Rn[iu] = ra; Rn = Rn + Rn.T
a_c = ra - ra.mean(); b_c = rb - rb.mean()
denom = np.sqrt((a_c ** 2).sum() * (b_c ** 2).sum())
errs = []
for _ in range(200):
    p = rng.permutation(Nm)
    v = Rn[np.ix_(p, p)][iu]
    fast = ((v - v.mean()) * b_c).sum() / denom
    slow = spearmanr(Dn[np.ix_(p, p)][iu], Db[iu])[0]
    errs.append(abs(fast - slow))
print(f"mantel   permuted rho max err (200 draws): {max(errs):.3e}")

# the permuted upper triangle must be a permutation of the original values
p = rng.permutation(Nm)
same = np.allclose(np.sort(Dn[np.ix_(p, p)][iu]), np.sort(Dn[iu]))
print(f"mantel   permutation preserves multiset  : {same}")

# the ISFC diagonal is by definition the ISC, so the two routines must agree
d_isfc = gi[:, :R]
print(f"isfc diagonal vs loo_isc                 : {np.abs(d_isfc - got).max():.3e}")
