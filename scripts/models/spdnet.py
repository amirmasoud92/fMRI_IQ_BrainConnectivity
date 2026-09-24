#!/usr/bin/env python
"""
SPD manifold network on the full-run covariance, against a fixed tangent-space baseline.

Architecture
------------
  C in SPD(215)                     shrunk full-run covariance (cortex+subcortex)
    -> BiMap   W1 C W1^T             215 -> 64, W1 semi-orthogonal (Stiefel)
    -> ReEig   max(lambda, eps)      manifold nonlinearity
    -> RiemBN                        log-Euclidean batch norm (running mean at eval)
    -> BiMap   W2 . W2^T             64 -> 32
    -> LogEig  U log(Lambda) U^T     tangent map -> Euclidean
    -> vec (diag || sqrt(2)*offdiag) -> Linear -> IST

~16 K parameters (about 23 per training subject).  For reference the original
BrainVAEGAT used 17.5 M (~20 000 per subject) and lost to Ridge.

Eigen-decompositions only ever run on 64x64 and 32x32 matrices, never on
215x215, so the backward pass through eigh stays cheap and numerically stable.
Symmetrisation plus diagonal jitter guard the 1/(lambda_i - lambda_j) term in
the eigh gradient.

Baselines on IDENTICAL folds
----------------------------
  static_tangent   tangent-space Ridge on the same covariances (the +0.369
                   benchmark this model must beat)
  spdnet_ridge     RidgeCV on the trained network's LogEig features, to test
                   whether the learned projection or the learned head is doing
                   the work

Usage
-----
  python scripts/models/spdnet.py --repeats 5
  python scripts/models/spdnet.py --repeats 1 --quick     # smoke test
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.nn.utils import parametrizations

from src.data.preprocessing import load_participants
from src.data.series import load_series
from src.stats.cv_inference import corrected_ttest_rel

warnings.filterwarnings("ignore")

CFG_PATH = "config/pipeline.yaml"
TARGET   = "IST_intelligence_total"
ALPHAS   = np.logspace(-2, 6, 25)
SEED     = 42
OUT_DIR  = Path("outputs/honest")
SERIES   = "legacy"          # set from --series in main()

SHRINK = 0.15          # matches the static_tangent baseline in the dFC run
D1, D2 = 64, 32        # BiMap output dims


# ---------------------------------------------------------------------------
# Symmetric matrix function with a stable backward
# ---------------------------------------------------------------------------

def _eigh64(X: torch.Tensor):
    """float64 eigh with a CPU fallback.

    The cuSOLVER float32 path fails to converge on these matrices ("too many
    repeated eigenvalues") even though they are well conditioned (median
    condition number ~2e2), so everything runs in double precision.
    """
    try:
        return torch.linalg.eigh(X)
    except Exception:
        w, V = torch.linalg.eigh(X.cpu())
        return w.to(X.device), V.to(X.device)


class _SymFuncm(torch.autograd.Function):
    """Spectral matrix function with a degeneracy-safe backward."""

    @staticmethod
    def forward(ctx, X, f, fp, tol):
        w, V = _eigh64(X)
        fw = f(w)
        ctx.save_for_backward(w, V, fw)
        ctx.fp, ctx.tol = fp, tol
        return V @ torch.diag_embed(fw) @ V.transpose(-1, -2)

    @staticmethod
    def backward(ctx, G):
        w, V, fw = ctx.saved_tensors
        G = 0.5 * (G + G.transpose(-1, -2))
        dw = w.unsqueeze(-1) - w.unsqueeze(-2)                 # (B, n, n)
        df = fw.unsqueeze(-1) - fw.unsqueeze(-2)
        near = dw.abs() < ctx.tol
        safe = torch.where(near, torch.ones_like(dw), dw)
        deriv = ctx.fp(w)
        loewner = torch.where(
            near,
            0.5 * (deriv.unsqueeze(-1) + deriv.unsqueeze(-2)),  # limit as li->lj
            df / safe,
        )
        gX = V @ (loewner * (V.transpose(-1, -2) @ G @ V)) @ V.transpose(-1, -2)
        return 0.5 * (gX + gX.transpose(-1, -2)), None, None, None


def sym_funcm(X: torch.Tensor, fn, jitter: float = 1e-5, fp=None,
              tol: float = 1e-7) -> torch.Tensor:
    """Apply fn to the eigenvalues of a batch of symmetric matrices."""
    in_dtype = X.dtype
    X = 0.5 * (X + X.transpose(-1, -2))
    Xd = X.to(torch.float64)
    if jitter:
        Xd = Xd + jitter * torch.eye(Xd.shape[-1], device=Xd.device, dtype=Xd.dtype)
    if fp is None:                       # no analytic derivative -> no grad path
        w, V = _eigh64(Xd)
        out = V @ torch.diag_embed(fn(w)) @ V.transpose(-1, -2)
    else:
        out = _SymFuncm.apply(Xd, fn, fp, tol)
    return out.to(in_dtype)


class BiMap(nn.Module):
    """X -> W X W^T with W semi-orthogonal (d_out x d_in), keeping the output SPD."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        lin = nn.Linear(d_in, d_out, bias=False)
        self.lin = parametrizations.orthogonal(lin)

    def forward(self, X):
        W = self.lin.weight                      # (d_out, d_in), W W^T = I
        return W @ X @ W.t()


class ReEig(nn.Module):
    """Eigenvalue rectification -- the manifold analogue of ReLU."""

    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, X):
        return sym_funcm(X, lambda w: w.clamp_min(self.eps),
                         fp=lambda w: (w > self.eps).to(w.dtype))


class RiemBN(nn.Module):
    """Log-Euclidean batch normalisation on SPD matrices.

    Maps to the tangent space at the identity, centres on the batch mean
    (running mean at eval time), rescales, and maps back.  The log-Euclidean
    mean is used rather than the affine-invariant Frechet mean: it is a close
    approximation here and far cheaper and more stable to differentiate.
    """

    def __init__(self, d: int, momentum: float = 0.1):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(d, d))
        self.register_buffer("initialised", torch.zeros(1))
        self.log_scale = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        L = sym_funcm(X, lambda w: w.clamp_min(1e-6).log(),
                      fp=lambda w: 1.0 / w.clamp_min(1e-6))
        if self.training:
            m = L.mean(0)
            with torch.no_grad():
                if self.initialised.item() == 0:
                    self.running_mean.copy_(m.detach())
                    self.initialised.fill_(1)
                else:
                    self.running_mean.mul_(1 - self.momentum).add_(
                        self.momentum * m.detach())
        else:
            m = self.running_mean
        # Bound the scale and the tangent eigenvalues: exp() on an unbounded
        # symmetric matrix overflows to inf and poisons the whole batch.
        L = (L - m) * self.log_scale.clamp(-2.0, 2.0).exp()
        return sym_funcm(L, lambda w: w.clamp(-12.0, 12.0).exp(), jitter=0.0,
                         fp=lambda w: w.clamp(-12.0, 12.0).exp()
                                      * ((w > -12.0) & (w < 12.0)).to(w.dtype))


class LogEig(nn.Module):
    """Matrix logarithm followed by upper-triangle vectorisation."""

    def forward(self, X):
        L = sym_funcm(X, lambda w: w.clamp_min(1e-6).log(),
                      fp=lambda w: 1.0 / w.clamp_min(1e-6))
        n = L.shape[-1]
        iu = torch.triu_indices(n, n, offset=1, device=L.device)
        diag = torch.diagonal(L, dim1=-2, dim2=-1)
        off = L[..., iu[0], iu[1]] * np.sqrt(2.0)
        return torch.cat([diag, off], dim=-1)


class SPDNet(nn.Module):
    def __init__(self, d_in=215, d1=D1, d2=D2, dropout=0.3):
        super().__init__()
        self.bimap1 = BiMap(d_in, d1)
        self.reeig  = ReEig()
        self.bn     = RiemBN(d1)
        self.bimap2 = BiMap(d1, d2)
        self.logeig = LogEig()
        self.drop   = nn.Dropout(dropout)
        self.head   = nn.Linear(d2 + d2 * (d2 - 1) // 2, 1)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=0.01)

    def features(self, C):
        h = self.bimap1(C)
        h = self.reeig(h)
        h = self.bn(h)
        h = self.bimap2(h)
        return self.logeig(h)

    def forward(self, C):
        f = self.features(C)
        return self.head(self.drop(f)).squeeze(-1), f


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def shrunk_cov(x, shrink=SHRINK):
    x = x - x.mean(0, keepdims=True)
    C = (x.T @ x) / max(len(x) - 1, 1)
    d = C.shape[0]
    return ((1 - shrink) * C + shrink * (np.trace(C) / d) * np.eye(d, dtype=np.float32))


def load_data(cfg):
    conn = Path(cfg["paths"]["connectivity_dir"])
    ctx, sub = {}, {}
    ctx, sub = load_series(conn, SERIES)
    parts = load_participants(cfg["paths"]["bids_root"])

    ids, cov, y, sex = [], [], [], []
    for s in sorted(set(ctx) & set(sub)):
        if ctx[s].shape[0] != sub[s].shape[0]:
            continue
        if s not in parts.index or pd.isna(parts.loc[s, TARGET]):
            continue
        ids.append(s)
        cov.append(shrunk_cov(np.hstack([ctx[s], sub[s]]).astype(np.float32)))
        y.append(float(parts.loc[s, TARGET]))
        sex.append(1 if str(parts.loc[s, "sex"]).lower().startswith("m") else 0)
    return ids, np.stack(cov), np.asarray(y, float), np.asarray(sex, int)


# ---------------------------------------------------------------------------
# Tangent baseline (GPU)
# ---------------------------------------------------------------------------

def frechet_mean(S, n_iter=12, tol=1e-7):
    G = S.mean(0)
    for _ in range(n_iter):
        Gs  = sym_funcm(G.unsqueeze(0), torch.sqrt, 0.0)[0]
        Gis = sym_funcm(G.unsqueeze(0), lambda w: w.rsqrt(), 0.0)[0]
        T = sym_funcm(Gis @ S @ Gis, torch.log, 0.0).mean(0)
        if torch.linalg.norm(T) < tol:
            break
        G = Gs @ sym_funcm(T.unsqueeze(0), torch.exp, 0.0)[0] @ Gs
        G = 0.5 * (G + G.t())
    return G


def tangent_vec(S, G):
    Gis = sym_funcm(G.unsqueeze(0), lambda w: w.rsqrt(), 0.0)[0]
    L = sym_funcm(Gis @ S @ Gis, torch.log, 0.0)
    n = G.shape[-1]
    iu = torch.triu_indices(n, n, offset=1, device=S.device)
    return torch.cat([torch.diagonal(L, dim1=-2, dim2=-1),
                      L[..., iu[0], iu[1]] * np.sqrt(2.0)], dim=-1)


def ridge_r(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    m = RidgeCV(alphas=ALPHAS, cv=5).fit(sc.transform(Xtr), ytr)
    return float(pearsonr(yte, m.predict(sc.transform(Xte)))[0])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(Ctr, ytr, Cva, yva, dev, epochs, lr=1e-3, wd=1e-3, patience=40):
    torch.manual_seed(SEED)
    model = SPDNet(d_in=Ctr.shape[-1]).to(dev)
    # Weight decay must NOT touch the Stiefel-constrained BiMap parameters or the
    # RiemBN scale.  AdamW decays the *unconstrained base tensor* behind the
    # orthogonal parametrisation, which drives W to zero -- the whole network
    # then emits an identical constant for every subject.
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if ("bimap" in n or "log_scale" in n) else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd},
         {"params": no_decay, "weight_decay": 0.0}], lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    mu, sd = ytr.mean(), ytr.std()
    t = torch.from_numpy(((ytr - mu) / sd).astype(np.float32)).to(dev)
    best, best_state, ctr = -np.inf, None, 0

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Ctr), device=dev)
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            if len(b) < 8:                       # RiemBN needs a usable batch
                continue
            opt.zero_grad()
            pred, _ = model(Ctr[b])
            loss = F.smooth_l1_loss(pred, t[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        model.eval()
        with torch.no_grad():
            pv = model(Cva)[0].cpu().numpy()
        r = pearsonr(yva, pv)[0] if np.std(pv) > 1e-8 else -1.0
        if r > best:
            best, ctr = r, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            ctr += 1
            if ctr >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-static", action="store_true")
    ap.add_argument("--series", choices=["legacy", "psc"], default="legacy",
                    help="region timeseries source (see src/data/series.py)")
    args = ap.parse_args()
    global OUT_DIR, SERIES
    SERIES = args.series
    if SERIES != "legacy":
        OUT_DIR = OUT_DIR / SERIES
    print(f"  series = {SERIES}   -> {OUT_DIR}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(CFG_PATH))

    print("Loading cortex + subcortex, building shrunk covariances ...", flush=True)
    ids, cov, y, sex = load_data(cfg)
    print(f"  {len(ids)} subjects, SPD({cov.shape[-1]})")

    if args.quick:
        k = np.arange(0, len(ids), 3)
        ids = [ids[i] for i in k]; cov, y, sex = cov[k], y[k], sex[k]
        print(f"  QUICK: {len(ids)} subjects")

    n_par = sum(p.numel() for p in SPDNet(d_in=cov.shape[-1]).parameters())
    print(f"  SPDNet parameters: {n_par:,}")

    C = torch.from_numpy(cov).to(dev)
    q = pd.qcut(y, 4, labels=False, duplicates="drop")
    strat = np.asarray(q) * 2 + sex
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats,
                                   random_state=SEED)

    rows = []
    for k, (tr, te) in enumerate(rskf.split(np.zeros(len(y)), strat), 1):
        t0 = time.time()
        rng = np.random.default_rng(SEED + k)
        perm = rng.permutation(len(tr))
        n_va = max(int(0.15 * len(tr)), 30)
        va_i, tr_i = tr[perm[:n_va]], tr[perm[n_va:]]

        model, val_r = train(C[tr_i], y[tr_i], C[va_i], y[va_i], dev, args.epochs)

        with torch.no_grad():
            pred = model(C[te])[0].cpu().numpy()
            f_tr = model.features(C[tr]).cpu().numpy()
            f_te = model.features(C[te]).cpu().numpy()
        r_net = float(pearsonr(y[te], pred)[0])
        r_net_ridge = ridge_r(f_tr, y[tr], f_te, y[te])

        r_static = np.nan
        if not args.skip_static:
            G = frechet_mean(C[torch.as_tensor(tr, device=dev)])
            T = tangent_vec(C, G).cpu().numpy()
            r_static = ridge_r(T[tr], y[tr], T[te], y[te])

        rows.append({"fold": k, "spdnet": r_net, "spdnet_ridge": r_net_ridge,
                     "static_tangent": r_static, "val_r": val_r,
                     "secs": time.time() - t0})
        print(f"  fold {k:2d}/{args.repeats*5}  spdnet={r_net:+.4f}  "
              f"spdnet_ridge={r_net_ridge:+.4f}  static={r_static:+.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "spdnet_folds.csv", index=False)

    print(f"\n{'='*70}\nRepeated {args.repeats}x5-fold CV  (n={len(ids)})\n{'='*70}")
    summ = []
    for c in ["spdnet", "spdnet_ridge", "static_tangent"]:
        v = df[c].dropna().values
        if len(v) == 0:
            continue
        summ.append({"model": c, "r_mean": v.mean(), "r_sd": v.std(ddof=1),
                     "n_folds": len(v)})
        print(f"  {c:16s} r = {v.mean():+.4f} +/- {v.std(ddof=1):.4f}")

    base = df["static_tangent"].values
    for c in ["spdnet", "spdnet_ridge"]:
        a = df[c].values
        m = ~np.isnan(a) & ~np.isnan(base)
        if m.sum() > 2:
            t, p = corrected_ttest_rel(a[m], base[m])
            wins = int((a[m] > base[m]).sum())
            print(f"\n  {c} vs static_tangent: delta = {(a[m]-base[m]).mean():+.4f}, "
                  f"t = {t:+.2f}, p = {p:.4g}, wins {wins}/{m.sum()} folds")

    pd.DataFrame(summ).to_csv(OUT_DIR / "spdnet_summary.csv", index=False)
    print(f"\nSaved -> {OUT_DIR}/spdnet_*.csv")


if __name__ == "__main__":
    main()
