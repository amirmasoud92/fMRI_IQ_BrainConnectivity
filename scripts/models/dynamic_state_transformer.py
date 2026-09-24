#!/usr/bin/env python
"""
Transformer over windowed connectivity states, against static tangent-space baselines.

Pipeline (per outer fold)
-------------------------
  timeseries (T, 215)
    -> PCA_ts   215 -> d_ts        [fit on train-fold timepoints]
    -> W proportional windows of L TRs
    -> shrunk covariance per window                    SPD(d_ts)
    -> tangent map at the train-fold Frechet mean      W x d_ts(d_ts+1)/2
    -> PCA_tan  -> d_in                                [fit on train fold]
    -> StateTransformer -> IST

StateTransformer
----------------
  Linear(d_in -> d) + learned positional embedding
  2-layer TransformerEncoder (pre-norm)
  K learnable state prototypes; soft assignment of each window -> occupancy
  Readout = [attention-pooled sequence || occupancy || dwell || transitions
             || switch-rate || group-sequence alignment]
  -> MLP -> IST

Baselines evaluated on the IDENTICAL folds
------------------------------------------
  static_tangent   tangent-space Ridge on the time-averaged covariance
                   (this is the +0.360 result the model must beat)
  dfc_mean_ridge   Ridge on the mean of the per-window tangent vectors
                   (isolates how much of any gain is dynamics vs just geometry)

Usage
-----
  python scripts/models/dynamic_state_transformer.py --repeats 5
  python scripts/models/dynamic_state_transformer.py --repeats 1 --quick   # smoke test
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
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

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

# Window / representation geometry
N_WIN    = 24     # windows per subject (proportionally spaced)
WIN_LEN  = 60     # TRs per window
D_TS     = 64     # timeseries PCA rank -> SPD(64) per window
D_IN     = 128    # tangent PCA rank fed to the transformer
SHRINK   = 0.15   # covariance shrinkage (windows are short -> rank deficient)


# ---------------------------------------------------------------------------
# SPD / tangent utilities (GPU, batched)
# ---------------------------------------------------------------------------

def _funcm(S: torch.Tensor, fn) -> torch.Tensor:
    """Apply fn to the eigenvalues of a batch of symmetric matrices."""
    w, V = torch.linalg.eigh(S)
    w = fn(w.clamp_min(1e-10))
    return V @ torch.diag_embed(w) @ V.transpose(-1, -2)


def frechet_mean(S: torch.Tensor, n_iter: int = 12, tol: float = 1e-7) -> torch.Tensor:
    """Affine-invariant Frechet mean of a batch of SPD matrices."""
    G = S.mean(0)
    for _ in range(n_iter):
        Gs  = _funcm(G, torch.sqrt)
        Gis = _funcm(G, lambda w: w.rsqrt())
        T = _funcm(Gis @ S @ Gis, torch.log).mean(0)
        if torch.linalg.norm(T) < tol:
            break
        G = Gs @ _funcm(T, torch.exp) @ Gs
        G = 0.5 * (G + G.transpose(-1, -2))
    return G


def tangent(S: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """Project SPD batch to the tangent space at G; return upper triangles."""
    Gis = _funcm(G, lambda w: w.rsqrt())
    L = _funcm(Gis @ S @ Gis, torch.log)
    n = G.shape[-1]
    iu = torch.triu_indices(n, n, offset=1, device=S.device)
    diag = torch.diagonal(L, dim1=-2, dim2=-1)
    off  = L[..., iu[0], iu[1]] * np.sqrt(2.0)
    return torch.cat([diag, off], dim=-1)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_timeseries(cfg):
    conn = Path(cfg["paths"]["connectivity_dir"])
    ctx, sub = {}, {}
    ctx, sub = load_series(conn, SERIES)

    parts = load_participants(cfg["paths"]["bids_root"])
    ids, ts, y, sex = [], [], [], []
    for s in sorted(set(ctx) & set(sub)):
        if ctx[s].shape[0] != sub[s].shape[0]:
            continue
        if s not in parts.index or pd.isna(parts.loc[s, TARGET]):
            continue
        ids.append(s)
        ts.append(np.hstack([ctx[s], sub[s]]).astype(np.float32))   # (T, 215)
        y.append(float(parts.loc[s, TARGET]))
        sex.append(1 if str(parts.loc[s, "sex"]).lower().startswith("m") else 0)
    return ids, ts, np.asarray(y, float), np.asarray(sex, int)


def window_starts(T, n_win=N_WIN, win_len=WIN_LEN):
    """Proportionally spaced window starts -> approximate stimulus alignment
    despite subject-specific scrubbing."""
    last = max(T - win_len, 0)
    return np.linspace(0, last, n_win).round().astype(int)


def subject_windows(x, n_win=N_WIN, win_len=WIN_LEN, shrink=SHRINK):
    """(T, d) timeseries -> (n_win, d, d) shrunk SPD covariances."""
    T, d = x.shape
    out = np.empty((n_win, d, d), np.float32)
    eye = np.eye(d, dtype=np.float32)
    for i, s0 in enumerate(window_starts(T, n_win, win_len)):
        seg = x[s0:s0 + win_len]
        seg = seg - seg.mean(0, keepdims=True)
        C = (seg.T @ seg) / max(len(seg) - 1, 1)
        out[i] = (1 - shrink) * C + shrink * (np.trace(C) / d) * eye
    return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class StateTransformer(nn.Module):
    """Temporal transformer over windowed connectivity + discrete state readout."""

    def __init__(self, d_in=D_IN, d=64, n_heads=4, n_layers=2, n_states=8,
                 n_win=N_WIN, dropout=0.3):
        super().__init__()
        self.n_states = n_states
        self.embed = nn.Linear(d_in, d)
        self.pos   = nn.Parameter(torch.zeros(1, n_win, d))
        nn.init.normal_(self.pos, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=2 * d,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d)

        # Learnable connectivity-state prototypes
        self.states = nn.Parameter(torch.randn(n_states, d) * 0.1)
        self.tau = nn.Parameter(torch.tensor(1.0))

        self.attn_pool = nn.Linear(d, 1)

        n_state_feats = n_states * 2 + n_states * n_states + 1 + 1
        self.head = nn.Sequential(
            nn.Linear(d + n_state_feats, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, 1))

    def occupancy(self, h):
        """Soft assignment of each window to a state prototype."""
        hn = F.normalize(h, dim=-1)
        sn = F.normalize(self.states, dim=-1)
        return F.softmax(hn @ sn.t() / self.tau.abs().clamp_min(0.05), dim=-1)

    def forward(self, x, group_occ=None):
        # x: (B, n_win, d_in)
        h = self.tf(self.embed(x) + self.pos)
        h = self.norm(h)

        occ = self.occupancy(h)                              # (B, W, K)
        mean_occ = occ.mean(1)                               # (B, K)
        # transition matrix between consecutive windows
        trans = torch.einsum("bwi,bwj->bij", occ[:, :-1], occ[:, 1:])
        trans = trans / trans.sum((1, 2), keepdim=True).clamp_min(1e-6)
        switch = 1.0 - torch.diagonal(trans, dim1=1, dim2=2).sum(-1, keepdim=True)
        dwell = torch.diagonal(trans, dim1=1, dim2=2)        # (B, K)

        # Alignment of this subject's state sequence to the group's
        if group_occ is None:
            align = torch.zeros(x.size(0), 1, device=x.device)
        else:
            a = occ.reshape(occ.size(0), -1)
            g = group_occ.reshape(1, -1).expand_as(a)
            a = a - a.mean(1, keepdim=True); g = g - g.mean(1, keepdim=True)
            align = (F.normalize(a, dim=1) * F.normalize(g, dim=1)).sum(1, keepdim=True)

        w = torch.softmax(self.attn_pool(h), dim=1)
        pooled = (h * w).sum(1)                              # (B, d)

        feats = torch.cat([pooled, mean_occ, dwell,
                           trans.reshape(trans.size(0), -1), switch, align], dim=-1)
        return self.head(feats).squeeze(-1), occ


# ---------------------------------------------------------------------------
# Fold logic
# ---------------------------------------------------------------------------

def build_fold_features(ts_list, tr_idx, dev):
    """Fit PCA_ts / Frechet mean / PCA_tan on the TRAIN fold; transform all."""
    # 1) timeseries PCA on train-fold timepoints
    pca_ts = PCA(n_components=D_TS, random_state=SEED)
    pca_ts.fit(np.vstack([ts_list[i] for i in tr_idx]))
    proj = [pca_ts.transform(x).astype(np.float32) for x in ts_list]

    # 2) windowed shrunk covariances -> (N, W, d, d)
    cov = np.stack([subject_windows(x) for x in proj])
    covt = torch.from_numpy(cov).to(dev)

    # 3) Frechet mean over TRAIN windows only, then tangent-project everything
    tr = torch.as_tensor(tr_idx, device=dev)
    G = frechet_mean(covt[tr].reshape(-1, D_TS, D_TS))
    N, W = covt.shape[:2]
    tan = tangent(covt.reshape(-1, D_TS, D_TS), G).reshape(N, W, -1).cpu().numpy()

    # 4) tangent PCA on train fold (fit over train subjects' windows)
    pca_tan = PCA(n_components=D_IN, random_state=SEED)
    pca_tan.fit(tan[tr_idx].reshape(-1, tan.shape[-1]))
    X = pca_tan.transform(tan.reshape(-1, tan.shape[-1])).reshape(N, W, D_IN)

    sc = StandardScaler().fit(X[tr_idx].reshape(-1, D_IN))
    X = sc.transform(X.reshape(-1, D_IN)).reshape(N, W, D_IN).astype(np.float32)
    return X, tan


def train_model(Xtr, ytr, Xva, yva, dev, epochs, lr=3e-4, wd=1e-2, patience=30):
    torch.manual_seed(SEED)
    model = StateTransformer().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    xtr = torch.from_numpy(Xtr).to(dev)
    ttr = torch.from_numpy(((ytr - ytr.mean()) / ytr.std()).astype(np.float32)).to(dev)
    xva = torch.from_numpy(Xva).to(dev)
    gocc = None
    best, best_state, ctr = -np.inf, None, 0

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr), device=dev)
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            opt.zero_grad()
            pred, occ = model(xtr[b], gocc)
            loss = F.smooth_l1_loss(pred, ttr[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        model.eval()
        with torch.no_grad():
            gocc = model(xtr, None)[1].mean(0)          # group state sequence
            pv = model(xva, gocc)[0].cpu().numpy()
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
    with torch.no_grad():
        gocc = model(xtr, None)[1].mean(0)
    return model, gocc, best


def ridge_r(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    m = RidgeCV(alphas=ALPHAS, cv=5).fit(sc.transform(Xtr), ytr)
    return float(pearsonr(yte, m.predict(sc.transform(Xte)))[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-static", action="store_true",
                    help="skip the 23k-dim static tangent Ridge baseline (slow)")
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

    print("Loading cortex + subcortex timeseries ...", flush=True)
    ids, ts, y, sex = load_timeseries(cfg)
    print(f"  {len(ids)} subjects, {ts[0].shape[1]} regions, "
          f"{N_WIN} windows x {WIN_LEN} TRs")

    if args.quick:
        keep = np.arange(0, len(ids), 3)
        ids = [ids[i] for i in keep]; ts = [ts[i] for i in keep]
        y, sex = y[keep], sex[keep]
        print(f"  QUICK: {len(ids)} subjects")

    q = pd.qcut(y, 4, labels=False, duplicates="drop")
    strat = np.asarray(q) * 2 + sex
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats,
                                   random_state=SEED)

    # Static (time-averaged) covariance for the benchmark baseline
    stat_cov = torch.from_numpy(
        np.stack([subject_windows(x, 1, x.shape[0])[0] for x in ts])).to(dev)

    rows = []
    for k, (tr, te) in enumerate(rskf.split(np.zeros(len(y)), strat), 1):
        t0 = time.time()
        X, tan = build_fold_features(ts, tr, dev)

        # inner split for early stopping (never touches the outer test fold)
        rng = np.random.default_rng(SEED + k)
        perm = rng.permutation(len(tr))
        n_va = max(int(0.15 * len(tr)), 30)
        va_i, tr_i = tr[perm[:n_va]], tr[perm[n_va:]]

        model, gocc, val_r = train_model(
            X[tr_i], y[tr_i], X[va_i], y[va_i], dev, args.epochs)
        with torch.no_grad():
            pred = model(torch.from_numpy(X[te]).to(dev), gocc)[0].cpu().numpy()
        r_model = float(pearsonr(y[te], pred)[0])

        # Baseline A: Ridge on the mean tangent vector (geometry, no dynamics)
        r_dfcmean = ridge_r(tan[tr].mean(1), y[tr], tan[te].mean(1), y[te])

        # Baseline B: static tangent Ridge on the full 215-region covariance
        r_static = np.nan
        if not args.skip_static:
            G = frechet_mean(stat_cov[torch.as_tensor(tr, device=dev)])
            st = tangent(stat_cov, G).cpu().numpy()
            r_static = ridge_r(st[tr], y[tr], st[te], y[te])

        rows.append({"fold": k, "state_transformer": r_model,
                     "dfc_mean_ridge": r_dfcmean, "static_tangent": r_static,
                     "val_r": val_r, "secs": time.time() - t0})
        print(f"  fold {k:2d}/{args.repeats*5}  transformer={r_model:+.4f}  "
              f"dfc_mean={r_dfcmean:+.4f}  static={r_static:+.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "dynamic_state_transformer_folds.csv", index=False)

    print(f"\n{'='*70}\nRepeated {args.repeats}x5-fold CV  (n={len(ids)})\n{'='*70}")
    summ = []
    for c in ["state_transformer", "dfc_mean_ridge", "static_tangent"]:
        v = df[c].dropna().values
        if len(v) == 0:
            continue
        summ.append({"model": c, "r_mean": v.mean(), "r_sd": v.std(ddof=1),
                     "n_folds": len(v)})
        print(f"  {c:20s} r = {v.mean():+.4f} +/- {v.std(ddof=1):.4f}")

    a = df["state_transformer"].values
    for c in ["static_tangent", "dfc_mean_ridge"]:
        b = df[c].values
        m = ~np.isnan(a) & ~np.isnan(b)
        if m.sum() > 2:
            t, p = corrected_ttest_rel(a[m], b[m])
            print(f"\n  paired t-test vs {c}: delta = {(a[m]-b[m]).mean():+.4f}, "
                  f"t = {t:+.2f}, p = {p:.4g}")

    pd.DataFrame(summ).to_csv(OUT_DIR / "dynamic_state_transformer_summary.csv",
                              index=False)
    print(f"\nSaved -> {OUT_DIR}/dynamic_state_transformer_*.csv")


if __name__ == "__main__":
    main()
