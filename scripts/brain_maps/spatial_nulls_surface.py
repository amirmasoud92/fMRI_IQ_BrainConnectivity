#!/usr/bin/env python
"""
Surface-based spin test for network enrichment, and variogram-matched surrogates for map-to-map correlations.

Outputs
-------
  outputs/honest/schaefer200_fsaverage5_sphere.csv   parcel sphere centroids
  outputs/honest/spatial_null_surface_haufe.csv      Part 1 (with the old nulls alongside)
  outputs/piop/spatial_null_map_pairs.csv            Part 2, every map pair
  outputs/piop/spatial_null_map_blocks.csv           Part 2, block means + encoding-ISC
  outputs/piop/spatial_null_variogram_fit.csv        surrogate variogram validation

Usage
-----
  python scripts/brain_maps/spatial_nulls_surface.py --spins 10000 --surrogates 5000 --jobs 8
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from nilearn import datasets, image, surface

from src.data.series import roi_networks
from src.io_utils import write_results

warnings.filterwarnings("ignore")

SEED = 20260914
H = Path("outputs/honest")
P = Path("outputs/piop")
CONN = Path("outputs/connectivity")
DROP_SUB = {"Background", "Left Cerebral White Matter", "Right Cerebral White Matter",
            "Left Cerebral Cortex", "Right Cerebral Cortex",
            "Left Lateral Ventricle", "Right Lateral Ventricle"}


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def parcel_sphere_centroids():
    """Sphere centroids of the 200 Schaefer parcels on fsaverage5, in extraction order."""
    sch = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7)
    labels = [l.decode() if isinstance(l, bytes) else str(l) for l in sch.labels]
    labels = [l for l in labels if l != "Background"]
    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    meta_xyz = np.load(CONN / "atlas_meta.npz", allow_pickle=True)["coords"]
    rows, fallback = [], 0
    for hemi, tag in (("left", "LH"), ("right", "RH")):
        tex = surface.vol_to_surf(sch.maps, fs[f"pial_{hemi}"], inner_mesh=fs[f"white_{hemi}"],
                                  interpolation="nearest_most_frequent", n_samples=10)
        tex = np.rint(tex).astype(int)
        sph = surface.load_surf_mesh(fs[f"sphere_{hemi}"]).coordinates
        pial = surface.load_surf_mesh(fs[f"pial_{hemi}"]).coordinates
        radius = float(np.linalg.norm(sph, axis=1).mean())
        for j, lab in enumerate(labels):
            if f"_{tag}_" not in lab:
                continue
            verts = np.flatnonzero(tex == j + 1)
            if verts.size:
                c = sph[verts].mean(0)
                src = "surface projection"
            else:                              # parcel too small to catch a vertex
                v = int(np.argmin(np.linalg.norm(pial - meta_xyz[j], axis=1)))
                c, src = sph[v], "nearest vertex to volumetric centroid"
                fallback += 1
            c = c / np.linalg.norm(c) * radius
            rows.append({"roi": j, "label": lab, "hemi": tag, "n_vertices": int(verts.size),
                         "x": c[0], "y": c[1], "z": c[2], "source": src})
    df = pd.DataFrame(rows).sort_values("roi").reset_index(drop=True)
    assert len(df) == 200 and (df.roi.to_numpy() == np.arange(200)).all(), "parcel coverage"
    return df, fallback


def volumetric_centroids_215():
    """World-space centroids of the 200 cortical + 15 subcortical regions, extraction order."""
    cort = np.load(CONN / "atlas_meta.npz", allow_pickle=True)["coords"].astype(float)
    ho = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    names = [l.decode() if isinstance(l, bytes) else str(l) for l in ho.labels]
    keep = [i for i, l in enumerate(names) if l not in DROP_SUB and i > 0]
    img = image.load_img(ho.maps)
    data = np.asarray(img.dataobj)
    aff = img.affine
    sub = []
    for i in keep:
        ijk = np.argwhere(data == i)
        xyz = (aff @ np.column_stack([ijk, np.ones(len(ijk))]).T).T[:, :3]
        sub.append(xyz.mean(0))
    xyz = np.vstack([cort, np.array(sub)])
    assert xyz.shape == (215, 3)
    return xyz, [names[i] for i in keep]


# ---------------------------------------------------------------------------
# Part 1: surface spin for network enrichment
# ---------------------------------------------------------------------------
def network_means(W, code, n_net):
    Hm = np.zeros((len(code), n_net))
    Hm[np.arange(len(code)), code] = 1.0
    n = Hm.sum(0)
    C = np.outer(n, n) - np.diag(n)
    return (Hm.T @ W @ Hm) / np.maximum(C, 1.0)


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def part1(n_spins):
    from neuromaps.nulls.spins import gen_spinsamples

    t0 = time.time()
    sph, fallback = parcel_sphere_centroids()
    write_results(sph, H / "schaefer200_fsaverage5_sphere.csv", script=__file__, args=sys.argv[1:],
                  atlas="Schaefer2018 200 parcels 7 networks, fsaverage5", fallback_parcels=int(fallback))
    coords = sph[["x", "y", "z"]].to_numpy()
    hemiid = (sph.hemi == "RH").to_numpy().astype(int)
    spins = gen_spinsamples(coords, hemiid, n_rotate=n_spins, method="hungarian", seed=SEED)
    spins = np.asarray(spins)
    if spins.shape[0] != 200:
        spins = spins.T
    perm_ok = all(np.array_equal(np.sort(spins[:, k]), np.arange(200)) for k in range(spins.shape[1]))
    hemi_ok = bool((hemiid[spins] == hemiid[:, None]).all())
    # What a spin must preserve is NEIGHBOURHOOD structure, not location: any random
    # rotation moves a parcel ~90 degrees. Metric: correlation between the original
    # inter-parcel sphere distances and the distances between the parcels each one
    # is reassigned to. A spin keeps this high; a label shuffle drives it to ~0.
    Dsph = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    same_h = hemiid[:, None] == hemiid[None, :]
    iu = np.triu_indices(200, 1)
    keep = same_h[iu]
    def preservation(perm):
        return float(np.corrcoef(Dsph[iu][keep], Dsph[np.ix_(perm, perm)][iu][keep])[0, 1])
    pres_spin = float(np.mean([preservation(spins[:, k]) for k in range(min(500, spins.shape[1]))]))
    rng = np.random.default_rng(SEED)
    pres_shuf = []
    for _ in range(500):
        perm = np.arange(200)
        for h in (0, 1):
            idx = np.flatnonzero(hemiid == h)
            perm[idx] = rng.permutation(idx)
        pres_shuf.append(preservation(perm))
    pres_shuf = float(np.mean(pres_shuf))
    print(f"  sphere centroids: 200 parcels, {fallback} via nearest-vertex fallback; "
          f"spins {spins.shape[1]}: permutation {perm_ok}, hemisphere-preserving {hemi_ok}; "
          f"neighbourhood preservation (distance corr) spin {pres_spin:.3f} vs label shuffle "
          f"{pres_shuf:.3f} ({time.time()-t0:.0f}s)", flush=True)
    assert perm_ok and hemi_ok, "invalid spins"

    labels = roi_networks(CONN)
    edges = pd.read_csv(H / "psc/haufe_edges.csv")
    csv_labels_ok = all(labels[i] == ni and labels[j] == nj for i, j, ni, nj in
                        edges[["edge_i", "edge_j", "net_i", "net_j"]].itertuples(index=False))
    assert csv_labels_ok, "haufe_edges.csv network labels do not match the fixed roi_networks labels"
    uniq = sorted(set(labels))
    code = np.array([uniq.index(l) for l in labels])
    old = pd.read_csv(H / "spatial_null_haufe.csv")
    roi = pd.read_csv(H / "psc/haufe_networks.csv")
    out = []
    for target, col in (("raw", "haufe_raw"), ("deconfounded", "haufe_deconfounded")):
        W = np.zeros((215, 215))
        W[edges.edge_i, edges.edge_j] = np.abs(edges[col].to_numpy())
        W = W + W.T
        obs = network_means(W, code, len(uniq))
        null = np.empty((spins.shape[1], len(uniq), len(uniq)))
        for k in range(spins.shape[1]):
            c = code.copy()
            c[:200] = code[:200][spins[:, k]]
            null[k] = network_means(W, c, len(uniq))
        mu, sd = null.mean(0), null.std(0)
        n_null = spins.shape[1] + 1
        for a in range(len(uniq)):
            for b in range(a, len(uniq)):
                upper = (null[:, a, b] >= obs[a, b]).sum()
                both = (np.abs(null[:, a, b] - mu[a, b]) >= abs(obs[a, b] - mu[a, b])).sum()
                valid = uniq[a] != "Subcortex" and uniq[b] != "Subcortex"
                # The enrichment claim ("this pair carries MORE weight than chance") is
                # one-sided, exactly as the centroid spin it replaces; the two-sided p is
                # what the ROI-label permutation reported and is kept for that comparison.
                rec = {"target": target, "net_i": uniq[a], "net_j": uniq[b],
                       "mean_abs_weight": obs[a, b],
                       "z_surface_spin": (obs[a, b] - mu[a, b]) / max(sd[a, b], 1e-15),
                       "p_surface_spin_upper": (upper + 1) / n_null,
                       "p_surface_spin_two_sided": (both + 1) / n_null,
                       "spin_valid": valid}
                o = old[(old.target == target) & (((old.net_i == uniq[a]) & (old.net_j == uniq[b])) |
                                                 ((old.net_i == uniq[b]) & (old.net_j == uniq[a])))]
                if len(o):
                    rec["z_centroid_spin"] = float(o.z_spin.iloc[0])
                    rec["p_centroid_spin_upper"] = float(o.p_spin.iloc[0])      # spatial_nulls.py: null >= obs
                q = roi[(roi.target == target) & (((roi.net_i == uniq[a]) & (roi.net_j == uniq[b])) |
                                                 ((roi.net_i == uniq[b]) & (roi.net_j == uniq[a])))]
                if len(q):
                    rec["z_label_perm"] = float(q.z_vs_null.iloc[0])
                    rec["p_label_perm_two_sided"] = float(q.p_perm.iloc[0])     # haufe_maps.py: |null - mu|
                out.append(rec)
    df = pd.DataFrame(out)
    # BH-FDR over the 28 spin-valid cortical pairs, per target, on the primary (upper) test
    df["q_surface_spin_upper"] = np.nan
    for target in df.target.unique():
        m = (df.target == target) & df.spin_valid
        df.loc[m, "q_surface_spin_upper"] = bh_fdr(df.loc[m, "p_surface_spin_upper"].to_numpy())
    write_results(df, H / "spatial_null_surface_haufe.csv", script=__file__, args=sys.argv[1:],
                  inputs=[H / "psc/haufe_edges.csv", H / "spatial_null_haufe.csv", H / "psc/haufe_networks.csv",
                          CONN / "atlas_meta.npz"],
                  seed=SEED, n_spins=int(spins.shape[1]))
    return df, {"fallback_parcels": fallback, "preservation_spin": pres_spin,
                "preservation_shuffle": pres_shuf}


# ---------------------------------------------------------------------------
# Part 2: BrainSMASH surrogates for map-map correlations
# ---------------------------------------------------------------------------
def variogram(x, D, bins):
    iu = np.triu_indices(len(x), 1)
    d, g = D[iu], 0.5 * (x[:, None] - x[None, :])[iu] ** 2
    idx = np.digitize(d, bins) - 1
    return np.array([g[idx == b].mean() if np.any(idx == b) else np.nan for b in range(len(bins) - 1)])


def part2(n_surr, n_jobs):
    from brainsmash.mapgen.base import Base

    maps = pd.read_csv(P / "amplitude_forward_maps.csv", index_col=0)
    locked = pd.read_csv(P / "amplitude_map_similarity.csv", index_col=0).to_numpy()
    recomputed = np.corrcoef(maps.to_numpy().T)
    gate = float(np.abs(recomputed - locked).max())
    print(f"  map gate: similarity from saved maps vs locked matrix max |diff| {gate:.3g}", flush=True)
    assert gate < 1e-6, "forward maps do not reproduce the locked similarity matrix"

    xyz, sub_names = volumetric_centroids_215()
    D = np.sqrt(((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(-1))
    cells = list(maps.columns)
    X = maps.to_numpy().T                                 # (11, 215)

    t0 = time.time()
    surr = {}
    fit_rows = []
    bins = np.quantile(D[np.triu_indices(215, 1)], np.linspace(0, 0.25, 11))
    for i, c in enumerate(cells):
        base = Base(x=X[i], D=D, seed=SEED + i, resample=True, n_jobs=n_jobs)
        S = base(n=n_surr)
        surr[c] = np.asarray(S)
        emp = variogram(X[i], D, bins)
        sv = np.nanmean([variogram(S[k], D, bins) for k in range(100)], axis=0)
        ok = np.isfinite(emp) & np.isfinite(sv)
        fit_rows.append({"map": c, "variogram_corr": float(np.corrcoef(emp[ok], sv[ok])[0, 1]),
                         "variogram_rel_rmse": float(np.sqrt(np.mean((sv[ok] - emp[ok]) ** 2)) / np.mean(emp[ok]))})
    fit = pd.DataFrame(fit_rows)
    meta = dict(script=__file__, args=sys.argv[1:], seed=SEED, n_surrogates=int(n_surr),
                inputs=[P / "amplitude_forward_maps.csv", P / "amplitude_map_similarity.csv",
                        CONN / "atlas_meta.npz", P / "encoding_accuracy.npz", H / "isc_regions.csv"])
    write_results(fit, P / "spatial_null_variogram_fit.csv", **meta)
    print(f"  surrogates: {len(cells)} maps x {n_surr} in {time.time()-t0:.0f}s; variogram fit "
          f"corr median {fit.variogram_corr.median():.3f}, rel RMSE median {fit.variogram_rel_rmse.median():.3f}",
          flush=True)

    def zc(A):
        A = A - A.mean(-1, keepdims=True)
        return A / np.linalg.norm(A, axis=-1, keepdims=True)

    Z = {c: zc(surr[c]) for c in cells}
    Xz = zc(X)
    pairs = []
    for i, j in combinations(range(len(cells)), 2):
        r_obs = float(Xz[i] @ Xz[j])
        null_i = Z[cells[i]] @ Xz[j]                     # surrogate i vs real j
        null_j = Z[cells[j]] @ Xz[i]
        p_i = (np.sum(np.abs(null_i) >= abs(r_obs)) + 1) / (n_surr + 1)
        p_j = (np.sum(np.abs(null_j) >= abs(r_obs)) + 1) / (n_surr + 1)
        # secondary, directional: maps AGREE (positive correlation) beyond chance
        u_i = (np.sum(null_i >= r_obs) + 1) / (n_surr + 1)
        u_j = (np.sum(null_j >= r_obs) + 1) / (n_surr + 1)
        pairs.append({"map_a": cells[i], "map_b": cells[j], "r": r_obs, "p_surrogate_a": p_i,
                      "p_surrogate_b": p_j, "p_conservative": max(p_i, p_j),
                      "p_conservative_upper": max(u_i, u_j),
                      "null_sd": float(np.std(np.concatenate([null_i, null_j]))),
                      "p_naive_parametric": float(2 * __import__("scipy.stats", fromlist=["t"]).t.sf(
                          abs(r_obs) * np.sqrt(213 / max(1 - r_obs ** 2, 1e-12)), 213))})
    pairs = pd.DataFrame(pairs)
    write_results(pairs, P / "spatial_null_map_pairs.csv", **meta)

    ds = {c: c.split("/")[0] for c in cells}
    def block(name, sel_a, sel_b, same):
        idx = [(i, j) for i, j in combinations(range(len(cells)), 2)
               if ((ds[cells[i]] in sel_a and ds[cells[j]] in sel_b) or
                   (ds[cells[j]] in sel_a and ds[cells[i]] in sel_b))]
        obs = float(np.mean([Xz[i] @ Xz[j] for i, j in idx]))
        # null: replace every map on the "a" side by an independent surrogate draw
        a_side = sorted({i if ds[cells[i]] in sel_a else j for i, j in idx})
        null = np.zeros(n_surr)
        for i, j in idx:
            ia, ib = (i, j) if i in a_side else (j, i)
            null += Z[cells[ia]] @ Xz[ib]
        null /= len(idx)
        return {"block": name, "pairs": len(idx), "mean_r": obs, "null_mean": float(null.mean()),
                "null_sd": float(null.std()), "z": (obs - null.mean()) / null.std(),
                "p_surrogate": (np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean())) + 1) / (n_surr + 1),
                "p_surrogate_upper": (np.sum(null >= obs) + 1) / (n_surr + 1),
                "note": "within-dataset maps share subjects: the spatial null does not address that dependence"
                        if same else ""}

    blocks = [block("ID1000 vs PIOP1", {"ID1000"}, {"PIOP1"}, False),
              block("ID1000 vs PIOP2", {"ID1000"}, {"PIOP2"}, False),
              block("PIOP1 vs PIOP2", {"PIOP1"}, {"PIOP2"}, False)]
    for d_ in ("PIOP1", "PIOP2"):
        idx = [(i, j) for i, j in combinations(range(len(cells)), 2) if ds[cells[i]] == d_ and ds[cells[j]] == d_]
        obs = float(np.mean([Xz[i] @ Xz[j] for i, j in idx]))
        null = np.mean([Z[cells[i]] @ Xz[j] for i, j in idx], axis=0)
        blocks.append({"block": f"within {d_}", "pairs": len(idx), "mean_r": obs,
                       "null_mean": float(null.mean()), "null_sd": float(null.std()),
                       "z": (obs - null.mean()) / null.std(),
                       "p_surrogate": (np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean())) + 1) / (n_surr + 1),
                       "p_surrogate_upper": (np.sum(null >= obs) + 1) / (n_surr + 1),
                       "note": "within-dataset maps share subjects: the spatial null does not address that dependence"})

    # encoding accuracy map vs ISC map. The registered claim (encoding_accuracy.py, P3) is
    # the SPEARMAN correlation between the ISC map and the subject-mean LATE-layer
    # encoding accuracy: rho = +0.396. Tested exactly as registered: surrogates of the
    # encoding map, rank-transformed, correlated with the ranked ISC map.
    from scipy.stats import rankdata, spearmanr
    enc = np.load(P / "encoding_accuracy.npz", allow_pickle=True)
    enc_map = enc["acc_late"].mean(0)
    isc = pd.read_csv(H / "isc_regions.csv").sort_values("roi")["mean_isc"].to_numpy()
    rho = float(spearmanr(isc, enc_map)[0])
    base = Base(x=enc_map, D=D, seed=SEED + 99, resample=True, n_jobs=n_jobs)
    Se = np.asarray(base(n=n_surr))
    Se_rank = zc(np.apply_along_axis(rankdata, 1, Se))
    null = Se_rank @ zc(rankdata(isc)[None, :])[0]
    blocks.append({"block": "encoding (late layer) map vs ISC map, Spearman", "pairs": 1, "mean_r": rho,
                   "null_mean": float(null.mean()), "null_sd": float(null.std()),
                   "z": (rho - null.mean()) / null.std(),
                   "p_surrogate": (np.sum(np.abs(null) >= abs(rho)) + 1) / (n_surr + 1),
                   "p_surrogate_upper": (np.sum(null >= rho) + 1) / (n_surr + 1),
                   "note": "registered statistic: Spearman(ISC map, mean late-layer accuracy) = +0.396"})
    B = pd.DataFrame(blocks)
    write_results(B, P / "spatial_null_map_blocks.csv", **meta)
    return pairs, B, fit, gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spins", type=int, default=10000)
    ap.add_argument("--surrogates", type=int, default=5000,
                    help="5,000 gives a p floor of 2e-4; each surrogate costs ~0.09 s single-threaded")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()
    print("PART 1: surface spin, Haufe network enrichment", flush=True)
    haufe, diag = part1(args.spins)
    pd.set_option("display.width", 220)
    v = haufe[haufe.spin_valid].copy()
    for target in ("raw", "deconfounded"):
        g = v[v.target == target].sort_values("z_surface_spin", ascending=False)
        n = len(g)
        print(f"\n  [{target}] cortical pairs with p < .05, like for like:"
              f"\n    enrichment (upper): surface spin {int((g.p_surface_spin_upper < 0.05).sum())}/{n} "
              f"(BH q < .05: {int((g.q_surface_spin_upper < 0.05).sum())}), "
              f"centroid spin {int((g.p_centroid_spin_upper < 0.05).sum())}/{n}"
              f"\n    two-sided:          surface spin {int((g.p_surface_spin_two_sided < 0.05).sum())}/{n}, "
              f"label permutation {int((g.p_label_perm_two_sided < 0.05).sum())}/{n}")
        print(g.head(8)[["net_i", "net_j", "mean_abs_weight", "z_surface_spin", "p_surface_spin_upper",
                         "q_surface_spin_upper", "p_centroid_spin_upper", "p_surface_spin_two_sided",
                         "p_label_perm_two_sided"]].round(4).to_string(index=False))
    print("\nPART 2: variogram-matched surrogates, map-map correlations", flush=True)
    pairs, B, fit, gate = part2(args.surrogates, args.jobs)
    print(B.round(4).to_string(index=False))
    sig = pairs[pairs.p_conservative < 0.05]
    print(f"\n  map pairs significant: naive parametric {int((pairs.p_naive_parametric < 0.05).sum())}/{len(pairs)}, "
          f"surrogate (conservative) {len(sig)}/{len(pairs)}")
    print(f"\nSaved -> {H}/spatial_null_surface_haufe.csv, schaefer200_fsaverage5_sphere.csv; "
          f"{P}/spatial_null_map_pairs.csv, spatial_null_map_blocks.csv, spatial_null_variogram_fit.csv "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
