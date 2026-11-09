"""
Geodesic estimators against a reference solver (Supplementary Table S2).

For each manifold (true synthetic metric; learned Phase 1 and Phase 2 metrics on a synthetic graph) and 200 random
pairs of data points, compare the training-time estimators with a reference solver:
  hybrid (P=64, 32-stencil)  = training-time estimator
  grid only (P=64)           = grid-graph estimator without local refinement
  straight line (T=21)       = straight-line estimator on the P=64 interpolated metric
  straight line, exact metric (T=41) = the same path with the exact decoder metric (interpolation error check)
  reference                  = discrete path-energy minimisation (K=64 segments, L-BFGS), exact metric,
                               best of two initialisations (fine-grid P=160 shortest path; straight
                               segment). It does not use any candidate estimator's value.
Writes results/estimator_validation.json
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aligning.metric import GridSpec, MetricField, pullback_metric
from aligning.geodesics import GridGeodesics, reference_geodesic, grid_path, straight_line_distances
from aligning.models import VAEConfig
from aligning.synthetic import (ManifoldSpec, true_metric_fn, true_geodesic_distances, generate, SyntheticConfig,
                                GraphConfig, PerturbConfig)
from aligning.train import train_phase1, GeoConfig, run_phase2
from experiments.common import ROOT, WORK, jsonable
from experiments.run_synthetic import VAE_CFG, P1_CFG, P2_CFG

torch.set_num_threads(2)


def validate(metric_fn, Z, n_pairs=200, seed=0, label=""):
    rng = np.random.default_rng(seed)
    n = Z.shape[0]
    pairs = set()
    while len(pairs) < n_pairs:
        i, j = rng.choice(n, 2, replace=False)
        pairs.add((min(i, j), max(i, j)))
    pairs = sorted(pairs)
    spec64 = GridSpec.from_points(Z, 64, 0.1)
    f64 = MetricField.from_function(lambda p: metric_fn(p).detach(), spec64)
    geo64 = GridGeodesics(spec64, Z, neighbourhood=32)
    with torch.no_grad():
        Dg = geo64.distances(f64).numpy()
        Dh = geo64.hybrid_distances(f64).numpy()
    spec_f = GridSpec.from_points(Z, 160, 0.1)
    f_f = MetricField.from_function(lambda p: metric_fn(p).detach(), spec_f)
    geo_f = GridGeodesics(spec_f, Z, neighbourhood=32)
    rows = []
    with torch.no_grad():
        Lt = straight_line_distances(f64, torch.tensor(Z[[i for i, _ in pairs]]), torch.tensor(Z[[j for _, j in pairs]]), T=21).numpy()
    for k, (i, j) in enumerate(pairs):
        p, q = Z[i], Z[j]
        with torch.no_grad():
            line_exact = straight_line_distances(_Exact(metric_fn), torch.tensor(p[None]), torch.tensor(q[None]), T=41).item()
        gp = grid_path(geo_f, f_f, i, j)
        cands = []
        for init in (gp, np.stack([p, q])):   # reference = best local optimum of the discrete energy (two inits)
            try:
                L, _ = reference_geodesic(lambda x: metric_fn(x), p, q, init, K=64, iters=200)
                cands.append(L)
            except Exception:
                pass
        ref = min(cands)   # independent of the candidate estimators
        rows.append(dict(i=int(i), j=int(j), ref=ref, hybrid=float(Dh[i, j]), grid=float(Dg[i, j]),
                         line=float(Lt[k]), line_exact_metric=float(line_exact)))
    ref = np.array([r["ref"] for r in rows])
    out = {"label": label, "n_pairs": len(rows)}
    for k in ["hybrid", "grid", "line", "line_exact_metric"]:
        est = np.array([r[k] for r in rows])
        rel = (est - ref) / ref
        out[k] = dict(median_abs_rel=float(np.median(np.abs(rel))), p95_abs_rel=float(np.quantile(np.abs(rel), 0.95)),
                      max_abs_rel=float(np.max(np.abs(rel))), mean_rel=float(rel.mean()),
                      pearson=float(np.corrcoef(est, ref)[0, 1]),
                      spearman=float(__import__("scipy.stats").stats.spearmanr(est, ref).correlation))
    out["rows"] = rows
    print(label, {k: (round(out[k]["median_abs_rel"], 4), round(out[k]["p95_abs_rel"], 4), round(out[k]["max_abs_rel"], 4)) for k in ["hybrid", "grid", "line", "line_exact_metric"]}, flush=True)
    return out


class _Exact:
    def __init__(self, fn):
        self.fn = fn
        self.G = torch.zeros(1, dtype=torch.float64)

    def interpolate(self, pts):
        return self.fn(pts).detach()


def main():
    results = []
    # (a) true metric of the main synthetic manifold (seed 1)
    seed = 101   
    rng = np.random.default_rng(10_000 + seed)
    man = ManifoldSpec.sample(rng, 20)
    Z = rng.normal(size=(500, 2))
    tm = true_metric_fn(man)
    results.append(validate(tm, Z, label="true metric (synthetic seed 101)"))
    # (b,c) learned metrics on condition R70 (seed 1)
    d_true = true_geodesic_distances(man, Z)
    data = generate(SyntheticConfig(graph=GraphConfig(bandwidth=0.1),
                                    perturb=PerturbConfig(region="scatter", m=70, intensity=1.0, kind="node_relabel")),
                    seed, d_true=d_true, manifold=man, Z=Z)
    vae, _ = train_phase1(data["X"], VAEConfig(input_dim=20, **VAE_CFG), P1_CFG, seed=seed)
    res = run_phase2(vae, data["X"], data["A"], GeoConfig(P=64), P2_CFG, seed=seed)
    Z1 = res["Z"]
    results.append(validate(lambda p: pullback_metric(vae.decoder, p, "expected"), Z1, label="Phase 1 learned metric (R70, seed 101)"))
    dec2 = res["decoder2"]
    results.append(validate(lambda p: pullback_metric(dec2, p, "expected"), Z1, label="Phase 2 learned metric (R70, seed 101)"))
    os.makedirs(os.path.join(WORK, "results"), exist_ok=True)
    with open(os.path.join(WORK, "results", "estimator_validation.json"), "w") as f:
        json.dump(jsonable(results), f)


if __name__ == "__main__":
    main()
