"""
Tax experiment, graphs and twists (Fig. 2): a GLOBAL smooth warp of the attribute geometry, and its
homophily-matched SCATTERED twin.

For one seed (manifold, positions and attributes as in run_twist.py) and one strength kappa:
  warp      graph generated from the geodesics of G_graph(u) = mu G + (lambda - mu) m m^T, with
            log lambda = kappa (f_a + f_s), log mu = kappa (f_a - f_s), axis angle theta(u):
            f_a, f_s, theta = smooth random Fourier fields over the whole domain (unit sd over the nodes).
  twin      the aligned graph (kappa = 0, from the attribute geodesics), with a fraction of its links moved to
            uniformly random non-adjacent pairs, the smallest fraction whose mean log X-distance over links reaches
            the warp graph's (observable attribute homophily matched); same edge count, no node isolated.
  *_boot    Phase 2 on a bootstrap graph drawn from the Phase 1 model fitted to that graph.
Phase 1 is shared by all variants of a seed. --variants '' only generates and saves the graphs.

usage: python experiments/run_global_twist.py --seed 601 --kappa 1.5 [--variants warp,warp_boot,twin,twin_boot] [--tag globaltax]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import asdict

import numpy as np
import torch

warnings.filterwarnings("ignore", message=".*encountered in matmul.*")
warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aligning.models import VAEConfig
from aligning.synthetic import SyntheticConfig, ManifoldSpec, generate, true_geodesic_distances, true_metric_fn
from aligning.train import train_phase1, GeoConfig, run_phase2, DistanceEngine
from aligning.metric import GridSpec, MetricField
from aligning.geodesics import GridGeodesics
from aligning.twist import TwistSpec, generate_twisted
from aligning import strain as ST
from experiments.common import ROOT, WORK, jsonable, triu, code_hash
from experiments.run_synthetic import VAE_CFG, P1_CFG
from experiments.run_twist import ARMS, GRAPH

P2 = ARMS["bce1"]
N_FOURIER, LENGTH = 24, 1.0


def fourier_field(rng, Z, length=LENGTH, K=N_FOURIER):
    W = rng.normal(scale=1.0 / length, size=(K, 2))
    b = rng.uniform(0, 2 * np.pi, K)
    a = rng.normal(size=K)
    vals = np.cos(Z @ W.T + b) @ a
    mu, sd = vals.mean(), vals.std()

    def f(u: torch.Tensor) -> torch.Tensor:
        Wt, bt, at = (torch.as_tensor(x, dtype=u.dtype) for x in (W, b, a))
        return ((torch.cos(u @ Wt.T + bt) @ at) - mu) / sd
    return f


def global_metric_fn(spec, seed, kappa, Z):
    rng = np.random.default_rng([seed, 2222])
    fa, fs, ft = (fourier_field(rng, Z) for _ in range(3))
    G_attr = true_metric_fn(spec)

    def fn(u: torch.Tensor) -> torch.Tensor:
        G = G_attr(u)
        if kappa == 0:
            return G
        th = np.pi * ft(u)
        a = torch.stack([torch.cos(th), torch.sin(th)], -1)
        lam = torch.exp(kappa * (fa(u) + fs(u)) / np.sqrt(2))
        mu = torch.exp(kappa * (fa(u) - fs(u)) / np.sqrt(2))
        Ga = torch.einsum("nij,nj->ni", G, a)
        m = Ga / torch.sqrt(torch.einsum("ni,ni->n", a, Ga)).unsqueeze(-1)
        return mu[:, None, None] * G + (lam - mu)[:, None, None] * m.unsqueeze(-1) * m.unsqueeze(-2)
    return fn


def geodesics(fn, Z, P=200):
    gs = GridSpec.from_points(Z, P, margin=0.05)
    with torch.no_grad():
        return GridGeodesics(gs, Z, neighbourhood=32).hybrid_distances(MetricField.from_function(fn, gs)).numpy()


def mean_log_xdist(A, X):
    i, j = np.where(np.triu(A > 0, 1))
    return float(np.mean(np.log(np.linalg.norm(X[i] - X[j], axis=1))))


def scattered_twin(A0, X, target, rng):
    """Move links of the aligned graph A0 to uniformly random non-adjacent pairs, in a fixed random order, until the
    mean log X-distance over links reaches `target`. Links whose removal would isolate a node are skipped."""
    A = (A0 > 0).astype(float)
    n = A.shape[0]
    i, j = np.where(np.triu(A, 1))
    order = rng.permutation(len(i))
    moved = 0
    if mean_log_xdist(A, X) >= target:
        return A, 0
    for k in order:
        a, b = i[k], j[k]
        if A[a].sum() <= 1 or A[b].sum() <= 1:
            continue
        while True:
            c, d = rng.integers(n, size=2)
            if c != d and A[c, d] == 0:
                break
        A[a, b] = A[b, a] = 0.0
        A[c, d] = A[d, c] = 1.0
        moved += 1
        if moved % 10 == 0 and mean_log_xdist(A, X) >= target:
            break
    return A, moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--kappa", type=float, required=True)
    ap.add_argument("--variants", default="warp,warp_boot,twin,twin_boot")
    ap.add_argument("--tag", default="globaltax")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    seed = args.seed
    outdir = os.path.join(WORK, "results", "synthetic", args.tag, f"seed{seed:03d}", f"k{args.kappa:.2f}")
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(10_000 + seed)
    manifold = ManifoldSpec.sample(rng, 20)
    Z = rng.normal(size=(500, 2))
    d_attr = true_geodesic_distances(manifold, Z)
    cfg = SyntheticConfig(graph=GRAPH)
    X = generate(cfg, seed, d_true=d_attr, manifold=manifold, Z=Z)["X"]
    t0 = time.time()
    d_graph = geodesics(global_metric_fn(manifold, seed, args.kappa, Z), Z)
    data_w = generate_twisted(cfg, seed, TwistSpec(kind="none"), manifold, Z, d_graph=d_graph)
    assert np.allclose(data_w["X"], X)
    data_0 = generate_twisted(cfg, seed, TwistSpec(kind="none"), manifold, Z, d_graph=d_attr)
    target = mean_log_xdist(data_w["A"], X)
    A_twin, moved = scattered_twin(data_0["A"], X, target, np.random.default_rng([seed, 3333, int(round(100 * args.kappa))]))
    graphs = dict(warp=data_w["A"], twin=A_twin)
    info = dict(seed=seed, kappa=args.kappa, homophily_target=target, homophily_twin=mean_log_xdist(A_twin, X),
                homophily_aligned=mean_log_xdist(data_0["A"], X), twin_links_moved=moved,
                n_edges={k: int(np.triu(v > 0, 1).sum()) for k, v in graphs.items()}, seconds_generation=time.time() - t0)
    json.dump(jsonable(info), open(os.path.join(outdir, "instance.json"), "w"))
    f_in = os.path.join(outdir, "inputs.pt")
    if os.path.exists(f_in):                       # deterministic; never rewritten (tax jobs may be reading it)
        old = torch.load(f_in, weights_only=False)
        assert np.array_equal(old["A_warp"], graphs["warp"].astype(np.float32)) and np.array_equal(old["A_twin"], graphs["twin"].astype(np.float32))
    else:
        torch.save(dict(X=X, Z_true=Z, d_attr=triu(d_attr).astype(np.float32), d_graph=triu(d_graph).astype(np.float32),
                        A_warp=graphs["warp"].astype(np.float32), A_twin=graphs["twin"].astype(np.float32)), f_in)
    variants = [v for v in args.variants.split(",") if v]
    if not variants:
        print(f"[global seed {seed} k {args.kappa:.2f}] generation only; twin moved {moved}; homophily target {target:.3f} twin {info['homophily_twin']:.3f}", flush=True)
        return
    vcfg = VAEConfig(input_dim=X.shape[1], **VAE_CFG)
    vae, _ = train_phase1(X, vcfg, P1_CFG, seed=seed)
    Z1 = vae.latent_means(torch.as_tensor(X, dtype=torch.float64)).numpy()
    D1 = None
    from dataclasses import replace
    p2 = P2 if args.epochs is None else replace(P2, epochs=args.epochs)
    for v in variants:
        base = v.split("_")[0]
        A = graphs[base]
        boot = None
        if v.endswith("_boot"):
            if D1 is None:
                with torch.no_grad():
                    D1 = DistanceEngine(Z1, GeoConfig()).distances(vae.decoder).numpy()
            bn = ST.bootstrap_null_graph(A, D1, np.random.default_rng([seed, 5151, int(round(100 * args.kappa)), base == "twin"]),
                                         target=p2.target)
            A = bn["A"]
            boot = {k: val for k, val in bn.items() if k != "A"}
        res = run_phase2(vae, X, A, GeoConfig(), p2, seed=seed)
        np.savez_compressed(os.path.join(outdir, f"{v}_arrays.npz"), A=A.astype(np.float32), d1=triu(res["D1"]).astype(np.float32),
                            d2=triu(res["D2"]).astype(np.float32))
        json.dump(jsonable(dict(variant=v, bootstrap=boot, code_sha256=code_hash(), epochs=res["epochs_run"],
                                loss_p1=res["loss_p1"], loss_p2=res["loss_p2"], seconds=res["seconds"])),
                  open(os.path.join(outdir, f"{v}.json"), "w"))
        print(f"[global seed {seed} k {args.kappa:.2f} {v}] phase2 {res['seconds']:.0f}s ({res['epochs_run']} ep) "
              f"loss {res['loss_p1']:.5f}->{res['loss_p2']:.5f}", flush=True)
    print(f"[global seed {seed} k {args.kappa:.2f}] twin moved {moved} links; homophily target {target:.3f} twin {info['homophily_twin']:.3f}", flush=True)


if __name__ == "__main__":
    main()
