"""
Node-relabelling block designs (Supplementary Section S4: scattered R70, compact D70 and band blocks).

One process = one seed. For that seed: one attribute manifold, one Phase 1 VAE (shared by all conditions since X
is identical across conditions), then for each condition: graph generation, Phase 2, method scores, baselines,
metrics. Null and perturbed conditions of a seed share every random number except the block's compatibilities
(paired design).

usage: python experiments/run_synthetic.py --seed 301 --conditions R70,R70N,D70,D70N,BAND,BANDN --p2 K=1,loss=bce --tag twist_blocks --no-pygod
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, replace

import warnings
import numpy as np
import torch

warnings.filterwarnings("ignore", message=".*encountered in matmul.*")   # spurious Accelerate FP flags
warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aligning.models import VAEConfig
from aligning.synthetic import (SyntheticConfig, GraphConfig, PerturbConfig, ManifoldSpec,
                                generate, true_geodesic_distances)
from aligning.train import train_phase1, Phase1Config, GeoConfig, Phase2Config, run_phase2
from experiments.common import (ROOT, WORK, method_scores, baseline_scores, partition_metrics, evaluate_all,
                                fidelity, jsonable, triu, code_hash)

# --------------------------------------------------------------------------- design
CONDITIONS = {
    # name: (kind, region, m, intensity, homophily_mix, bandwidth)
    # --- position-free node relabelling ---------------------------------------------
    "R70":     ("node_relabel", "scatter", 70, 1.00, 0.0, 0.1),   # main condition
    "R70N":    ("node_relabel", "scatter", 70, 0.00, 0.0, 0.1),   # its paired null
    "R70A25":  ("node_relabel", "scatter", 70, 0.25, 0.0, 0.1),   # intensity sweep
    "R70A50":  ("node_relabel", "scatter", 70, 0.50, 0.0, 0.1),
    "R35":     ("node_relabel", "scatter", 35, 1.00, 0.0, 0.1),   # size sweep (+ nulls)
    "R35N":    ("node_relabel", "scatter", 35, 0.00, 0.0, 0.1),
    "R140":    ("node_relabel", "scatter", 140, 1.00, 0.0, 0.1),
    "R140N":   ("node_relabel", "scatter", 140, 0.00, 0.0, 0.1),
    "R70H30":  ("node_relabel", "scatter", 70, 1.00, 0.3, 0.1),   # homophily: random mixing
    "R70H30N": ("node_relabel", "scatter", 70, 0.00, 0.3, 0.1),
    "R70H60":  ("node_relabel", "scatter", 70, 1.00, 0.6, 0.1),
    "R70H60N": ("node_relabel", "scatter", 70, 0.00, 0.6, 0.1),
    "R70BW3":  ("node_relabel", "scatter", 70, 1.00, 0.0, 0.3),   # homophily: kernel width (original 0.3)
    "R70BW3N": ("node_relabel", "scatter", 70, 0.00, 0.0, 0.3),
    # --- the paper's perturbation (within-block shuffle, done exactly) ---------------------
    "D70":     ("block_shuffle", "disk", 70, 1.00, 0.0, 0.1),     # compact interior region
    "D70N":    ("block_shuffle", "disk", 70, 0.00, 0.0, 0.1),
    "BAND":    ("block_shuffle", "band", 70, 1.00, 0.0, 0.3),     # replication of the original design
    "BANDN":   ("block_shuffle", "band", 70, 0.00, 0.0, 0.3),
    "BAND1":   ("block_shuffle", "band", 70, 1.00, 0.0, 0.1),
    "BAND1N":  ("block_shuffle", "band", 70, 0.00, 0.0, 0.1),
}

VAE_CFG = dict(enc_hidden=[16, 16], dec_hidden=[16], activation="elu", enc_dropout=0.2,
               lambda_comp_variance=5.0, lambda_decoder_variance=100.0, beta_end=2.0)
P1_CFG = Phase1Config(epochs=1500, lr=5e-3, weight_decay=1e-5)
P2_CFG = Phase2Config(K=15, c=4.0, target="max", epochs=400, lr=1e-2, patience=40, rel_tol=1e-4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--conditions", default="all")
    ap.add_argument("--estimator", default="hybrid")
    ap.add_argument("--P", type=int, default=64)
    ap.add_argument("--metric", default="expected")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--no-pygod", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--p2", default="", help="Phase2Config overrides, e.g. K=5,c=16,freeze_logvar=1")
    ap.add_argument("--vae", default="", help="VAEConfig overrides, e.g. dec_hidden=64,latent_dim=3")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    p2 = P2_CFG
    for kv in filter(None, args.p2.split(",")):
        k, v = kv.split("=")
        cur = getattr(p2, k)
        p2 = replace(p2, **{k: type(cur)(float(v)) if not isinstance(cur, (bool, str)) else (v == "1" if isinstance(cur, bool) else v)})
    vcfg_kw = dict(VAE_CFG)
    for kv in filter(None, args.vae.split(",")):
        k, v = kv.split("=")
        vcfg_kw[k] = [int(v)] if k in ("dec_hidden",) else (int(v) if k == "latent_dim" else float(v))
    gcfg = GeoConfig(estimator=args.estimator, P=args.P, metric=args.metric)
    conds = list(CONDITIONS) if args.conditions == "all" else args.conditions.split(",")
    outdir = os.path.join(WORK, "results", "synthetic", args.tag, f"seed{args.seed:03d}")
    os.makedirs(outdir, exist_ok=True)

    seed = args.seed
    rng = np.random.default_rng(10_000 + seed)
    manifold = ManifoldSpec.sample(rng, 20)
    Z = rng.normal(size=(500, 2))
    t0 = time.time()
    d_true = true_geodesic_distances(manifold, Z)
    base = generate(SyntheticConfig(), seed, d_true=d_true, manifold=manifold, Z=Z)
    X = base["X"]
    vcfg = VAEConfig(input_dim=X.shape[1], **vcfg_kw)
    vae, h1 = train_phase1(X, vcfg, P1_CFG, seed=seed)
    print(f"[seed {seed}] manifold+phase1 {time.time()-t0:.0f}s", flush=True)
    # phase-1 artefacts shared by all conditions of this seed (atomic write, once)
    f1 = os.path.join(outdir, "phase1.npz")
    if not os.path.exists(f1):
        Z1 = vae.latent_means(torch.as_tensor(X, dtype=torch.float64)).numpy()
        tmp = os.path.join(outdir, f"phase1.{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, Z_true=Z, X=X, Z1=Z1.astype(np.float32),
                            d_true=triu(d_true).astype(np.float32), loss_phase1=h1.astype(np.float32))
        os.replace(tmp, f1)
    chash = code_hash()

    for name in conds:
        fn = os.path.join(outdir, f"{name}.json")
        if os.path.exists(fn):
            continue
        kind, region, m, alpha, h, bw = CONDITIONS[name]
        cfg = SyntheticConfig(graph=GraphConfig(homophily_mix=h, bandwidth=bw),
                              perturb=PerturbConfig(region=region, m=m, intensity=alpha, kind=kind))
        data = generate(cfg, seed, d_true=d_true, manifold=manifold, Z=Z)
        assert np.allclose(data["X"], X)
        A, y = data["A"], data["labels"]
        t1 = time.time()
        try:
            res = run_phase2(vae, X, A, gcfg, p2, seed=seed)
            if not (np.all(np.isfinite(res["D1"])) and np.all(np.isfinite(res["D2"])) and np.all(res["D2"][np.triu_indices(len(y), 1)] > 0)):
                raise RuntimeError("non-finite or zero distances")
        except Exception as e:   # failures are recorded (failure rate is reported), never silently dropped
            with open(os.path.join(outdir, f"{name}.FAILED.json"), "w") as f:
                json.dump(dict(seed=seed, condition=name, error=repr(e)), f)
            print(f"[seed {seed}] {name}: FAILED {e!r}", flush=True)
            continue
        t_p2 = time.time() - t1
        scores = method_scores(res, A)
        bsc, tb = baseline_scores(A, X, res["Z"], res["D1"], res["At"], res["times"], res["weights"], vae, seed,
                                  d_true=d_true, pygod=not args.no_pygod)
        err = bsc.pop("_pygod_error", None)
        scores.update(bsc)
        metrics = evaluate_all(scores, y)
        deg = (A > 0).sum(1)
        out = dict(
            seed=seed, condition=name, kind=kind, region=region, m=m, intensity=alpha, homophily_mix=h, bandwidth=bw,
            code_sha256=chash,
            estimator=args.estimator, P=args.P, metric=args.metric, tag=args.tag,
            phase2=asdict(p2), vae=asdict(vcfg), phase1=asdict(P1_CFG),
            metrics=metrics, partitions=partition_metrics(A, y, seed), pygod_error=err,
            fidelity_phase1=fidelity(res["D1"], d_true),
            graph=dict(n_edges=int((triu(A) > 0).sum()), mean_degree=float(deg.mean()),
                       deg_block=float(deg[y == 1].mean()), deg_rest=float(deg[y == 0].mean()),
                       n_isolated_fixed=data["n_isolated_fixed"], n_components_before=data["n_components_before"]),
            phase2_summary=dict(loss_p1=res["loss_p1"], loss_p2=res["loss_p2"], scale_p1=res["scale_p1"], amp_p1=res["amp_p1"], amp_p2=res["amp_p2"], calib_at_boundary=res["calib_at_boundary"], calib_r2_vs_zero=res["calib_r2_vs_zero"],
                                scale_p2=res["scale_p2"], epochs=res["epochs_run"], seconds=t_p2,
                                times=res["times"].tolist()),
            timings=tb,
        )
        with open(fn, "w") as f:
            json.dump(jsonable(out), f)
        np.savez_compressed(os.path.join(outdir, f"{name}_arrays.npz"),
                            y=y, d1=triu(res["D1"]).astype(np.float32), d2=triu(res["D2"]).astype(np.float32),
                            **{f"score_{k}": v.astype(np.float32) for k, v in scores.items()},
                            loss_curve=res["history"][:, 0].astype(np.float32))
        # progress line: training diagnostics only
        print(f"[seed {seed}] {name}: done phase2 {t_p2:.0f}s ({res['epochs_run']} ep) loss {res['loss_p1']:.5f}->{res['loss_p2']:.5f} "
              f"s {res['scale_p1']:.3g}->{res['scale_p2']:.3g} calibR2 {res['calib_r2_vs_zero']:.3f}", flush=True)


if __name__ == "__main__":
    main()
