"""
Planted-disagreement experiment (Fig. 3).

One process = one (seed, arm). For that seed: one attribute manifold and attributes, one Phase 1 VAE, then for
each design: the graph generated from the twisted metric, Phase 2, the learned strain at the nodes, the
Phase-1-only competitors, and everything the aggregation needs to compare them with the known truth. Designs of a
seed share every random number (attributes, Phase 1, the uniform draws of the graph) except the twist itself.

The runner prints training diagnostics only (loss, scale, epochs); the agreement with the truth is computed by
experiments/aggregate_twist.py.

Arms: bce1 = single-scale cross-entropy (primary); mse15 = squared error over 15 diffusion times (secondary).
Designs "<kind>_boot" run Phase 2 on a bootstrap graph of the <kind> graph (strain.bootstrap_null_graph).

usage: python experiments/run_twist.py --seed 301 --arm bce1 [--designs none,shear,barrier,corridor,shear_boot,...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import asdict, replace

import numpy as np
import torch

warnings.filterwarnings("ignore", message=".*encountered in matmul.*")
warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aligning.models import VAEConfig
from aligning.synthetic import SyntheticConfig, GraphConfig, ManifoldSpec, generate, true_geodesic_distances
from aligning.train import train_phase1, GeoConfig, Phase2Config, run_phase2
from aligning import heat, strain as ST
from aligning.twist import TwistSpec, KINDS, generate_twisted, true_strain, graph_geodesic_distances
from aligning import baselines as B
from experiments.common import ROOT, WORK, jsonable, triu, code_hash
from experiments.run_synthetic import VAE_CFG, P1_CFG, P2_CFG

ARMS = {
    # single-scale cross-entropy: PRIMARY
    "bce1": replace(P2_CFG, K=1, loss="bce"),
    # multi-scale squared error: SECONDARY
    "mse15": P2_CFG,
}
GRAPH = GraphConfig(bandwidth=0.1)          # as in the main benchmark (R70): sharp kernel, mean degree 22
KAPPA = float(np.log(9.0))                  # lengths x3 along the twist axis at the centre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arm", default="bce1", choices=list(ARMS))
    ap.add_argument("--designs", default="none,shear,barrier,corridor")
    ap.add_argument("--kappa", type=float, default=KAPPA)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--tag", default="twist")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    p2 = ARMS[args.arm] if args.epochs is None else replace(ARMS[args.arm], epochs=args.epochs)
    gcfg = GeoConfig()
    designs = args.designs.split(",")
    assert all(d.replace("_boot", "") in KINDS for d in designs), designs
    outdir = os.path.join(WORK, "results", "synthetic", args.tag, args.arm, f"seed{args.seed:03d}")
    os.makedirs(outdir, exist_ok=True)

    seed = args.seed
    rng = np.random.default_rng(10_000 + seed)          # same manifold and positions as run_synthetic.py
    manifold = ManifoldSpec.sample(rng, 20)
    Z = rng.normal(size=(500, 2))
    t0 = time.time()
    d_attr = true_geodesic_distances(manifold, Z)
    cfg = SyntheticConfig(graph=GRAPH)
    X = generate(cfg, seed, d_true=d_attr, manifold=manifold, Z=Z)["X"]
    vcfg = VAEConfig(input_dim=X.shape[1], **VAE_CFG)
    vae, h1 = train_phase1(X, vcfg, P1_CFG, seed=seed)
    Xt = torch.as_tensor(X, dtype=torch.float64)
    Z1 = vae.latent_means(Xt).numpy()
    J = ST.dz_du(vae.encoder, manifold, X, Z)
    print(f"[seed {seed} {args.arm}] manifold+phase1 {time.time()-t0:.0f}s", flush=True)
    f1 = os.path.join(outdir, "phase1.pt")
    if not os.path.exists(f1):
        torch.save(dict(vae_state=vae.state_dict(), vcfg=asdict(vcfg), Z_true=Z, X=X, Z1=Z1, J_dzdu=J,
                        d_attr=triu(d_attr).astype(np.float32)), f1)
    chash = code_hash()

    D1_cache = {}
    for name in designs:
        fn = os.path.join(outdir, f"{name}.json")
        if os.path.exists(fn):
            continue
        boot = name.endswith("_boot")
        kind = name[:-len("_boot")] if boot else name
        tw = TwistSpec(kind="none") if kind == "none" else \
            TwistSpec.draw(kind, Z, np.random.default_rng([seed, 4242, KINDS.index(kind)]), args.kappa)
        t1 = time.time()
        d_graph = d_attr if kind == "none" else graph_geodesic_distances(manifold, tw, Z)
        data = generate_twisted(cfg, seed, tw, manifold, Z, d_graph=d_graph)
        assert np.allclose(data["X"], X)
        A = data["A"]
        truth = true_strain(manifold, tw, Z)
        boot_info = None
        if boot:
            # bootstrap null of the observed design graph: drawn from the Phase 1 latent-space model fitted to
            # it (independent random numbers); the twist of this graph is what a graph agreeing with the
            # learned attribute geometry would produce (excess twist)
            if "D1" not in D1_cache:
                from aligning.train import DistanceEngine
                with torch.no_grad():
                    D1_cache["D1"] = DistanceEngine(Z1, gcfg).distances(vae.decoder).numpy()
            bn = ST.bootstrap_null_graph(A, D1_cache["D1"], np.random.default_rng([seed, 5151, KINDS.index(kind)]),
                                         target=p2.target)
            A = bn["A"]
            boot_info = {k: v for k, v in bn.items() if k != "A"}
        t_gen = time.time() - t1
        try:
            res = run_phase2(vae, X, A, gcfg, p2, seed=seed)
            iu = np.triu_indices(len(Z), 1)
            if not (np.all(np.isfinite(res["D2"])) and np.all(res["D2"][iu] > 0)):
                raise RuntimeError("non-finite or zero distances")
        except Exception as e:
            with open(os.path.join(outdir, f"{name}.FAILED.json"), "w") as f:
                json.dump(dict(seed=seed, design=name, arm=args.arm, error=repr(e)), f)
            print(f"[seed {seed} {args.arm}] {name}: FAILED {e!r}", flush=True)
            continue
        dec1, dec2 = vae.decoder, res["decoder2"]
        learned = ST.learned_strain(dec1, dec2, Z1)
        major_u = np.linalg.solve(J, learned["major_z"][..., None])[..., 0]
        # Phase-1-only competitors: link moment tensor with a single-Gaussian Phase 1 kernel
        t1k = heat.heat_times(A, 1, 4.0)
        w1k = heat.kernel_weights(t1k, 2)
        At = res["At"]
        s0, a0, _, _ = heat.calibrate(res["D1"], At, t1k, w1k)
        s0, a0, _ = heat.calibrate_bce(res["D1"], At, t1k, w1k, init=(s0, a0))
        P1 = ST.phase1_link_probabilities(res["D1"], At, t1k, w1k, s0, a0)
        mom = {}
        for k_s in (0, 10):
            m = ST.link_moment_strain(Z1, A, P1, learned["G1"], smooth_k=k_s)
            mom[k_s] = dict(area=m["area"], aniso=m["aniso"],
                            major_u=np.linalg.solve(J, np.nan_to_num(m["major_z"])[..., None])[..., 0])
        paths = ST.graph_path_lengths(A, res["D1"])
        deg = (A > 0).sum(1)
        out = dict(seed=seed, design=name, arm=args.arm, twist=asdict(tw), bootstrap=boot_info, code_sha256=chash,
                   phase2=asdict(p2), graph_cfg=asdict(GRAPH),
                   graph=dict(n_edges=int((triu(A) > 0).sum()), mean_degree=float(deg.mean()),
                              n_isolated_fixed=data["n_isolated_fixed"]),
                   phase2_summary=dict(loss_p1=res["loss_p1"], loss_p2=res["loss_p2"], scale_p1=res["scale_p1"],
                                       scale_p2=res["scale_p2"], amp_p1=res["amp_p1"], amp_p2=res["amp_p2"],
                                       epochs=res["epochs_run"], seconds=res["seconds"],
                                       calib_r2_vs_zero=res["calib_r2_vs_zero"]),
                   moment_calibration=dict(scale=s0, amp=a0), seconds_generation=t_gen)
        np.savez_compressed(
            os.path.join(outdir, f"{name}_arrays.npz"),
            A=A.astype(np.float32),
            d1=triu(res["D1"]).astype(np.float32), d2=triu(res["D2"]).astype(np.float32),
            d_graph=triu(d_graph).astype(np.float32), d_attr=triu(d_attr).astype(np.float32),
            hop=triu(paths["hop"]).astype(np.float32), sp_d1=triu(paths["sp_d1"]).astype(np.float32),
            t_lam=truth["lam"], t_mu=truth["mu"], t_phi=truth["phi"], t_area=truth["area"], t_aniso=truth["aniso"],
            t_major_u=truth["major_u"], G_attr=truth["G_attr"],
            area=learned["area"], aniso=learned["aniso"], L=learned["L"], major_u=major_u,
            G1=learned["G1"], G2=learned["G2"],
            mom_area=mom[0]["area"], mom_aniso=mom[0]["aniso"], mom_major_u=mom[0]["major_u"],
            moms_area=mom[10]["area"], moms_aniso=mom[10]["aniso"], moms_major_u=mom[10]["major_u"],
            p1_edge_length=B.edge_length_score(res["D1"], A),
            local_cosine_homophily=B.local_cosine_homophily(A, X),
            loss_curve=res["history"][:, 0].astype(np.float32))
        torch.save(dict(decoder2=dec2.state_dict(), scale_p2=res["scale_p2"], amp_p2=res["amp_p2"],
                        times=res["times"], weights=res["weights"]), os.path.join(outdir, f"{name}_decoder2.pt"))
        with open(fn, "w") as f:
            json.dump(jsonable(out), f)
        print(f"[seed {seed} {args.arm}] {name}: gen {t_gen:.0f}s phase2 {res['seconds']:.0f}s ({res['epochs_run']} ep) "
              f"loss {res['loss_p1']:.5f}->{res['loss_p2']:.5f} s {res['scale_p1']:.3g}->{res['scale_p2']:.3g}", flush=True)


if __name__ == "__main__":
    main()
