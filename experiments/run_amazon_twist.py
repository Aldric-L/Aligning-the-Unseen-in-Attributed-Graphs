"""
Twist of the product network (Fig. 4): Phase 1 (VAE_REAL, P1_REAL) on the product attributes, then the
single-scale cross-entropy Phase 2 on a binary layer. Phase 1 is deterministic given the seed (its latent means are
hashed in the output, so the variants of a seed can be checked to share it).

Variants: S | C | S_boot | C_boot (bootstrap graph from the Phase 1 model fitted to that layer).
Attributes: text (X.npy, primary) | popularity (X_popularity.npy, control).

usage: python experiments/run_amazon_twist.py --seed 1 --variant S [--attributes text]
"""
from __future__ import annotations

import argparse
import hashlib
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
from aligning.train import train_phase1, GeoConfig, run_phase2, DistanceEngine
from aligning import heat, strain as ST
from experiments.common import ROOT, WORK, jsonable, triu, code_hash, VAE_REAL, P1_REAL, P2_REAL

P2_AMZ = replace(P2_REAL, K=1, loss="bce", target="max")     # cross-entropy arm, binary target
DATA = os.path.join(WORK, "data", "Amazon", "Musical_Instruments")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--variant", required=True, choices=["S", "C", "S_boot", "C_boot"])
    ap.add_argument("--attributes", default="text", choices=["text", "popularity"])
    ap.add_argument("--P", type=int, default=96)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=None, help="override (smoke tests only)")
    ap.add_argument("--tag", default="amazon")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    X = np.load(os.path.join(DATA, "X.npy" if args.attributes == "text" else "X_popularity.npy"))
    boot = args.variant.endswith("_boot")
    layer = args.variant.split("_")[0]
    A = np.load(os.path.join(DATA, f"A_{layer}.npy"))
    p2 = P2_AMZ if args.epochs is None else replace(P2_AMZ, epochs=args.epochs)
    outdir = os.path.join(WORK, "results", args.tag, args.attributes, args.variant)
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    vcfg = VAEConfig(input_dim=X.shape[1], **VAE_REAL)
    vae, h1 = train_phase1(X, vcfg, P1_REAL, seed=args.seed)
    Z1_ = vae.latent_means(torch.as_tensor(X, dtype=torch.float64)).numpy()
    z1_hash = hashlib.sha256(np.round(Z1_, 8).tobytes()).hexdigest()[:16]
    boot_info = None
    if boot:
        with torch.no_grad():
            D1_ = DistanceEngine(Z1_, GeoConfig(P=args.P)).distances(vae.decoder).numpy()
        bn = ST.bootstrap_null_graph(A, D1_, np.random.default_rng([args.seed, 5151, ["S", "C"].index(layer)]), target=p2.target)
        A = bn["A"]
        boot_info = {k: v for k, v in bn.items() if k != "A"}
    t1 = time.time()
    res = run_phase2(vae, X, A, GeoConfig(P=args.P), p2, seed=args.seed)
    Z1 = res["Z"]
    learned = ST.learned_strain(vae.decoder, res["decoder2"], Z1)
    t1k = heat.heat_times(A, 1, 4.0)
    w1k = heat.kernel_weights(t1k, 2)
    s0, a0, _, _ = heat.calibrate(res["D1"], res["At"], t1k, w1k)
    s0, a0, _ = heat.calibrate_bce(res["D1"], res["At"], t1k, w1k, init=(s0, a0))
    P1 = ST.phase1_link_probabilities(res["D1"], res["At"], t1k, w1k, s0, a0)
    mom = ST.link_moment_strain(Z1, A, P1, learned["G1"], smooth_k=10)
    out = dict(seed=args.seed, variant=args.variant, attributes=args.attributes, P=args.P, phase2=asdict(p2),
               code_sha256=code_hash(), z1_hash=z1_hash, bootstrap=boot_info, vae=VAE_REAL, phase1=asdict(P1_REAL),
               n=int(A.shape[0]), n_edges=int((triu(A) > 0).sum()),
               phase2_summary=dict(loss_p1=res["loss_p1"], loss_p2=res["loss_p2"], scale_p1=res["scale_p1"],
                                   amp_p1=res["amp_p1"], amp_p2=res["amp_p2"], scale_p2=res["scale_p2"],
                                   epochs=res["epochs_run"], seconds=res["seconds"],
                                   calib_r2_vs_zero=res["calib_r2_vs_zero"]),
               moment_calibration=dict(scale=s0, amp=a0), seconds_phase1=t1 - t0, seconds_total=time.time() - t0)
    with open(os.path.join(outdir, f"seed{args.seed:03d}.json"), "w") as f:
        json.dump(jsonable(out), f)
    np.savez_compressed(os.path.join(outdir, f"seed{args.seed:03d}_arrays.npz"),
                        Z1=Z1.astype(np.float32), d1=triu(res["D1"]).astype(np.float32),
                        d2=triu(res["D2"]).astype(np.float32),
                        area=learned["area"], aniso=learned["aniso"], major_z=learned["major_z"],
                        G1=learned["G1"], G2=learned["G2"],
                        mom_area=mom["area"], mom_aniso=mom["aniso"], mom_major_z=mom["major_z"],
                        A=A.astype(np.float32), loss_curve=res["history"][:, 0].astype(np.float32))
    print(f"[amazon {args.attributes} {args.variant} seed {args.seed}] z1 {z1_hash} phase1 {t1-t0:.0f}s "
          f"phase2 {res['seconds']:.0f}s ({res['epochs_run']} ep) loss {res['loss_p1']:.5f}->{res['loss_p2']:.5f} "
          f"s {res['scale_p1']:.3g}->{res['scale_p2']:.3g}", flush=True)


if __name__ == "__main__":
    main()
