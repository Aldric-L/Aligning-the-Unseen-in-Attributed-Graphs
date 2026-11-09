"""Shared pieces of the experiment runners: method scores + all baselines for one dataset."""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from aligning import baselines as B
from aligning import heat
from aligning.evaluation import detection_metrics
from aligning.scores import ratio_scores, legacy_scores, edge_restricted
from aligning.train import Phase1Config, Phase2Config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the code
# data/ and results/ live under WORK: this repository by default, or the directory named by ALIGNING_WORKDIR
WORK = os.path.abspath(os.environ.get("ALIGNING_WORKDIR", ROOT))

# Phase 1 and Phase 2 configuration for real networks (Supplementary Table S1)
VAE_REAL = dict(enc_hidden=[96, 64], dec_hidden=[128], activation="relu", enc_dropout=0.2, clip_logvar=30.0,
                lambda_comp_variance=100.0, lambda_decoder_variance=20.0, beta_end=1.0)
P1_REAL = Phase1Config(epochs=1200, lr=1e-3, weight_decay=1e-5)
P2_REAL = Phase2Config(K=15, c=4.0, target="logmed", epochs=400, lr=1e-2, patience=40, rel_tol=1e-4)


def triu(M):
    iu = np.triu_indices(M.shape[0], 1)
    return M[iu]


def method_scores(res: dict, A: np.ndarray) -> dict:
    D1, D2 = res["D1"], res["D2"]
    Zr, S_ratio, S_signed = ratio_scores(D1, D2)
    S_edge, S_edge_signed = edge_restricted(Zr, A)
    _, S_legacy = legacy_scores(D1, D2)
    R2 = heat.residuals(D2, res["At"], res["times"], res["weights"], res["scale_p2"], res["amp_p2"])
    return {
        "distortion_ratio": S_ratio,            # primary
        "distortion_legacy": S_legacy,          # paper Eq. (5)
        "distortion_signed": S_signed,          # descriptive (expansion > 0)
        "distortion_edge": S_edge,              # mean |Z| over edges
        "distortion_edge_contraction": -S_edge_signed,  # edges pulled together -> high
        "p2_residual": np.abs(R2).sum(1) / (D1.shape[0] - 1),  # descriptive
    }


def baseline_scores(A, X, Z1, D1, At, times, weights, vae, seed, d_true=None, pygod=True):
    out = {}
    t = {}
    t0 = time.time()
    out["degree_dev"] = B.degree_deviation(A)
    out["strength_dev"] = B.strength_deviation(A)
    out["clustering_dev"] = B.clustering_deviation(A)
    out["iforest_X"] = B.iforest(X, seed)
    out["lof_X"] = B.lof(X)
    out["iforest_latent"] = B.iforest(Z1, seed)
    with torch.no_grad():
        Xt = torch.as_tensor(X, dtype=torch.float64)
        mx, _ = vae.decoder(torch.as_tensor(Z1, dtype=torch.float64))
        out["vae_recon"] = torch.linalg.norm(Xt - mx, dim=1).numpy()
    out["local_cosine_homophily"] = B.local_cosine_homophily(A, X)
    out["local_assortativity"] = B.multiscale_local_assortativity(A, X)
    t["simple"] = time.time() - t0
    t0 = time.time()
    out["p1_residual"] = B.kernel_residual_score(D1, At, times, weights)               # Phase-1-only control
    out["euclid_latent_residual"] = B.kernel_residual_score(squareform(pdist(Z1)), At, times, weights)
    Xs = (X - X.mean(0)) / X.std(0)
    out["attr_residual"] = B.kernel_residual_score(squareform(pdist(Xs)), At, times, weights)
    out["p1_eccentricity"] = B.eccentricity_score(D1)
    out["p1_edge_length"] = B.edge_length_score(D1, A)                                  # Phase-1-only edge control
    out["euclid_latent_edge_length"] = B.edge_length_score(squareform(pdist(Z1)), A)
    out["attr_edge_length"] = B.edge_length_score(squareform(pdist(Xs)), A)
    if d_true is not None:
        out["ORACLE_true_geometry_residual"] = B.kernel_residual_score(d_true, At, times, weights)
        out["ORACLE_true_edge_length"] = B.edge_length_score(d_true, A)
    t["kernel"] = time.time() - t0
    if pygod:
        t0 = time.time()
        try:
            out.update(B.pygod_scores(A, X, seed))
        except Exception as e:  # recorded, never silently dropped
            out["_pygod_error"] = repr(e)
        t["pygod"] = time.time() - t0
    return out, t


def partition_metrics(A, y, seed):
    out = {}
    for k, lab in B.partitions(A, seed).items():
        out[k] = dict(ari=float(adjusted_rand_score(y, lab)), nmi=float(normalized_mutual_info_score(y, lab)),
                      n_parts=int(len(np.unique(lab))))
    return out


def evaluate_all(scores: dict, y: np.ndarray) -> dict:
    return {k: detection_metrics(v, y) for k, v in scores.items() if isinstance(v, np.ndarray)}


def fidelity(D1, d_true):
    a, b = triu(D1), triu(d_true)
    return dict(spearman=float(spearmanr(a, b).correlation), pearson=float(pearsonr(a, b)[0]),
                pearson_log=float(pearsonr(np.log(a), np.log(b))[0]))


def code_hash() -> str:
    """sha256 of the method + experiment code, stored in every result file."""
    import hashlib
    h = hashlib.sha256()
    for sub in ("aligning", "experiments"):
        d = os.path.join(ROOT, sub)
        for f in sorted(os.listdir(d)):
            if f.endswith(".py"):
                h.update(open(os.path.join(d, f), "rb").read())
    return h.hexdigest()


def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o
