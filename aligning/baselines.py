"""
Baselines. Every baseline returns a CONTINUOUS node score oriented a priori so that
high = anomalous (no post-hoc sign choice). Hard partitions (spectral, Louvain) are
evaluated separately with ARI/NMI because AUROC is undefined for hard 0/1 labels.

Families
  graph-only        degree deviation, local clustering deviation
  attribute-only    isolation forest / LOF on X, VAE reconstruction error, VAE-latent iforest
  homophily         local cosine homophily ("cosine assortativity"), multiscale
                    local assortativity (Peel et al. 2018, PPR-weighted, vector attributes)
  kernel controls   Phase-1-only residual, Euclidean-latent residual,
                    attribute-space residual, Phase-1 eccentricity (confound probe)
  learned (PyGOD)   DOMINANT, AnomalyDAE, CoLA with library defaults (no tuning on labels)
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from .scores import robust_z
from . import heat


# ----------------------------------------------------------------------------- graph-only
def degree_deviation(A: np.ndarray) -> np.ndarray:
    return np.abs(robust_z((A > 0).sum(1).astype(float)))


def strength_deviation(A: np.ndarray) -> np.ndarray:
    return np.abs(robust_z(A.sum(1)))


def clustering_deviation(A: np.ndarray) -> np.ndarray:
    G = nx.from_numpy_array((A > 0).astype(int))
    c = np.array([nx.clustering(G, i) for i in range(A.shape[0])])
    return np.abs(robust_z(c))


def partitions(A: np.ndarray, seed: int) -> Dict[str, np.ndarray]:
    from sklearn.cluster import SpectralClustering
    out = {}
    sc = SpectralClustering(n_clusters=2, affinity="precomputed", random_state=seed, assign_labels="kmeans")
    out["spectral_k2"] = sc.fit_predict(A)
    G = nx.from_numpy_array(A)
    comms = nx.community.louvain_communities(G, weight="weight", seed=seed)
    lab = np.zeros(A.shape[0], dtype=int)
    for k, c in enumerate(comms):
        lab[list(c)] = k
    out["louvain"] = lab
    return out


# ----------------------------------------------------------------------------- attribute-only
def iforest(F: np.ndarray, seed: int) -> np.ndarray:
    Fs = StandardScaler().fit_transform(F)
    return -IsolationForest(n_estimators=500, random_state=seed).fit(Fs).score_samples(Fs)


def lof(F: np.ndarray, k: int = 20) -> np.ndarray:
    Fs = StandardScaler().fit_transform(F)
    m = LocalOutlierFactor(n_neighbors=k).fit(Fs)
    return -m.negative_outlier_factor_


# ----------------------------------------------------------------------------- homophily
def local_cosine_homophily(A: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Local cosine assortativity, vectorised; score = - homophily."""
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    C = Xn @ Xn.T
    n = A.shape[0]
    nb = (A > 0)
    nb_mean = (C * nb).sum(1) / np.maximum(nb.sum(1), 1)
    all_mean = (C.sum(1) - np.diag(C)) / (n - 1)
    return -(nb_mean - all_mean)


def ppr_matrix(A: np.ndarray, restart: float) -> np.ndarray:
    """Personalised PageRank: row l = stationary distribution of a walk restarting at l."""
    s = A.sum(1)
    W = A / np.maximum(s, 1e-300)[:, None]            # row-stochastic
    n = A.shape[0]
    return restart * np.linalg.inv(np.eye(n) - (1 - restart) * W)


def multiscale_local_assortativity(A: np.ndarray, X: np.ndarray,
                                   restarts=(0.1, 0.3, 0.5, 0.7, 0.9)) -> np.ndarray:
    """Local assortativity in the spirit of Peel, Delvenne & Lambiotte (PNAS 2018),
    generalised to vector attributes: for each attribute dimension (globally standardised
    over edge endpoints), local Pearson-type assortativity
        r_l = sum_i w_l(i) sum_j (A_ij / s_i) zeta_i zeta_j,
    with w_l the personalised PageRank distribution around l, averaged over restart
    probabilities ("TotalRank"-like multiscale summary) and over dimensions.
    Score = - r_l (low local assortativity = misaligned)."""
    s = A.sum(1)
    W = A / np.maximum(s, 1e-300)[:, None]
    # edge-endpoint standardisation (as in Newman's assortativity)
    ends = np.repeat(np.arange(A.shape[0]), (A > 0).sum(1))
    Zx = (X - X[ends].mean(0)) / X[ends].std(0)
    neigh = W @ Zx                                   # (n, D): weighted mean neighbour value
    local_prod = (Zx * neigh)                        # (n, D): zeta_i * mean_j zeta_j
    r = np.zeros(A.shape[0])
    for a in restarts:
        R = ppr_matrix(A, a)                         # (n, n): row l = w_l
        r += (R @ local_prod).mean(1)
    return -r / len(restarts)


# ----------------------------------------------------------------------------- kernel controls
def kernel_residual_score(D: np.ndarray, At: np.ndarray, times: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Calibrate the heat-kernel scale on distances D, return mean_j |A~_ij - H_s(D_ij)|."""
    s, a, _, _ = heat.calibrate(D, At, times, weights)
    R = heat.residuals(D, At, times, weights, s, a)
    return np.abs(R).sum(1) / (D.shape[0] - 1)


def edge_length_score(D: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Mean distance (under geometry D) from a node to its graph neighbours, normalised by the
    median edge length: high = the node's links contradict the geometry. With D = true geodesic
    distances this is the benchmark's oracle; with D = Phase 1 geodesics it is the Phase-1-only
    edge control (no deformation); with Euclidean latent / attribute distances it tests whether
    the Riemannian metric / the VAE matter."""
    nb = A > 0
    med = np.median(D[np.triu(nb, 1)])
    return (D * nb).sum(1) / np.maximum(nb.sum(1), 1) / med


def eccentricity_score(D: np.ndarray) -> np.ndarray:
    n = D.shape[0]
    L = np.log(np.where(np.eye(n, dtype=bool), 1.0, D))
    return L.sum(1) / (n - 1)


# ----------------------------------------------------------------------------- PyGOD detectors
class _FullBatchLoader:
    """Drop-in for torch_geometric NeighborLoader as used by PyGOD in full-batch mode
    (batch_size = N, all neighbours): yields the whole graph once with
    batch_size = N and n_id = arange(N). Needed because pyg-lib / torch-sparse are not
    available for this torch build; mathematically identical to PyGOD's full batch."""

    def __init__(self, data, num_neighbors=None, batch_size=None, **kw):
        import torch
        self.data = data.clone()
        self.data.batch_size = data.num_nodes
        self.data.n_id = torch.arange(data.num_nodes)

    def __iter__(self):
        yield self.data

    def __len__(self):
        return 1


def _patch_pygod():
    import pygod.detector.base as B
    B.NeighborLoader = _FullBatchLoader


def pygod_scores(A: np.ndarray, X: np.ndarray, seed: int, which=("DOMINANT", "AnomalyDAE", "CoLA"),
                 epochs: int = 100) -> Dict[str, np.ndarray]:
    import torch
    from torch_geometric.data import Data
    from pygod import detector as D
    _patch_pygod()
    torch.manual_seed(seed)
    np.random.seed(seed)
    ei = np.array(np.nonzero(A > 0))
    data = Data(x=torch.tensor(StandardScaler().fit_transform(X), dtype=torch.float32),
                edge_index=torch.tensor(ei, dtype=torch.long))
    out = {}
    for name in which:
        torch.manual_seed(seed)
        det = getattr(D, name)(epoch=epochs, verbose=0, gpu=-1)
        det.fit(data)
        out[name.lower()] = det.decision_score_.detach().cpu().numpy().astype(float) \
            if hasattr(det.decision_score_, "detach") else np.asarray(det.decision_score_, dtype=float)
    return out
