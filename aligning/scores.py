"""
Node-level distortion scores of the pair distortion.

Ratio score:

    delta_ij = log d2_ij - log d1_ij                      (i < j)
    Z_ij     = 0.6745 (delta_ij - median(delta)) / MAD(delta)
    S_i      = mean_{j != i} |Z_ij|

  * log-ratio: invariant to any global rescaling of either phase (a uniform inflation of the Phase 2
    manifold gives delta = const, hence Z = 0). A score built on |d2 - d1| is not: under a uniform
    inflation by c, log|d2 - d1| = log(c-1) + log d1, a Phase 1 eccentricity that ignores the graph;
  * the sign has a meaning: Z > 0 pair pushed apart by the graph, Z < 0 pair pulled together;
  * node aggregate = mean absolute deformation of the pairs incident to i.

Difference score (legacy_scores, kept for comparison):
    delta_ij = log|d2_ij - d1_ij|, same robust z, S_i = sum_{j != i} Z_ij.
"""
from __future__ import annotations

import numpy as np

MAD_CONST = 0.6744897501960817  # Phi^{-1}(3/4)


def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        return np.zeros_like(x)
    return MAD_CONST * (x - med) / mad


def _pairs_to_matrix(vals: np.ndarray, n: int) -> np.ndarray:
    M = np.zeros((n, n))
    iu = np.triu_indices(n, 1)
    M[iu] = vals
    return M + M.T


def ratio_scores(D1: np.ndarray, D2: np.ndarray):
    """Returns (Z matrix, node score S_i = mean_j |Z_ij|, signed node score mean_j Z_ij)."""
    n = D1.shape[0]
    iu = np.triu_indices(n, 1)
    delta = np.log(D2[iu]) - np.log(D1[iu])
    Z = _pairs_to_matrix(robust_z(delta), n)
    S = np.abs(Z).sum(1) / (n - 1)
    S_signed = Z.sum(1) / (n - 1)
    return Z, S, S_signed


def edge_restricted(Z: np.ndarray, A: np.ndarray):
    """Aggregate pair deformations over the observed edges of each node only:
    S^E_i = mean_{j in N(i)} |Z_ij| and signed mean_{j in N(i)} Z_ij (negative = edges contracted).
    Why: the misalignment injected by node relabelling lives on
    edges; the all-pairs mean dilutes it ~20x (mean degree 22 of 499 pairs)."""
    nb = A > 0
    deg = nb.sum(1)
    k = np.maximum(deg, 1)
    S, Ss = (np.abs(Z) * nb).sum(1) / k, (Z * nb).sum(1) / k
    S = np.where(deg > 0, S, np.nan)          # undefined for isolated nodes
    Ss = np.where(deg > 0, Ss, np.nan)
    return S, Ss


def legacy_scores(D1: np.ndarray, D2: np.ndarray, eps: float = 1e-12):
    """Paper Eq. (5) as written: delta = log|d2-d1|, S_i = sum_{j!=i} Z_ij."""
    n = D1.shape[0]
    iu = np.triu_indices(n, 1)
    delta = np.log(np.abs(D2[iu] - D1[iu]) + eps)
    Z = _pairs_to_matrix(robust_z(delta), n)
    return Z, Z.sum(1)


def label_free_flags(score: np.ndarray, threshold: float = 3.5) -> np.ndarray:
    """Iglewicz & Hoaglin (1993) rule: flag nodes whose modified z-score exceeds 3.5.
    Applied identically to every method's continuous node score (high = anomalous)."""
    return robust_z(score) > threshold
