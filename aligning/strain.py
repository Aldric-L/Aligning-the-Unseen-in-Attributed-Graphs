"""
Reading the twist: the strain field of Phase 2 relative to Phase 1, and Phase-1-only competitors
(Methods, "Phase 2" and "Competitors").

Learned strain at a latent point z (coordinate-free quantities):

    R_hat(z) = G1(z)^{-1} G2(z)          G1, G2 = pullback metrics of the Phase 1 / Phase 2 decoders

  generalised eigenvalues l1 >= l2 > 0 (eigh(G2, G1)) and G1-orthonormal eigenvectors v1, v2:
    log-area change   0.5 log(l1 l2)       (defined up to a global constant: the kernel scale is free)
    anisotropy        0.5 log(l1 / l2)     (scale-free)
    major axis        v1                   (direction the twist lengthens most, relative to the rest)

Directions are compared with the truth in TRUE coordinates u: a latent direction v maps to
du = (dz/du)^{-1} v, with dz/du = J_encoder(x) J_immersion(u) (exact for the frozen encoder), and the
angle is measured in the true attribute metric G_attr(u) (the metric in which the truth's eigenvectors
are orthogonal). Errors lie in [0, 90] degrees. A direction drawn uniformly in THAT metric gives 45, but
estimators are not uniform in it (a direction uniform in latent coordinates gives ~76 on pilot seed 900),
so chance is evaluated by rotating the true axis (aggregate_twist.py), never assumed to be 45.

Excess twist: the twist of the observed graph relative to the twist of a null graph,
    R_exc = G2_null^{-1} G2_obs,
removes the part of the deformation that any graph agreeing with the attributes would also produce
(Phase 2 fitting sampling noise, kernel misfit, and the correction of Phase 1's own metric errors when
the null is generated from the true attribute geometry).

Phase-1-only competitor ("link moment tensor"): at node i, with latent displacements dz_ij = z_j - z_i,
    C_i      = sum_{j in N(i)} dz dz^T / deg_i                                (observed links)
    C0_i     = sum_j p_ij dz dz^T / sum_j p_ij,  p = Phase-1-calibrated kernel on D1 (what the links would
               look like if the graph followed the Phase 1 geometry; corrects for density and boundaries)
If links are isotropic in the graph metric, C ~ G_graph^{-1} and C0 ~ G1^{-1}, so
    R_mom = C0 C^{-1}  ~  G1^{-1} G_graph
is a closed-form, training-free estimate of the same strain (the first-order, endpoint-only version of
the twist: no path integration, no smoothing). Optionally smoothed over the k nearest latent neighbours
(log-Euclidean mean).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.linalg as sla
import torch

from .metric import pullback_metric
from .synthetic import ManifoldSpec, immersion


def metric_at(decoder, Z: np.ndarray, kind: str = "expected") -> np.ndarray:
    with torch.no_grad():
        return pullback_metric(decoder, torch.as_tensor(Z, dtype=torch.float64), kind).numpy()


def generalized_eig(Gnum: np.ndarray, Gden: np.ndarray):
    """Per point: eigenvalues (descending) and Gden-orthonormal eigenvectors of Gden^{-1} Gnum."""
    n = Gnum.shape[0]
    L = np.empty((n, 2))
    V = np.empty((n, 2, 2))
    for i in range(n):
        w, v = sla.eigh(Gnum[i], Gden[i])
        L[i] = w[::-1]
        V[i] = v[:, ::-1]
    return L, V


def invariants(L: np.ndarray) -> dict:
    L = np.clip(L, 1e-300, None)
    return dict(area=0.5 * np.log(L[:, 0] * L[:, 1]), aniso=0.5 * np.log(L[:, 0] / L[:, 1]))


def learned_strain(dec1, dec2, Z: np.ndarray, kind: str = "expected") -> dict:
    G1, G2 = metric_at(dec1, Z, kind), metric_at(dec2, Z, kind)
    L, V = generalized_eig(G2, G1)
    out = invariants(L)
    out.update(L=L, major_z=V[:, :, 0], G1=G1, G2=G2)
    return out


def dz_du(encoder, spec: ManifoldSpec, X: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Jacobian of u -> z = mu_enc(f(u)) at the nodes (encoder in eval mode, attribute noise fixed)."""
    from torch.func import jacrev, vmap
    was = encoder.training
    encoder.eval()
    Xt = torch.as_tensor(X, dtype=torch.float64)
    Ut = torch.as_tensor(U, dtype=torch.float64)
    with torch.no_grad():
        Jf = vmap(jacrev(lambda p: immersion(spec, p)))(Ut)                          # (n, D, 2)
        Je = vmap(jacrev(lambda x: encoder(x.unsqueeze(0))[0].squeeze(0)))(Xt)       # (n, 2, D)
    encoder.train(was)
    return (Je @ Jf).numpy()                                                         # (n, 2, 2)


def axis_error_deg(v_z: np.ndarray, J: np.ndarray, a_u: np.ndarray, G_attr: np.ndarray) -> np.ndarray:
    """Angle (degrees, in [0, 90]) between latent directions v_z mapped to u-coordinates and true axes
    a_u, measured in the true attribute metric. NaN where the truth is undefined."""
    v_u = np.linalg.solve(J, v_z[..., None])[..., 0]
    num = np.abs(np.einsum("ni,nij,nj->n", v_u, G_attr, a_u))
    den = np.sqrt(np.einsum("ni,nij,nj->n", v_u, G_attr, v_u) * np.einsum("ni,nij,nj->n", a_u, G_attr, a_u))
    with np.errstate(invalid="ignore"):
        return np.degrees(np.arccos(np.clip(num / den, 0.0, 1.0)))


def _log_spd(M):
    w, v = np.linalg.eigh(M)
    return (v * np.log(np.clip(w, 1e-300, None))[..., None, :]) @ np.swapaxes(v, -1, -2)


def _exp_sym(M):
    w, v = np.linalg.eigh(M)
    return (v * np.exp(w)[..., None, :]) @ np.swapaxes(v, -1, -2)


def link_moments(Z: np.ndarray, W: np.ndarray) -> np.ndarray:
    """C_i = sum_j W_ij dz_ij dz_ij^T / sum_j W_ij (W = adjacency indicator, or link probabilities)."""
    dZ = Z[None, :, :] - Z[:, None, :]                                    # (n, n, 2): z_j - z_i
    W = W.copy()
    np.fill_diagonal(W, 0.0)
    return np.einsum("ij,ija,ijb->iab", W, dZ, dZ) / np.maximum(W.sum(1), 1e-300)[:, None, None]


def link_moment_strain(Z: np.ndarray, A: np.ndarray, P1: np.ndarray, G1: np.ndarray,
                       smooth_k: int = 0, C0: Optional[np.ndarray] = None) -> dict:
    """Training-free strain estimate R_mom = C0 C^{-1} (see module docstring), returned in the same
    form as learned_strain (eigenvalues of G1^{-1} G_hat with G_hat = G1 R_mom, G1-orthonormal axes).
    Nodes with fewer than 2 links get NaN. smooth_k > 0: log-Euclidean mean of the G1-whitened
    strain over the node and its smooth_k nearest latent neighbours. If C0 is given (e.g. the link
    moments of a null graph: the excess version of the estimator), P1 is ignored."""
    n = Z.shape[0]
    nb = (A > 0).astype(float)
    deg = nb.sum(1)
    C = link_moments(Z, nb)
    if C0 is None:
        C0 = link_moments(Z, P1)
    ok = (deg >= 2) & (np.linalg.det(C0) > 0)
    R = np.full((n, 2, 2), np.nan)
    for i in np.nonzero(ok)[0]:
        try:
            R[i] = C0[i] @ np.linalg.inv(C[i])
        except np.linalg.LinAlgError:
            pass
    ok &= np.all(np.isfinite(R.reshape(n, -1)), 1)
    # symmetric representation in a G1-orthonormal frame: S = W R W^{-1}, W = G1^{1/2}
    W = np.array([sla.sqrtm(G1[i]).real for i in range(n)])
    S = np.full((n, 2, 2), np.nan)
    for i in np.nonzero(ok)[0]:
        Si = W[i] @ R[i] @ np.linalg.inv(W[i])
        S[i] = 0.5 * (Si + Si.T)
    if smooth_k:
        from scipy.spatial import cKDTree
        idx = np.nonzero(ok)[0]
        logS = np.zeros_like(S)
        logS[idx] = _log_spd(_ensure_spd(S[idx]))
        _, nn = cKDTree(Z[idx]).query(Z[idx], k=min(smooth_k + 1, len(idx)))
        sm = logS[idx][nn].mean(1)
        S = np.full_like(S, np.nan)
        S[idx] = _exp_sym(sm)
    L = np.full((n, 2), np.nan)
    major = np.full((n, 2), np.nan)
    for i in np.nonzero(ok)[0]:
        w, v = np.linalg.eigh(_ensure_spd(S[i:i + 1])[0])
        L[i] = w[::-1]
        major[i] = np.linalg.solve(W[i], v[:, ::-1][:, 0])          # back to latent coordinates
    out = invariants(np.where(np.isfinite(L), L, 1.0))
    for k in out:
        out[k] = np.where(ok, out[k], np.nan)
    out.update(L=L, major_z=major)
    return out


def _ensure_spd(S, floor=1e-12):
    S = 0.5 * (S + np.swapaxes(S, -1, -2))
    w, v = np.linalg.eigh(S)
    return (v * np.clip(w, floor, None)[..., None, :]) @ np.swapaxes(v, -1, -2)


def phase1_link_probabilities(D1: np.ndarray, At: np.ndarray, times: np.ndarray, weights: np.ndarray,
                              scale: float, amp: float) -> np.ndarray:
    t = times[None, None, :]
    w = weights[None, None, :]
    H = amp * (w * np.exp(-((scale * D1)[..., None] ** 2) / (4 * t))).sum(-1)
    np.fill_diagonal(H, 0.0)
    return H


def bootstrap_null_graph(A: np.ndarray, D1: np.ndarray, rng: np.random.Generator, target: str = "max") -> dict:
    """A graph drawn from the Phase 1 latent-space model fitted to the observed graph A: the graph one
    would observe if links followed the Phase 1 (attribute) geometry exactly. Model: single-scale kernel
    H = a exp(-(s d1)^2 / 4t), (s, a) fitted by cross-entropy on (D1, A~) as in Phase 2 (K = 1). Since
    H models the normalised weight A~, a pair is linked with probability p = min(1, H / w_bar)
    (w_bar = mean normalised weight of the observed links) and given a weight resampled from the observed
    positive weights, so that E[A~_null] = H wherever p < 1. The result is rescaled to the observed
    maximum weight (the target normalisation divides by it). Independent random numbers: nothing of the
    observed graph is reused except its weight distribution and the fitted (s, a)."""
    from . import heat
    n = A.shape[0]
    At = heat.normalize_target(A, target)
    np.fill_diagonal(At, 0.0)
    t1 = heat.heat_times(A, 1, 4.0)
    w1 = heat.kernel_weights(t1, 2)              # K = 1: the single weight is 1 whatever the dimension
    s0, a0, _, _ = heat.calibrate(D1, At, t1, w1)
    s0, a0, _ = heat.calibrate_bce(D1, At, t1, w1, init=(s0, a0))
    H = phase1_link_probabilities(D1, At, t1, w1, s0, a0)
    iu = np.triu_indices(n, 1)
    pos = A[iu][A[iu] > 0]
    wbar = float(At[iu][A[iu] > 0].mean())
    p = np.minimum(1.0, H[iu] / wbar)
    e = rng.random(p.shape[0]) < p
    B = np.zeros((n, n))
    B[iu[0][e], iu[1][e]] = rng.choice(pos, size=int(e.sum()), replace=True)
    B = B + B.T
    if B.max() > 0:
        B *= A.max() / B.max()
    return dict(A=B, scale=s0, amp=a0, wbar=wbar, frac_p_clipped=float((H[iu] / wbar > 1).mean()))


def graph_path_lengths(A: np.ndarray, D1: np.ndarray) -> dict:
    """Graph-only and graph-attribute distances: hop count, and shortest path with each link
    measured by its Phase 1 geodesic length (a training-free fusion of the two geometries)."""
    from scipy.sparse.csgraph import shortest_path
    nb = A > 0
    hop = shortest_path(nb.astype(float), directed=False, unweighted=True)
    W = np.where(nb, D1, 0.0)
    sp1 = shortest_path(W, directed=False)
    return dict(hop=hop, sp_d1=sp1)
