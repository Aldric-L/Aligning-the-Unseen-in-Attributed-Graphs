"""
Deterministic synthetic attributed graphs.

Everything is seeded and the latent-space graph model is explicit:

  manifold   z ~ N(0, I_2);  x = f(z) + N(0, noise^2 I_D),
             f = soft swiss roll (r0=1.5, spread=0.5, twist=2) in R^3, extended to R^D by
             f_k(a_k u + b_k v), f_k in {sin, cos, tanh, exp(-x^2/10), x/(1+x^2)}, (a_k, b_k) ~ N(0,1);
  homophily  s_ij = exp(-d_ij^2 / (bw * median d^2)), s < 0.2 set to 0,
             d = TRUE geodesic distance on (Z, pullback metric of f), computed with the
             hybrid estimator on the exact metric (rel. error ~0.2%);
  graph      P(A_ij=1) = min(1, c theta_i theta_j [(1-h) s'_ij + h mean(s)]) (theta = 1 by default),
             c calibrated to a target expected mean degree; weights = compatibility of
             the realised edge (random-component edges get weights drawn from the size-biased
             compatibility distribution); non-giant components are attached to the giant
             component through their most compatible pair (connected graphs);
  perturbation  block B (|B| = m): "disk" = m nearest (true geodesic) neighbours of a
             uniformly drawn centre, "band" = the m largest v, or "scatter" = m random nodes;
             pi = a uniform derangement of B; "block_shuffle": s'_ab = s_{pi(a) pi(b)} for a,b in B;
             "node_relabel": node a in B takes the full compatibility profile of pi(a).
             Intensity alpha: each affected pair independently uses the perturbed profile with
             probability alpha (alpha=1: exact relabelling; alpha=0: no perturbation). Labels: y = 1 on B.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np
import torch

from .metric import GridSpec, MetricField
from .geodesics import GridGeodesics

_FUNCS_T = [torch.sin, torch.cos, torch.tanh,
            lambda x: torch.exp(-0.1 * x ** 2),
            lambda x: x / (1 + x ** 2)]
FUNC_NAMES = ["sin", "cos", "tanh", "gauss", "rational"]


@dataclass
class ManifoldSpec:
    D: int = 20
    r0: float = 1.5
    spread: float = 0.5
    twist: float = 2.0
    ext: list = field(default_factory=list)  # [(func_id, a, b)] for coordinates 3..D-1

    @staticmethod
    def sample(rng: np.random.Generator, D: int = 20) -> "ManifoldSpec":
        ext = []
        for _ in range(3, D):
            a, b = rng.normal(size=2)
            ext.append((int(rng.integers(0, len(_FUNCS_T))), float(a), float(b)))
        return ManifoldSpec(D=D, ext=ext)


def immersion(spec: ManifoldSpec, uv: torch.Tensor) -> torch.Tensor:
    u, v = uv[..., 0], uv[..., 1]
    r = spec.r0 + torch.nn.functional.softplus(spec.spread * u)
    th = spec.twist * v
    cols = [r * torch.cos(th), r * torch.sin(th), th]
    for fid, a, b in spec.ext:
        cols.append(_FUNCS_T[fid](a * u + b * v))
    return torch.stack(cols, -1)


def true_metric_fn(spec: ManifoldSpec):
    from torch.func import jacrev, vmap
    J = vmap(jacrev(lambda p: immersion(spec, p)))

    def fn(pts: torch.Tensor) -> torch.Tensor:
        Jp = J(pts)  # (n, D, 2)
        return Jp.transpose(-1, -2) @ Jp
    return fn


def true_geodesic_distances(spec: ManifoldSpec, Z: np.ndarray, P: int = 200, neighbourhood: int = 32) -> np.ndarray:
    gs = GridSpec.from_points(Z, P, margin=0.05)
    field_ = MetricField.from_function(true_metric_fn(spec), gs)
    with torch.no_grad():
        return GridGeodesics(gs, Z, neighbourhood=neighbourhood).hybrid_distances(field_).numpy()


@dataclass
class GraphConfig:
    bandwidth: float = 0.3        # s = exp(-d^2 / (bandwidth * median d^2))
    threshold: float = 0.2        # s < threshold -> 0
    mean_degree: float = 22.0
    theta_sigma: float = 0.0      # lognormal sd of degree propensities (0 = homogeneous)
    homophily_mix: float = 0.0    # h


@dataclass
class PerturbConfig:
    region: str = "disk"          # "disk" | "band" | "scatter"
    m: int = 70
    intensity: float = 1.0        # alpha; 0 = null case
    kind: str = "block_shuffle"   # "block_shuffle" (paper's operation, done exactly) | "node_relabel"


@dataclass
class SyntheticConfig:
    n: int = 500
    D: int = 20
    noise_std: float = 0.01
    graph: GraphConfig = field(default_factory=GraphConfig)
    perturb: PerturbConfig = field(default_factory=PerturbConfig)


def _calibrate_c(base: np.ndarray, target_mean_degree: float) -> float:
    n = base.shape[0]
    iu = np.triu_indices(n, 1)
    b = base[iu]
    lo, hi = 0.0, 1.0
    while (2 * np.minimum(1.0, hi * b).sum() / n) < target_mean_degree:
        hi *= 2
        if hi > 1e8:
            raise RuntimeError("cannot reach target degree")
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if 2 * np.minimum(1.0, mid * b).sum() / n < target_mean_degree:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def generate(cfg: SyntheticConfig, seed: int, d_true: Optional[np.ndarray] = None,
             manifold: Optional[ManifoldSpec] = None, Z: Optional[np.ndarray] = None) -> dict:
    """Generate one synthetic attributed graph. If (manifold, Z, d_true) are given, they are
    reused (lets several perturbation conditions share the same attribute manifold)."""
    rng = np.random.default_rng(seed)
    if manifold is None:
        manifold = ManifoldSpec.sample(rng, cfg.D)
    if Z is None:
        Z = rng.normal(size=(cfg.n, 2))
    n = Z.shape[0]
    with torch.no_grad():
        X = immersion(manifold, torch.as_tensor(Z, dtype=torch.float64)).numpy()
    X = X + rng.normal(scale=cfg.noise_std, size=X.shape)
    if d_true is None:
        d_true = true_geodesic_distances(manifold, Z)
    iu = np.triu_indices(n, 1)
    g = cfg.graph
    s = np.exp(-d_true ** 2 / (g.bandwidth * np.median(d_true[iu] ** 2)))
    s[s < g.threshold] = 0.0
    np.fill_diagonal(s, 0.0)

    # perturbation block
    p = cfg.perturb
    if p.region == "disk":
        centre = int(rng.integers(n))
        block = np.argsort(d_true[centre])[: p.m]
    elif p.region == "band":
        block = np.argsort(-Z[:, 1])[: p.m]
    elif p.region == "scatter":
        block = rng.choice(n, size=p.m, replace=False)
    else:
        raise ValueError(p.region)
    block = np.sort(block)
    perm = rng.permutation(block)          # drawn in every condition (paired null/perturbed design)
    while len(block) > 1 and np.any(perm == block):   # derangement: every labelled node is really relabelled
        perm = rng.permutation(block)
    s_used = s.copy()
    idx = np.arange(n)                     # relabelling map (identity outside the block)
    if p.kind == "node_relabel":
        idx[block] = perm
    # Intensity alpha in (0,1) mixes the GENERATIVE PROCESSES, not the kernels: each
    # affected unordered pair draws, with probability alpha, the perturbed profile, which then sets both
    # its edge probability and its weight. Mixing kernel VALUES ((1-a)s + a s_pi) scaled the weights of
    # block edges by a or 1-a and leaked the labels into A. The draws use an independent stream so the
    # main stream (and the null/perturbed pairing) is unchanged.
    mix = None
    if 0 < p.intensity < 1:
        mrng = np.random.default_rng([seed, 7_777])
        mix = np.triu(mrng.random((n, n)) < p.intensity, 1)
        mix = mix | mix.T
    if p.intensity > 0:
        if p.kind == "block_shuffle":
            # within-block relabelling: s'_ab = s_{pi(a) pi(b)} for a, b in B (exact multiset at alpha = 1)
            sub = s[np.ix_(block, block)]
            sub_pi = s[np.ix_(perm, perm)]
            new_sub = sub_pi if mix is None else np.where(mix[np.ix_(block, block)], sub_pi, sub)
            np.fill_diagonal(new_sub, 0.0)
            s_used[np.ix_(block, block)] = new_sub
        elif p.kind == "node_relabel":
            # full-profile relabelling: node a in B connects like node pi(a) (all its compatibilities,
            # inside and outside B) while keeping its own attributes. At alpha = 1 the graph distribution
            # is invariant (a relabelled sample): graph-only statistics are blind by construction.
            s_pi = s[np.ix_(idx, idx)]
            s_used = s_pi if mix is None else np.where(mix, s_pi, s)
            s_used = s_used.copy()
            np.fill_diagonal(s_used, 0.0)
        else:
            raise ValueError(p.kind)
    labels = np.zeros(n, dtype=int)
    labels[block] = 1

    # latent-space graph (homogeneous propensities unless theta_sigma > 0)
    theta = np.exp(rng.normal(scale=g.theta_sigma, size=n)) if g.theta_sigma > 0 else np.ones(n)
    theta = theta / theta.mean()
    sbar = s_used[iu].mean()
    base = theta[:, None] * theta[None, :] * ((1 - g.homophily_mix) * s_used + g.homophily_mix * sbar)
    np.fill_diagonal(base, 0.0)
    c = _calibrate_c(base, g.mean_degree)
    P_edge = np.minimum(1.0, c * base)
    U = rng.random(size=iu[0].shape[0])
    edge = U < P_edge[iu]
    # weights: compatibility for homophilic edges; random-component edges get a weight drawn from the
    # SIZE-BIASED compatibility distribution (homophilic edges are realised with probability ∝ s, so their
    # weights are size-biased too): weights do not reveal which component produced an edge
    hom_part = (1 - g.homophily_mix) * s_used[iu]
    tot = hom_part + g.homophily_mix * sbar
    from_hom = rng.random(size=edge.shape[0]) < np.where(tot > 0, hom_part / np.maximum(tot, 1e-300), 0.0)
    pos_vals = s_used[iu][s_used[iu] > 0]
    w = np.where(from_hom, s_used[iu], rng.choice(pos_vals, size=edge.shape[0], p=pos_vals / pos_vals.sum()))
    A = np.zeros((n, n))
    A[iu[0][edge], iu[1][edge]] = w[edge]
    A = A + A.T
    # connectivity repair: attach every non-giant component to the giant component through
    # its most compatible pair (fallback: closest pair in the RELABELLED geometry)
    from scipy.sparse.csgraph import connected_components
    d_eff = d_true[np.ix_(idx, idx)] if (p.kind == "node_relabel" and p.intensity >= 1) else d_true
    n_comp, comp = connected_components(A > 0, directed=False)
    n_repairs = 0
    if n_comp > 1:
        giant = np.argmax(np.bincount(comp))
        G = np.where(comp == giant)[0]
        for cid in range(n_comp):
            if cid == giant:
                continue
            Cn = np.where(comp == cid)[0]
            sub = s_used[np.ix_(Cn, G)]
            if sub.max() > 0:
                a_, b_ = np.unravel_index(np.argmax(sub), sub.shape)
                wij = sub[a_, b_]
            else:
                a_, b_ = np.unravel_index(np.argmin(d_eff[np.ix_(Cn, G)]), (len(Cn), len(G)))
                wij = float(np.median(pos_vals))
            A[Cn[a_], G[b_]] = A[G[b_], Cn[a_]] = wij
            n_repairs += 1
    return dict(X=X, Z=Z, A=A, labels=labels, block=block, compat=s, compat_used=s_used,
                d_true=d_true, manifold=manifold, c=c, n_isolated_fixed=int(n_repairs), n_components_before=int(n_comp),
                config=asdict(cfg), seed=seed)
