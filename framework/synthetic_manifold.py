# synthetic_manifold.py
# Purpose: Generate N points on a smooth 2D manifold immersed in R^D,
#          with an explicit closed-form immersion and a Gaussian latent prior.
#          Manifolds provided below are all homeomorphic to R^2 (no holes),
#          making them friendly to VAEs with a Gaussian prior.

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Literal, Optional
import json
from pathlib import Path

ManifoldName = Literal["paraboloid", "helicoid", "catenoid_like", "soft_swiss"]


@dataclass
class ManifoldConfig:
    name: ManifoldName = "helicoid"
    D: int = 8                         # ambient dimension (>= 3)
    noise_std: float = 0.0             # optional ambient Gaussian noise
    prior_scale: float = 1.0           # scale for N(0, prior_scale^2 I_2)
    seed: Optional[int] = None


# -------------------------
# Core Immersion Definitions
# -------------------------
# Each immersion is a C^∞ map f: R^2 -> R^D with full rank everywhere (immersion),
# and with image homeomorphic to R^2 (no holes).
# --- EXPORTS & SCALABLE EXTENSIONS -------------------------------------------

def _extend_dims(base_xyz: np.ndarray, u: np.ndarray, v: np.ndarray, D: int, seed: int | None = None) -> np.ndarray:
    """
    Extend the base immersion in R^3 to R^D by generating smooth nonlinear
    functions of (u,v). Works for arbitrary D >= 3.
    """
    n = u.shape[0]
    if D <= 3:
        return base_xyz

    rng = np.random.default_rng(seed)
    out = np.zeros((n, D))
    out[:, :3] = base_xyz

    k = 3
    while k < D:
        a, b = rng.normal(size=2)
        combo = a * u + b * v
        funcs = [
            np.sin, np.cos, np.tanh,
            lambda x: np.exp(-0.1 * x**2),
            lambda x: x / (1 + x**2),
        ]
        f = funcs[rng.integers(0, len(funcs))]
        out[:, k] = f(combo)
        k += 1
    return out


def export_immersion(name: ManifoldName):
    """
    Return the immersion callable f(u,v,D[,seed]) and a true-metric helper.
    The metric is G = J^T J computed via finite differences.
    """
    if name not in IMMERSIONS:
        raise ValueError(f"Unknown manifold '{name}'. Available: {list(IMMERSIONS.keys())}")
    f = IMMERSIONS[name]

    def true_metric(u: float, v: float, D: int, eps: float = 1e-5, seed: int | None = None) -> np.ndarray:
        f0 = f(np.array([u]), np.array([v]), D) if seed is None else f(np.array([u]), np.array([v]), D)
        f0 = f0[0]
        fu = (f(np.array([u + eps]), np.array([v]), D)[0] - f0) / eps
        fv = (f(np.array([u]), np.array([v + eps]), D)[0] - f0) / eps
        J = np.stack([fu, fv], axis=1)  # D x 2
        return J.T @ J  # 2 x 2

    return f, true_metric


# --- REPARAM-INVARIANT BENCHMARKS --------------------------------------------

def _knn_indices(X: np.ndarray, k: int) -> list[np.ndarray]:
    """
    Return for each row i the indices of its k nearest neighbors in Euclidean space of X.
    """
    from numpy.linalg import norm
    n = X.shape[0]
    idxs = []
    for i in range(n):
        d = norm(X - X[i], axis=1)
        nn = np.argsort(d)[1:k+1]
        idxs.append(nn)
    return idxs


def estimate_local_jacobian(u_samples: np.ndarray, z_samples: np.ndarray, k: int = 20) -> np.ndarray:
    """
    Estimate the Jacobian Dφ(u,v) at each sample by local linear regression on k-NN in the *true* latent space.
    We fit z - z_i ≈ A_i ( [u - u_i, v - v_i]^T ) and return A_i (2x2).
    Inputs
      - u_samples: (N,2) true seeds (u,v)
      - z_samples: (N,2) VAE latent codes (z1,z2)
    Output
      - Jphi: (N,2,2) estimated Jacobians
    """
    N = u_samples.shape[0]
    idxs = _knn_indices(u_samples, k=k)
    Jphi = np.zeros((N, 2, 2))
    for i in range(N):
        nbr = idxs[i]
        U = u_samples[nbr] - u_samples[i]        # (k,2)
        Z = z_samples[nbr] - z_samples[i]        # (k,2)
        # Solve least squares: U @ A^T ≈ Z  -> A^T = argmin ||U A^T - Z||
        # Using pseudoinverse for stability
        A_T, *_ = np.linalg.lstsq(U, Z, rcond=None)
        A = A_T.T
        Jphi[i] = A
    return Jphi


def metric_pullback_alignment_error(
    seeds_uv: np.ndarray,
    codes_z: np.ndarray,
    true_metric_func: Callable[[float,float,int], np.ndarray],
    vae_metric_func: Callable[[np.ndarray], np.ndarray],
    D: int,
    k: int = 20,
) -> dict:
    """
    Reparam-invariant metric test:
      For each i, compare G_true(u_i) with (Dφ_i)^T G_vae(z_i) Dφ_i, where Dφ_i is estimated locally.
    Returns summary stats and the per-point errors.
    """
    N = seeds_uv.shape[0]
    Jphi = estimate_local_jacobian(seeds_uv, codes_z, k=k)  # (N,2,2)

    errs = np.zeros(N)
    vols_true = np.zeros(N)
    vols_vae = np.zeros(N)

    for i in range(N):
        u, v = seeds_uv[i]
        z = codes_z[i]
        Gt = true_metric_func(float(u), float(v), D)              # 2x2
        Gv = vae_metric_func(z)                                   # 2x2
        A  = Jphi[i]                                              # 2x2
        Gv_pull = A.T @ Gv @ A
        # relative Frobenius error (scale-invariant alternative below)
        num = np.linalg.norm(Gt - Gv_pull, ord='fro')
        den = max(1e-12, np.linalg.norm(Gt, ord='fro'))
        errs[i] = num / den

        # volume elements (reparam invariant)
        vols_true[i] = np.sqrt(max(1e-12, np.linalg.det(Gt)))
        vols_vae[i]  = np.sqrt(max(1e-12, np.linalg.det(Gv)))

    summary = {
        "mean_rel_error": float(np.mean(errs)),
        "median_rel_error": float(np.median(errs)),
        "p90_rel_error": float(np.quantile(errs, 0.90)),
        "volume_corr": float(np.corrcoef(vols_true, vols_vae)[0,1]),
        "errors": errs,
        "vol_true": vols_true,
        "vol_vae": vols_vae,
        "Jphi": Jphi,
    }
    return summary


# --- GEODESIC GRAPH BENCHMARK ------------------------------------------------

def _edge_length(metric_func: Callable[[np.ndarray], np.ndarray], p: np.ndarray, q: np.ndarray) -> float:
    """
    Small-step Riemannian length using midpoint rule:
        L(p,q) ≈ sqrt( (q-p)^T G(mid) (q-p) )
    where G(mid) is the metric at (p+q)/2.
    """
    mid = 0.5 * (p + q)
    G = metric_func(mid)
    d = (q - p)
    return float(np.sqrt(max(1e-12, d @ (G @ d))))


def geodesic_graph_distances(
    points: np.ndarray,
    metric_func: Callable[[np.ndarray], np.ndarray],
    k: int = 12,
) -> np.ndarray:
    """
    Approximate pairwise geodesic distances in a 2D latent domain by a k-NN graph and Dijkstra.
    points: (N,2) in the coordinate system where metric_func is defined (u,v or z)
    metric_func: returns 2x2 metric at a given point
    """
    from heapq import heappush, heappop
    from numpy.linalg import norm

    N = points.shape[0]
    # Build kNN adjacency with Riemannian edge weights
    dE = np.sqrt(((points[:,None,:]-points[None,:,:])**2).sum(axis=2))
    nbrs = np.argsort(dE, axis=1)[:, 1:k+1]

    # adjacency list with weights
    adj = [[] for _ in range(N)]
    for i in range(N):
        for j in nbrs[i]:
            w = _edge_length(metric_func, points[i], points[j])
            adj[i].append((j, w))
            adj[j].append((i, w))  # undirected

    # Dijkstra from each node (can be subsampled in practice)
    INF = 1e18
    dist = np.full((N, N), INF)

    for src in range(N):
        dist[src, src] = 0.0
        pq = [(0.0, src)]
        visited = np.zeros(N, dtype=bool)
        while pq:
            dcur, u = heappop(pq)
            if visited[u]: 
                continue
            visited[u] = True
            for v, w in adj[u]:
                nd = dcur + w
                if nd < dist[src, v]:
                    dist[src, v] = nd
                    heappush(pq, (nd, v))
    return dist


def geodesic_benchmark(
    seeds_uv: np.ndarray,
    codes_z: np.ndarray,
    true_metric_func: Callable[[float,float,int], np.ndarray],
    vae_metric_func: Callable[[np.ndarray], np.ndarray],
    D: int,
    k_graph: int = 12,
    subsample: int | None = 300,
) -> dict:
    """
    Reparam-invariant geodesic test via k-NN graph distances.
    We compare geodesic distance matrices computed in the true coordinates (u,v) and in the VAE coordinates z.

    Returns correlation between vectorized upper triangles, plus raw distances.
    """
    N = seeds_uv.shape[0]
    idx = np.arange(N)
    if subsample is not None and subsample < N:
        rng = np.random.default_rng(0)
        idx = rng.choice(N, size=subsample, replace=False)

    U = seeds_uv[idx]
    Z = codes_z[idx]

    # Wrap metric funcs for vector input
    def Gtrue_at(pt: np.ndarray) -> np.ndarray:
        return true_metric_func(float(pt[0]), float(pt[1]), D)

    def Gvae_at(pt: np.ndarray) -> np.ndarray:
        return vae_metric_func(pt)

    D_true = geodesic_graph_distances(U, Gtrue_at, k=k_graph)
    D_vae  = geodesic_graph_distances(Z, Gvae_at,  k=k_graph)

    # Compare upper-triangle entries
    iu, ju = np.triu_indices(D_true.shape[0], k=1)
    v1, v2 = D_true[iu, ju], D_vae[iu, ju]
    corr = np.corrcoef(v1, v2)[0,1]

    return {
        "pairwise_corr": float(corr),
        "D_true": D_true,
        "D_vae": D_vae,
        "indices": idx,
    }

def get_geodesic_distances(
    seeds_uv: np.ndarray,
    true_metric_func: Callable[[float,float,int], np.ndarray],
    D: int,
    k_graph: int = 12,
    subsample: int | None = 300):
    """
    We compute the geodesic distances matrix in the true coordinates (u,v)
    """
    N = seeds_uv.shape[0]
    idx = np.arange(N)
    if subsample is not None and subsample < N:
        rng = np.random.default_rng(0)
        idx = rng.choice(N, size=subsample, replace=False)

    U = seeds_uv[idx]

    # Wrap metric funcs for vector input
    def Gtrue_at(pt: np.ndarray) -> np.ndarray:
        return true_metric_func(float(pt[0]), float(pt[1]), D)

    D_true = geodesic_graph_distances(U, Gtrue_at, k=k_graph)

    return D_true



def immersion_paraboloid(u: np.ndarray, v: np.ndarray, D: int, a: float = 0.5) -> np.ndarray:
    """
    Paraboloid: f(u, v) = (u, v, (u^2 + v^2)/(2a)), a>0.
    Smooth, injective, rank 2 everywhere, image ≅ R^2 (no holes).
    """
    z = (u**2 + v**2) / (2.0 * a)
    base = np.stack([u, v, z], axis=1)
    return _extend_dims(base, u, v, D)


def immersion_helicoid(u: np.ndarray, v: np.ndarray, D: int, pitch: float = 0.5) -> np.ndarray:
    """
    Helicoid: f(r, θ) = (r cos θ, r sin θ, pitch * θ), (r, θ) ∈ R^2.
    Rank 2 everywhere (including r=0); image ≅ R^2 (no holes).
    """
    x = u * np.cos(v)
    y = u * np.sin(v)
    z = pitch * v
    base = np.stack([x, y, z], axis=1)
    return _extend_dims(base, u, v, D)


def immersion_catenoid_like(u: np.ndarray, v: np.ndarray, D: int, a: float = 1.0) -> np.ndarray:
    """
    "Unwrapped" catenoid-like sheet that’s diffeomorphic to R^2:
    Use (u, v) -> (cosh(u) cos v, cosh(u) sin v, a*u). Unlike the true catenoid
    (a minimal surface with neck), the domain here is all of R^2 and image ≅ R^2.
    Rank 2 everywhere; no holes.
    """
    ch = np.cosh(u)
    x = ch * np.cos(v)
    y = ch * np.sin(v)
    z = a * u
    base = np.stack([x, y, z], axis=1)
    return _extend_dims(base, u, v, D)


def immersion_soft_swiss(u: np.ndarray, v: np.ndarray, D: int,
                         r0: float = 1.5, spread: float = 0.5, twist: float = 2.0) -> np.ndarray:
    """
    Soft, infinite 'swiss-roll-like' sheet with NO holes and NO hard boundary:
    r(u) = r0 + softplus(spread*u), θ(v) = twist * v,
    f(u, v) = (r(u) cos θ, r(u) sin θ, θ).
    Homeomorphic to R^2 (since u,v ∈ R), smooth, rank 2 everywhere.
    """
    # Smooth strictly positive radius
    r = r0 + np.log1p(np.exp(spread * u))     # softplus
    theta = twist * v
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = theta
    base = np.stack([x, y, z], axis=1)
    return _extend_dims(base, u, v, D)


# Registry
IMMERSIONS: Dict[ManifoldName, Callable[..., np.ndarray]] = {
    "paraboloid": immersion_paraboloid,
    "helicoid": immersion_helicoid,
    "catenoid_like": immersion_catenoid_like,
    "soft_swiss": immersion_soft_swiss,
}


# -------------------------
# Public API
# -------------------------

def sample_latent(n: int, scale: float = 1.0, seed: Optional[int] = None) -> np.ndarray:
    """
    Sample Z ~ N(0, scale^2 I_2). This is exactly the VAE's standard Gaussian prior (scaled).
    """
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=scale, size=(n, 2))


def generate_manifold_data(
    n: int,
    config: ManifoldConfig = ManifoldConfig(),
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate N samples on a chosen 2D manifold immersed in R^D.

    Returns
    -------
    X : np.ndarray, shape (n, D)
        Ambient coordinates in R^D.
    Z : np.ndarray, shape (n, 2)
        Latent coordinates drawn from a Gaussian prior.
    info : dict
        Metadata including the chosen manifold and parameters.
    """
    if config.D < 3:
        raise ValueError("Ambient dimension D must be at least 3 for a proper immersion.")

    # Sample Gaussian latent (VAE-friendly)
    Z = sample_latent(n, scale=config.prior_scale, seed=config.seed)
    u, v = Z[:, 0], Z[:, 1]

    # Build immersion
    if config.name not in IMMERSIONS:
        raise ValueError(f"Unknown manifold '{config.name}'. Available: {list(IMMERSIONS.keys())}")

    X = IMMERSIONS[config.name](u, v, config.D)

    # Optional ambient noise (does NOT change the underlying manifold, just perturbs samples)
    if config.noise_std > 0:
        rng = np.random.default_rng(config.seed)
        X = X + rng.normal(0.0, config.noise_std, size=X.shape)

    info = {
        "manifold": config.name,
        "D": config.D,
        "noise_std": config.noise_std,
        "prior_scale": config.prior_scale,
        "seed": config.seed,
        "note": (
            "Immersion is smooth, rank-2 everywhere, and image is homeomorphic to R^2 (no holes); "
            "compatible with a VAE Gaussian prior."
        ),
    }
    return X, Z, info


# -------------------------
# Utilities (optional)
# -------------------------

def approx_jacobian_rank(
    z: np.ndarray,
    f: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
    D: int,
    eps: float = 1e-4,
    n_check: int = 64,
    seed: Optional[int] = None
) -> float:
    """
    Numerically estimate average local Jacobian rank over a subset of points,
    to sanity-check immersion rank=2. Uses finite differences.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(z.shape[0], size=min(n_check, z.shape[0]), replace=False)
    u, v = z[idx, 0], z[idx, 1]

    def F(u_, v_):
        return f(u_, v_, D)

    X = F(u, v)
    # finite differences along u and v
    Xu = (F(u + eps, v) - X) / eps
    Xv = (F(u, v + eps) - X) / eps

    ranks = []
    for i in range(X.shape[0]):
        J = np.stack([Xu[i], Xv[i]], axis=1)  # D x 2
        s = np.linalg.svd(J, compute_uv=False)
        # numerical rank with tolerance
        tol = max(J.shape) * np.finfo(float).eps * s[0]
        rank = int((s > tol).sum())
        ranks.append(rank)
    return float(np.mean(ranks))

def save_dataset(
    filename: str | Path,
    X: np.ndarray,
    Z_true: np.ndarray,
    config: ManifoldConfig,
    Z_vae: np.ndarray | None = None,
):
    """
    Save synthetic dataset to compressed .npz plus metadata.json
    so it can be reloaded later for benchmarking.
    """
    filename = Path(filename)
    np.savez_compressed(
        filename.with_suffix(".npz"),
        X=X,
        Z_true=Z_true,
        Z_vae=Z_vae if Z_vae is not None else np.array([]),
    )
    # Save metadata separately in JSON for readability
    meta = {
        "manifold": config.name,   # immersion id
        "D": config.D,
        "noise_std": config.noise_std,
        "prior_scale": config.prior_scale,
        "seed": config.seed,
    }
    with open(filename.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_dataset(
    filename: str | Path,
) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray | None, Callable, Callable]:
    """
    Load synthetic dataset, metadata, and immersion.

    Returns
    -------
    X : ndarray, shape (N,D)
        Ambient points.
    Z_true : ndarray, shape (N,2)
        Ground truth latent seeds.
    meta : dict
        Metadata dictionary describing the generator configuration.
    Z_vae : ndarray or None
        VAE latent codes if present.
    immersion : callable
        Immersion f(u,v,D) -> ndarray(N,D).
    true_metric : callable
        Function true_metric(u,v,D) -> ndarray(2,2).
    """
    filename = Path(filename)
    data = np.load(filename.with_suffix(".npz"))
    with open(filename.with_suffix(".json"), "r") as f:
        meta = json.load(f)

    X = data["X"]
    Z_true = data["Z_true"]
    Z_vae = data["Z_vae"]
    if Z_vae.size == 0:
        Z_vae = None

    # Recover immersion and metric
    immersion = IMMERSIONS[meta["manifold"]]

    def true_metric(u: float, v: float, D: int, eps: float = 1e-5) -> np.ndarray:
        f0 = immersion(np.array([u]), np.array([v]), D)[0]
        fu = (immersion(np.array([u + eps]), np.array([v]), D)[0] - f0) / eps
        fv = (immersion(np.array([u]), np.array([v + eps]), D)[0] - f0) / eps
        J = np.stack([fu, fv], axis=1)
        return J.T @ J

    return X, Z_true, meta, Z_vae, immersion, true_metric


def get_metric(manifold_name: ManifoldName) -> Callable[[float, float, int], np.ndarray]:
    """
    Utility to get the true metric function for a given manifold name.
    """
    if manifold_name not in IMMERSIONS:
        raise ValueError(f"Unknown manifold '{manifold_name}'. Available: {list(IMMERSIONS.keys())}")
    immersion = IMMERSIONS[manifold_name]   

    def true_metric(u: float, v: float, D: int, eps: float = 1e-5) -> np.ndarray:
        f0 = immersion(np.array([u]), np.array([v]), D)[0]
        fu = (immersion(np.array([u + eps]), np.array([v]), D)[0] - f0) / eps
        fv = (immersion(np.array([u]), np.array([v + eps]), D)[0] - f0) / eps
        J = np.stack([fu, fv], axis=1)
        return J.T @ J
    return true_metric

# -------------------------
# Example usage
# -------------------------

if __name__ == "__main__":
    # Configure a helicoid immersed in R^8, with a standard Gaussian latent prior
    cfg = ManifoldConfig(name="helicoid", D=8, noise_std=0.01, prior_scale=1.0, seed=42)

    X, Z, info = generate_manifold_data(n=10_000, config=cfg)
    print("Info:", info)
    print("X shape:", X.shape, "Z shape:", Z.shape)

    # Optional: quick numerical check of immersion rank
    rank_avg = approx_jacobian_rank(Z, IMMERSIONS[cfg.name], D=cfg.D, n_check=128, seed=0)
    print(f"Average numerical Jacobian rank (expect ~2): {rank_avg:.2f}")
