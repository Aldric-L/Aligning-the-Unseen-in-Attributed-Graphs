"""
Geodesic distance estimators on a 2-D/3-D latent space equipped with a MetricField.

1. GridGeodesics  — grid-graph estimator (Methods, "Geodesic distances"):
   * edge weights are recomputed from the *current* metric at every call;
   * the same quadrature (Simpson on the speed) is used for path selection and for
     the differentiable length;
   * 16-neighbourhood stencil in 2-D (26 in 3-D) to reduce metrication bias
     (8-neighbourhood: up to ~8% overestimation on off-axis directions);
   * query points are attached to the surrounding 4^d grid nodes and to nearby
     queries; their metric is interpolated from the grid (no first-come cache);
   * shortest paths are computed in C (scipy) and the gradient is the exact
     (almost-everywhere) gradient of the min-over-paths, d L / d w_e =
     sum over pairs whose shortest path uses e (Danskin), accumulated along the
     shortest-path trees with a numba kernel.

2. straight_line_distances — straight-line estimator (not used for results), with
   Simpson quadrature on the interpolated metric.

3. reference_geodesic — validation-only solver: discrete path-energy minimisation
   (L-BFGS) initialised from a fine-grid shortest path, metric evaluated exactly.

4. gaussian_curvature — true Gaussian curvature of a 2-D metric field (Brioschi),
   via automatic differentiation.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numba
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
import torch

from .metric import GridSpec, MetricField

# --------------------------------------------------------------------------------------
# Grid-graph estimator
# --------------------------------------------------------------------------------------

def _stencil(d: int, neighbourhood: int) -> np.ndarray:
    """Half stencil (one orientation per undirected edge)."""
    if d == 2:
        base = [(1, 0), (0, 1), (1, 1), (1, -1)]
        if neighbourhood == 8:
            return np.array(base)
        knight = [(1, 2), (2, 1), (1, -2), (2, -1)]
        if neighbourhood == 16:
            return np.array(base + knight)
        if neighbourhood == 32:
            far = [(1, 3), (3, 1), (1, -3), (3, -1), (2, 3), (3, 2), (2, -3), (3, -2)]
            return np.array(base + knight + far)
        raise ValueError("2-D neighbourhood must be 8, 16 or 32")
    if d == 3:
        offs = []
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                for c in (-1, 0, 1):
                    o = (a, b, c)
                    if o > (0, 0, 0):
                        offs.append(o)
        return np.array(offs)
    raise ValueError("only d in {2,3} supported")


@numba.njit(cache=True)
def _tree_backward(order, pred, query_nodes, grad_out, indptr, indices, eids, n_edges):
    S, V = pred.shape
    Q = query_nodes.shape[0]
    grad_w = np.zeros(n_edges)
    acc = np.zeros(V)
    for s in range(S):
        for v in range(V):
            acc[v] = 0.0
        for q in range(Q):
            acc[query_nodes[q]] += grad_out[s, q]
        for k in range(V):
            v = order[s, k]
            u = pred[s, v]
            if u < 0:
                continue
            a = acc[v]
            if a == 0.0:
                continue
            e = -1
            for p in range(indptr[v], indptr[v + 1]):
                if indices[p] == u:
                    e = eids[p]
                    break
            grad_w[e] += a
            acc[u] += a
    return grad_w


class _ShortestPaths(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, graph, sources):
        wn = w.detach().cpu().numpy().astype(np.float64)
        data = wn[graph.csr_eid]
        csr = sp.csr_matrix((data, graph.csr_indices, graph.csr_indptr), shape=(graph.n_total, graph.n_total))
        dist, pred = dijkstra(csr, directed=True, indices=sources, return_predecessors=True)
        D = dist[:, graph.query_nodes]
        if not np.all(np.isfinite(D)):
            raise RuntimeError("disconnected geodesic graph (infinite distance)")
        ctx.graph = graph
        ctx.pred = pred.astype(np.int64)
        ctx.dist = dist
        ctx.dtype, ctx.device = w.dtype, w.device
        return torch.as_tensor(D, dtype=w.dtype, device=w.device)

    @staticmethod
    def backward(ctx, grad_D):
        g = ctx.graph
        order = np.argsort(-ctx.dist, axis=1, kind="stable").astype(np.int64)
        gw = _tree_backward(order, ctx.pred, g.query_nodes.astype(np.int64),
                            grad_D.detach().cpu().numpy().astype(np.float64),
                            g.csr_indptr.astype(np.int64), g.csr_indices.astype(np.int64),
                            g.csr_eid.astype(np.int64), g.n_edges)
        return torch.as_tensor(gw, dtype=ctx.dtype, device=ctx.device), None, None


class GridGeodesics:
    """Static structure of the extended grid graph for a fixed set of query points.

    Build once per (grid, query points); call `distances(field)` at every training
    step: edge weights are recomputed from the current metric field.
    """

    def __init__(self, spec: GridSpec, queries: np.ndarray, neighbourhood: int = 16,
                 attach_radius_cells: int = 1, qq_radius_cells: float = 3.0):
        self.spec = spec
        d, P = spec.d, spec.P
        self.d = d
        Q = queries.shape[0]
        lo, h = spec.bounds[:, 0], spec.steps
        q = np.clip(queries, spec.bounds[:, 0], spec.bounds[:, 1])
        self.queries = q
        n_g = spec.n_nodes
        self.n_grid = n_g
        self.n_total = n_g + Q
        self.query_nodes = np.arange(n_g, n_g + Q)

        coords = spec.node_coords()
        strides = spec.strides()
        # grid-grid edges
        multi = np.stack(np.meshgrid(*[np.arange(P + 1)] * d, indexing="ij"), -1).reshape(-1, d)
        ea, eb = [], []
        for o in _stencil(d, neighbourhood):
            m2 = multi + o
            ok = np.all((m2 >= 0) & (m2 <= P), axis=1)
            ea.append((multi[ok] * strides).sum(1))
            eb.append((m2[ok] * strides).sum(1))
        ea, eb = np.concatenate(ea), np.concatenate(eb)
        pa, pb = coords[ea], coords[eb]
        # query -> surrounding grid nodes (4^d block when attach_radius_cells=1)
        rel = (q - lo) / h
        base = np.clip(np.floor(rel), 0, P - 1).astype(np.int64)
        rng = np.arange(-attach_radius_cells, attach_radius_cells + 2)
        qa, qb = [], []
        for off in np.stack(np.meshgrid(*[rng] * d, indexing="ij"), -1).reshape(-1, d):
            m = base + off
            ok = np.all((m >= 0) & (m <= P), axis=1)
            qa.append(np.nonzero(ok)[0] + n_g)
            qb.append((m[ok] * strides).sum(1))
        qa, qb = np.concatenate(qa), np.concatenate(qb)
        # query <-> query
        tree = cKDTree(q / h)  # in cell units
        pairs = np.array(sorted(tree.query_pairs(qq_radius_cells)), dtype=np.int64).reshape(-1, 2)
        a_all = np.concatenate([ea, qa, pairs[:, 0] + n_g])
        b_all = np.concatenate([eb, qb, pairs[:, 1] + n_g])
        pos = np.concatenate([coords, q], axis=0)
        self.edge_a, self.edge_b = a_all, b_all
        self.n_edges = a_all.shape[0]
        self.vec = pos[b_all] - pos[a_all]
        self.pos_a, self.pos_b = pos[a_all], pos[b_all]
        self.mid = 0.5 * (self.pos_a + self.pos_b)
        # symmetric CSR with edge ids
        rows = np.concatenate([a_all, b_all])
        cols = np.concatenate([b_all, a_all])
        eid = np.concatenate([np.arange(self.n_edges), np.arange(self.n_edges)])
        order = np.lexsort((cols, rows))
        rows, cols, eid = rows[order], cols[order], eid[order]
        self.csr_indptr = np.searchsorted(rows, np.arange(self.n_total + 1)).astype(np.int64)
        self.csr_indices = cols.astype(np.int64)
        self.csr_eid = eid.astype(np.int64)
        self._grid_edge = a_all < n_g  # endpoint a is a grid node for grid-grid and grid-query? (qa is query)

    def edge_weights(self, field: MetricField) -> torch.Tensor:
        dt = field.G.dtype
        dev = field.G.device
        v = torch.as_tensor(self.vec, dtype=dt, device=dev)
        # metric at endpoints: grid nodes read directly, queries interpolated
        def metric_at(nodes: np.ndarray, pos: np.ndarray) -> torch.Tensor:
            out = torch.empty((nodes.shape[0], self.d, self.d), dtype=dt, device=dev)
            isg = nodes < self.n_grid
            if isg.any():
                out[torch.as_tensor(isg, device=dev)] = field.G[torch.as_tensor(nodes[isg], device=dev)]
            if (~isg).any():
                out[torch.as_tensor(~isg, device=dev)] = field.interpolate(torch.as_tensor(pos[~isg], dtype=dt, device=dev))
            return out
        Ga = metric_at(self.edge_a, self.pos_a)
        Gb = metric_at(self.edge_b, self.pos_b)
        Gm = field.interpolate(torch.as_tensor(self.mid, dtype=dt, device=dev))
        def speed(G):
            return torch.sqrt(torch.clamp(torch.einsum("ei,eij,ej->e", v, G, v), min=1e-18))
        return (speed(Ga) + 4.0 * speed(Gm) + speed(Gb)) / 6.0

    def distances(self, field: MetricField, sources: Optional[np.ndarray] = None) -> torch.Tensor:
        """Grid shortest-path lengths (n_sources x Q), differentiable w.r.t. field.G."""
        w = self.edge_weights(field)
        if sources is None:
            sources = self.query_nodes
        else:
            sources = self.query_nodes[sources]
        return _ShortestPaths.apply(w, self, np.asarray(sources))

    def hybrid_distances(self, field: MetricField, T: int = 21) -> torch.Tensor:
        """min(grid shortest path, straight segment) for all query pairs (Q x Q).

        Both terms are Riemannian lengths of actual curves, hence upper bounds of the
        geodesic distance; their minimum is a tighter upper bound. On the hyperbolic
        half-plane (closed-form distances) this gives median |rel. error| ~0.2%,
        p95 ~0.5%, max ~1.2% with P=100 and the 32-stencil (tests/test_geometry.py).
        """
        Dg = self.distances(field)
        Z = torch.as_tensor(self.queries, dtype=field.G.dtype, device=field.G.device)
        Dl = pairwise_straight_line(field, Z, T=T)
        D = torch.minimum(Dg, Dl)
        return 0.5 * (D + D.T)


# --------------------------------------------------------------------------------------
# Straight-line (linear interpolation) estimator
# --------------------------------------------------------------------------------------

def simpson_weights(T: int, dtype=torch.float64) -> torch.Tensor:
    if T % 2 == 0:
        raise ValueError("Simpson rule needs an odd number of points")
    w = torch.ones(T, dtype=dtype)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w / (3.0 * (T - 1))


def straight_line_distances(field: MetricField, U: torch.Tensor, V: torch.Tensor, T: int = 21) -> torch.Tensor:
    """d(u,v) = int_0^1 sqrt((v-u)^T G(u + t(v-u)) (v-u)) dt  (Simpson, T odd)."""
    dt = field.G.dtype
    U, V = U.to(dt), V.to(dt)
    t = torch.linspace(0.0, 1.0, T, dtype=dt, device=U.device)
    diff = V - U                                  # (B,d)
    pts = U[None] + t[:, None, None] * diff[None]  # (T,B,d)
    G = field.interpolate(pts.reshape(-1, U.shape[1])).reshape(T, U.shape[0], U.shape[1], U.shape[1])
    q = torch.einsum("bi,tbij,bj->tb", diff, G, diff)
    speed = torch.sqrt(torch.clamp(q, min=1e-18))
    return (simpson_weights(T, dt).to(U.device)[:, None] * speed).sum(0)


def pairwise_straight_line(field: MetricField, Z: torch.Tensor, T: int = 21, chunk: int = 200_000) -> torch.Tensor:
    n = Z.shape[0]
    iu, ju = torch.triu_indices(n, n, offset=1)
    out = []
    for s in range(0, iu.shape[0], chunk):
        out.append(straight_line_distances(field, Z[iu[s:s + chunk]], Z[ju[s:s + chunk]], T))
    vals = torch.cat(out)
    D = torch.zeros((n, n), dtype=vals.dtype, device=vals.device)
    D[iu, ju] = vals
    return D + D.T


# --------------------------------------------------------------------------------------
# Reference solver (validation only)
# --------------------------------------------------------------------------------------

def reference_geodesic(metric_fn: Callable[[torch.Tensor], torch.Tensor], p: np.ndarray, q: np.ndarray,
                       init_path: np.ndarray, K: int = 64, iters: int = 300) -> Tuple[float, np.ndarray]:
    """Minimise the discrete energy sum_k K * dx_k^T G(mid_k) dx_k with fixed endpoints.

    metric_fn maps (n,d) points to (n,d,d) metric tensors exactly (no grid).
    init_path: (m,d) polyline from p to q (e.g. fine-grid Dijkstra path).
    Returns the Riemannian length of the optimised path (midpoint rule) and the path.
    """
    # resample initial polyline uniformly in Euclidean arc length
    seg = np.linalg.norm(np.diff(init_path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] == 0:
        return 0.0, init_path
    tgt = np.linspace(0.0, s[-1], K + 1)
    path = np.stack([np.interp(tgt, s, init_path[:, k]) for k in range(init_path.shape[1])], 1)
    path[0], path[-1] = p, q
    P0 = torch.tensor(path[0], dtype=torch.float64)
    P1 = torch.tensor(path[-1], dtype=torch.float64)
    inner = torch.tensor(path[1:-1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([inner], lr=1.0, max_iter=iters, tolerance_grad=1e-10,
                            tolerance_change=1e-14, history_size=50, line_search_fn="strong_wolfe")

    def full():
        return torch.cat([P0[None], inner, P1[None]], 0)

    def energy():
        X = full()
        dx = X[1:] - X[:-1]
        G = metric_fn(0.5 * (X[1:] + X[:-1]))
        return K * torch.einsum("ki,kij,kj->", dx, G, dx)

    def closure():
        opt.zero_grad()
        e = energy()
        e.backward()
        return e

    opt.step(closure)
    with torch.no_grad():
        X = full()
        dx = X[1:] - X[:-1]
        G = metric_fn(0.5 * (X[1:] + X[:-1]))
        L = torch.sqrt(torch.clamp(torch.einsum("ki,kij,kj->k", dx, G, dx), min=0)).sum().item()
    return L, X.numpy()


def grid_path(geo: GridGeodesics, field: MetricField, i: int, j: int) -> np.ndarray:
    """Polyline of the grid shortest path between query i and query j (no gradient)."""
    with torch.no_grad():
        w = geo.edge_weights(field).cpu().numpy()
    csr = sp.csr_matrix((w[geo.csr_eid], geo.csr_indices, geo.csr_indptr), shape=(geo.n_total, geo.n_total))
    src = geo.query_nodes[i]
    _, pred = dijkstra(csr, directed=True, indices=[src], return_predecessors=True)
    pos = np.concatenate([geo.spec.node_coords(), geo.queries], 0)
    v = geo.query_nodes[j]
    chain = [v]
    while chain[-1] != src:
        chain.append(pred[0, chain[-1]])
    return pos[np.array(chain[::-1])]


# --------------------------------------------------------------------------------------
# Gaussian curvature (2-D), Brioschi formula with autodiff
# --------------------------------------------------------------------------------------

def gaussian_curvature(metric_fn: Callable[[torch.Tensor], torch.Tensor], pts: torch.Tensor) -> torch.Tensor:
    """True Gaussian curvature K at pts (n,2) for a smooth metric function metric_fn: (n,2)->(n,2,2)."""
    from torch.func import jacfwd, vmap

    def comps(z):  # z: (2,) -> (3,) = (E, F, G)
        g = metric_fn(z[None])[0]
        return torch.stack([g[0, 0], g[0, 1], g[1, 1]])

    J1 = vmap(jacfwd(comps))(pts)              # (n,3,2): d/du, d/dv
    J2 = vmap(jacfwd(jacfwd(comps)))(pts)      # (n,3,2,2)
    c = vmap(comps)(pts)
    E, F, G = c[:, 0], c[:, 1], c[:, 2]
    Eu, Ev = J1[:, 0, 0], J1[:, 0, 1]
    Fu, Fv = J1[:, 1, 0], J1[:, 1, 1]
    Gu, Gv = J1[:, 2, 0], J1[:, 2, 1]
    Evv = J2[:, 0, 1, 1]
    Fuv = J2[:, 1, 0, 1]
    Guu = J2[:, 2, 0, 0]
    M1 = torch.stack([
        torch.stack([-0.5 * Evv + Fuv - 0.5 * Guu, 0.5 * Eu, Fu - 0.5 * Ev], -1),
        torch.stack([Fv - 0.5 * Gu, E, F], -1),
        torch.stack([0.5 * Gv, F, G], -1)], -2)
    zero = torch.zeros_like(E)
    M2 = torch.stack([
        torch.stack([zero, 0.5 * Ev, 0.5 * Gu], -1),
        torch.stack([0.5 * Ev, E, F], -1),
        torch.stack([0.5 * Gu, F, G], -1)], -2)
    return (torch.linalg.det(M1) - torch.linalg.det(M2)) / (E * G - F ** 2) ** 2


def log_volume_element(G: torch.Tensor) -> torch.Tensor:
    """log sqrt(det G) — the 'magnification factor'."""
    return 0.5 * torch.log(torch.clamp(torch.linalg.det(G), min=1e-300))
