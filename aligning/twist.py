"""
Metric-twist benchmark: attributed graphs whose GRAPH geometry is a known deformation of the
ATTRIBUTE geometry.

Deforming ("twisting") the attribute manifold until a link probability of its geodesic distances fits
the graph reveals how the two geometries differ. A node-relabelling benchmark has no such ground truth:
there the graph is the attribute geometry plus random rewiring, so the only thing to find is "which
nodes were rewired". Here the graph is generated from a smooth Riemannian
metric G_graph that differs from the attribute metric G_attr by a known strain field, so what the
twist should recover is defined exactly:

    relative metric (strain)   R(u) = G_attr(u)^{-1} G_graph(u)

Its generalised eigenvalues (lambda, mu) are the squared length-stretch factors of the graph geometry
relative to the attribute geometry along its eigen-directions; they do not depend on coordinates.

Construction (u = true manifold coordinates, G = G_attr(u), a = twist axis in u-coordinates):

    m       = G a / sqrt(a^T G a)                     (unit covector of the axis in the G metric)
    G_graph = mu G + (lambda - mu) m m^T              =>  R a = lambda a,  R w = mu w  for a^T G w = 0

with profile phi(u) in [0, 1] (Gaussian bump) and strength kappa:

    "shear"     disk around a centre c, random axis:   lambda = exp(+kappa phi), mu = exp(-kappa phi)
                (area-preserving: det R = 1; the graph prefers one direction, local link density kept)
    "barrier"   strip around a line through c, axis = strip normal:   lambda = exp(+kappa phi), mu = 1
                (crossing the strip is longer for the graph: fewer links across, "river / segregation")
    "corridor"  strip around a line through c, axis = strip tangent:  lambda = exp(-kappa phi), mu = 1
                (moving along the strip is shorter for the graph: long links along it, "transit line")
    "none"      R = I (null)

The graph is then generated exactly as in aligning.synthetic.generate (same kernel, threshold, degree
calibration, weights, connectivity repair, no relabelling), from TRUE geodesic distances under G_graph.
The attributes are generated from the attribute immersion as before, so the encoder never sees the twist.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch

from .metric import GridSpec, MetricField
from .geodesics import GridGeodesics
from .synthetic import ManifoldSpec, immersion, true_metric_fn, SyntheticConfig, GraphConfig, PerturbConfig, generate

KINDS = ("none", "shear", "barrier", "corridor")


@dataclass
class TwistSpec:
    kind: str = "none"
    kappa: float = float(np.log(9.0))   # log of the maximal squared-length stretch (x3 lengths)
    centre: tuple = (0.0, 0.0)          # u-coordinates
    angle: float = 0.0                  # radians; shear: axis direction; strip: tangent direction
    width: float = 0.5                  # Gaussian profile sd (u units)

    @staticmethod
    def draw(kind: str, Z: np.ndarray, rng: np.random.Generator, kappa: float, width: Optional[float] = None) -> "TwistSpec":
        """Random placement: centre = a node with |u| < 1 (interior), axis angle ~ U[0, pi)."""
        inner = np.nonzero(np.linalg.norm(Z, axis=1) < 1.0)[0]
        c = Z[int(rng.choice(inner))]
        ang = float(rng.uniform(0.0, np.pi))
        if width is None:
            width = 0.5 if kind == "shear" else 0.3
        return TwistSpec(kind=kind, kappa=float(kappa), centre=(float(c[0]), float(c[1])), angle=ang, width=float(width))


def _profile_axis(tw: TwistSpec, u: torch.Tensor):
    """phi(u) in [0,1] and the twist axis a(u) (unit vector in u-coordinates) for every point."""
    c = torch.as_tensor(tw.centre, dtype=u.dtype)
    t = torch.tensor([np.cos(tw.angle), np.sin(tw.angle)], dtype=u.dtype)
    nrm = torch.tensor([-np.sin(tw.angle), np.cos(tw.angle)], dtype=u.dtype)
    r = u - c
    if tw.kind == "shear":
        phi = torch.exp(-(r ** 2).sum(-1) / (2 * tw.width ** 2))
        axis = t
    else:
        dist = r @ nrm
        phi = torch.exp(-dist ** 2 / (2 * tw.width ** 2))
        axis = nrm if tw.kind == "barrier" else t
    return phi, axis.expand(u.shape[0], 2)


def eigen_factors(tw: TwistSpec, phi: torch.Tensor):
    """(lambda along the axis, mu along its G-orthogonal complement), squared-length factors."""
    k = tw.kappa
    if tw.kind == "none":
        return torch.ones_like(phi), torch.ones_like(phi)
    if tw.kind == "shear":
        return torch.exp(k * phi), torch.exp(-k * phi)
    if tw.kind == "barrier":
        return torch.exp(k * phi), torch.ones_like(phi)
    if tw.kind == "corridor":
        return torch.exp(-k * phi), torch.ones_like(phi)
    raise ValueError(tw.kind)


def graph_metric_fn(spec: ManifoldSpec, tw: TwistSpec):
    """u -> G_graph(u), (n, 2, 2)."""
    G_attr = true_metric_fn(spec)

    def fn(u: torch.Tensor) -> torch.Tensor:
        G = G_attr(u)
        if tw.kind == "none":
            return G
        phi, a = _profile_axis(tw, u)
        lam, mu = eigen_factors(tw, phi)
        Ga = torch.einsum("nij,nj->ni", G, a)
        m = Ga / torch.sqrt(torch.einsum("ni,ni->n", a, Ga)).unsqueeze(-1)
        return mu[:, None, None] * G + (lam - mu)[:, None, None] * m.unsqueeze(-1) * m.unsqueeze(-2)
    return fn


def graph_geodesic_distances(spec: ManifoldSpec, tw: TwistSpec, Z: np.ndarray, P: int = 200,
                             neighbourhood: int = 32) -> np.ndarray:
    gs = GridSpec.from_points(Z, P, margin=0.05)
    field_ = MetricField.from_function(graph_metric_fn(spec, tw), gs)
    with torch.no_grad():
        return GridGeodesics(gs, Z, neighbourhood=neighbourhood).hybrid_distances(field_).numpy()


def true_strain(spec: ManifoldSpec, tw: TwistSpec, Z: np.ndarray) -> dict:
    """Ground truth at the nodes: eigenvalues (lambda, mu), the axis (u-coordinates), the
    log-area change 0.5 log(lambda mu), the anisotropy 0.5 |log(lambda/mu)| and the MAJOR axis
    (eigenvector of the larger eigenvalue) in u-coordinates. For "none" the axis is undefined (NaN)."""
    u = torch.as_tensor(Z, dtype=torch.float64)
    with torch.no_grad():
        G = true_metric_fn(spec)(u)
        if tw.kind == "none":
            n = Z.shape[0]
            return dict(lam=np.ones(n), mu=np.ones(n), phi=np.zeros(n), area=np.zeros(n), aniso=np.zeros(n),
                        major_u=np.full((n, 2), np.nan), G_attr=G.numpy())
        phi, a = _profile_axis(tw, u)
        lam, mu = eigen_factors(tw, phi)
        # G-orthogonal complement of a, in u-coordinates: w = J a with J = rotation in the G metric
        Ga = torch.einsum("nij,nj->ni", G, a)
        w = torch.stack([-Ga[:, 1], Ga[:, 0]], -1)          # a^T G w = 0
        major = torch.where((lam >= mu).unsqueeze(-1), a, w)
    return dict(lam=lam.numpy(), mu=mu.numpy(), phi=phi.numpy(),
                area=0.5 * np.log(lam.numpy() * mu.numpy()),
                aniso=0.5 * np.abs(np.log(lam.numpy() / mu.numpy())),
                major_u=major.numpy(), G_attr=G.numpy())


def generate_twisted(cfg: SyntheticConfig, seed: int, tw: TwistSpec, manifold: ManifoldSpec, Z: np.ndarray,
                     d_graph: Optional[np.ndarray] = None) -> dict:
    """One twisted attributed graph: attributes from the attribute immersion (identical to the
    untwisted sample of the same seed), graph from the geodesics of G_graph (no relabelling)."""
    if d_graph is None:
        d_graph = graph_geodesic_distances(manifold, tw, Z)
    cfg0 = SyntheticConfig(n=cfg.n, D=cfg.D, noise_std=cfg.noise_std, graph=cfg.graph,
                           perturb=PerturbConfig(region="scatter", m=cfg.perturb.m, intensity=0.0, kind="node_relabel"))
    data = generate(cfg0, seed, d_true=d_graph, manifold=manifold, Z=Z)
    data["d_graph"] = d_graph
    data["twist"] = asdict(tw)
    return data
