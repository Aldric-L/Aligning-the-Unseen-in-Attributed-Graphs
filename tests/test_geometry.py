"""Correctness tests for aligning.models / metric / geodesics.

Run:  python -m pytest -q tests/
"""
import numpy as np
import pytest
import torch

from aligning.models import VariationalDecoder, set_seed
from aligning.metric import GridSpec, MetricField, pullback_metric
from aligning.geodesics import (GridGeodesics, straight_line_distances, pairwise_straight_line,
                                reference_geodesic, grid_path, gaussian_curvature)

torch.set_default_dtype(torch.float64)


# ----------------------------------------------------------------------------- Jacobians
@pytest.mark.parametrize("act", ["elu", "relu", "tanh"])
@pytest.mark.parametrize("clip", [None, 0.5])
def test_decoder_jacobian_matches_autograd(act, clip):
    set_seed(0)
    dec = VariationalDecoder(2, 7, [16, 9], activation=act, clip_logvar=clip).double()
    z = torch.randn(25, 2) * 1.5
    J_mu, J_sig = dec.jacobians(z)
    for i in range(z.shape[0]):
        f_mu = lambda zz: dec(zz[None])[0][0]
        f_sig = lambda zz: torch.exp(0.5 * dec(zz[None])[1][0])
        Jm = torch.autograd.functional.jacobian(f_mu, z[i])
        Js = torch.autograd.functional.jacobian(f_sig, z[i])
        assert torch.allclose(J_mu[i], Jm, atol=1e-10)
        assert torch.allclose(J_sig[i], Js, atol=1e-10)


def test_metric_kinds_differ_and_expected_is_sum():
    set_seed(1)
    dec = VariationalDecoder(2, 5, [8]).double()
    z = torch.randn(10, 2)
    Jm, Js = dec.jacobians(z)
    G = pullback_metric(dec, z, "expected")
    assert torch.allclose(G, Jm.transpose(1, 2) @ Jm + Js.transpose(1, 2) @ Js)
    Gl = pullback_metric(dec, z, "legacy")
    assert not torch.allclose(G, Gl)


# ----------------------------------------------------------------------------- interpolation
def test_interpolation_exact_on_bilinear_field():
    spec = GridSpec(np.array([[-1.0, 2.0], [0.0, 3.0]]), P=12)
    def fn(x):  # bilinear in each component -> interpolation must be exact
        a = 1 + 0.3 * x[:, 0] + 0.2 * x[:, 1] + 0.1 * x[:, 0] * x[:, 1]
        b = 0.05 * x[:, 0]
        c = 2 + 0.4 * x[:, 1]
        return torch.stack([torch.stack([a, b], -1), torch.stack([b, c], -1)], -2)
    field = MetricField.from_function(fn, spec)
    pts = torch.rand(200, 2) * torch.tensor([3.0, 3.0]) + torch.tensor([-1.0, 0.0])
    assert torch.allclose(field.interpolate(pts), fn(pts), atol=1e-12)


# ----------------------------------------------------------------------------- geodesics
def poincare(x):
    y = x[:, 1]
    eye = torch.eye(2, dtype=x.dtype).expand(x.shape[0], 2, 2)
    return eye / (y ** 2)[:, None, None]


def poincare_dist(p, q):
    return np.arccosh(1 + ((p - q) ** 2).sum(-1) / (2 * p[..., 1] * q[..., 1]))


@pytest.fixture(scope="module")
def poincare_setup():
    rng = np.random.default_rng(0)
    spec = GridSpec(np.array([[-1.5, 1.5], [0.4, 2.6]]), P=100)
    Q = np.stack([rng.uniform(-1.3, 1.3, 60), rng.uniform(0.5, 2.5, 60)], 1)
    field = MetricField.from_function(poincare, spec)
    return spec, Q, field


def test_grid_geodesics_accuracy_poincare(poincare_setup):
    spec, Q, field = poincare_setup
    iu = np.triu_indices(len(Q), 1)
    true = poincare_dist(Q[iu[0]], Q[iu[1]])
    # pure grid (32-stencil): small systematic over-estimation, larger on short pairs
    D = GridGeodesics(spec, Q, neighbourhood=32).distances(field).detach().numpy()
    rel = (D[iu] - true) / true
    assert np.median(np.abs(rel)) < 0.005 and np.quantile(np.abs(rel), 0.95) < 0.03
    # hybrid min(grid, straight segment): the default estimator
    H = GridGeodesics(spec, Q, neighbourhood=32).hybrid_distances(field).detach().numpy()
    relh = (H[iu] - true) / true
    assert np.median(np.abs(relh)) < 0.004, np.median(np.abs(relh))
    assert np.quantile(np.abs(relh), 0.95) < 0.01, np.quantile(np.abs(relh), 0.95)
    assert np.max(np.abs(relh)) < 0.03, np.max(np.abs(relh))
    assert np.min(relh) > -0.01  # upper bound up to interpolation/quadrature error


def test_grid8_has_metrication_bias(poincare_setup):
    spec, Q, field = poincare_setup
    D8 = GridGeodesics(spec, Q, neighbourhood=8).distances(field).detach().numpy()
    iu = np.triu_indices(len(Q), 1)
    true = poincare_dist(Q[iu[0]], Q[iu[1]])
    rel8 = (D8[iu] - true) / true
    assert np.median(rel8) > 0.005  # systematic over-estimation, documented


def test_straight_line_is_upper_bound(poincare_setup):
    spec, Q, field = poincare_setup
    Zt = torch.tensor(Q)
    D = pairwise_straight_line(field, Zt, T=41).numpy()
    iu = np.triu_indices(len(Q), 1)
    true = poincare_dist(Q[iu[0]], Q[iu[1]])
    assert np.all(D[iu] >= true * (1 - 2e-3))  # any path length >= geodesic distance


def test_reference_solver_recovers_closed_form(poincare_setup):
    spec, Q, field = poincare_setup
    geo = GridGeodesics(spec, Q, neighbourhood=16)
    errs = []
    for (i, j) in [(0, 1), (2, 7), (10, 33), (5, 50)]:
        path = grid_path(geo, field, i, j)
        L, _ = reference_geodesic(poincare, Q[i], Q[j], path, K=64)
        errs.append(abs(L - poincare_dist(Q[i], Q[j])) / poincare_dist(Q[i], Q[j]))
    assert max(errs) < 2e-3, errs


def test_dijkstra_gradient_matches_finite_differences():
    """Gradient of sum_ij c_ij d_ij w.r.t. the grid metric equals finite differences."""
    rng = np.random.default_rng(3)
    spec = GridSpec(np.array([[0.0, 1.0], [0.0, 1.0]]), P=10)
    Q = rng.uniform(0.05, 0.95, size=(6, 2))
    nodes = torch.tensor(spec.node_coords())
    base = 1.0 + 0.5 * torch.sin(3 * nodes[:, :1]) * torch.cos(2 * nodes[:, 1:])
    theta = base.clone().squeeze(1).requires_grad_(True)
    geo = GridGeodesics(spec, Q, neighbourhood=16)
    C = torch.tensor(rng.normal(size=(6, 6)))

    def loss(th):
        G = th[:, None, None] * torch.eye(2).expand(th.shape[0], 2, 2)
        D = geo.distances(MetricField(spec, G))
        return (C * D).sum()

    l = loss(theta)
    l.backward()
    g = theta.grad.numpy()
    eps = 1e-6
    idx = rng.choice(theta.shape[0], 15, replace=False)
    for k in idx:
        tp = theta.detach().clone(); tp[k] += eps
        tm = theta.detach().clone(); tm[k] -= eps
        fd = (loss(tp).item() - loss(tm).item()) / (2 * eps)
        assert abs(fd - g[k]) < 1e-5 * max(1.0, abs(fd)), (k, fd, g[k])


def test_straight_line_gradient():
    spec = GridSpec(np.array([[0.0, 1.0], [0.0, 1.0]]), P=8)
    nodes = torch.tensor(spec.node_coords())
    theta = (1.0 + nodes[:, 0] ** 2).clone().requires_grad_(True)
    U = torch.tensor([[0.1, 0.2], [0.3, 0.9]]); V = torch.tensor([[0.8, 0.7], [0.6, 0.1]])
    def f(th):
        G = th[:, None, None] * torch.eye(2).expand(th.shape[0], 2, 2)
        return straight_line_distances(MetricField(spec, G), U, V, T=21).sum()
    assert torch.autograd.gradcheck(f, (theta,), eps=1e-6, atol=1e-6)


# ----------------------------------------------------------------------------- curvature
def test_curvature_poincare_is_minus_one():
    pts = torch.tensor([[0.0, 1.0], [0.5, 2.0], [-1.0, 0.7]])
    K = gaussian_curvature(poincare, pts)
    assert torch.allclose(K, -torch.ones(3), atol=1e-8)


def test_curvature_sphere_is_one():
    def sphere(x):  # spherical coordinates (theta, phi), radius 1
        s = torch.sin(x[:, 0]) ** 2
        z = torch.zeros_like(s)
        return torch.stack([torch.stack([torch.ones_like(s), z], -1), torch.stack([z, s], -1)], -2)
    pts = torch.tensor([[0.7, 0.1], [1.5, 2.0], [2.2, -1.0]])
    assert torch.allclose(gaussian_curvature(sphere, pts), torch.ones(3), atol=1e-8)
