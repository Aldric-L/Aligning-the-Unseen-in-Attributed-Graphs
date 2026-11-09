"""Tests for scores (invariance claims), heat-kernel calibration and the synthetic generator."""
import numpy as np
import pytest
import torch

from aligning.scores import ratio_scores, legacy_scores, robust_z, edge_restricted
from aligning import heat
from aligning.synthetic import SyntheticConfig, PerturbConfig, GraphConfig, ManifoldSpec, generate


def _rand_dist(n=60, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.normal(size=(n, 2))
    D = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
    return D


def test_ratio_score_invariant_to_independent_rescalings():
    D1 = _rand_dist(seed=1)
    D2 = D1 * np.exp(0.3 * np.random.default_rng(2).normal(size=D1.shape))
    D2 = 0.5 * (D2 + D2.T)
    np.fill_diagonal(D2, 0)
    Z, S, Ss = ratio_scores(D1, D2)
    Zb, Sb, Ssb = ratio_scores(3.7 * D1, 0.21 * D2)
    assert np.allclose(Z, Zb) and np.allclose(S, Sb) and np.allclose(Ss, Ssb)


def test_uniform_inflation_gives_zero_ratio_score_but_not_legacy():
    D1 = _rand_dist(seed=3)
    Z, S, _ = ratio_scores(D1, 5.0 * D1)
    assert np.allclose(S, 0.0)
    Zl, Sl = legacy_scores(D1, 5.0 * D1)
    # legacy score then equals a monotone function of Phase-1 eccentricity (D1 confound)
    ecc = np.log(np.where(np.eye(len(D1), dtype=bool), 1, D1)).sum(1)
    assert np.corrcoef(Sl, ecc)[0, 1] > 0.99


def test_edge_restricted_mean():
    Z = np.arange(16, dtype=float).reshape(4, 4)
    Z = Z + Z.T
    A = np.zeros((4, 4)); A[0, 1] = A[1, 0] = 1; A[0, 2] = A[2, 0] = 2
    S, Ss = edge_restricted(Z, A)
    assert np.isclose(S[0], (abs(Z[0, 1]) + abs(Z[0, 2])) / 2) and np.isnan(S[3])   # isolated -> undefined


def test_calibration_recovers_scale_and_amplitude():
    D = _rand_dist(n=120, seed=4)
    times = np.logspace(-1, 1, 7)
    w = heat.kernel_weights(times, 2)
    s0, a0 = 1.7, 0.35
    h = (w[None, None] * np.exp(-((s0 * D)[..., None] ** 2) / (4 * times[None, None]))).sum(-1)
    At = a0 * h
    np.fill_diagonal(At, 0)
    s, a, loss, hit = heat.calibrate(D, At, times, w)
    assert abs(s / s0 - 1) < 1e-3 and abs(a / a0 - 1) < 1e-3 and loss < 1e-10 and not hit


def test_bce_calibration_close_to_truth():
    D = _rand_dist(n=200, seed=5)
    times = np.logspace(-1, 1, 7)
    w = heat.kernel_weights(times, 2)
    s0, a0 = 1.2, 0.6
    h = (w[None, None] * np.exp(-((s0 * D)[..., None] ** 2) / (4 * times[None, None]))).sum(-1)
    rng = np.random.default_rng(6)
    U = rng.random(D.shape)
    A = (np.triu(U, 1) < np.triu(a0 * h, 1)).astype(float)
    A = A + A.T
    s, a, _ = heat.calibrate_bce(D, A, times, w, init=(1.0, 0.5))
    assert abs(np.log(s / s0)) < 0.15 and abs(a - a0) < 0.12


def test_heat_times_use_graph_laplacian():
    A = np.zeros((6, 6))
    for i in range(5):
        A[i, i + 1] = A[i + 1, i] = 1.0
    t = heat.heat_times(A, K=5, c=4.0)
    ev = heat.normalized_laplacian_spectrum(A)
    pos = ev[ev > 1e-8]
    assert np.isclose(t[0], 1 / pos.max()) and np.isclose(t[-1], 4 / pos.min())


@pytest.mark.parametrize("kind,region", [("node_relabel", "scatter"), ("block_shuffle", "disk"), ("block_shuffle", "band")])
def test_generator_pairing_and_multiset(kind, region):
    rng = np.random.default_rng(123)
    man = ManifoldSpec.sample(rng, 20)
    Z = rng.normal(size=(150, 2))
    d = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1))  # stand-in geometry (structure test only)
    cfg = lambda a: SyntheticConfig(n=150, graph=GraphConfig(bandwidth=0.3, mean_degree=10),
                                    perturb=PerturbConfig(region=region, m=20, intensity=a, kind=kind))
    g1, g0 = generate(cfg(1.0), 7, d_true=d, manifold=man, Z=Z), generate(cfg(0.0), 7, d_true=d, manifold=man, Z=Z)
    assert np.array_equal(g1["block"], g0["block"]) and np.allclose(g1["X"], g0["X"])
    iu = np.triu_indices(150, 1)
    assert np.allclose(np.sort(g1["compat_used"][iu]), np.sort(g1["compat"][iu]))      # exact multiset
    assert np.allclose(g1["compat_used"], g1["compat_used"].T) and np.all(np.diag(g1["compat_used"]) == 0)
    assert np.isclose(g1["c"], g0["c"])                                                   # common random numbers
    b = g1["block"]
    assert not np.allclose(g1["compat_used"][np.ix_(b, b)], g1["compat"][np.ix_(b, b)])  # block really changed


def test_intermediate_intensity_does_not_leak_through_weights():
    """At alpha in (0,1) the perturbed pairs take whole profiles, so the weights of the block
    nodes' edges are compatibilities of the same distribution as elsewhere (not scaled by alpha or 1-alpha)."""
    rng = np.random.default_rng(321)
    man = ManifoldSpec.sample(rng, 20)
    Z = rng.normal(size=(300, 2))
    d = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1))
    g = generate(SyntheticConfig(n=300, graph=GraphConfig(bandwidth=0.1, mean_degree=15),
                                 perturb=PerturbConfig(region="scatter", m=40, intensity=0.5, kind="node_relabel")),
                 11, d_true=d, manifold=man, Z=Z)
    A, y = g["A"], g["labels"]
    nb = A > 0
    mean_w = (A * nb).sum(1) / np.maximum(nb.sum(1), 1)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, -mean_w)
    assert 0.3 < auc < 0.7, auc     # the leaky kernel mixing gave ~0.95-0.97


def test_generated_graphs_are_connected():
    from scipy.sparse.csgraph import connected_components
    rng = np.random.default_rng(5)
    man = ManifoldSpec.sample(rng, 20)
    Z = rng.normal(size=(300, 2))
    d = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1))
    g = generate(SyntheticConfig(n=300, graph=GraphConfig(bandwidth=0.1, mean_degree=12),
                                 perturb=PerturbConfig(region="scatter", m=30, intensity=1.0, kind="node_relabel")),
                 3, d_true=d, manifold=man, Z=Z)
    assert connected_components(g["A"] > 0, directed=False)[0] == 1
