"""Tests of the metric-twist construction and of the strain read-outs (aligning/twist.py, strain.py)."""
import copy

import numpy as np
import scipy.linalg as sla
import torch

from aligning.models import VariationalDecoder
from aligning.strain import learned_strain, axis_error_deg, link_moment_strain, generalized_eig
from aligning.synthetic import ManifoldSpec, true_metric_fn
from aligning.twist import TwistSpec, graph_metric_fn, true_strain


def _setup(kind, seed=0):
    rng = np.random.default_rng(seed)
    spec = ManifoldSpec.sample(rng, 8)
    Z = rng.normal(size=(40, 2))
    tw = TwistSpec.draw(kind, Z, rng, kappa=np.log(9.0))
    return spec, Z, tw


def test_relative_metric_has_the_prescribed_eigenstructure():
    for kind in ("shear", "barrier", "corridor"):
        spec, Z, tw = _setup(kind)
        u = torch.as_tensor(Z, dtype=torch.float64)
        Ga = true_metric_fn(spec)(u).numpy()
        Gg = graph_metric_fn(spec, tw)(u).numpy()
        t = true_strain(spec, tw, Z)
        for i in range(Z.shape[0]):
            w = sla.eigh(Gg[i], Ga[i], eigvals_only=True)
            assert np.allclose(np.sort(w), np.sort([t["lam"][i], t["mu"][i]]), rtol=1e-8)
            # the major axis is an eigenvector of Ga^{-1} Gg with the larger eigenvalue
            a = t["major_u"][i]
            Ra = np.linalg.solve(Ga[i], Gg[i] @ a)
            assert np.allclose(Ra, max(t["lam"][i], t["mu"][i]) * a, rtol=1e-7, atol=1e-10)
        if kind == "shear":
            assert np.allclose(t["area"], 0.0, atol=1e-12)


def test_null_twist_is_the_attribute_metric():
    spec, Z, _ = _setup("shear")
    u = torch.as_tensor(Z, dtype=torch.float64)
    assert torch.allclose(graph_metric_fn(spec, TwistSpec("none"))(u), true_metric_fn(spec)(u))


def test_axis_error_zero_for_exact_map_and_invariant_to_scaling():
    rng = np.random.default_rng(3)
    n = 30
    a_u = rng.normal(size=(n, 2))
    J = rng.normal(size=(n, 2, 2)) + 3 * np.eye(2)
    M = rng.normal(size=(n, 2, 2))
    G = M @ np.swapaxes(M, 1, 2) + 0.5 * np.eye(2)
    v_z = np.einsum("nij,nj->ni", J, a_u) * rng.uniform(-3, 3, size=(n, 1))
    assert np.allclose(axis_error_deg(v_z, J, a_u, G), 0.0, atol=1e-5)
    # a G-orthogonal direction is at 90 degrees
    Ga = np.einsum("nij,nj->ni", G, a_u)
    w = np.stack([-Ga[:, 1], Ga[:, 0]], 1)
    assert np.allclose(axis_error_deg(np.einsum("nij,nj->ni", J, w), J, a_u, G), 90.0, atol=1e-5)


def test_learned_strain_is_identity_for_identical_decoders():
    torch.manual_seed(0)
    dec = VariationalDecoder(2, 6, [16]).double()
    Z = np.random.default_rng(0).normal(size=(25, 2))
    s = learned_strain(dec, copy.deepcopy(dec), Z)
    assert np.allclose(s["L"], 1.0, atol=1e-8) and np.allclose(s["aniso"], 0.0, atol=1e-8)


def test_link_moment_recovers_a_constant_anisotropic_graph_metric():
    """Points on a dense lattice, Euclidean Phase 1 metric (G1 = I), links drawn with probability
    exp(-d_g^2) where d_g uses G_graph = diag(4, 1): the estimator should return eigenvalues with
    ratio ~4 and the major axis along x."""
    rng = np.random.default_rng(1)
    g = np.linspace(-3, 3, 31)
    Z = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    n = Z.shape[0]
    dZ = Z[None] - Z[:, None]
    Gg = np.diag([4.0, 1.0])
    dg2 = np.einsum("ija,ab,ijb->ij", dZ, Gg, dZ)
    d12 = (dZ ** 2).sum(-1)
    P = np.exp(-dg2 / 0.5)
    np.fill_diagonal(P, 0)
    U = rng.random((n, n))
    A = np.triu(U < P, 1)
    A = (A | A.T).astype(float)
    P1 = np.exp(-d12 / 0.5)                       # the Phase 1 model (isotropic) at the right scale
    np.fill_diagonal(P1, 0)
    G1 = np.tile(np.eye(2), (n, 1, 1))
    m = link_moment_strain(Z, A, P1, G1, smooth_k=10)
    inner = np.all(np.abs(Z) < 2.0, 1)
    ratio = np.exp(2 * np.nanmedian(m["aniso"][inner]))
    assert 3.0 < ratio < 5.5, ratio
    maj = m["major_z"][inner]
    cosx = np.abs(maj[:, 0]) / np.linalg.norm(maj, axis=1)
    assert np.nanmedian(cosx) > 0.95


def test_generalized_eig_matches_scipy():
    rng = np.random.default_rng(5)
    M = rng.normal(size=(10, 2, 2)); N = rng.normal(size=(10, 2, 2))
    A = M @ np.swapaxes(M, 1, 2) + np.eye(2); Bm = N @ np.swapaxes(N, 1, 2) + np.eye(2)
    L, V = generalized_eig(A, Bm)
    for i in range(10):
        assert np.allclose(L[i], sla.eigh(A[i], Bm[i], eigvals_only=True)[::-1])
        assert np.allclose(A[i] @ V[i][:, 0], L[i, 0] * Bm[i] @ V[i][:, 0])


# ---------------------------------------------------------------- planted strain in learned coordinates
def test_e2_equals_planted_strain_with_a_perfect_phase1():
    from experiments.aggregate_twist import learned_relative_truth
    spec, Z, tw = _setup("shear", seed=4)
    rng = np.random.default_rng(4)
    J = rng.normal(size=(Z.shape[0], 2, 2)) + 3 * np.eye(2)
    t = true_strain(spec, tw, Z)
    Ga = t["G_attr"]
    Ji = np.linalg.inv(J)
    G1 = np.einsum("nji,njk,nkl->nil", Ji, Ga, Ji)            # Phase 1 = the true attribute metric in z
    e2 = learned_relative_truth(spec, tw, Z, J, G1)
    assert np.allclose(e2["aniso"], t["aniso"], atol=1e-8)
    assert np.allclose(e2["area"], t["area"], atol=1e-8)
    err = axis_error_deg(e2["major_z"], J, t["major_u"], Ga)
    big = t["aniso"] > 1e-3
    assert np.nanmax(err[big]) < 1e-3


def test_corrected_moment_is_exact_when_c0_is_not_proportional_to_g1_inverse():
    from experiments.aggregate_twist import moment_invariants
    rng = np.random.default_rng(6)
    n = 20
    M, N, P = (rng.normal(size=(n, 2, 2)) for _ in range(3))
    C = M @ np.swapaxes(M, 1, 2) + 0.2 * np.eye(2)
    C0 = N @ np.swapaxes(N, 1, 2) + 0.2 * np.eye(2)
    G1 = P @ np.swapaxes(P, 1, 2) + 0.2 * np.eye(2)
    Z = rng.normal(size=(n, 2))
    m = moment_invariants(Z, C, C0, G1, np.full(n, 5), smooth_k=0)
    for i in range(n):
        lam = sla.eigh(C0[i], C[i], eigvals_only=True)
        assert np.isclose(m["area"][i], 0.5 * (np.log(np.linalg.det(C0[i])) - np.log(np.linalg.det(C[i]))))
        assert np.isclose(m["aniso"][i], 0.5 * np.log(lam.max() / lam.min()))
        # eigenvalues of C0 C^-1 are the generalised eigenvalues: no clipping, no indefinite matrices
        assert np.allclose(np.sort(np.linalg.eigvals(C0[i] @ np.linalg.inv(C[i])).real), np.sort(lam))


def test_stratified_spearman():
    from experiments.aggregate_twist import stratified_spearman
    rng = np.random.default_rng(7)
    s = rng.normal(size=2000)
    y = rng.normal(size=2000)
    assert np.isclose(stratified_spearman(2 * y + 1, y, s), 1.0)
    assert abs(stratified_spearman(rng.normal(size=2000), y, s)) < 0.1
    # a shared dependence on s is REDUCED, not removed, by 10 strata (residual within-stratum variation of s):
    # this is why H7 is a paired difference against the "none" twist, which carries the same residual bias
    x, y2 = s + rng.normal(size=2000), s + rng.normal(size=2000)
    from scipy.stats import spearmanr
    assert stratified_spearman(x, y2, s) < 0.5 * spearmanr(x, y2).correlation
    assert np.isnan(stratified_spearman(y[:40], y[:40], s[:40]))


def test_rotation_chance_is_unbiased_for_an_uninformed_axis():
    """An estimated axis drawn independently of the planted angle: mean(error - rotation chance) ~ 0."""
    rng = np.random.default_rng(8)
    rot = np.radians(np.arange(5, 180, 5))
    diffs = []
    n = 60
    J = np.tile(np.eye(2), (n, 1, 1))
    G = np.tile(np.diag([1.0, 6.0]), (n, 1, 1))              # anisotropic metric, as on the manifold
    est_ang = rng.uniform(0, np.pi)
    for _ in range(400):
        theta = rng.uniform(0, np.pi)
        v = np.tile([np.cos(est_ang + 0.2 * rng.normal()), np.sin(est_ang)], (n, 1))
        truth = lambda a: np.tile([np.cos(a), np.sin(a)], (n, 1))
        e = np.median(axis_error_deg(v, J, truth(theta), G))
        ch = np.mean([np.median(axis_error_deg(v, J, truth(theta + r), G)) for r in rot])
        diffs.append(e - ch)
    diffs = np.array(diffs)
    assert abs(diffs.mean()) < 3 * diffs.std() / np.sqrt(len(diffs)) + 0.5
