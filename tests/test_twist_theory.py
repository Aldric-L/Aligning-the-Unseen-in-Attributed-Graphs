"""Checks of the first-variation (stress) formulas of the twist against automatic differentiation."""
import numpy as np
import torch

from aligning import heat


def _pair(d, a, s, amp, loss):
    times = torch.tensor([1.0], dtype=torch.float64)
    weights = torch.tensor([1.0], dtype=torch.float64)
    D = torch.tensor([[0.0, d], [d, 0.0]], dtype=torch.float64, requires_grad=True)
    A = torch.tensor([[0.0, a], [a, 0.0]], dtype=torch.float64)
    f = heat.pair_loss_bce if loss == "bce" else heat.pair_loss
    L = f(D, A, times, weights, torch.tensor(s, dtype=torch.float64), torch.tensor(amp, dtype=torch.float64))
    L.backward()
    g = D.grad[0, 1] + D.grad[1, 0]           # dL/dd for the single pair (the matrix is symmetric)
    H = amp * np.exp(-(s * d) ** 2 / 4.0)
    return float(g), H


def test_bce_pair_force_matches_09():
    """dl/dd = (s^2 d / 2t) (A - H) / (1 - H) for the likelihood with one scale (t = 1 here)."""
    for d, a, s, amp in [(0.3, 1.0, 1.7, 0.6), (2.5, 1.0, 1.1, 0.4), (0.8, 0.0, 2.0, 0.7), (1.2, 0.35, 0.9, 0.5)]:
        g, H = _pair(d, a, s, amp, "bce")
        pred = (s ** 2 * d / 2.0) * (a - H) / (1.0 - H)
        assert np.isclose(g, pred, rtol=1e-9, atol=1e-12), (g, pred)


def test_mse_pair_force_matches_09():
    """dl/dd = 2 (A - H) H (s^2 d / 2t): vanishes for long links (H -> 0), unlike the likelihood."""
    for d, a, s, amp in [(0.3, 1.0, 1.7, 0.6), (2.5, 1.0, 1.1, 0.4), (0.8, 0.0, 2.0, 0.7)]:
        g, H = _pair(d, a, s, amp, "mse")
        pred = 2.0 * (a - H) * H * (s ** 2 * d / 2.0)
        assert np.isclose(g, pred, rtol=1e-9, atol=1e-14), (g, pred)
    # a long misaligned link: the squared-error force collapses, the likelihood force grows like d
    g_mse, _ = _pair(6.0, 1.0, 1.0, 0.5, "mse")
    g_bce, _ = _pair(6.0, 1.0, 1.0, 0.5, "bce")
    assert abs(g_mse) < 1e-3 * g_bce and g_bce > 2.9          # ratio ~1e-4 at d = 6


def test_first_variation_of_length_in_a_constant_metric():
    """d = sqrt(dz^T G dz)  =>  dd/dG = dz dz^T / (2 d): the stress of a straight geodesic in a constant field."""
    rng = np.random.default_rng(0)
    M = rng.normal(size=(2, 2))
    G = torch.tensor(M @ M.T + np.eye(2), dtype=torch.float64, requires_grad=True)
    dz = torch.tensor(rng.normal(size=2), dtype=torch.float64)
    d = torch.sqrt(dz @ G @ dz)
    d.backward()
    pred = torch.outer(dz, dz) / (2 * d.detach())
    assert torch.allclose(G.grad, pred, rtol=1e-10)


def test_stress_is_unbiased_under_the_null():
    """E[r] = 0 when links are drawn with probability H (soft labels: E[A~] = H)."""
    rng = np.random.default_rng(1)
    H = rng.uniform(0.01, 0.9, size=200000)
    A = (rng.random(H.shape) < H).astype(float)
    r = (A - H) / (1 - H)
    assert abs(r.mean()) < 4 * r.std() / np.sqrt(r.size)
