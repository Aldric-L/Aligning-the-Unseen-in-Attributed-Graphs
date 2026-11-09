"""
Phase 2 objective: a function of latent geodesic distances matched to the (normalised) observed adjacency.

Multi-scale squared-error form:

    H_{s,a}(d) = a sum_k w_k exp(-(s d)^2 / (4 t_k)),   w_k ∝ (4 pi t_k)^(-dim/2),  sum_k w_k = 1
    L          = mean_{i<j} ( A~_ij - H_{s,a}(d_ij) )^2

Single-scale cross-entropy form (used in the paper, K = 1, t = 1/lambda_max):

    H_{s,a}(d) = a exp(-(s d)^2 / (4 t)),   L = -sum_{i<j} [A~ log H + (1 - A~) log(1 - H)]

  * heat times come from the spectrum of the normalised Laplacian of the observed weighted graph,
    t in [1/lambda_max, c/lambda_2];
  * the kernel is normalised so that H(0) = a and the target is normalised to [0, 1];
  * two global constants: an amplitude a and a scale s converting latent geodesic units into
    diffusion-time units. They are calibrated on the Phase 1 distances and free during Phase 2, so that
    a uniform rescaling of the manifold costs nothing and only non-uniform deformations are rewarded;
  * the diagonal is excluded (no self-loops).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def normalized_laplacian_spectrum(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    deg = A.sum(1)
    inv = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-300)), 0.0)
    L = np.eye(A.shape[0]) - inv[:, None] * A * inv[None, :]
    return np.clip(np.linalg.eigvalsh(L), 0.0, None)


def heat_times(A: np.ndarray, K: int = 15, c: float = 4.0, tol: float = 1e-8) -> np.ndarray:
    ev = normalized_laplacian_spectrum(A)
    pos = ev[ev > tol]
    lam2, lmax = pos.min(), pos.max()
    return np.logspace(np.log10(1.0 / lmax), np.log10(c / lam2), K)


def normalize_target(A: np.ndarray, mode: str = "max") -> np.ndarray:
    """Target normalisation. All modes are symmetric and map non-edges to 0.
      max      A / max A
      logmed   log1p(A / m) / log1p(max A / m), m = median positive weight (unit-free log)
      log      log1p(A) / log1p(max A)  (unit-dependent; kept for reference)
      binary   1[A > 0]
      rank     empirical CDF of the positive weights in (0, 1], computed on the upper triangle and
               symmetrised (average ranks for ties)"""
    A = np.asarray(A, dtype=np.float64)
    if mode == "max":
        return A / A.max()
    if mode == "logmed":
        m = np.median(A[A > 0])
        return np.log1p(A / m) / np.log1p(A.max() / m)
    if mode == "log":
        return np.log1p(A) / np.log1p(A.max())
    if mode == "binary":
        return (A > 0).astype(np.float64)
    if mode == "rank":
        from scipy.stats import rankdata
        n = A.shape[0]
        iu = np.triu_indices(n, 1)
        v = A[iu]
        out_u = np.zeros_like(v)
        pos = v > 0
        out_u[pos] = rankdata(v[pos], method="average") / pos.sum()
        out = np.zeros_like(A)
        out[iu] = out_u
        return out + out.T
    raise ValueError(mode)


def kernel_weights(times: np.ndarray, latent_dim: int) -> np.ndarray:
    w = (4 * np.pi * times) ** (-latent_dim / 2.0)
    return w / w.sum()


def heat_kernel(d: torch.Tensor, times: torch.Tensor, weights: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Normalised multi-scale kernel h_s(d) = sum_k w_k exp(-(s d)^2 / 4 t_k), h_s(0) = 1."""
    x = (scale * d).unsqueeze(-1) ** 2
    return (weights * torch.exp(-x / (4.0 * times))).sum(-1)


def pair_loss(D: torch.Tensor, At: torch.Tensor, times: torch.Tensor, weights: torch.Tensor,
              scale: torch.Tensor, amp: torch.Tensor) -> torch.Tensor:
    """mean_{i<j} (A~_ij - a h_s(d_ij))^2."""
    n = D.shape[0]
    iu, ju = torch.triu_indices(n, n, offset=1, device=D.device)
    return ((At[iu, ju] - amp * heat_kernel(D[iu, ju], times, weights, scale)) ** 2).mean()


def _h_np(d: np.ndarray, times: np.ndarray, weights: np.ndarray, s: float) -> np.ndarray:
    return (weights[None, :] * np.exp(-((s * d)[:, None] ** 2) / (4 * times[None, :]))).sum(1)


def calibrate(D1: np.ndarray, At: np.ndarray, times: np.ndarray, weights: np.ndarray,
              grid: Tuple[float, float, int] = (-12.0, 12.0, 241)) -> Tuple[float, float, float, bool]:
    """Least-squares calibration of the two global constants of the kernel on distances D1:
        (s*, a*) = argmin_{s>0, a>=0} mean_{i<j} (A~_ij - a h_s(D1_ij))^2.
    For fixed s the optimal amplitude is closed-form, a*(s) = max(0, <A~,h_s>/<h_s,h_s>), so the
    search is 1-D in log s (grid + golden-section refinement).
    Returns (s*, a*, loss*, hit_boundary). hit_boundary=True flags a degenerate calibration.

    Why a free amplitude: with the amplitude fixed at h(0)=1, least squares
    preferred h -> 0 whenever close pairs are linked with probability < ~0.5, which froze
    Phase 2 in weakly homophilic graphs (calibration at the grid boundary, zero gradients)."""
    iu = np.triu_indices(D1.shape[0], 1)
    d, a = D1[iu], At[iu]
    aa = float((a * a).mean())

    def f(logs):
        h = _h_np(d, times, weights, np.exp(logs))
        hh = float((h * h).mean())
        amp = max(0.0, float((a * h).mean()) / hh) if hh > 0 else 0.0
        return aa - 2 * amp * float((a * h).mean()) + amp * amp * hh, amp

    xs = np.linspace(*grid)
    vals = np.array([f(x)[0] for x in xs])
    k = int(np.argmin(vals))
    hit = k in (0, len(xs) - 1)
    lo, hi = xs[max(k - 1, 0)], xs[min(k + 1, len(xs) - 1)]
    g = (np.sqrt(5) - 1) / 2
    for _ in range(60):
        c1, c2 = hi - g * (hi - lo), lo + g * (hi - lo)
        if f(c1)[0] < f(c2)[0]:
            hi = c2
        else:
            lo = c1
    logs = 0.5 * (lo + hi)
    loss, amp = f(logs)
    return float(np.exp(logs)), float(amp), float(loss), bool(hit)


def residuals(D: np.ndarray, At: np.ndarray, times: np.ndarray, weights: np.ndarray, scale: float, amp: float) -> np.ndarray:
    t = times[None, None, :]
    w = weights[None, None, :]
    H = amp * (w * np.exp(-((scale * D)[..., None] ** 2) / (4 * t))).sum(-1)
    R = At - H
    np.fill_diagonal(R, 0.0)
    return R


# ------------------------------------------------------------------------------------------
# Cross-entropy (latent-space-model) objective, single scale: the objective used in the paper
# ------------------------------------------------------------------------------------------
_EPS = 1e-9


def _log_h(d: torch.Tensor, times: torch.Tensor, weights: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """log h_s(d) computed with log-sum-exp (no underflow at large distances)."""
    x = (scale * d).unsqueeze(-1) ** 2
    return torch.logsumexp(torch.log(weights) - x / (4.0 * times), dim=-1)


def pair_loss_bce(D: torch.Tensor, At: torch.Tensor, times: torch.Tensor, weights: torch.Tensor,
                  scale: torch.Tensor, amp: torch.Tensor) -> torch.Tensor:
    """mean_{i<j} [ -A~ log H - (1 - A~) log(1 - H) ],  H = a h_s(d) with a in (0, 1).

    Soft-label Bernoulli cross-entropy, i.e. the likelihood of a latent-space network model
    (Hoff, Raftery & Handcock 2002) whose distances are the manifold geodesics. Unlike the squared
    error, -log H grows like (s d)^2 / (4 t_max) for an observed edge at large distance, so
    geometrically implausible links keep a non-vanishing gradient. log H is computed with
    log-sum-exp and log(1-H) = log1p(-exp(log H)) (H <= a < 1): no clamping, no vanishing gradient."""
    n = D.shape[0]
    iu, ju = torch.triu_indices(n, n, offset=1, device=D.device)
    logH = torch.log(amp) + _log_h(D[iu, ju], times, weights, scale)
    a = At[iu, ju]
    return -(a * logH + (1 - a) * torch.log1p(-torch.exp(logH))).mean()


def calibrate_bce(D1: np.ndarray, At: np.ndarray, times: np.ndarray, weights: np.ndarray,
                  init: Tuple[float, float]) -> Tuple[float, float, float]:
    """(s*, a*) minimising the cross-entropy on Phase 1 distances (L-BFGS on (log s, logit a))."""
    iu = np.triu_indices(D1.shape[0], 1)
    d = torch.as_tensor(D1[iu]); a = torch.as_tensor(At[iu])
    T = torch.as_tensor(times); Wt = torch.as_tensor(weights)
    ls = torch.tensor(np.log(init[0]), dtype=torch.float64, requires_grad=True)
    la = torch.tensor(np.log(min(max(init[1], 1e-6), 0.99) / (1 - min(max(init[1], 1e-6), 0.99))), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([ls, la], lr=1.0, max_iter=200, line_search_fn="strong_wolfe")

    def f():
        logH = torch.nn.functional.logsigmoid(la) + _log_h(d, T, Wt, torch.exp(ls))
        return -(a * logH + (1 - a) * torch.log1p(-torch.exp(logH))).mean()

    def closure():
        opt.zero_grad()
        l = f()
        l.backward()
        return l

    opt.step(closure)
    with torch.no_grad():
        return float(torch.exp(ls)), float(torch.sigmoid(la)), float(f())
