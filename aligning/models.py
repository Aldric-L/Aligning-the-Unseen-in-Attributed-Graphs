"""
VAE used in Phase 1 (and whose decoder is deformed in Phase 2).

MLP encoder, variational MLP decoder (mean and log-variance heads), standard Gaussian prior and sigmoid
KL annealing (Supplementary Section S2), with:
  * explicit seeding,
  * an analytic, batched, differentiable decoder Jacobian that is *checked* against
    autograd in tests/,
  * the correct expected pullback metric of a Gaussian decoder (see metric.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ACTS = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "softplus": nn.Softplus}


def _act_and_derivative(name: str, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (phi(a), phi'(a)) for a supported activation."""
    if name == "elu":
        out = F.elu(a)
        der = torch.where(a > 0, torch.ones_like(a), out + 1.0)  # alpha=1: d/da (e^a - 1) = e^a
        return out, der
    if name == "relu":
        return F.relu(a), (a > 0).to(a.dtype)
    if name == "tanh":
        out = torch.tanh(a)
        return out, 1.0 - out ** 2
    if name == "softplus":
        return F.softplus(a), torch.sigmoid(a)
    raise ValueError(f"unsupported activation {name}")


def _xavier(layer: nn.Linear) -> nn.Linear:
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class MLPEncoder(nn.Module):
    """q(z|x) = N(mu(x), diag(exp(logvar(x)))); MLP on attributes only (no graph)."""

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int,
                 activation: str = "elu", dropout: float = 0.2):
        super().__init__()
        self.activation = activation
        self.dropout = dropout
        dims = [input_dim] + list(hidden_dims)
        self.layers = nn.ModuleList([_xavier(nn.Linear(a, b)) for a, b in zip(dims[:-1], dims[1:])])
        self.mu = _xavier(nn.Linear(dims[-1], latent_dim))
        self.logvar = _xavier(nn.Linear(dims[-1], latent_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = x
        for lin in self.layers:
            h, _ = _act_and_derivative(self.activation, lin(h))
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.mu(h), self.logvar(h)


class VariationalDecoder(nn.Module):
    """p(x|z) = N(mu(z), diag(sigma(z)^2)), shared MLP trunk + mean and log-variance heads."""

    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: List[int],
                 activation: str = "elu", clip_logvar: Optional[float] = None):
        super().__init__()
        self.activation = activation
        self.clip_logvar = clip_logvar
        dims = [latent_dim] + list(hidden_dims)
        self.layers = nn.ModuleList([_xavier(nn.Linear(a, b)) for a, b in zip(dims[:-1], dims[1:])])
        self.mean_head = _xavier(nn.Linear(dims[-1], output_dim))
        self.logvar_head = _xavier(nn.Linear(dims[-1], output_dim))

    def _trunk(self, z: torch.Tensor, with_jac: bool):
        h = z
        J = None
        if with_jac:
            n, d = z.shape
            J = torch.eye(d, dtype=z.dtype, device=z.device).expand(n, d, d)
        for lin in self.layers:
            a = lin(h)
            h, der = _act_and_derivative(self.activation, a)
            if with_jac:
                J = der.unsqueeze(-1) * torch.matmul(lin.weight, J)  # (n,H,d)
        return h, J

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h, _ = self._trunk(z, with_jac=False)
        mu = self.mean_head(h)
        logvar = self.logvar_head(h)
        if self.clip_logvar is not None:
            logvar = torch.clamp(logvar, -self.clip_logvar, self.clip_logvar)
        return mu, logvar

    def jacobians(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Analytic Jacobians of the mean and of the standard deviation sigma=exp(logvar/2).

        Returns J_mu (n,D,d) and J_sigma (n,D,d). Differentiable w.r.t. the parameters.
        The derivative of the clamp is taken into account (zero where clipping is active).
        """
        h, Jh = self._trunk(z, with_jac=True)
        J_mu = torch.matmul(self.mean_head.weight, Jh)
        lv = self.logvar_head(h)
        J_lv = torch.matmul(self.logvar_head.weight, Jh)
        if self.clip_logvar is not None:
            inside = (lv.abs() < self.clip_logvar).to(lv.dtype)
            lv = torch.clamp(lv, -self.clip_logvar, self.clip_logvar)
            J_lv = J_lv * inside.unsqueeze(-1)
        sigma = torch.exp(0.5 * lv)
        J_sigma = 0.5 * sigma.unsqueeze(-1) * J_lv
        return J_mu, J_sigma


@dataclass
class VAEConfig:
    input_dim: int
    latent_dim: int = 2
    enc_hidden: List[int] = field(default_factory=lambda: [16, 16])
    dec_hidden: List[int] = field(default_factory=lambda: [16])
    activation: str = "elu"
    enc_dropout: float = 0.2
    clip_logvar: Optional[float] = None
    # weights of the two variance-matching terms of the Phase 1 objective (Supplementary Section S2)
    lambda_comp_variance: float = 5.0
    lambda_decoder_variance: float = 100.0
    # KL annealing (sigmoid, steepness 5)
    beta_start: float = 0.0
    beta_end: float = 2.0


class VAE(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = MLPEncoder(cfg.input_dim, cfg.enc_hidden, cfg.latent_dim, cfg.activation, cfg.enc_dropout)
        self.decoder = VariationalDecoder(cfg.latent_dim, cfg.input_dim, cfg.dec_hidden, cfg.activation, cfg.clip_logvar)

    @torch.no_grad()
    def latent_means(self, x: torch.Tensor) -> torch.Tensor:
        was = self.training
        self.eval()
        mu, _ = self.encoder(x)
        self.train(was)
        return mu

    def recon_loss(self, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Phase 1 reconstruction objective: Gaussian likelihood plus the two variance-matching terms."""
        eps = 1e-8
        std = torch.exp(0.5 * logvar) + eps
        nll = -torch.distributions.Normal(mu, std).log_prob(x).sum(1).mean()
        D = x.shape[1]
        var_emp = ((x - x.mean(1, keepdim=True)) ** 2).sum(1) / D
        var_mu = ((mu - mu.mean(1, keepdim=True)) ** 2).sum(1) / D
        comp = ((var_mu - var_emp) ** 2).mean()
        resid = torch.clamp(var_emp - var_mu, min=eps)
        dec = ((torch.exp(logvar).mean(1) - resid) ** 2).mean()
        return nll + self.cfg.lambda_comp_variance * comp + self.cfg.lambda_decoder_variance * dec

    @staticmethod
    def kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    def beta(self, step: int, total: int) -> float:
        p = min(1.0, step / max(total, 1))
        s = 1.0 / (1.0 + np.exp(-5.0 * (p - 0.5)))
        return self.cfg.beta_start + s * (self.cfg.beta_end - self.cfg.beta_start)


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
