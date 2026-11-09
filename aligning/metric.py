"""
Pullback Riemannian metric of the variational decoder, and its representation
as a field sampled on a regular latent grid with multilinear interpolation.

Metric choices
--------------
"expected" : G = J_mu^T J_mu + J_sigma^T J_sigma
             (expected metric of a Gaussian decoder, Arvanitidis et al. 2018). DEFAULT.
"mean"     : G = J_mu^T J_mu (deterministic mean decoder).
"legacy"   : G = (J_mu + J_sigma)^T (J_mu + J_sigma), an incorrect variant kept for comparison only.

Grid representation
-------------------
The decoder is evaluated (with autograd) only on the (P+1)^d grid nodes; any other
point gets its metric by multilinear interpolation of the node values.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


def pullback_metric(decoder, z: torch.Tensor, kind: str = "expected") -> torch.Tensor:
    J_mu, J_sigma = decoder.jacobians(z)
    if kind == "expected":
        return J_mu.transpose(-1, -2) @ J_mu + J_sigma.transpose(-1, -2) @ J_sigma
    if kind == "mean":
        return J_mu.transpose(-1, -2) @ J_mu
    if kind == "legacy":
        J = J_mu + J_sigma
        return J.transpose(-1, -2) @ J
    raise ValueError(kind)


@dataclass
class GridSpec:
    bounds: np.ndarray  # (d, 2)
    P: int              # number of cells per dimension -> (P+1)^d nodes

    @property
    def d(self) -> int:
        return self.bounds.shape[0]

    @property
    def steps(self) -> np.ndarray:
        return (self.bounds[:, 1] - self.bounds[:, 0]) / self.P

    @property
    def n_nodes(self) -> int:
        return (self.P + 1) ** self.d

    def strides(self) -> np.ndarray:
        s = [1]
        for _ in range(self.d - 1):
            s.insert(0, s[0] * (self.P + 1))
        return np.array(s, dtype=np.int64)

    def node_index(self, multi: np.ndarray) -> np.ndarray:
        return (multi * self.strides()).sum(-1)

    def node_coords(self) -> np.ndarray:
        axes = [np.linspace(self.bounds[k, 0], self.bounds[k, 1], self.P + 1) for k in range(self.d)]
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.stack([m.ravel() for m in mesh], axis=1)

    @staticmethod
    def from_points(points: np.ndarray, P: int, margin: float = 0.1) -> "GridSpec":
        lo, hi = points.min(0), points.max(0)
        pad = (hi - lo) * margin
        return GridSpec(np.stack([lo - pad, hi + pad], axis=1).astype(np.float64), P)


class MetricField:
    """Metric sampled on grid nodes; multilinear interpolation elsewhere."""

    def __init__(self, spec: GridSpec, G_nodes: torch.Tensor):
        self.spec = spec
        self.G = G_nodes  # (n_nodes, d, d), may carry autograd
        self._lo = torch.tensor(spec.bounds[:, 0], dtype=G_nodes.dtype, device=G_nodes.device)
        self._h = torch.tensor(spec.steps, dtype=G_nodes.dtype, device=G_nodes.device)
        self._strides = torch.tensor(spec.strides(), device=G_nodes.device)

    @staticmethod
    def from_decoder(decoder, spec: GridSpec, kind: str = "expected", dtype=torch.float64) -> "MetricField":
        nodes = torch.tensor(spec.node_coords(), dtype=dtype)
        G = pullback_metric(decoder, nodes, kind)
        return MetricField(spec, G)

    @staticmethod
    def from_function(fn, spec: GridSpec, dtype=torch.float64) -> "MetricField":
        nodes = torch.tensor(spec.node_coords(), dtype=dtype)
        return MetricField(spec, fn(nodes))

    def interpolate(self, pts: torch.Tensor) -> torch.Tensor:
        """Multilinear interpolation of G at pts (n,d). Points are clamped to the box."""
        d, P = self.spec.d, self.spec.P
        pts = pts.to(self.G.dtype)
        rel = (pts - self._lo) / self._h
        rel = torch.clamp(rel, 0.0, float(P))
        base = torch.clamp(torch.floor(rel), 0, P - 1)
        frac = rel - base
        base = base.long()
        out = 0.0
        for corner in range(2 ** d):
            bits = [(corner >> k) & 1 for k in range(d)]
            w = torch.ones(pts.shape[0], dtype=self.G.dtype, device=pts.device)
            idx = torch.zeros(pts.shape[0], dtype=torch.long, device=pts.device)
            for k, b in enumerate(bits):
                w = w * (frac[:, k] if b else (1.0 - frac[:, k]))
                idx = idx + (base[:, k] + b) * self._strides[k]
            out = out + w[:, None, None] * self.G[idx]
        return out
