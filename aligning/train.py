"""
Phase 1 and Phase 2 training.

Phase 1: VAE on the attributes alone (objective in models.py, sigmoid KL annealing), seeded.

Phase 2 (see heat.py for the objective):
  * encoder frozen; latent coordinates = posterior means;
  * every decoder parameter (mean and variance heads) is trainable; optionally the
    log-variance head can be frozen (ablation);
  * the metric field and ALL edge weights are recomputed from the current decoder
    at every step;
  * early stopping on the training loss only.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np
import torch

from .models import VAE, VAEConfig, set_seed
from .metric import GridSpec, MetricField
from .geodesics import GridGeodesics, pairwise_straight_line
from . import heat

DTYPE = torch.float64


# ----------------------------------------------------------------------------- Phase 1
@dataclass
class Phase1Config:
    epochs: int = 1500
    lr: float = 5e-3
    weight_decay: float = 1e-5


def train_phase1(X: np.ndarray, vcfg: VAEConfig, pcfg: Phase1Config, seed: int, verbose: bool = False):
    set_seed(seed)
    vae = VAE(vcfg).to(DTYPE)
    Xt = torch.as_tensor(X, dtype=DTYPE)
    opt = torch.optim.Adam(vae.parameters(), lr=pcfg.lr, weight_decay=pcfg.weight_decay)
    hist = []
    for ep in range(pcfg.epochs):
        vae.train()
        opt.zero_grad()
        mu, logvar = vae.encoder(Xt)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        mx, lvx = vae.decoder(z)
        rec = vae.recon_loss(Xt, mx, lvx)
        kl = vae.kl(mu, logvar)
        beta = vae.beta(ep, pcfg.epochs)
        loss = rec + beta * kl
        loss.backward()
        opt.step()
        hist.append((float(rec), float(kl), float(beta)))
        if verbose and (ep + 1) % 250 == 0:
            print(f"[phase1 seed={seed}] ep {ep+1}: rec={float(rec):.3f} kl={float(kl):.3f} beta={beta:.3f}")
    vae.eval()
    return vae, np.array(hist)


# ----------------------------------------------------------------------------- distances
@dataclass
class GeoConfig:
    estimator: str = "hybrid"   # "hybrid" (default) | "grid" | "line"
    P: int = 64                 # grid cells per dimension
    neighbourhood: int = 32     # 2-D stencil: 8 | 16 | 32
    T: int = 21                 # Simpson points for straight segments
    metric: str = "expected"    # "expected" | "mean" | "legacy"
    margin: float = 0.1         # relative margin of the latent bounding box


class DistanceEngine:
    def __init__(self, Z: np.ndarray, gcfg: GeoConfig):
        self.Z = np.asarray(Z, dtype=np.float64)
        self.cfg = gcfg
        self.spec = GridSpec.from_points(self.Z, gcfg.P, gcfg.margin)
        self.Zt = torch.as_tensor(self.Z, dtype=DTYPE)
        self.geo = None
        if gcfg.estimator in ("hybrid", "grid"):
            self.geo = GridGeodesics(self.spec, self.Z, neighbourhood=gcfg.neighbourhood)

    def field(self, decoder) -> MetricField:
        return MetricField.from_decoder(decoder, self.spec, self.cfg.metric, dtype=DTYPE)

    def distances(self, decoder) -> torch.Tensor:
        f = self.field(decoder)
        if self.cfg.estimator == "hybrid":
            return self.geo.hybrid_distances(f, T=self.cfg.T)
        if self.cfg.estimator == "grid":
            D = self.geo.distances(f)
            return 0.5 * (D + D.T)
        if self.cfg.estimator == "line":
            return pairwise_straight_line(f, self.Zt, T=self.cfg.T)
        raise ValueError(self.cfg.estimator)


# ----------------------------------------------------------------------------- Phase 2
@dataclass
class Phase2Config:
    K: int = 15                 # number of diffusion times (paper: 15)
    c: float = 4.0              # t_max = c / lambda_2 (paper: 4)
    target: str = "max"         # normalisation of A (heat.normalize_target)
    epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 0.0   # an L2 pull towards 0 combined with the free
                                # global scale s created a data-flat "shrink the decoder, grow s" drift
                                # that changed node rankings (Spearman ~0.83 between wd 1e-5 and 0)
    learn_scale: bool = True    # learn the two global kernel constants (s, a) during Phase 2
    freeze_logvar: bool = False
    grad_clip: float = 1.0
    patience: int = 30
    rel_tol: float = 1e-4
    loss: str = "mse"            # "mse" (multi-scale squared error) | "bce" (single-scale cross-entropy, used in the paper)


def run_phase2(vae: VAE, X: np.ndarray, A: np.ndarray, gcfg: GeoConfig, p2: Phase2Config,
               seed: int, verbose: bool = False, Z: Optional[np.ndarray] = None) -> dict:
    set_seed(seed)
    Xt = torch.as_tensor(X, dtype=DTYPE)
    if Z is None:
        Z = vae.latent_means(Xt).numpy()
    eng = DistanceEngine(Z, gcfg)
    dec1 = vae.decoder
    with torch.no_grad():
        D1 = eng.distances(dec1).numpy()
    times = heat.heat_times(A, p2.K, p2.c)
    weights = heat.kernel_weights(times, Z.shape[1])
    At = heat.normalize_target(A, p2.target)
    np.fill_diagonal(At, 0.0)
    s0, a0, loss_p1, calib_boundary = heat.calibrate(D1, At, times, weights)
    iu_ = np.triu_indices(At.shape[0], 1)
    calib_r2 = 1.0 - loss_p1 / float(np.mean(At[iu_] ** 2))      # R^2 relative to predicting 0 for every pair
    if p2.loss == "bce":   # cross-entropy objective, calibrated from the MSE solution
        s0, a0, loss_p1 = heat.calibrate_bce(D1, At, times, weights, init=(s0, a0))
    loss_fn = heat.pair_loss_bce if p2.loss == "bce" else heat.pair_loss

    dec2 = copy.deepcopy(dec1)
    for p in dec2.parameters():
        p.requires_grad_(True)
    if p2.freeze_logvar:
        for p in dec2.logvar_head.parameters():
            p.requires_grad_(False)
    log_s = torch.tensor(np.log(s0), dtype=DTYPE, requires_grad=p2.learn_scale)
    log_a = torch.tensor(np.log(max(a0, 1e-12)), dtype=DTYPE, requires_grad=p2.learn_scale)
    if p2.loss == "bce":   # keep a in (0,1): parametrise by its logit
        log_a = torch.tensor(np.log(a0 / (1 - a0)), dtype=DTYPE, requires_grad=p2.learn_scale)
    amp_of = (lambda la: torch.sigmoid(la)) if p2.loss == "bce" else (lambda la: torch.exp(la))
    params = [p for p in dec2.parameters() if p.requires_grad] + ([log_s, log_a] if p2.learn_scale else [])
    opt = torch.optim.Adam(params, lr=p2.lr, weight_decay=0.0)
    T_ = torch.as_tensor(times, dtype=DTYPE)
    W_ = torch.as_tensor(weights, dtype=DTYPE)
    A_ = torch.as_tensor(At, dtype=DTYPE)

    hist, best, since = [], np.inf, 0
    t0 = time.time()
    for ep in range(p2.epochs):
        opt.zero_grad()
        D = eng.distances(dec2)
        loss = loss_fn(D, A_, T_, W_, torch.exp(log_s), amp_of(log_a))
        # coupled L2 on decoder parameters (0 by default)
        reg = 0.5 * p2.weight_decay * sum((p ** 2).sum() for p in dec2.parameters() if p.requires_grad) if p2.weight_decay else 0.0
        (loss + reg).backward()
        if p2.grad_clip:
            torch.nn.utils.clip_grad_norm_(params, p2.grad_clip)
        opt.step()
        l = float(loss)
        hist.append((l, float(torch.exp(log_s)), float(amp_of(log_a))))
        if l < best * (1 - p2.rel_tol):
            best, since = l, 0
        else:
            since += 1
        if verbose and (ep + 1) % 25 == 0:
            print(f"[phase2 seed={seed}] ep {ep+1}: loss={l:.6f} (P1-calibrated {loss_p1:.6f}) s={float(torch.exp(log_s)):.4g} a={float(amp_of(log_a)):.4g} {time.time()-t0:.0f}s")
        if since >= p2.patience:
            break
    with torch.no_grad():
        D2 = eng.distances(dec2).numpy()
        loss_final = float(loss_fn(torch.as_tensor(D2), A_, T_, W_, torch.exp(log_s), amp_of(log_a)))
    return dict(Z=Z, D1=D1, D2=D2, times=times, weights=weights, At=At, scale_p1=s0, amp_p1=a0,
                calib_at_boundary=calib_boundary, calib_r2_vs_zero=calib_r2,
                scale_p2=float(torch.exp(log_s)), amp_p2=float(amp_of(log_a)), loss_p1=loss_p1, loss_p2=loss_final,
                history=np.array(hist), epochs_run=len(hist), seconds=time.time() - t0,
                decoder2=dec2, engine=eng)
