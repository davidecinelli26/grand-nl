"""VAE with a planar normalizing flow on the latent space (GRAnD-PF).

The anomaly score of GRAnD is the free energy (negative ELBO) of a sample,

    F(x) = ||x - x_hat||^2 + log q_K(z_K | x) - log p(z_K),

where z_0 ~ q_0(z | x) = N(mu(x), sigma(x)^2) is pushed through K planar flows
and log q_K(z_K | x) = log q_0(z_0 | x) - sum_k log |det J_k|.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from grand.config import ModelConfig

_LOG_2PI = math.log(2.0 * math.pi)


class ForwardOutput(NamedTuple):
    recon: torch.Tensor  # (B, D) reconstruction in [0, 1]
    z: torch.Tensor  # (B, d) latent code after the flows
    kl: torch.Tensor  # (B,) single-sample estimate of log q_K(z_K | x) - log p(z_K)


class PlanarFlow(nn.Module):
    """f(z) = z + u_hat * tanh(w^T z + b) (Rezende & Mohamed, 2015).

    The reference implementation uses u directly, which makes f invertible
    only when w^T u >= -1 happens to hold. Here u is reparametrised as u_hat
    (Appendix A of the paper) so that the condition holds by construction and
    log |det J| is the log-density correction of a true bijection.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.empty(1, dim).uniform_(-0.01, 0.01))
        self.u = nn.Parameter(torch.empty(1, dim).uniform_(-0.01, 0.01))
        self.b = nn.Parameter(torch.empty(1).uniform_(-0.01, 0.01))

    def u_hat(self) -> torch.Tensor:
        wu = (self.w * self.u).sum()
        m_wu = -1.0 + F.softplus(wu)
        return self.u + (m_wu - wu) * self.w / (self.w.pow(2).sum() + 1e-12)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u = self.u_hat()
        h = torch.tanh(z @ self.w.t() + self.b)  # (B, 1)
        psi = (1.0 - h.pow(2)) * self.w  # (B, d)
        log_det = torch.log((1.0 + psi @ u.t()).abs().squeeze(1) + 1e-9)
        return z + u * h, log_det


class _ResBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, 4 * width)
        self.fc2 = nn.Linear(4 * width, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(F.gelu(self.fc1(self.norm(x))))
        return x + self.drop(self.fc2(h))


def _bodies(input_dim: int, cfg: ModelConfig) -> tuple[nn.Module, nn.Module, int]:
    """Encoder and decoder trunks, and the width of their last hidden layer."""
    if cfg.arch == "mlp":
        encoder = nn.Sequential(nn.Linear(input_dim, cfg.hidden_dim), nn.ReLU())
        decoder = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim), nn.ReLU(), nn.Linear(cfg.hidden_dim, input_dim)
        )
        return encoder, decoder, cfg.hidden_dim
    if cfg.arch == "resmlp":
        width = 2 ** round(math.log2(max(input_dim, 16)))

        def blocks() -> list[nn.Module]:
            return [_ResBlock(width, cfg.resmlp_dropout) for _ in range(cfg.resmlp_depth)]

        encoder = nn.Sequential(
            nn.Linear(input_dim, width), nn.GELU(), nn.Dropout(cfg.resmlp_dropout), *blocks(), nn.LayerNorm(width)
        )
        decoder = nn.Sequential(nn.Linear(cfg.latent_dim, width), nn.GELU(), *blocks(), nn.Linear(width, input_dim))
        return encoder, decoder, width
    raise ValueError(f"unknown arch {cfg.arch!r}")


class GRAnD(nn.Module):
    """Variational autoencoder with K planar flows; the decoder ends in a sigmoid."""

    def __init__(self, input_dim: int, cfg: ModelConfig | None = None):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.input_dim = input_dim
        self.cfg = cfg
        self.encoder, decoder, width = _bodies(input_dim, cfg)
        self.decoder = nn.Sequential(decoder, nn.Sigmoid())
        self.mu = nn.Linear(width, cfg.latent_dim)
        self.logvar = nn.Linear(width, cfg.latent_dim)
        self.flows = nn.ModuleList(PlanarFlow(cfg.latent_dim) for _ in range(cfg.n_flows))

    def forward(self, x: torch.Tensor, eps: torch.Tensor | None = None) -> ForwardOutput:
        """One stochastic pass. Passing `eps` fixes the sample and makes it deterministic."""
        h = self.encoder(x)
        mu, logvar = self.mu(h), self.logvar(h)
        std = torch.exp(0.5 * logvar)
        if eps is None:
            eps = torch.randn_like(std)
        z = mu + eps * std
        # log N(z0; mu, std^2), written through eps to avoid a division by std
        log_q = (-0.5 * eps.pow(2) - torch.log(std) - 0.5 * _LOG_2PI).sum(dim=1)
        for flow in self.flows:
            z, log_det = flow(z)
            log_q = log_q - log_det
        log_p = (-0.5 * z.pow(2) - 0.5 * _LOG_2PI).sum(dim=1)
        return ForwardOutput(self.decoder(z), z, log_q - log_p)


def free_energy(x: torch.Tensor, out: ForwardOutput, kl_weight: float = 1.0) -> torch.Tensor:
    """Per-sample free energy: squared reconstruction error plus the weighted KL term."""
    return (out.recon - x).pow(2).sum(dim=1) + kl_weight * out.kl
