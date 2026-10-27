"""Loss terms: the GRAnD margin, its smooth variant and the AE-SAD corruption."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def softabs(x: torch.Tensor, tau: float) -> torch.Tensor:
    """Smooth |x|: tau * (softplus(x/tau) + softplus(-x/tau)) - 2 tau log 2.

    Differentiable everywhere, equal to 0 at x = 0 and to |x| as tau -> 0.
    """
    tau = max(float(tau), 1e-8)
    return tau * (F.softplus(x / tau) + F.softplus(-x / tau)) - 2.0 * tau * math.log(2.0)


def margin_loss(energy: torch.Tensor, m: float, kind: str = "abs", tau: float = 1.0) -> torch.Tensor:
    """Per-sample |m - F(x)| (paper, Eq. 7) or its smooth version."""
    if kind == "abs":
        return (m - energy).abs()
    if kind == "softabs":
        return softabs(m - energy, tau)
    raise ValueError(f"unknown margin {kind!r}")


def feature_swap(x: torch.Tensor, p: float) -> torch.Tensor:
    """AE-SAD corruption F(x): each feature is replaced, with probability p,
    by the same feature of another sample in the batch.

    Unlike F(x) = 1 - x of the AE-SAD paper, this keeps every marginal
    distribution intact (one-hot blocks stay valid values) and only breaks the
    joint structure, which is what tells NSL-KDD connections apart.
    """
    partner = x[torch.randperm(x.size(0), device=x.device)]
    mask = torch.rand_like(x) < p
    return torch.where(mask, partner, x)


def ramp(epoch: int, warmup: int, length: int) -> float:
    """0 up to `warmup`, then linear up to 1 over `length` epochs."""
    if epoch <= warmup:
        return 0.0 if (warmup > 0 or length > 0) else 1.0
    if length <= 0 or epoch >= warmup + length:
        return 1.0
    return (epoch - warmup) / length
