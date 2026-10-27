"""Local explanations of the free-energy score.

The score splits into a reconstruction part, which decomposes over features,
and a KL part, which does not. When the reconstruction part dominates, the
per-feature squared errors are the explanation; when the KL part dominates,
Integrated Gradients (Sundararajan et al., 2017) attributes the whole energy
to the input features.

Both use the same fixed latent draws as `evaluate.score`, so the function
being explained is exactly the score being reported, and IG satisfies
completeness: the attributions sum to F(x) - F(baseline) up to the
discretisation error of the path integral.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from grand.model import GRAnD, free_energy

ONE_HOT_GROUPS = ("protocol_type_", "service_", "flag_")


def _fixed_eps(model: GRAnD, n_samples: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(n_samples, model.cfg.latent_dim, generator=gen)


def _mean_energy(model: GRAnD, x: torch.Tensor, eps: torch.Tensor, kl_weight: float):
    """Mean free energy, reconstruction error and squared error per feature of each row of x."""
    energy = torch.zeros(x.size(0))
    recon = torch.zeros(x.size(0))
    per_feature = torch.zeros_like(x)
    for e in eps:
        out = model(x, e.expand(x.size(0), -1))
        sq = (out.recon - x).pow(2)
        energy = energy + free_energy(x, out, kl_weight)
        recon = recon + sq.sum(dim=1)
        per_feature = per_feature + sq
    k = eps.size(0)
    return energy / k, recon / k, per_feature / k


@dataclass
class Explanation:
    energy: float
    reconstruction: float  # part of the energy due to reconstruction
    kl: float  # remaining part, kl_weight * KL term
    squared_error: np.ndarray  # per feature, sums to `reconstruction`
    integrated_gradients: np.ndarray  # per feature, sums to energy - baseline energy (approx.)
    completeness_gap: float  # |sum(IG) - (F(x) - F(baseline))|

    @property
    def reconstruction_share(self) -> float:
        return self.reconstruction / self.energy if self.energy else float("nan")


def explain(
    model: GRAnD,
    x: np.ndarray,
    *,
    kl_weight: float = 1.0,
    baseline: np.ndarray | None = None,
    steps: int = 128,
    n_samples: int = 16,
    seed: int = 0,
) -> Explanation:
    """Explain the score of a single sample x (1-D array)."""
    model.eval()
    x_t = torch.as_tensor(x, dtype=torch.float32).reshape(1, -1)
    b_t = torch.zeros_like(x_t) if baseline is None else torch.as_tensor(baseline, dtype=torch.float32).reshape(1, -1)
    eps = _fixed_eps(model, n_samples, seed)

    with torch.no_grad():
        energy, recon, per_feature = _mean_energy(model, x_t, eps, kl_weight)
        energy_b, _, _ = _mean_energy(model, b_t, eps, kl_weight)

    # Integrated Gradients, midpoint rule along the straight path b -> x
    alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps
    path = (b_t + alphas[:, None] * (x_t - b_t)).requires_grad_(True)
    path_energy, _, _ = _mean_energy(model, path, eps, kl_weight)
    (grads,) = torch.autograd.grad(path_energy.sum(), path)
    ig = ((x_t - b_t) * grads.mean(dim=0, keepdim=True)).squeeze(0).detach().numpy()

    e, r = float(energy), float(recon)
    return Explanation(
        energy=e,
        reconstruction=r,
        kl=e - r,
        squared_error=per_feature.squeeze(0).numpy(),
        integrated_gradients=ig,
        completeness_gap=abs(float(ig.sum()) - (e - float(energy_b))),
    )


def group_one_hot(feature_names: list[str], values: np.ndarray) -> dict[str, float]:
    """Sum the contributions of each one-hot block (protocol_type, service, flag)."""
    grouped: dict[str, float] = defaultdict(float)
    for name, v in zip(feature_names, values):
        key = next((g[:-1] for g in ONE_HOT_GROUPS if name.startswith(g)), name)
        grouped[key] += float(v)
    return dict(sorted(grouped.items(), key=lambda kv: -abs(kv[1])))
