"""Thresholds that split the training energies into L, U and S.

GRAnD (paper, Sec. 4.1) uses Peaks-Over-Threshold:

    u   = min(Q3 + alpha * IQR, Q_{1-q})
    t_q = u + (sigma / xi) * ((q n / N_u)^(-xi) - 1)

where (xi, sigma) are the maximum-likelihood parameters of a Generalized
Pareto Distribution fitted on the N_u exceedances over u. Equation 4 of the
paper prints the ratio upside down; the reference code and the standard POT
quantile formula (Siffer et al., KDD 2017) use sigma / xi, as here.

The reference code fits the GPD through R (rpy2 + the POT package); this
module uses scipy's MLE, which estimates the same model without R.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import genpareto

from grand.config import TrainConfig


@dataclass(frozen=True)
class Thresholds:
    u: float
    tq: float
    n_exceed: int = 0
    shape: float = float("nan")
    scale: float = float("nan")
    tq_pot: float = float("nan")  # uncapped POT estimate
    fallback: str | None = None  # why the empirical quantile replaced the GPD, if it did


def pot_thresholds(
    energy: np.ndarray,
    alpha: float,
    q: float,
    *,
    cap: bool = False,
    min_exceedances: int = 30,
    shape_eps: float = 1e-6,
) -> Thresholds:
    """POT thresholds. With `cap`, t_q is bounded by the empirical quantile Q_{1-q}."""
    e = np.asarray(energy, dtype=np.float64)
    n = e.size
    q1, q3 = np.quantile(e, [0.25, 0.75])
    t_emp = float(np.quantile(e, 1.0 - q))
    u = float(min(q3 + alpha * (q3 - q1), t_emp))

    excess = e[e > u] - u
    n_exceed = int(excess.size)
    if n_exceed < min_exceedances:
        return Thresholds(u, max(t_emp, u), n_exceed, fallback="few_exceedances")

    shape, _, scale = genpareto.fit(excess, floc=0.0)
    p = q * n / n_exceed  # <= 1 because u <= Q_{1-q}
    if abs(shape) < shape_eps:
        tq_pot = u - scale * np.log(p)
    else:
        tq_pot = u + (scale / shape) * (p ** (-shape) - 1.0)
    if not (np.isfinite(tq_pot) and scale > 0):
        return Thresholds(u, max(t_emp, u), n_exceed, shape, scale, fallback="bad_fit")

    tq = min(tq_pot, t_emp) if cap else tq_pot
    return Thresholds(u, float(max(tq, u)), n_exceed, float(shape), float(scale), float(tq_pot))


def quantile_thresholds(energy: np.ndarray, q_low: float, q: float) -> Thresholds:
    """Non-parametric alternative: u = Q_{1-q_low}, t_q = Q_{1-q}."""
    e = np.asarray(energy, dtype=np.float64)
    u, tq = np.quantile(e, [1.0 - q_low, 1.0 - q])
    return Thresholds(float(u), float(max(tq, u)), int((e > u).sum()))


class ThresholdPolicy:
    """Computes (u, t_q) epoch after epoch, with optional EMA smoothing."""

    def __init__(self, cfg: TrainConfig):
        if cfg.threshold not in ("pot", "pot_cap", "quantile"):
            raise ValueError(f"unknown threshold policy {cfg.threshold!r}")
        self.cfg = cfg
        self._ema: tuple[float, float] | None = None

    def __call__(self, energy: np.ndarray) -> Thresholds:
        cfg = self.cfg
        if cfg.threshold == "quantile":
            th = quantile_thresholds(energy, cfg.q_low, cfg.q)
        else:
            th = pot_thresholds(
                energy, cfg.alpha, cfg.q, cap=cfg.threshold == "pot_cap", min_exceedances=cfg.min_exceedances
            )
        if cfg.threshold_ema > 0.0:
            if self._ema is not None:
                b = cfg.threshold_ema
                u = b * self._ema[0] + (1 - b) * th.u
                tq = max(b * self._ema[1] + (1 - b) * th.tq, u)
                th = Thresholds(u, tq, th.n_exceed, th.shape, th.scale, th.tq_pot, th.fallback)
            self._ema = (th.u, th.tq)
        return th
