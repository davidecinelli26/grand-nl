"""Structural early stopping.

GRAnD optimises a moving target: the loss depends on the thresholds, which
depend on the energies, which depend on the model. A loss plateau alone does
not mean the system has settled, so training stops only when, over a sliding
window of epochs, the loss, both thresholds and the L/U/S fractions are all
stable at once. No labels are needed.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from grand.config import TrainConfig


def relative_range(values) -> float:
    a = np.asarray(values, dtype=np.float64)
    return float((a.max() - a.min()) / (abs(a.mean()) + 1e-12))


def absolute_range(values) -> float:
    a = np.asarray(values, dtype=np.float64)
    return float(a.max() - a.min())


class StructuralEarlyStopping:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        # never stop while a scheduled loss term is still ramping up
        self.warmup = max(cfg.es_warmup, cfg.schedules_end())
        w = cfg.es_window
        self.loss, self.u, self.tq = deque(maxlen=w), deque(maxlen=w), deque(maxlen=w)
        self.fractions = deque(maxlen=w)  # (pL, pU, pS)

    def step(self, epoch: int, loss: float, u: float, tq: float, n_l: int, n_u: int, n_s: int) -> bool:
        n = max(n_l + n_u + n_s, 1)
        self.loss.append(loss)
        self.u.append(u)
        self.tq.append(tq)
        self.fractions.append((n_l / n, n_u / n, n_s / n))
        if epoch <= self.warmup or len(self.loss) < self.cfg.es_window:
            return False
        frac_range = max(absolute_range(col) for col in zip(*self.fractions))
        return (
            relative_range(self.loss) < self.cfg.es_loss_rel
            and relative_range(self.u) < self.cfg.es_u_rel
            and relative_range(self.tq) < self.cfg.es_tq_rel
            and frac_range < self.cfg.es_frac_abs
        )
