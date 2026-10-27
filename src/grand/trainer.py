"""GRAnD training loop (paper, Algorithm of Sec. 4) with the optional extensions.

Each epoch:
  1. gradient steps on L (minimise F), on U (push F towards the margin m,
     weighted by the eCDF of the energy inside U) and on S (push F towards m);
  2. the energies of the whole training set are recomputed and the thresholds
     (u, t_q) re-split it into L = {F < u}, U = {u <= F < t_q}, S = {F >= t_q}.

Training labels are optional and only feed the diagnostics (how many injected
attacks end up in S or U): they never touch the loss, the thresholds or the
split, which a test checks.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.stats import rankdata

from grand.config import TrainConfig
from grand.early_stopping import StructuralEarlyStopping
from grand.losses import feature_swap, margin_loss, ramp
from grand.model import GRAnD, free_energy
from grand.thresholds import ThresholdPolicy, Thresholds


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class History:
    loss: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    u: list[float] = field(default_factory=list)
    tq: list[float] = field(default_factory=list)
    n_l: list[int] = field(default_factory=list)
    n_u: list[int] = field(default_factory=list)
    n_s: list[int] = field(default_factory=list)
    # diagnostics against the injected contamination (only when labels are given)
    s_precision: list[float] = field(default_factory=list)  # attacks among S
    s_recall: list[float] = field(default_factory=list)  # attacks caught in S
    us_recall: list[float] = field(default_factory=list)  # attacks caught in U or S
    l_contamination: list[float] = field(default_factory=list)  # attacks left in L
    stop_reason: str = "max_epochs"
    train_time: float = 0.0

    @property
    def epochs(self) -> int:
        return len(self.loss)


@dataclass
class Split:
    L: np.ndarray  # indices
    U: np.ndarray
    S: np.ndarray
    w_u: np.ndarray  # eCDF weights of U, aligned with U


def split(energy: np.ndarray, th: Thresholds) -> Split:
    """Partition by energy; U gets the eCDF weight of its energy within U (paper, Eq. 6)."""
    L = np.flatnonzero(energy < th.u)
    S = np.flatnonzero(energy >= th.tq)
    U = np.flatnonzero((energy >= th.u) & (energy < th.tq))
    w_u = rankdata(energy[U], method="max") / max(U.size, 1)
    return Split(L, U, S, w_u.astype(np.float32))


@torch.no_grad()
def energies(model: GRAnD, X: torch.Tensor, kl_weight: float, batch_size: int = 8192) -> np.ndarray:
    """One stochastic free-energy evaluation per sample, as in the reference rejection step."""
    was_training = model.training
    model.eval()
    out = [free_energy(xb, model(xb), kl_weight) for xb in X.split(batch_size)]
    model.train(was_training)
    return torch.cat(out).cpu().numpy()


def _batches(n: int, batch_size: int, device) -> list[torch.Tensor]:
    return list(torch.randperm(n, device=device).split(batch_size))


def train_epoch(
    model: GRAnD, opt: torch.optim.Optimizer, X: torch.Tensor, part: Split, cfg: TrainConfig, epoch: int
) -> float:
    """One epoch over L, then U, then S (reference order). Returns the loss per training sample."""
    model.train()
    device = X.device
    lam_us = ramp(epoch, cfg.us_warmup, cfg.us_ramp)
    lam_a = cfg.aesad_lambda * ramp(epoch, cfg.aesad_warmup, cfg.aesad_ramp) if cfg.aesad else 0.0
    total = 0.0

    def step(loss: torch.Tensor) -> None:
        nonlocal total
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += float(loss.detach())

    L = torch.as_tensor(part.L, device=device)
    for idx in _batches(L.numel(), cfg.batch_size, device):
        xb = X[L[idx]]
        step(free_energy(xb, model(xb), cfg.kl_weight).sum())

    if part.U.size and lam_us > 0:
        U = torch.as_tensor(part.U, device=device)
        w_u = torch.as_tensor(part.w_u, device=device)
        for idx in _batches(U.numel(), cfg.batch_size, device):
            xb, wb = X[U[idx]], w_u[idx]
            out = model(xb)
            e = free_energy(xb, out, cfg.kl_weight)
            loss = lam_us * (wb * margin_loss(e, cfg.m, cfg.margin, cfg.softabs_tau)).sum()
            if lam_a > 0:
                # negative learning only on the most suspicious part of the batch
                top = torch.topk(e.detach(), k=max(1, int(cfg.aesad_u_top * xb.size(0)))).indices
                target = feature_swap(xb[top], cfg.aesad_p_swap)
                loss = loss + lam_us * lam_a * (wb[top] * (out.recon[top] - target).pow(2).sum(dim=1)).sum()
            step(loss)

    if part.S.size and lam_us > 0:
        S = torch.as_tensor(part.S, device=device)
        for idx in _batches(S.numel(), cfg.batch_size, device):
            xb = X[S[idx]]
            out = model(xb)
            e = free_energy(xb, out, cfg.kl_weight)
            loss = lam_us * margin_loss(e, cfg.m, cfg.margin, cfg.softabs_tau).sum()
            if lam_a > 0:
                target = feature_swap(xb, cfg.aesad_p_swap)
                loss = loss + lam_us * lam_a * (out.recon - target).pow(2).sum(dim=1).sum()
            step(loss)

    return total / X.size(0)


def fit(
    model: GRAnD,
    X: np.ndarray,
    cfg: TrainConfig,
    *,
    y: np.ndarray | None = None,
    device: str | torch.device = "cpu",
    verbose: bool = False,
) -> tuple[History, Thresholds]:
    """Train `model` on X. Returns the history and the thresholds of the last split."""
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    n = X_t.size(0)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=cfg.plateau_factor, patience=cfg.plateau_patience, threshold=1e-4
    )
    policy = ThresholdPolicy(cfg)
    stopper = StructuralEarlyStopping(cfg) if cfg.early_stopping else None
    hist = History()
    attacks = None if y is None else np.asarray(y) == 1

    part = Split(np.arange(n), np.empty(0, int), np.empty(0, int), np.empty(0, np.float32))
    start = time.perf_counter()
    for epoch in range(1, cfg.max_epochs + 1):
        loss = train_epoch(model, opt, X_t, part, cfg, epoch)
        sched.step(loss)
        lr = opt.param_groups[0]["lr"]

        e = energies(model, X_t, cfg.kl_weight)
        th = policy(e)
        new = split(e, th)
        # the vanilla baseline keeps training on everything; its split is only observed
        part = new if cfg.rejection else Split(np.arange(n), new.U[:0], new.S[:0], new.w_u[:0])

        hist.loss.append(loss)
        hist.lr.append(lr)
        hist.u.append(th.u)
        hist.tq.append(th.tq)
        hist.n_l.append(int(new.L.size))
        hist.n_u.append(int(new.U.size))
        hist.n_s.append(int(new.S.size))
        if attacks is not None and attacks.any():
            n_att = attacks.sum()
            hist.s_precision.append(float(attacks[new.S].mean()) if new.S.size else float("nan"))
            hist.s_recall.append(float(attacks[new.S].sum() / n_att))
            hist.us_recall.append(float((attacks[new.S].sum() + attacks[new.U].sum()) / n_att))
            hist.l_contamination.append(float(attacks[new.L].mean()))
        if verbose:
            print(
                f"epoch {epoch:3d}  loss {loss:9.4f}  L/U/S {new.L.size}/{new.U.size}/{new.S.size}"
                f"  u {th.u:.3f}  tq {th.tq:.3f}  lr {lr:.1e}"
            )

        if stopper and stopper.step(epoch, loss, th.u, th.tq, new.L.size, new.U.size, new.S.size):
            hist.stop_reason = "structural_convergence"
            break
        if lr <= cfg.min_lr:
            hist.stop_reason = "min_lr"
            break

    hist.train_time = time.perf_counter() - start
    return hist, th
