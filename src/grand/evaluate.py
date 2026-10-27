"""Anomaly scores and metrics.

The free energy is a random quantity (it depends on the latent sample), so a
single forward pass gives a noisy score and two evaluations of the same model
disagree. Scores here are the Monte Carlo mean of the free energy over
`n_samples` latent draws taken from a seeded generator: deterministic,
reproducible and closer to the expectation that defines the ELBO.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from grand.model import GRAnD, free_energy


@torch.no_grad()
def score_components(
    model: GRAnD,
    X: np.ndarray,
    *,
    n_samples: int = 16,
    seed: int = 0,
    batch_size: int = 8192,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Mean reconstruction error and mean KL term of each row of X over `n_samples` seeded latent draws."""
    model.eval()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    d = model.cfg.latent_dim
    recon, kl = [], []
    for xb in torch.as_tensor(X, dtype=torch.float32).split(batch_size):
        xb = xb.to(device)
        r = torch.zeros(xb.size(0), device=device)
        k = torch.zeros(xb.size(0), device=device)
        for _ in range(n_samples):
            out = model(xb, torch.randn(xb.size(0), d, generator=gen).to(device))
            r += free_energy(xb, out, kl_weight=0.0)
            k += out.kl
        recon.append(r / n_samples)
        kl.append(k / n_samples)
    return torch.cat(recon).cpu().numpy(), torch.cat(kl).cpu().numpy()


def score(model: GRAnD, X: np.ndarray, *, kl_weight: float = 1.0, **kwargs) -> np.ndarray:
    """Mean free energy of each row of X over `n_samples` seeded latent draws."""
    recon, kl = score_components(model, X, **kwargs)
    return recon + kl_weight * kl


def ranking_metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Threshold-free metrics. AUPRC is reported with its chance level (the attack rate)."""
    y = np.asarray(y).astype(int)
    return {
        "auroc": float(roc_auc_score(y, scores)),
        "auprc": float(average_precision_score(y, scores)),
        "auprc_chance": float(y.mean()),
    }


def alarm_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Metrics of the decision `score >= threshold`."""
    y = np.asarray(y).astype(int)
    pred = np.asarray(scores) >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "fpr": fp / max(fp + tn, 1),
    }
