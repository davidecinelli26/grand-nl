"""Configuration of the model, of training, and of the ablation variants."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ModelConfig:
    """Architecture of the VAE + planar-flow model.

    The defaults reproduce the reference implementation: 122-32-8 encoder,
    8-32-122 decoder, eight planar flows in the 8-dimensional latent space.
    """

    arch: str = "mlp"  # "mlp" (reference) | "resmlp" (residual MLP blocks)
    hidden_dim: int = 32
    latent_dim: int = 8
    n_flows: int = 8
    resmlp_depth: int = 2
    resmlp_dropout: float = 0.1


@dataclass(frozen=True)
class TrainConfig:
    """Training procedure. Defaults follow the reference notebook."""

    # Optimisation
    batch_size: int = 512
    lr: float = 1e-3
    max_epochs: int = 100
    plateau_factor: float = 0.5
    plateau_patience: int = 10
    min_lr: float = 1e-5  # training stops once the learning rate falls to this value

    # Free energy F(x) = ||x - x_hat||^2 + kl_weight * KL term
    kl_weight: float = 1.0

    # GRAnD rejection strategy (paper, Sec. 4)
    rejection: bool = True  # False trains a vanilla VAE + planar flow on all data
    alpha: float = 1.5  # u = Q3 + alpha * IQR
    q: float = 1e-3  # POT risk parameter
    m: float = 100.0  # energy margin for U and S
    threshold: str = "pot"  # "pot" | "pot_cap" | "quantile"
    q_low: float = 1e-2  # "quantile" only: u = Q_{1 - q_low}
    min_exceedances: int = 30  # fewer exceedances than this: no GPD fit, empirical quantile
    threshold_ema: float = 0.0  # EMA smoothing of (u, tq) across epochs, 0 = off
    margin: str = "abs"  # "abs" (paper) | "softabs" (smooth approximation)
    softabs_tau: float = 1.0

    # Linear warm-up of the U and S terms
    us_warmup: int = 0
    us_ramp: int = 0

    # AE-SAD negative learning (Angiulli et al.) on S and on the top of U
    aesad: bool = False
    aesad_lambda: float = 1.0
    aesad_p_swap: float = 0.3  # fraction of features swapped by the corruption F(x)
    aesad_u_top: float = 0.15  # fraction of each U batch, by energy, that gets the term
    aesad_warmup: int = 10
    aesad_ramp: int = 10

    # Structural early stopping
    early_stopping: bool = False
    es_warmup: int = 15
    es_window: int = 5
    es_loss_rel: float = 5e-2
    es_u_rel: float = 3e-2
    es_tq_rel: float = 6e-2
    es_frac_abs: float = 1e-3

    def schedules_end(self) -> int:
        """Last epoch at which a scheduled loss term is still ramping up."""
        end = self.us_warmup + self.us_ramp
        if self.aesad:
            end = max(end, self.aesad_warmup + self.aesad_ramp)
        return end


# Ablation variants. Each one adds a single mechanism to the GRAnD baseline,
# and "grand+all" is exactly the union of the three, so every effect can be
# read against the same reference run (same seed, same data, same init).
VARIANTS: dict[str, dict] = {
    "vanilla": {"rejection": False},
    "grand": {},
    "grand+es": {"early_stopping": True},
    "grand+potcap": {"threshold": "pot_cap"},
    "grand+aesad": {"aesad": True},
    "grand+all": {"early_stopping": True, "threshold": "pot_cap", "aesad": True},
    # sensitivity check: the course code weighted the KL term by 0.05
    "grand-kl0.05": {"kl_weight": 0.05},
}


def variant_config(name: str, **overrides) -> TrainConfig:
    """TrainConfig of a named variant, with optional field overrides."""
    if name not in VARIANTS:
        raise KeyError(f"unknown variant {name!r}; choose from {sorted(VARIANTS)}")
    return replace(TrainConfig(), **{**VARIANTS[name], **overrides})
