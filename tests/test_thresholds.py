import numpy as np
import pytest
from scipy.stats import genpareto

from grand.config import TrainConfig
from grand.thresholds import ThresholdPolicy, pot_thresholds, quantile_thresholds


def test_pot_recovers_the_true_tail_quantile():
    """On GPD data every exceedance is GPD too, so t_q must match the true (1-q) quantile."""
    rng = np.random.default_rng(0)
    xi, q = 0.2, 1e-3
    e = genpareto.rvs(xi, scale=1.0, size=400_000, random_state=rng)
    th = pot_thresholds(e, alpha=1.5, q=q)
    assert th.fallback is None
    assert th.shape == pytest.approx(xi, abs=0.05)
    assert th.tq == pytest.approx(genpareto.ppf(1 - q, xi), rel=0.05)


@pytest.mark.parametrize("cap", [False, True])
def test_threshold_ordering(cap):
    rng = np.random.default_rng(1)
    e = np.concatenate([rng.gamma(2.0, 1.0, 50_000), rng.normal(40, 5, 500)])  # nominal + outliers
    th = pot_thresholds(e, alpha=1.5, q=1e-3, cap=cap)
    t_emp = np.quantile(e, 1 - 1e-3)
    assert th.u <= t_emp + 1e-9
    assert th.tq >= th.u
    if cap:
        assert th.tq <= t_emp + 1e-9


def test_few_exceedances_fall_back_to_the_empirical_quantile():
    e = np.arange(100, dtype=float)
    th = pot_thresholds(e, alpha=1.5, q=0.01, min_exceedances=30)
    assert th.fallback == "few_exceedances"
    assert th.tq == pytest.approx(np.quantile(e, 0.99))


def test_quantile_thresholds():
    e = np.linspace(0, 1, 10_001)
    th = quantile_thresholds(e, q_low=0.1, q=0.01)
    assert th.u == pytest.approx(0.9) and th.tq == pytest.approx(0.99)


def test_ema_smooths_thresholds_across_epochs():
    policy = ThresholdPolicy(TrainConfig(threshold="quantile", q_low=0.1, q=0.01, threshold_ema=0.5))
    first = policy(np.linspace(0, 1, 10_001))
    second = policy(np.linspace(0, 3, 10_001))
    assert second.u == pytest.approx(0.5 * first.u + 0.5 * 2.7)
    assert second.tq >= second.u
