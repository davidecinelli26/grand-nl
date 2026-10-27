import numpy as np
import pytest

from grand.config import ModelConfig, variant_config
from grand.evaluate import ranking_metrics, score
from grand.model import GRAnD
from grand.thresholds import Thresholds
from grand.trainer import fit, set_seed, split

SMALL = ModelConfig(hidden_dim=16, latent_dim=4, n_flows=2)


def test_split_is_a_partition_with_ecdf_weights():
    e = np.random.default_rng(0).gamma(2, 1, 5000)
    part = split(e, Thresholds(u=4.0, tq=9.0))
    idx = np.concatenate([part.L, part.U, part.S])
    assert np.array_equal(np.sort(idx), np.arange(e.size))
    assert (e[part.L] < 4).all() and (e[part.S] >= 9).all()
    assert ((e[part.U] >= 4) & (e[part.U] < 9)).all()
    assert part.w_u.max() == pytest.approx(1.0) and (part.w_u > 0).all()
    order = np.argsort(e[part.U])
    assert np.all(np.diff(part.w_u[order]) >= 0)  # weight grows with energy


def _train(toy_data, variant, labels=True, **kw):
    X, y, _, _ = toy_data
    set_seed(0)
    model = GRAnD(X.shape[1], SMALL)
    cfg = variant_config(variant, max_epochs=12, batch_size=128, q=0.01, **kw)
    hist, th = fit(model, X, cfg, y=y if labels else None)
    return model, hist, th


def test_labels_never_influence_training(toy_data):
    _, with_y, _ = _train(toy_data, "grand+aesad", aesad_warmup=2, aesad_ramp=2)
    _, without_y, _ = _train(toy_data, "grand+aesad", labels=False, aesad_warmup=2, aesad_ramp=2)
    assert with_y.loss == without_y.loss and with_y.tq == without_y.tq and with_y.n_s == without_y.n_s
    assert with_y.us_recall and not without_y.us_recall


@pytest.mark.parametrize("variant", ["vanilla", "grand", "grand+all"])
def test_end_to_end_detects_toy_anomalies(toy_data, variant):
    _, _, X_test, y_test = toy_data
    model, hist, _ = _train(toy_data, variant, aesad_warmup=2, aesad_ramp=2, es_warmup=5)
    auroc = ranking_metrics(y_test, score(model, X_test, n_samples=4))["auroc"]
    assert auroc > 0.9
    assert len(hist.loss) == len(hist.tq) == len(hist.n_s) <= 12


def test_rejection_isolates_injected_attacks(toy_data):
    _, hist, _ = _train(toy_data, "grand")
    assert hist.us_recall[-1] > 0.5  # most injected attacks leave L
    assert hist.l_contamination[-1] < 0.1  # below the injected 10%
