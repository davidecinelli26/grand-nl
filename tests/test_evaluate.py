import numpy as np
import pytest
import torch

from grand.evaluate import alarm_metrics, ranking_metrics, score
from grand.model import GRAnD


def test_scores_are_reproducible_and_seeded():
    torch.manual_seed(0)
    model = GRAnD(10)
    X = np.random.default_rng(0).uniform(size=(50, 10)).astype(np.float32)
    a, b = score(model, X, seed=3), score(model, X, seed=3)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, score(model, X, seed=4))


def test_metrics():
    y = np.array([0, 0, 1, 1])
    m = ranking_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["auroc"] == 1.0 and m["auprc"] == 1.0 and m["auprc_chance"] == 0.5
    a = alarm_metrics(y, np.array([0.1, 0.9, 0.8, 0.2]), threshold=0.5)
    assert a == pytest.approx({"precision": 0.5, "recall": 0.5, "f1": 0.5, "fpr": 0.5})
