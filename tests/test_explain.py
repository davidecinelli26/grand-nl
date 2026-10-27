import numpy as np
import pytest
import torch

from grand.explain import explain, group_one_hot
from grand.model import GRAnD


def test_integrated_gradients_completeness_and_decomposition():
    torch.manual_seed(0)
    model = GRAnD(15)
    x = np.random.default_rng(0).uniform(size=15).astype(np.float32)
    ex = explain(model, x, steps=256)
    assert ex.squared_error.sum() == pytest.approx(ex.reconstruction, rel=1e-5)
    assert ex.reconstruction + ex.kl == pytest.approx(ex.energy, rel=1e-6)
    assert ex.completeness_gap < 0.02 * abs(ex.integrated_gradients.sum()) + 1e-3


def test_group_one_hot():
    names = ["duration", "service_http", "service_ftp", "flag_SF"]
    g = group_one_hot(names, np.array([0.1, 0.5, 0.25, -1.0]))
    assert g == {"flag": -1.0, "service": 0.75, "duration": 0.1}
