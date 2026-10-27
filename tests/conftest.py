import numpy as np
import pytest


@pytest.fixture
def toy_data():
    """Nominal points near a 2-D linear manifold of [0, 1]^12; attacks uniform in the cube."""
    rng = np.random.default_rng(0)
    basis = rng.uniform(0, 1, (2, 12))

    def nominal(n):
        t = rng.uniform(0, 1, (n, 2))
        return np.clip(0.2 + 0.3 * (t @ basis) + rng.normal(0, 0.01, (n, 12)), 0, 1)

    def attacks(n):
        return rng.uniform(0, 1, (n, 12))

    X_train = np.vstack([nominal(1800), attacks(200)]).astype(np.float32)
    y_train = np.r_[np.zeros(1800), np.ones(200)].astype(int)
    X_test = np.vstack([nominal(500), attacks(500)]).astype(np.float32)
    y_test = np.r_[np.zeros(500), np.ones(500)].astype(int)
    return X_train, y_train, X_test, y_test
