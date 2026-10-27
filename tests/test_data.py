from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from grand.data import CATEGORICAL, COLUMNS, Preprocessor, contaminate, prepare

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def _frame(n, rng):
    df = pd.DataFrame({c: rng.uniform(0, 100, n) for c in COLUMNS[:-2]})
    df["protocol_type"] = rng.choice(["tcp", "udp", "icmp"], n)
    df["service"] = rng.choice(["http", "ftp", "smtp"], n)
    df["flag"] = rng.choice(["SF", "S0"], n)
    return df


@pytest.mark.parametrize("c", [0.0, 0.05, 0.10, 0.15])
def test_contamination_fraction_is_exact(c):
    rng = np.random.default_rng(0)
    X = _frame(3000, rng)
    y = (rng.uniform(size=3000) < 0.5).astype(int)
    Xc, yc = contaminate(X, y, c, np.random.default_rng(1))
    assert (yc == 0).sum() == (y == 0).sum()  # every normal record is kept
    assert yc.mean() == pytest.approx(c, abs=1 / len(yc))
    assert len(Xc) == len(yc)


def test_preprocessor_fits_on_training_data_only():
    rng = np.random.default_rng(0)
    vocab, train, test = _frame(500, rng), _frame(300, rng), _frame(300, rng)
    test.loc[0, "service"] = "never_seen"
    test.loc[1, "duration"] = 1e6
    pre = Preprocessor(vocab).fit(train)
    Xtr, Xte = pre.transform(train), pre.transform(test)
    n_num = len(COLUMNS) - 2 - len(CATEGORICAL)
    assert Xtr.shape[1] == len(pre.feature_names) == n_num + 3 + 3 + 2
    assert Xtr.min() >= 0 and Xtr.max() <= 1 + 1e-6
    assert Xte[1, pre.feature_names.index("duration")] > 1  # not clipped
    service = [i for i, f in enumerate(pre.feature_names) if f.startswith("service_")]
    assert Xte[0, service].sum() == 0  # unknown category -> all-zero block


@pytest.mark.skipif(not (RAW / "KDDTrain+.txt").exists(), reason="NSL-KDD not downloaded")
def test_nsl_kdd_has_122_features_at_every_contamination():
    for c in (0.0, 0.15):
        d = prepare(RAW, c, seed=0)
        assert d.X_train.shape[1] == d.X_test.shape[1] == 122
        assert d.y_train.mean() == pytest.approx(c, abs=1e-4)
        assert len(d.y_test) == 22544
