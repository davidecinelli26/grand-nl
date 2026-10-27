from grand.config import TrainConfig
from grand.early_stopping import StructuralEarlyStopping


def _first_stop(cfg, signal):
    es = StructuralEarlyStopping(cfg)
    for epoch in range(1, 100):
        loss, u, tq, n_s = signal(epoch)
        if es.step(epoch, loss, u, tq, 1000 - n_s - 50, 50, n_s):
            return epoch
    return None


def test_stops_after_warmup_on_a_flat_system():
    cfg = TrainConfig(early_stopping=True, es_warmup=15, es_window=5)
    assert _first_stop(cfg, lambda e: (1.0, 2.0, 3.0, 10)) == 16


def test_does_not_stop_while_thresholds_drift():
    cfg = TrainConfig(early_stopping=True)
    assert _first_stop(cfg, lambda e: (1.0, 2.0 + 0.1 * e, 3.0 + 0.1 * e, 10)) is None


def test_does_not_stop_while_the_split_moves():
    cfg = TrainConfig(early_stopping=True)
    assert _first_stop(cfg, lambda e: (1.0, 2.0, 3.0, 10 + e)) is None


def test_waits_for_scheduled_terms():
    cfg = TrainConfig(early_stopping=True, es_warmup=15, aesad=True, aesad_warmup=20, aesad_ramp=10)
    assert _first_stop(cfg, lambda e: (1.0, 2.0, 3.0, 10)) == 31
