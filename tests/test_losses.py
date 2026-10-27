import torch

from grand.losses import feature_swap, margin_loss, ramp, softabs


def test_softabs_approximates_abs():
    x = torch.linspace(-5, 5, 101)
    assert softabs(torch.zeros(1), 1.0).abs().item() < 1e-6
    assert torch.allclose(softabs(x, 1e-3), x.abs(), atol=1e-2)
    assert (softabs(x, 1.0) <= x.abs() + 1e-6).all()


def test_margin_loss_kinds():
    e = torch.tensor([90.0, 100.0, 130.0])
    assert torch.allclose(margin_loss(e, 100.0, "abs"), torch.tensor([10.0, 0.0, 30.0]))
    assert torch.allclose(margin_loss(e, 100.0, "softabs", 1e-3), torch.tensor([10.0, 0.0, 30.0]), atol=1e-2)


def test_feature_swap_keeps_marginals():
    torch.manual_seed(0)
    x = torch.rand(64, 10)
    y = feature_swap(x, 0.5)
    for j in range(10):
        assert set(y[:, j].tolist()) <= set(x[:, j].tolist())
    assert torch.equal(feature_swap(x, 0.0), x)
    assert not torch.equal(y, x)


def test_ramp():
    assert [ramp(e, 2, 4) for e in range(8)] == [0, 0, 0, 0.25, 0.5, 0.75, 1, 1]
    assert ramp(1, 0, 0) == 1.0 and ramp(50, 10, 10) == 1.0
