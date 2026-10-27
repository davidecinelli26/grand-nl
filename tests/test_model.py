import math

import pytest
import torch

from grand.config import ModelConfig
from grand.model import GRAnD, PlanarFlow, free_energy


def test_planar_flow_log_det_matches_autograd_jacobian():
    torch.manual_seed(0)
    flow = PlanarFlow(5)
    with torch.no_grad():  # large parameters, raw w.u far below -1
        flow.w.copy_(torch.randn(1, 5) * 2)
        flow.u.copy_(-3 * flow.w)
        flow.b.fill_(0.3)
    z = torch.randn(4, 5)
    _, log_det = flow(z)
    for i in range(z.size(0)):
        jac = torch.autograd.functional.jacobian(lambda v: flow(v.unsqueeze(0))[0].squeeze(0), z[i])
        assert log_det[i].item() == pytest.approx(torch.linalg.slogdet(jac).logabsdet.item(), abs=1e-4)


def test_planar_flow_is_always_invertible():
    torch.manual_seed(1)
    for _ in range(200):
        flow = PlanarFlow(8)
        with torch.no_grad():
            flow.w.normal_(0, 3)
            flow.u.normal_(0, 3)
        assert (flow.w * flow.u_hat()).sum().item() >= -1 - 1e-4  # float32 rounding when w.u << -1


@pytest.mark.parametrize("arch", ["mlp", "resmlp"])
def test_forward_shapes_and_fixed_noise(arch):
    torch.manual_seed(0)
    model = GRAnD(20, ModelConfig(arch=arch)).eval()
    x = torch.rand(7, 20)
    eps = torch.randn(7, 8)
    a, b = model(x, eps), model(x, eps)
    assert a.recon.shape == (7, 20) and a.z.shape == (7, 8) and a.kl.shape == (7,)
    assert torch.equal(a.kl, b.kl) and torch.equal(a.recon, b.recon)
    assert ((a.recon >= 0) & (a.recon <= 1)).all()
    assert torch.isfinite(free_energy(x, a)).all()


def test_kl_term_is_unbiased_without_flows():
    """With K = 0 the KL term is a single-sample estimate of KL(N(mu, s^2) || N(0, I))."""
    torch.manual_seed(0)
    model = GRAnD(6, ModelConfig(n_flows=0)).eval()
    x = torch.rand(1, 6)
    with torch.no_grad():
        h = model.encoder(x)
        mu, logvar = model.mu(h), model.logvar(h)
        exact = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum().item()
        estimate = model(x.expand(20000, -1)).kl.mean().item()
    assert estimate == pytest.approx(exact, abs=0.05 + 0.02 * abs(exact))


def test_free_energy_weights_the_kl_term():
    torch.manual_seed(0)
    model = GRAnD(4).eval()
    x, eps = torch.rand(3, 4), torch.randn(3, 8)
    out = model(x, eps)
    recon = (out.recon - x).pow(2).sum(1)
    assert torch.allclose(free_energy(x, out, 0.0), recon)
    assert torch.allclose(free_energy(x, out, 2.0), recon + 2.0 * out.kl)
    assert not math.isnan(free_energy(x, out).sum().item())
