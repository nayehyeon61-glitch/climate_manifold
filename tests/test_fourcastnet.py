import math

import pytest
import torch
from torch import nn

from climate_manifold._vendor.fourcastnet.afnonet import AFNO2D, AFNONet
from climate_manifold.downstream.fourcastnet import FourCastNetPredictor
from climate_manifold.spatial import SpatialDecoder, SpatialEncoder


def sample(grid=(2, 16, 32), history_steps=3, batch=2):
    history = torch.randn(batch, history_steps, math.prod(grid))
    leads = torch.tensor([6., 12., 24.])
    origins = torch.full((batch,), 1_600_000_000_000_000_000, dtype=torch.int64)
    return history, leads, origins


def test_official_afno_core_raw_small_grid_forward_backward():
    torch.manual_seed(2)
    predictor = FourCastNetPredictor((2, 16, 32), 3, hidden=16, depth=2,
                                     information_channels=1)
    assert isinstance(predictor.core, AFNONet)
    assert isinstance(predictor.core.blocks[0].filter, AFNO2D)
    history, leads, origins = sample()
    history.requires_grad_()
    information = torch.randn(2, 16 * 32, requires_grad=True)
    output, auxiliary = predictor(history, leads, origins, information=information)
    assert output.shape == (2, 3, 2 * 16 * 32)
    assert torch.isfinite(output).all()
    assert auxiliary is None
    output.square().mean().backward()
    assert predictor.core.blocks[0].filter.w1.grad.abs().sum() > 0
    assert history.grad[:, 0].abs().sum() > 0
    assert information.grad.abs().sum() > 0


def test_joint_manifold_forecast_receives_gradients():
    torch.manual_seed(5)
    raw_grid, latent_grid = (2, 16, 32), (4, 8, 16)
    encoder = SpatialEncoder(raw_grid, latent_grid, hidden_dim=8, factor=2, periodic_lon=True)
    decoder = SpatialDecoder(latent_grid, raw_grid, hidden_dim=8, factor=2, periodic_lon=True)
    predictor = FourCastNetPredictor(latent_grid, 3, hidden=16, depth=1)
    history, leads, origins = sample(raw_grid)
    output, _ = predictor(encoder(history), leads, origins)
    decoded = decoder(output)
    decoded.square().mean().backward()
    for model in (encoder, predictor.core, decoder):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_missing_requested_leads_still_use_fixed_autoregressive_steps():
    predictor = FourCastNetPredictor((2, 4, 8), 3, hidden=8, depth=1)
    history, _, origins = sample((2, 4, 8))
    calls = []

    class AddOneCore(nn.Module):
        def forward(self, inputs):
            calls.append(inputs.detach().clone())
            return inputs[:, :2] + 1.

    predictor.core = AddOneCore()
    output, _ = predictor(history, torch.tensor([6., 24.]), origins)
    assert len(calls) == 4
    torch.testing.assert_close(output[:, 0], history[:, -1] + 1.)
    torch.testing.assert_close(output[:, 1], history[:, -1] + 4.)
    # All observed fields are kept at their origin times, not shifted by 6h.
    torch.testing.assert_close(calls[0][:, 2:8], calls[-1][:, 2:8])
    torch.testing.assert_close(calls[0][0, 8:11, 0, 0], torch.tensor([-2., -1., 0.]))


def test_prediction_independent_of_lead_selection_and_history_spacing_is_used():
    torch.manual_seed(8)
    predictor = FourCastNetPredictor((2, 4, 8), 3, hidden=8, depth=1).eval()
    other_spacing = FourCastNetPredictor((2, 4, 8), 3, hidden=8, depth=1, history_dt_hours=6.).eval()
    other_spacing.load_state_dict(predictor.state_dict())
    history, leads, origins = sample((2, 4, 8))
    full, _ = predictor(history, leads, origins)
    isolated, _ = predictor(history, leads[-1:], origins)
    torch.testing.assert_close(full[:, -1:], isolated)
    changed, _ = other_spacing(history, leads, origins)
    assert not torch.allclose(full, changed)


@pytest.mark.parametrize('leads', [torch.tensor([3.]), torch.tensor([6., 7.]),
                                 torch.tensor([0.]), torch.tensor([12., 6.]),
                                 torch.tensor([float('nan')]), torch.tensor([])])
def test_invalid_leads_rejected(leads):
    predictor = FourCastNetPredictor((2, 4, 8), 3, hidden=8, depth=1)
    history, _, origins = sample((2, 4, 8))
    with pytest.raises(ValueError, match='Lead hours'):
        predictor(history, leads, origins)


@pytest.mark.parametrize('configuration', [dict(latent_grid=(2, 3, 8)), dict(hidden=0),
                                         dict(patch_size=0), dict(depth=0),
                                         dict(forecast_step_hours=float('inf')),
                                         dict(history_dt_hours=-6), dict(information_channels=-1)])
def test_invalid_configuration_rejected(configuration):
    options = dict(latent_grid=(2, 4, 8), history_steps=3, hidden=8, depth=1)
    options.update(configuration)
    with pytest.raises(ValueError):
        FourCastNetPredictor(**options)


def test_future_information_cannot_enter_predictor():
    predictor = FourCastNetPredictor((2, 4, 8), 3, hidden=8, depth=1, information_channels=1)
    history, leads, origins = sample((2, 4, 8))
    with pytest.raises(ValueError, match='origin information'):
        predictor(history, leads, origins, information=torch.randn(2, 3, 32))
    with pytest.raises(TypeError):
        predictor(history, leads, origins, future=torch.randn(2, 3, 64))
