import math

import pytest
import torch

from climate_manifold.downstream.simvp import SimVPPredictor
from climate_manifold._vendor.openstl.gsta import PaddedConv2d
from climate_manifold.spatial import SpatialDecoder, SpatialEncoder


def data(grid=(2, 3, 5), steps=3, batch=2):
    history = torch.randn(batch, steps, math.prod(grid))
    leads = torch.tensor([6., 12., 30., 120.])
    origins = torch.full((batch,), 1_600_000_000_000_000_000, dtype=torch.int64)
    return history, leads, origins


@pytest.mark.parametrize('grid', [(2, 1, 1), (2, 2, 4), (2, 9, 18)])
@pytest.mark.parametrize('periodic', [False, True])
def test_tiny_odd_grids_and_twenty_leads(grid, periodic):
    model = SimVPPredictor(grid, 6, hidden=8, periodic_lon=periodic)
    history, _, origins = data(grid, steps=6, batch=1)
    leads = torch.arange(6., 121., 6.)
    output, auxiliary = model(history, leads, origins)
    assert output.shape == (1, 20, math.prod(grid))
    assert torch.isfinite(output).all()
    assert auxiliary is None
    output.square().mean().backward()
    assert model.translator.enc[0].block.attn.proj_1.weight.grad.abs().sum() > 0


def test_shared_manifold_encoder_predictor_decoder_gradients():
    torch.manual_seed(3)
    raw_grid, latent_grid = (2, 6, 10), (4, 3, 5)
    encoder = SpatialEncoder(raw_grid, latent_grid, hidden_dim=8, factor=2, periodic_lon=True)
    decoder = SpatialDecoder(latent_grid, raw_grid, hidden_dim=8, factor=2, periodic_lon=True)
    predictor = SimVPPredictor(latent_grid, 3, hidden=16, periodic_lon=True)
    history, leads, origins = data(raw_grid)
    predicted, _ = predictor(encoder(history), leads, origins)
    decoded = decoder(predicted)
    decoded.square().mean().backward()
    for module in (encoder, predictor.encoder, predictor.translator, predictor.lead_query,
                   predictor.decoder, decoder):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def test_each_lead_is_direct_and_does_not_depend_on_other_requested_leads():
    model = SimVPPredictor((2, 3, 5), 3, hidden=8).eval()
    history, leads, origins = data()
    entire, _ = model(history, leads, origins)
    isolated, _ = model(history, leads[2:3], origins)
    torch.testing.assert_close(entire[:, 2:3], isolated, atol=1e-6, rtol=1e-5)
    # Public interface has no future observations/targets argument.
    with pytest.raises(TypeError):
        model(history, leads, origins, future=torch.ones_like(history))


def test_physical_lead_history_spacing_and_calendar_change_forecast():
    torch.manual_seed(4)
    model = SimVPPredictor((2, 3, 5), 3, hidden=8, history_dt_hours=24.).eval()
    different_spacing = SimVPPredictor((2, 3, 5), 3, hidden=8, history_dt_hours=6.).eval()
    different_spacing.load_state_dict(model.state_dict())
    history, leads, origins = data()
    reference, _ = model(history, leads, origins)
    shifted_leads, _ = model(history, leads + 3, origins)
    shifted_calendar, _ = model(history, leads, origins + 6 * 3_600_000_000_000)
    shifted_spacing, _ = different_spacing(history, leads, origins)
    for altered in (shifted_leads, shifted_calendar, shifted_spacing):
        assert not torch.allclose(reference, altered)
    assert model.history_dt_hours == 24.


def test_single_history_state_still_has_lead_conditioning():
    torch.manual_seed(7)
    model = SimVPPredictor((2, 3, 5), 1, hidden=8).eval()
    history, leads, origins = data(steps=1)
    output, _ = model(history, leads, origins)
    assert not torch.allclose(output[:, 0], output[:, -1])


def test_raw_information_is_origin_only_and_reaches_prediction():
    torch.manual_seed(8)
    model = SimVPPredictor((2, 3, 5), 3, hidden=8, information_channels=2).eval()
    history, leads, origins = data()
    information = torch.randn(2, 30, requires_grad=True)
    output, _ = model(history, leads, origins, information=information)
    output.square().mean().backward()
    assert information.grad is not None and information.grad.abs().sum() > 0
    changed, _ = model(history, leads, origins, information=information.detach() + 2.)
    assert not torch.allclose(output, changed)
    for invalid in (None, information[:, None], information[:, :15], torch.full_like(information, float('nan'))):
        with pytest.raises(ValueError, match='origin information'):
            model(history, leads, origins, information=invalid)
    latent = SimVPPredictor((2, 3, 5), 3, hidden=8)
    with pytest.raises(ValueError, match='encoder only'):
        latent(history, leads, origins, information=information)


@pytest.mark.parametrize('lead', [torch.tensor([]), torch.tensor([0.]), torch.tensor([-6.]),
                                torch.tensor([12., 6.]), torch.tensor([6., 6.]),
                                torch.tensor([float('nan')]), torch.tensor([[6.]])])
def test_invalid_physical_leads_are_rejected(lead):
    model = SimVPPredictor((2, 3, 5), 3, hidden=8)
    history, _, origins = data()
    with pytest.raises(ValueError, match='Lead hours'):
        model(history, lead, origins)


@pytest.mark.parametrize('kwargs', [dict(history_dt_hours=0.), dict(history_dt_hours=float('inf')),
                                   dict(history_steps=0), dict(hidden=True),
                                   dict(information_channels=-1), dict(latent_grid=(2, 0, 5))])
def test_invalid_configuration_is_rejected(kwargs):
    config = dict(latent_grid=(2, 3, 5), history_steps=3, hidden=8)
    config.update(kwargs)
    with pytest.raises(ValueError):
        SimVPPredictor(**config)


def test_invalid_history_or_origin_is_rejected():
    model = SimVPPredictor((2, 3, 5), 3, hidden=8)
    history, leads, origins = data()
    for invalid in (history[:, :2], history[..., :20], torch.full_like(history, float('nan'))):
        with pytest.raises(ValueError, match='History'):
            model(invalid, leads, origins)
    with pytest.raises(ValueError, match='origin timestamp'):
        model(history, leads, origins[:, None])


def test_geographic_large_kernel_wraps_tiny_longitude_more_than_once():
    torch.manual_seed(10)
    layer = PaddedConv2d(2, 2, 7, padding=9, dilation=3, groups=2, periodic_lon=True)
    x = torch.randn(1, 2, 1, 4)
    predicted = layer(x)
    expected = predicted.roll(1, dims=-1)
    actual = layer(x.roll(1, dims=-1))
    assert actual.shape == x.shape
    torch.testing.assert_close(expected, actual)
