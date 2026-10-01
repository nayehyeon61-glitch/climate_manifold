"""Signed field statistics retain locations, magnitude, and safe empty signs."""
from types import SimpleNamespace

import pytest
import torch

from climate_manifold.downstream.statistical_objective import (
    make_statistical_config, spatial_statistical_loss,
    statistical_config_from_args, statistical_config_from_payload,
    validate_statistical_config,
)
from climate_manifold.downstream.reconstruction_objective import reconstruction_constraint_losses
from test_pinn_training import pinn_prepared
from test_reconstruction_objective import split_prepared, pair_weights


def signed(prediction, target, area=None):
    if area is None:
        area = torch.ones(prediction.shape[-1], dtype=prediction.dtype)
    return spatial_statistical_loss(prediction, target, area, make_statistical_config('signed_measure'))


def test_signed_measure_distinguishes_locations_and_strength():
    target = torch.tensor([[1., 2., -1., -2.]])
    relocated = torch.tensor([[2., 1., -2., -1.]])
    # Identical value histograms do not identify the same geographic pattern.
    assert spatial_statistical_loss(relocated, target, torch.ones(4))['loss'] == 0
    relocated_scores = signed(relocated, target)
    assert relocated_scores['signed_spatial_js'] > 0
    assert relocated_scores['signed_mass_mse'] == 0
    stronger_scores = signed(2 * target, target)
    assert stronger_scores['signed_mass_mse'] > 0
    assert stronger_scores['loss'] > 0
    for value in signed(target, target).values():
        assert value == 0


def test_signed_measure_area_scaling_and_finite_gradients():
    target = torch.tensor([[1., -3., 2., -.5]], dtype=torch.float64, requires_grad=True)
    prediction = torch.tensor([[.5, -1., 4., -.2]], dtype=torch.float64, requires_grad=True)
    area = torch.tensor([1., 2., 3., 4.], dtype=torch.float64)
    scores = signed(prediction, target, area)
    torch.testing.assert_close(scores['loss'], signed(prediction, target, area * 5)['loss'])
    assert not torch.isclose(scores['loss'], signed(prediction, target)['loss'])
    assert torch.autograd.gradcheck(lambda x: signed(x, target, area)['loss'], (prediction,))
    scores['loss'].backward()
    assert target.grad is None
    assert torch.isfinite(prediction.grad).all() and prediction.grad.abs().sum() > 0


@pytest.mark.parametrize('prediction_values,target_values', [
    ([0., 0., 0.], [0., 0., 0.]),
    ([0., 0., 0.], [1., 2., 3.]),
    ([1., 2., 3.], [-1., -2., -3.]),
    ([-1., -2., -3.], [-1., -2., -3.]),
])
def test_empty_signs_are_finite(prediction_values, target_values):
    prediction = torch.tensor([prediction_values], requires_grad=True)
    target = torch.tensor([target_values])
    scores = signed(prediction, target)
    assert all(torch.isfinite(value) for value in scores.values())
    scores['loss'].backward()
    assert torch.isfinite(prediction.grad).all()


def test_signed_configuration_is_explicit_and_preserves_legacy_options():
    cfg = make_statistical_config('signed_measure')
    assert cfg['zero_reference'] == 'normalized_field_zero'
    assert validate_statistical_config(cfg) == cfg
    payload = {'constraint_pair': 'statistical', 'statistical_loss': 'signed_measure',
               'statistical_loss_config': cfg}
    assert statistical_config_from_payload(payload) == cfg
    assert statistical_config_from_args(SimpleNamespace(statistical_loss='signed_measure'), 'statistical') == cfg
    with pytest.raises(ValueError, match='--kl-'):
        statistical_config_from_args(SimpleNamespace(statistical_loss='signed_measure', kl_bins=32), 'statistical')
    with pytest.raises(ValueError, match='Inconsistent'):
        validate_statistical_config({**cfg, 'zero_reference': 'zero_physical_pressure'})


@pytest.mark.parametrize('decoder_mode', ['information_only', 'separate_surface_and_information'])
def test_guided_statistical_route_updates_encoder_without_forecast_decoder(
        split_prepared, monkeypatch, decoder_mode):
    pipe, batch = split_prepared
    pipe.bridge.mode = 'guided'
    manifold = pipe.bridge.manifold

    def forbidden(*args, **kwargs):
        raise AssertionError('Statistical guide supervision must bypass forecast model and decoder')

    monkeypatch.setattr(pipe.predictor, 'forward', forbidden)
    monkeypatch.setattr(manifold.core.manifold.decoder, 'forward', forbidden)
    values = reconstruction_constraint_losses(
        pipe, batch, pair_weights('statistical'), 'statistical', decoder_mode=decoder_mode,
        statistical_config=make_statistical_config('signed_measure'))
    torch.testing.assert_close(values['statistical_total'], values['statistical_signed_measure'])
    torch.testing.assert_close(values['statistical_total'],
                               values['statistical_signed_spatial_js'] + values['statistical_signed_mass_mse'])
    assert values['statistical_kl_entropy'] == values['information_spatial_quantile'] == 0
    assert values['pinn_total'] == values['static'] == 0
    values['statistical_total'].backward()
    assert any(p.grad is not None and bool(p.grad.abs().sum())
               for p in manifold.core.manifold.encoder.parameters())
    for module in (pipe.predictor, manifold.core.manifold.decoder, manifold.pinn):
        assert all(p.grad is None for p in module.parameters())


def test_guided_statistical_route_rejects_shared_forecast_decoder(split_prepared):
    pipe, batch = split_prepared
    pipe.bridge.mode = 'guided'
    with pytest.raises(ValueError, match='Guided constraints require'):
        reconstruction_constraint_losses(pipe, batch, pair_weights('statistical'), 'statistical',
                                          decoder_mode='surface_and_information')
