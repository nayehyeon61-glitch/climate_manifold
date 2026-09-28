"""Temporal quantile transport follows the forecast, with detached labels."""
from types import SimpleNamespace

import pytest
import torch

from test_pinn_training import pinn_prepared
from test_reconstruction_objective import split_prepared
from climate_manifold.downstream.statistical_flow import (
    forecast_statistical_flow_losses, make_statistical_flow_config,
    quantile_transport_velocity_loss, statistical_flow_config_from_args,
    statistical_flow_config_from_payload, weighted_spatial_quantiles,
)


def translated_path(speed, leads):
    origin = torch.tensor([[[-2., -1., 1., 3.]]], dtype=torch.float64)
    path = origin[:, None] + speed * leads[None, :, None, None] / 24.
    return origin, path


def test_velocity_match_measures_rate_per_day_with_nonuniform_intervals():
    leads = torch.tensor([6., 18., 30.], dtype=torch.float64)
    origin, target = translated_path(1., leads)
    _, prediction = translated_path(3., leads)
    area = torch.tensor([.1, .2, .3, .4])
    loss = quantile_transport_velocity_loss(prediction, target, origin, area, leads)
    assert loss.item() == pytest.approx(4.)
    # The same value trajectory taking twice as long has one-quarter squared rate error.
    stretched = quantile_transport_velocity_loss(prediction, target, origin, area, 2*leads)
    assert stretched.item() == pytest.approx(1.)


def test_observed_origin_anchor_penalizes_constant_distribution_offset():
    leads = torch.tensor([6., 12.])
    origin, target = translated_path(1., leads)
    prediction = target + 1.
    # First velocity error = 1 / .25 = 4; next velocity error = 0.
    loss = quantile_transport_velocity_loss(prediction, target, origin, torch.ones(4), leads)
    assert loss.item() == pytest.approx(8.)


def test_transport_quantiles_keep_area_and_variable_distributions_separate():
    values = torch.tensor([[[[0., 2.], [10., 12.]], [[1., 3.], [11., 13.]]]])
    expected = torch.tensor([[[[0., 2., 2., 2.], [10., 12., 12., 12.]],
                              [[1., 3., 3., 3.], [11., 13., 13., 13.]]]])
    actual = weighted_spatial_quantiles(values, torch.tensor([.25, .75]), quantiles=4)
    torch.testing.assert_close(actual, expected)


def test_quantile_transport_is_marginal_not_geographic_motion():
    leads = torch.tensor([6., 12.])
    origin, target = translated_path(1., leads)
    assert quantile_transport_velocity_loss(
        target.flip(-1), target, origin, torch.ones(4), leads) == 0


def test_velocity_objective_differentiates_only_prediction():
    leads = torch.tensor([6., 12.], dtype=torch.float64)
    origin, target = translated_path(1., leads)
    _, predicted = translated_path(2., leads)
    origin.requires_grad_()
    target.requires_grad_()
    predicted.requires_grad_()
    area = torch.tensor([.1, .2, .3, .4], dtype=torch.float64, requires_grad=True)
    loss = quantile_transport_velocity_loss(predicted, target, origin, area, leads)
    loss.backward()
    assert predicted.grad is not None and predicted.grad.abs().sum() > 0
    assert torch.isfinite(predicted.grad).all()
    assert origin.grad is target.grad is area.grad is None
    assert torch.autograd.gradcheck(
        lambda x: quantile_transport_velocity_loss(x, target, origin, area, leads), (predicted,))


@pytest.mark.parametrize('leads', ([0., 6.], [6., 6.], [12., 6.], [6., float('nan')], [6.]))
def test_invalid_time_axes_fail(leads):
    origin, target = translated_path(1., torch.tensor([6., 12.]))
    with pytest.raises(ValueError, match='lead hours'):
        quantile_transport_velocity_loss(target, target, origin, torch.ones(4), leads)


def _forecast_case(split_prepared):
    pipe, original = split_prepared
    batch = {**original,
             'targets': original['targets'].detach().clone().requires_grad_(),
             'statistical_flow_information_targets': original['information_targets'].detach().clone().requires_grad_()}
    leads = torch.tensor([6., 12.])
    origin_ns = batch['origin_time_ns']
    prediction = pipe(batch['history'], batch['information'], origin_ns, leads,
                      reconstruct_origin=False)
    return pipe, batch, leads, prediction


@pytest.mark.parametrize('mode', ('separate_surface_and_information', 'surface_and_information', 'information_only'))
def test_flow_reaches_forecast_and_selected_decoders_without_future_encoding(split_prepared, monkeypatch, mode):
    pipe, batch, leads, prediction = _forecast_case(split_prepared)
    manifold = pipe.bridge.manifold
    before = prediction['mean'].detach().clone()

    def forbidden(*args, **kwargs):
        raise AssertionError('Flow loss must reuse the prediction, never rerun F or encode a future label')

    monkeypatch.setattr(manifold, 'raw_encode', forbidden)
    monkeypatch.setattr(pipe.predictor, 'forward', forbidden)
    if mode != 'surface_and_information':
        monkeypatch.setattr(manifold.core.manifold.decoder, 'forward', forbidden)
    if mode != 'separate_surface_and_information':
        monkeypatch.setattr(pipe.reconstruction_decoder, 'forward', forbidden)
    values = forecast_statistical_flow_losses(pipe, prediction, batch, leads,
                                               make_statistical_flow_config(.2), decoder_mode=mode)
    assert values['statistical_flow'] > 0
    torch.testing.assert_close(values['statistical_flow_regularization'], .2*values['statistical_flow'])
    values['statistical_flow_regularization'].backward()
    active_modules = [manifold.core.manifold.encoder, manifold.information, manifold.info_head, pipe.predictor]
    if mode == 'surface_and_information':
        active_modules.append(manifold.core.manifold.decoder)
    else:
        assert all(parameter.grad is None for parameter in manifold.core.manifold.decoder.parameters())
    if mode == 'separate_surface_and_information':
        active_modules.append(pipe.reconstruction_decoder)
    else:
        assert all(parameter.grad is None for parameter in pipe.reconstruction_decoder.parameters())
    if mode == 'information_only':
        assert values['statistical_flow_surface'] == 0
        torch.testing.assert_close(values['statistical_flow'], values['statistical_flow_information'])
    for module in active_modules:
        gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(gradient.abs().sum() for gradient in gradients) > 0
    assert batch['targets'].grad is batch['statistical_flow_information_targets'].grad is None
    assert all(parameter.grad is None for parameter in manifold.pinn.parameters())
    torch.testing.assert_close(prediction['mean'], before, rtol=0, atol=0)


def test_future_label_changes_only_flow_supervision_and_static_labels_are_ignored(split_prepared):
    pipe, batch, leads, prediction = _forecast_case(split_prepared)
    config = make_statistical_flow_config(.1)
    baseline = forecast_statistical_flow_losses(pipe, prediction, batch, leads, config)
    before = {key: value.detach().clone() for key, value in prediction.items() if isinstance(value, torch.Tensor)}
    changed = {**batch, 'statistical_flow_information_targets': batch['statistical_flow_information_targets'] + 5.}
    alternative = forecast_statistical_flow_losses(pipe, prediction, changed, leads, config)
    assert not torch.equal(baseline['statistical_flow_information'], alternative['statistical_flow_information'])
    torch.testing.assert_close(baseline['statistical_flow_surface'], alternative['statistical_flow_surface'])
    for key in before:
        torch.testing.assert_close(prediction[key], before[key], rtol=0, atol=0)

    manifold = pipe.bridge.manifold
    shape = manifold.info_metadata['shape']
    poisoned = batch['statistical_flow_information_targets'].detach().clone()
    fields = poisoned.reshape(*poisoned.shape[:2], *shape)
    for index, variable in enumerate(manifold.info_metadata['variables']):
        if variable['kind'] == 'static':
            fields[:, :, index] = float('nan')
    same = forecast_statistical_flow_losses(pipe, prediction,
        {**batch, 'statistical_flow_information_targets': poisoned}, leads, config)
    for key in baseline:
        torch.testing.assert_close(same[key], baseline[key])


def test_auxiliary_decoders_receive_raw_not_normalized_latent_coordinates(split_prepared):
    pipe, batch = split_prepared
    manifold = pipe.bridge.manifold
    with torch.no_grad():
        manifold.core.latent_mean.fill_(3.)
        manifold.core.latent_scale.fill_(2.)
    pipe, batch, leads, prediction = _forecast_case((pipe, batch))
    captured = []
    hooks = [module.register_forward_pre_hook(lambda _module, args: captured.append(args[0].detach().clone()))
             for module in (pipe.reconstruction_decoder, manifold.info_head)]
    try:
        forecast_statistical_flow_losses(pipe, prediction, batch, leads, make_statistical_flow_config(.1))
    finally:
        for hook in hooks:
            hook.remove()
    assert len(captured) == 2
    expected = 2. * prediction['predicted_latent'] + 3.
    for value in captured:
        torch.testing.assert_close(value, expected)


@pytest.mark.parametrize('dt', (torch.tensor([[6., 18.], [6., 18.]]),
                               torch.tensor([[6.], [6.]]),
                               torch.tensor([[6., 0.], [6., 6.]]),
                               torch.tensor([[6., float('nan')], [6., 6.]])))
def test_target_time_mismatch_fails_before_auxiliary_decoding(split_prepared, monkeypatch, dt):
    pipe, batch, leads, prediction = _forecast_case(split_prepared)

    def forbidden(*args, **kwargs):
        raise AssertionError('Mismatched target times must fail before decoding')

    monkeypatch.setattr(pipe.bridge.manifold.info_head, 'forward', forbidden)
    with pytest.raises(ValueError, match='dt_hours'):
        forecast_statistical_flow_losses(pipe, prediction, {**batch, 'dt_hours': dt},
                                          leads, make_statistical_flow_config(.1))


def test_disabled_flow_requires_no_future_labels_or_pipeline():
    values = forecast_statistical_flow_losses(None, {'mean': torch.zeros(2, 3, 4)}, {}, [], None)
    assert all(value == 0 and not value.requires_grad for value in values.values())


def test_config_is_independent_of_w2_kl_and_preserves_old_payloads():
    assert statistical_flow_config_from_payload({}) is None
    assert statistical_flow_config_from_args(SimpleNamespace(), None) is None
    configs = []
    for kind in ('w2', 'kl_entropy'):
        args = SimpleNamespace(statistical_flow_weight=.1, statistical_loss=kind,
                               bridge='latent', training_mode='joint', anchor='none')
        config = statistical_flow_config_from_args(args, 'pinn_statistical')
        configs.append(config)
        payload = {'constraint_pair': 'pinn_statistical', 'statistical_loss': kind,
                   'statistical_flow_config': config, 'config': vars(args)}
        assert statistical_flow_config_from_payload(payload) == config
    assert configs[0] == configs[1] == make_statistical_flow_config(.1)


@pytest.mark.parametrize('weight', (-1., float('nan'), float('inf'), True))
def test_invalid_weights_fail(weight):
    with pytest.raises(ValueError, match='finite and nonnegative'):
        make_statistical_flow_config(weight)


@pytest.mark.parametrize('changes,pair', (
    ({}, 'pinn_static'), ({'bridge': 'raw'}, 'pinn_statistical'),
    ({'training_mode': 'frozen'}, 'statistical_static'),
    ({'anchor': 'origin'}, 'pinn_statistical'),
))
def test_flow_rejects_incompatible_training_routes(changes, pair):
    args = SimpleNamespace(statistical_flow_weight=.1, **changes)
    with pytest.raises(ValueError, match='requires'):
        statistical_flow_config_from_args(args, pair)


def test_flow_config_detects_unsupported_or_inconsistent_metadata():
    config = make_statistical_flow_config(.2)
    for changed in ({**config, 'origin': 'learned'}, {**config, 'time_unit': 'hours'},
                    {**config, 'weight': 0.}, {**config, 'quantiles': 0}):
        with pytest.raises(ValueError):
            statistical_flow_config_from_payload({'constraint_pair': 'pinn_statistical',
                                                 'statistical_flow_config': changed})
    with pytest.raises(ValueError, match='positive'):
        statistical_flow_config_from_args(SimpleNamespace(statistical_flow_quantiles=32), 'pinn_statistical')
    with pytest.raises(ValueError, match='requires'):
        statistical_flow_config_from_payload({'constraint_pair': None, 'statistical_flow_config': config})
