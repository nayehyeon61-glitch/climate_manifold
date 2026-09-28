"""Observed constraints use a dedicated surface D_rec and bypass forecast D/F."""
from dataclasses import replace

import pytest
import torch

from test_pinn_training import pinn_prepared
from climate_manifold.downstream.joint_objective import JointObjectiveWeights, spatial_quantile_loss
from climate_manifold.downstream.pipeline import ForecastPipeline, PredictorConfig
from climate_manifold.downstream.reconstruction_objective import (
    DECODER_MODES, PAIRS, reconstruction_constraint_losses,
)
from climate_manifold.spatial import SpatialClimateManifold


def pair_weights(pair):
    return JointObjectiveWeights(
        reconstruction=.1, physics=0., information=0.,
        distribution=.2 if 'statistical' in pair else 0.,
        static=.3 if 'static' in pair.split('_') else 0.,
        pinn=.01 if 'pinn' in pair else 0.,
    )


@pytest.fixture
def split_prepared(pinn_prepared):
    old, previous, data, _ = pinn_prepared
    config = replace(old.config, representation_kind='spatial', latent_channels=3,
                     spatial_downsample=2, spatial_hidden_dim=8)
    manifold = SpatialClimateManifold(
        config, data['schema'], data['mean'], data['scale'], data['statistics'],
        data['information_metadata'], pinn_config=old.pinn.config,
        information_mean=data['information_mean'], information_scale=data['information_scale'],
    )
    pipe = ForecastPipeline(manifold, PredictorConfig(
        model='neural_ode', bridge='latent', training_mode='joint',
        latent_layout='spatial', hidden_dim=8), schema=data['schema'],
        separate_reconstruction_decoder=True)
    origin = config.history_span_steps - 1
    information = torch.stack([torch.as_tensor(data['information'][i-1:i+1].copy())
                               for i in (origin, origin + 1)])
    batch = {**previous,
             'constraint_states': previous['history'][:, -2:].clone().requires_grad_(),
             'constraint_information': information.requires_grad_(),
             'constraint_dt_hours': torch.full((2, 1), 6.)}
    return pipe, batch


@pytest.mark.parametrize('pair', PAIRS)
@pytest.mark.parametrize('decoder_mode', DECODER_MODES)
def test_auxiliary_only_gradients_update_representation_never_predictor(
        split_prepared, monkeypatch, pair, decoder_mode):
    pipe, batch = split_prepared
    manifold = pipe.bridge.manifold

    def forbidden(*args, **kwargs):
        raise AssertionError('Forecast F must never enter observed reconstruction constraints')

    monkeypatch.setattr(pipe.predictor, 'forward', forbidden)
    if decoder_mode != 'surface_and_information':
        def surface_forbidden(*args, **kwargs):
            raise AssertionError('Forecast D must not enter separated constraints')

        monkeypatch.setattr(manifold.core.manifold.decoder, 'forward', surface_forbidden)
    if decoder_mode != 'separate_surface_and_information':
        def reconstruction_forbidden(*args, **kwargs):
            raise AssertionError('Dedicated D_rec must not enter legacy constraints')

        monkeypatch.setattr(pipe.reconstruction_decoder, 'forward', reconstruction_forbidden)
    values = reconstruction_constraint_losses(pipe, batch, pair_weights(pair), pair,
                                              decoder_mode=decoder_mode)
    values['regularization'].backward()
    groups = [manifold.core.manifold.encoder, manifold.information, manifold.info_head]
    if decoder_mode == 'surface_and_information':
        groups.append(manifold.core.manifold.decoder)
    else:
        assert all(p.requires_grad and p.grad is None
                   for p in manifold.core.manifold.decoder.parameters())
    if decoder_mode == 'separate_surface_and_information':
        groups.append(pipe.reconstruction_decoder)
    else:
        assert all(p.grad is None for p in pipe.reconstruction_decoder.parameters())
    if decoder_mode == 'information_only':
        assert values['reconstruction_surface'] == values['statistical_surface'] == 0
    if 'pinn' in pair:
        groups.append(manifold.pinn.closure_head)
        assert values['pinn_tendency'] == 0
        assert 'pinn_surface_tendency' not in values
    else:
        assert all(p.grad is None for p in manifold.pinn.parameters())
    for module in groups:
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients) > 0
    assert all(p.grad is None for p in pipe.predictor.parameters())
    assert batch['constraint_states'].grad is None
    assert batch['constraint_information'].grad is None
    assert values['physics'] == values['information'] == 0


@pytest.mark.parametrize('pair', PAIRS)
@pytest.mark.parametrize('decoder_mode', DECODER_MODES)
def test_forecast_inputs_outputs_and_future_targets_are_not_read(split_prepared, pair, decoder_mode):
    pipe, batch = split_prepared
    weights = pair_weights(pair)
    expected = reconstruction_constraint_losses(pipe, batch, weights, pair, decoder_mode=decoder_mode)
    minimal = {key: value for key, value in batch.items() if key.startswith('constraint_')}
    # Even NaNs in all unrelated forecast inputs/targets/predictions are ignored.
    poisoned = {key: torch.full_like(value, float('nan')) if value.is_floating_point() else None
                for key, value in batch.items() if not key.startswith('constraint_')}
    poisoned.update(predicted_latent=torch.tensor(float('nan')), mean=torch.tensor(float('nan')))
    for candidate in (minimal, {**minimal, **poisoned}):
        actual = reconstruction_constraint_losses(pipe, candidate, weights, pair, decoder_mode=decoder_mode)
        assert expected.keys() == actual.keys()
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)


def test_each_observation_is_encoded_with_its_colocated_information(split_prepared, monkeypatch):
    pipe, batch = split_prepared
    manifold = pipe.bridge.manifold
    original = manifold.raw_encode
    calls = []

    def capture(states, information):
        calls.append((states.detach().clone(), information.detach().clone()))
        return original(states, information)

    monkeypatch.setattr(manifold, 'raw_encode', capture)
    reconstruction_constraint_losses(pipe, batch, pair_weights('statistical_static'), 'statistical_static')
    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0], batch['constraint_states'])
    torch.testing.assert_close(calls[0][1], batch['constraint_information'])
    assert not torch.equal(calls[0][1][:, 0], calls[0][1][:, 1])


@pytest.mark.parametrize('decoder_mode', DECODER_MODES)
def test_pointwise_and_statistical_objectives_exclude_static_outputs(split_prepared, decoder_mode):
    pipe, batch = split_prepared
    manifold = pipe.bridge.manifold
    pair, weights = 'statistical_static', pair_weights('statistical_static')
    baseline = reconstruction_constraint_losses(pipe, batch, weights, pair, decoder_mode=decoder_mode)
    mask = torch.tensor([v['kind'] == 'static' for v in manifold.info_metadata['variables']])
    shift = mask[:, None, None].expand(manifold.info_metadata['shape']).flatten().float() * 50

    def change_only_static(module, args, output):
        return output + shift.to(output)

    handle = manifold.info_head.register_forward_hook(change_only_static)
    changed = reconstruction_constraint_losses(pipe, batch, weights, pair, decoder_mode=decoder_mode)
    handle.remove()
    for key in ('reconstruction', 'reconstruction_information', 'reconstruction_surface',
                'information_spatial_quantile', 'statistical_information', 'statistical_surface'):
        torch.testing.assert_close(baseline[key], changed[key], rtol=0, atol=0)
    assert changed['static'] > baseline['static']


@pytest.mark.parametrize('decoder_mode', DECODER_MODES)
def test_losses_use_area_and_variable_mean_and_static_origin_target(split_prepared, monkeypatch, decoder_mode):
    pipe, batch = split_prepared
    manifold = pipe.bridge.manifold
    pair, weights = 'statistical_static', pair_weights('statistical_static')
    z = manifold.raw_encode(batch['constraint_states'], batch['constraint_information']).detach()
    # Keep the prediction fixed while perturbing labels to isolate target selection.
    monkeypatch.setattr(manifold, 'raw_encode', lambda *args: z)
    values = reconstruction_constraint_losses(pipe, batch, weights, pair, decoder_mode=decoder_mode)
    area = manifold.temporal.area.flatten()
    cells = len(area)
    surface_decoder = (pipe.reconstruction_decoder if decoder_mode == 'separate_surface_and_information'
                       else manifold.core.manifold.decode)
    surface = surface_decoder(z).reshape(2, 2, -1, cells)
    truth = batch['constraint_states'].reshape_as(surface)
    decoded = manifold.info_head(z).reshape(2, 2, -1, cells)
    information = batch['constraint_information'].reshape_as(decoded)
    static = torch.tensor([v['kind'] == 'static' for v in manifold.info_metadata['variables']])
    surface_mse = ((surface - truth).square() * area).sum(-1).mean()
    dynamic_mse = ((decoded[:, :, ~static] - information[:, :, ~static]).square() * area).sum(-1).mean()
    static_mse = ((decoded[:, :, static] - information[:, 1:2, static]).square() * area).sum(-1).mean()
    expected_mse = (.5 * (surface_mse + dynamic_mse)
                    if decoder_mode != 'information_only' else dynamic_mse)
    torch.testing.assert_close(values['reconstruction'], expected_mse)
    torch.testing.assert_close(values['static'], static_mse)
    quantile = spatial_quantile_loss(decoded[:, :, ~static], information[:, :, ~static], area)
    if decoder_mode != 'information_only':
        quantile = .5 * (spatial_quantile_loss(surface, truth, area) + quantile)
    torch.testing.assert_close(values['information_spatial_quantile'], quantile)
    changed = batch['constraint_information'].detach().clone().reshape_as(decoded)
    changed[:, 0, static] += 500
    altered = reconstruction_constraint_losses(pipe, {**batch, 'constraint_information': changed.reshape_as(
        batch['constraint_information'])}, weights, pair, decoder_mode=decoder_mode)
    # Both endpoints use the same static origin target, not the earlier label.
    for key in values:
        torch.testing.assert_close(values[key], altered[key], rtol=0, atol=0)


@pytest.mark.parametrize('pair', PAIRS)
@pytest.mark.parametrize('decoder_mode', DECODER_MODES)
def test_pair_activation_and_weighted_total(split_prepared, pair, decoder_mode):
    pipe, batch = split_prepared
    weights = pair_weights(pair)
    values = reconstruction_constraint_losses(pipe, batch, weights, pair, decoder_mode=decoder_mode)
    for name, field in (('pinn_total', 'pinn'), ('static', 'static'),
                        ('information_spatial_quantile', 'distribution')):
        assert bool(values[name] > 0) == bool(getattr(weights, field))
    expected = (weights.reconstruction * values['reconstruction'] + weights.pinn * values['pinn_total']
                + weights.static * values['static'] + weights.distribution * values['information_spatial_quantile'])
    torch.testing.assert_close(values['regularization'], expected)


def test_default_bypasses_surface_decoder_without_freezing_forecast_decoder(split_prepared, monkeypatch):
    pipe, batch = split_prepared
    manifold = pipe.bridge.manifold

    def forbidden(*args, **kwargs):
        raise AssertionError('Default auxiliary route must not invoke surface D or predictor F')

    # Calling the default must use D_rec instead of forecast D and bypass F.
    with monkeypatch.context() as isolated:
        isolated.setattr(manifold.core.manifold.decoder, 'forward', forbidden)
        isolated.setattr(pipe.predictor, 'forward', forbidden)
        values = reconstruction_constraint_losses(pipe, batch, pair_weights('pinn_statistical'),
                                                  'pinn_statistical')
        values['regularization'].backward()
    assert values['reconstruction_surface'] > 0
    assert values['statistical_surface'] > 0
    torch.testing.assert_close(values['reconstruction'], .5 * (
        values['reconstruction_surface'] + values['reconstruction_information']))
    torch.testing.assert_close(values['information_spatial_quantile'], .5 * (
        values['statistical_surface'] + values['statistical_information']))
    assert any(p.grad is not None and bool(p.grad.abs().sum() > 0)
               for p in pipe.reconstruction_decoder.parameters())
    assert all(p.requires_grad and p.grad is None for p in manifold.core.manifold.decoder.parameters())
    assert all(p.grad is None for p in pipe.predictor.parameters())

    pipe.zero_grad(set_to_none=True)
    prediction = pipe(batch['history'], batch['information'], batch['origin_time_ns'], torch.tensor([6., 12.]))
    (prediction['mean'] - batch['targets'][:, :2]).square().mean().backward()
    for module in (manifold.core.manifold.encoder, manifold.core.manifold.decoder, pipe.predictor):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients) > 0
    assert all(p.grad is None for p in manifold.info_head.parameters())
    assert all(p.grad is None for p in pipe.reconstruction_decoder.parameters())


def test_separate_surface_statistics_respond_to_msl_labels(split_prepared, pinn_prepared, monkeypatch):
    pipe, batch = split_prepared
    manifold = pipe.bridge.manifold
    schema = pinn_prepared[2]['schema']
    msl = next(i for i, variable in enumerate(schema['variables']) if variable['name'] == 'msl')
    latent = manifold.raw_encode(batch['constraint_states'], batch['constraint_information']).detach()
    monkeypatch.setattr(manifold, 'raw_encode', lambda *args: latent)
    weights = pair_weights('pinn_statistical')
    baseline = reconstruction_constraint_losses(pipe, batch, weights, 'pinn_statistical')
    changed_states = batch['constraint_states'].detach().clone()
    fields = changed_states.reshape(2, 2, *manifold.config.grid)
    fields[:, :, msl] += 100
    changed = reconstruction_constraint_losses(pipe, {**batch, 'constraint_states': changed_states},
                                               weights, 'pinn_statistical')
    assert changed['statistical_surface'] > baseline['statistical_surface']
    assert changed['reconstruction_surface'] > baseline['reconstruction_surface']
    for key in ('statistical_information', 'reconstruction_information', 'pinn_total'):
        torch.testing.assert_close(baseline[key], changed[key], rtol=0, atol=0)
    torch.testing.assert_close(changed['information_spatial_quantile'], .5 * (
        changed['statistical_surface'] + changed['statistical_information']))
    # Surface W2 itself trains D_rec, not the forecast decoder or information D_I.
    changed['statistical_surface'].backward()
    assert any(p.grad is not None and bool(p.grad.abs().sum() > 0)
               for p in pipe.reconstruction_decoder.parameters())
    for module in (manifold.core.manifold.decoder, manifold.info_head, pipe.predictor):
        assert all(p.grad is None for p in module.parameters())


def test_missing_dedicated_decoder_fails_before_encoding(split_prepared, monkeypatch):
    pipe, batch = split_prepared
    pipe.reconstruction_decoder = None

    def forbidden(*args, **kwargs):
        raise AssertionError('Missing D_rec must fail before encoding')

    monkeypatch.setattr(pipe.bridge.manifold, 'raw_encode', forbidden)
    with pytest.raises(ValueError, match='dedicated reconstruction_decoder'):
        reconstruction_constraint_losses(pipe, batch, pair_weights('pinn_statistical'), 'pinn_statistical')


@pytest.mark.parametrize('mode', ['', 'both', 'information', None])
def test_unknown_decoder_mode_is_rejected_before_any_network_call(split_prepared, monkeypatch, mode):
    pipe, batch = split_prepared

    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid mode must fail before encoding')

    monkeypatch.setattr(pipe.bridge.manifold, 'raw_encode', forbidden)
    with pytest.raises(ValueError, match='decoder_mode'):
        reconstruction_constraint_losses(pipe, batch, pair_weights('pinn_statistical'),
                                         'pinn_statistical', decoder_mode=mode)


@pytest.mark.parametrize('changes,message', [
    ({'reconstruction': 0.}, 'positive common reconstruction'),
    ({'physics': .1}, 'physics=information=0'),
    ({'information': .1}, 'physics=information=0'),
    ({'pinn': .1}, 'pinn weight to be zero'),
    ({'distribution': 0.}, 'distribution weight to be positive'),
    ({'static': 0.}, 'static weight to be positive'),
])
def test_pair_weights_reject_hidden_or_missing_constraints(split_prepared, changes, message):
    pipe, batch = split_prepared
    with pytest.raises(ValueError, match=message):
        reconstruction_constraint_losses(pipe, batch, replace(pair_weights('statistical_static'), **changes),
                                         'statistical_static')


@pytest.mark.parametrize('key,value', [
    ('constraint_states', None),
    ('constraint_states', torch.zeros(2, 1, 128)),
    ('constraint_information', torch.zeros(2, 1, 1)),
    ('constraint_dt_hours', torch.tensor([6., 6.])),
    ('constraint_dt_hours', torch.tensor([[6.], [12.]])),
    ('constraint_dt_hours', torch.tensor([[6.], [float('nan')]])),
])
def test_observed_pair_shapes_and_fixed_dt_are_validated(split_prepared, key, value):
    pipe, batch = split_prepared
    with pytest.raises(ValueError, match=key):
        reconstruction_constraint_losses(pipe, {**batch, key: value}, pair_weights('statistical_static'),
                                         'statistical_static')


def test_missing_required_representation_and_pinn_fail_explicitly(split_prepared):
    pipe, batch = split_prepared
    with pytest.raises(ValueError, match='constraint_pair'):
        reconstruction_constraint_losses(pipe, batch, pair_weights('pinn_static'), 'all')
    pipe.bridge.manifold.pinn = None
    with pytest.raises(ValueError, match='enabled Hybrid PINN'):
        reconstruction_constraint_losses(pipe, batch, pair_weights('pinn_static'), 'pinn_static')
    pipe.bridge.mode = 'raw'
    with pytest.raises(ValueError, match='unanchored latent bridge'):
        reconstruction_constraint_losses(pipe, batch, pair_weights('statistical_static'), 'statistical_static')
