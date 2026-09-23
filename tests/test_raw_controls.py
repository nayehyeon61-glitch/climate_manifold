"""Direct-data controls use the same temporal cores without any learned E/D."""
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from test_pinn_training import pinn_prepared
from test_spatial_pipeline import spatial_manifold
from climate_manifold.downstream.pipeline import ForecastPipeline, PredictorConfig
from climate_manifold.downstream.protocol import validate_experiment
from climate_manifold.downstream.spatial_baselines import SpatialHistoryPredictor
from climate_manifold.downstream.latent_climode import LatentClimODEPredictor


def raw_config(family, **kwargs):
    return PredictorConfig(model=family, bridge='raw', training_mode='joint',
                           latent_layout='spatial', raw_backend='matched',
                           hidden_dim=8, climode_step_hours=6, **kwargs)


@pytest.mark.parametrize('family', ['mlp', 'neural_ode', 'climode'])
def test_raw_controls_have_no_manifold_and_use_observed_information(pinn_prepared, family):
    original, batch, data, _ = pinn_prepared
    # A plain data/config contract suffices: no representation module, encoder,
    # decoder, pretrained checkpoint or external ClimODE constants exist here.
    contract = SimpleNamespace(config=original.config, info_metadata=original.info_metadata)
    pipe = ForecastPipeline(contract, raw_config(family), schema=data['schema'])
    assert pipe.bridge.manifold is None
    assert all(key.startswith('predictor.') for key in pipe.state_dict())
    assert {id(p) for p in pipe.parameters()} == {id(p) for p in pipe.predictor.parameters()}
    assert pipe.predictor.information_dim == batch['information'].shape[-1]
    history = batch['history'].clone().requires_grad_(True)
    information = batch['information'].clone().requires_grad_(True)
    output = pipe(history, information, batch['origin_time_ns'], torch.tensor([6., 12.]))
    assert output['mean'].shape == (2, 2, original.config.state_dim)
    assert output['std'] is output['predicted_latent'] is output['history_latent'] is None
    torch.testing.assert_close(output['reconstructed_origin'], history[:, -1])
    (output['mean'] - batch['targets'][:, :2]).square().mean().backward()
    assert torch.isfinite(history.grad).all() and (history.grad.abs().sum((0, 2)) > 0).all()
    assert torch.isfinite(information.grad).all() and information.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in pipe.predictor.parameters())
    with torch.no_grad():
        original_mean = pipe(history, information, batch['origin_time_ns'], torch.tensor([6.]))['mean']
        changed_mean = pipe(history, information + .5, batch['origin_time_ns'], torch.tensor([6.]))['mean']
    assert not torch.equal(original_mean, changed_mean)
    with pytest.raises(ValueError, match='origin information'):
        pipe(history, information[:, None].expand(-1, 2, -1), batch['origin_time_ns'], torch.tensor([6.]))
    poisoned = information.detach().clone()
    poisoned[:, 0] = float('nan')
    with pytest.raises(ValueError, match='origin information'):
        pipe(history, poisoned, batch['origin_time_ns'], torch.tensor([6.]))
    protocol = validate_experiment(pipe.config, 'primary')
    assert protocol['path'] == 'observations -> field predictor'
    assert protocol['predictor_variant'] == ('raw_transport_climode' if family == 'climode' else 'raw_spatial_' + family)
    if family == 'climode':
        assert pipe.predictor.spatial_factor == 1
        assert pipe.predictor.max_speed == pipe.config.latent_max_speed * original.config.spatial_downsample


@pytest.mark.parametrize('family', ['mlp', 'neural_ode', 'climode'])
def test_raw_and_latent_use_same_core_and_checkpoint_roundtrip(pinn_prepared, family):
    original, batch, data, _ = pinn_prepared
    manifold = spatial_manifold(original, data)
    raw = ForecastPipeline(manifold, raw_config(family), schema=data['schema'])
    latent = ForecastPipeline(manifold, replace(raw.config, bridge='latent'), schema=data['schema'])
    assert type(raw.predictor) is type(latent.predictor)
    leads = torch.tensor([6., 12.])
    with torch.no_grad():
        expected = raw(batch['history'], batch['information'], batch['origin_time_ns'], leads)
        manifold_forecast = latent(batch['history'], batch['information'], batch['origin_time_ns'], leads)
    assert expected['mean'].shape == manifold_forecast['mean'].shape
    restored = ForecastPipeline(manifold, PredictorConfig(**asdict(raw.config)), schema=data['schema'])
    restored.load_state_dict(raw.state_dict(), strict=True)
    with torch.no_grad():
        actual = restored(batch['history'], batch['information'], batch['origin_time_ns'], leads)
    torch.testing.assert_close(actual['mean'], expected['mean'], atol=0, rtol=0)


@pytest.mark.parametrize('family', ['mlp', 'climode'])
def test_raw_conditioning_can_be_disabled_without_unused_information_parameters(pinn_prepared, family):
    original, batch, data, _ = pinn_prepared
    pipe = ForecastPipeline(original, raw_config(family, condition_information=False), schema=data['schema'])
    assert pipe.predictor.information_channels == 0
    with torch.no_grad():
        a = pipe(batch['history'], None, batch['origin_time_ns'], torch.tensor([6.]))['mean']
        b = pipe(batch['history'], torch.full_like(batch['information'], float('nan')),
                 batch['origin_time_ns'], torch.tensor([6.]))['mean']
    torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_historical_raw_config_keeps_legacy_and_matched_requires_explicit_grid():
    assert PredictorConfig(bridge='raw').raw_backend == 'legacy'
    for overrides in ({'training_mode': 'frozen'}, {'latent_layout': 'global'}, {'model': 'persistence'}):
        options = dict(model='neural_ode', bridge='raw', training_mode='joint',
                       latent_layout='spatial', raw_backend='matched')
        options.update(overrides)
        with pytest.raises(ValueError, match='Matched raw'):
            PredictorConfig(**options)


@pytest.mark.parametrize('family', ['neural_ode', 'climode'])
def test_latent_core_rejects_silent_information_bypass(pinn_prepared, family):
    original, batch, data, _ = pinn_prepared
    if family == 'climode':
        model = LatentClimODEPredictor(original.config.grid, data['schema'], hidden=8, spatial_factor=1)
    else:
        model = SpatialHistoryPredictor(original.config.grid, original.config.history_steps, hidden=8)
    with pytest.raises(ValueError, match='through the encoder'):
        model(batch['history'], torch.tensor([6.]), batch['origin_time_ns'], batch['information'])
