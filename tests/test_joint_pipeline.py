"""Joint forecasts must train the actual representation and temporal predictor."""
from dataclasses import asdict

import numpy as np
import pytest
import torch

from test_pinn_training import pinn_prepared
from test_downstream import seal
from climate_manifold.downstream.pipeline import ForecastPipeline, PredictorConfig
from climate_manifold.downstream.protocol import experiment_contract
from climate_manifold.downstream.plain_ae import new_plain_ae


def assert_gradient(module):
    gradients = [p.grad for p in module.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(value).all() for value in gradients)
    assert sum(value.abs().sum() for value in gradients) > 0


@pytest.mark.parametrize('kind', ['mlp', 'neural_ode'])
@pytest.mark.parametrize('representation', ['climate_manifold', 'plain_ae'])
def test_future_loss_jointly_updates_encoder_predictor_decoder(pinn_prepared, kind, representation):
    a, batch, _, _ = pinn_prepared
    ae = (new_plain_ae({'config': asdict(a.config), 'information_metadata': a.info_metadata})
          if representation == 'plain_ae' else None)
    pipe = ForecastPipeline(a, PredictorConfig(model=kind, hidden_dim=24,
        training_mode='joint', representation=representation), representation=ae)
    selected = pipe.bridge.manifold
    assert not bool(selected.core.manifold_ready)
    assert selected.training
    torch.testing.assert_close(selected.core.latent_mean, torch.zeros(a.config.manifold_dim))
    torch.testing.assert_close(selected.core.latent_scale, torch.ones(a.config.manifold_dim))
    groups = [selected.core.manifold.encoder, pipe.predictor,
              selected.core.manifold.decoder, selected.information]
    before = [[p.detach().clone() for p in module.parameters()] for module in groups]
    output = pipe(batch['history'], batch['information'], batch['origin_time_ns'], torch.tensor([6., 12.]))
    assert output['history_latent'].requires_grad
    output['history_latent'].retain_grad()
    # Forecast error alone, with no reconstruction or latent supervision, must
    # supply gradients all the way through the representation.
    (output['mean']-batch['targets'][:, :2]).square().mean().backward()
    for module in groups:
        assert_gradient(module)
    assert output['history_latent'].grad[:, :-1].abs().sum() > 0
    parameters = [p for p in pipe.parameters() if p.requires_grad]
    torch.optim.Adam(parameters, lr=1e-3).step()
    for module, previous in zip(groups, before):
        assert any(not torch.equal(old, new) for old, new in zip(previous, module.parameters()))
    if representation == 'climate_manifold':
        for unused in (a.a_sampler, a.a_context, a.core.manifold.latent_drift):
            assert all(not p.requires_grad and p.grad is None for p in unused.parameters())
        assert all(p.requires_grad for p in a.info_head.parameters())
        assert all(p.requires_grad for p in a.pinn.parameters())
    pipe.eval()
    with torch.no_grad():
        output = pipe(batch['history'], batch['information'], batch['origin_time_ns'], torch.tensor([6.]))
    assert not output['mean'].requires_grad
    assert not output['history_latent'].requires_grad


def test_historical_config_retains_frozen_behavior(pinn_prepared):
    a, batch, data, _ = pinn_prepared
    seal(a, data)
    config = PredictorConfig(**{'model': 'mlp', 'hidden_dim': 24})
    pipe = ForecastPipeline(a, config)
    assert config.training_mode == 'frozen'
    pipe.train()
    assert not a.training
    output = pipe(batch['history'], batch['information'], batch['origin_time_ns'], torch.tensor([6.]))
    assert not output['history_latent'].requires_grad
    output['mean'].square().mean().backward()
    assert all(p.grad is None for p in a.parameters())
    assert_gradient(pipe.predictor)
    assert experiment_contract(config)['representation_frozen']
    assert not experiment_contract(PredictorConfig(training_mode='joint'))['representation_frozen']


def test_joint_decoded_climode_receives_forecast_gradient(pinn_prepared):
    pytest.importorskip('torchdiffeq')
    a, batch, data, _ = pinn_prepared
    constants = {**data['schema']['variables'][0]['coords'],
        'orography': np.ones((4, 8)), 'lsm': np.zeros((4, 8)), 'orography_units': 'm'}
    config = PredictorConfig(model='climode', bridge='decoded', training_mode='joint',
        climode_attention=False, climode_step_hours=6, velocity_iterations=1)
    pipe = ForecastPipeline(a, config, constants, data['schema'])
    output = pipe(batch['history'], batch['information'], batch['origin_time_ns'], torch.tensor([6., 12.]))
    (output['mean']-batch['targets'][:, :2]).square().mean().backward()
    for module in (a.core.manifold.encoder, a.core.manifold.decoder, a.information, pipe.predictor):
        assert_gradient(module)
    contract = experiment_contract(config)
    assert contract['suite'] == 'auxiliary'
    assert contract['prediction_space'] == 'field'
    assert contract['training_mode'] == 'joint'
    assert not contract['representation_frozen']


def test_joint_rejects_residual_bypass_and_nontrainable_predictor():
    for kwargs in ({'anchor': 'origin'}, {'model': 'persistence', 'bridge': 'raw'}):
        with pytest.raises(ValueError, match='trainable predictor and anchor=none'):
            PredictorConfig(training_mode='joint', **kwargs)
    with pytest.raises(ValueError, match='not a reshaped global latent'):
        PredictorConfig(training_mode='joint', model='climode', bridge='latent')
