"""Joint E--spatial predictor--D path and gradient/causality contracts."""
from dataclasses import replace

import pytest
import torch
from torch import nn

from test_pinn_training import pinn_prepared
from climate_manifold.spatial import SpatialClimateManifold
from climate_manifold.downstream.pipeline import ForecastPipeline, PredictorConfig
from climate_manifold.downstream.protocol import validate_experiment
from climate_manifold.downstream.spatial_baselines import SpatialHistoryPredictor


def spatial_manifold(original, data):
    config = replace(original.config, representation_kind='spatial', latent_channels=3,
                     spatial_downsample=2, spatial_hidden_dim=8)
    return SpatialClimateManifold(config, data['schema'], data['mean'], data['scale'],
                                  data['statistics'], data['information_metadata'])


@pytest.mark.parametrize('family', ['mlp', 'neural_ode', 'climode'])
def test_actual_spatial_forecast_path_and_future_loss_reach_all_components(pinn_prepared, family):
    original, batch, data, _ = pinn_prepared
    manifold = spatial_manifold(original, data)
    config = PredictorConfig(model=family, bridge='latent', training_mode='joint',
                             latent_layout='spatial', hidden_dim=8, climode_step_hours=6)
    contract = validate_experiment(config, 'primary')
    assert contract['path'] == 'encoder -> predictor -> decoder'
    assert not contract['representation_frozen']
    # In particular latent ClimODE needs no external physical-grid constants.
    model = ForecastPipeline(manifold, config, schema=data['schema'])
    events, snapshots = [], {}
    def encoded(module, args, output):
        events.append('E')
    def predicted(module, args, output):
        events.append('F')
        snapshots['predictor_input'] = args[0]
        snapshots['predicted_latent'] = output[0]
    def decoding(module, args):
        events.append('D')
        snapshots.setdefault('first_decoder_input', args[0])
    handles = [manifold.core.manifold.encoder.register_forward_hook(encoded),
               model.predictor.register_forward_hook(predicted),
               manifold.core.manifold.decoder.register_forward_pre_hook(decoding)]
    history = batch['history'].clone().requires_grad_(True)
    out = model(history, batch['information'], batch['origin_time_ns'], torch.tensor([6., 12.]))
    for handle in handles:
        handle.remove()
    assert events[:3] == ['E', 'F', 'D']
    assert snapshots['predictor_input'].shape == (2, manifold.config.history_steps, 3*2*4)
    torch.testing.assert_close(snapshots['first_decoder_input'], snapshots['predicted_latent'])
    assert out['mean'].shape == (2, 2, original.config.state_dim)
    assert out['predicted_latent'].shape == (2, 2, 24)
    assert out['std'] is None
    loss = (out['mean'] - batch['targets'][:, :2]).square().mean()
    loss.backward()
    for module in (manifold.core.manifold.encoder, model.predictor,
                   manifold.core.manifold.decoder, manifold.information):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients) > 0
    # All observed times contribute, including the earliest history field.
    assert (history.grad.abs().sum((0, 2)) > 0).all()
    assert torch.isfinite(history.grad).all()


@pytest.mark.parametrize('family', ['mlp', 'neural_ode'])
def test_spatial_context_has_no_global_dense_bottleneck_and_uses_each_history(family):
    torch.manual_seed(21)
    torch.set_num_threads(1)
    model = SpatialHistoryPredictor((3, 4, 8), history_steps=6, hidden=8, kind=family)
    assert not any(isinstance(layer, nn.Linear) for layer in model.modules())
    history = torch.randn(2, 6, 96, requires_grad=True)
    mean, std = model(history, torch.tensor([6., 12.]), torch.tensor([0, 0]))
    mean.square().mean().backward()
    assert mean.shape == (2, 2, 96) and std is None
    assert (history.grad.abs().sum((0, 2)) > 0).all()
    with pytest.raises(ValueError, match='history length'):
        model(history[:, -1:], torch.tensor([6.]), torch.tensor([0, 0]))


def test_spatial_climode_contract_rejects_relabelled_global_latent_and_frozen_mode(pinn_prepared):
    original, _, data, _ = pinn_prepared
    with pytest.raises(ValueError, match='not a reshaped global latent'):
        PredictorConfig(model='climode', bridge='latent', training_mode='joint')
    with pytest.raises(ValueError, match='joint'):
        PredictorConfig(model='climode', bridge='latent', latent_layout='spatial')
    config = PredictorConfig(model='climode', bridge='latent', latent_layout='spatial', training_mode='joint')
    with pytest.raises(ValueError, match='actual manifold'):
        ForecastPipeline(original, config, schema=data['schema'])
    with pytest.raises(ValueError, match='Primary'):
        validate_experiment(PredictorConfig(model='climode', bridge='decoded', training_mode='joint'), 'primary')


def test_latent_gaussian_variance_is_not_forwarded_through_nonlinear_decoder(pinn_prepared):
    original, batch, data, _ = pinn_prepared
    manifold = spatial_manifold(original, data)
    model = ForecastPipeline(manifold, PredictorConfig(model='mlp', bridge='latent',
                             latent_layout='spatial', training_mode='joint', hidden_dim=8))
    class GaussianLatent(nn.Module):
        def forward(self, history, lead_hours, origin_ns, information):
            predicted = history[:, -1, None].expand(-1, len(lead_hours), -1)
            return predicted, torch.ones_like(predicted)
    model.predictor = GaussianLatent()
    with pytest.raises(ValueError, match='Latent variance'):
        model(batch['history'], batch['information'], batch['origin_time_ns'], torch.tensor([6.]))
