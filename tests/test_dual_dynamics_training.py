"""Causal and joint-training gates for the optional, independent E_d branch.

Tiny synthetic runs check the experimental wiring, not forecasting skill.
"""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from test_pinn_training import pinn_prepared
from test_spatial_training import _args, _assert_finite_gradient, _capture
from test_guided_training import _batch, _inputs
from climate_manifold.downstream.evaluate import evaluate
from climate_manifold.downstream.dynamics_objective import dynamics_auxiliary_losses
from climate_manifold.downstream.pipeline import ForecastPipeline, PredictorConfig
from climate_manifold.downstream.statistical_objective import make_statistical_config
from climate_manifold.downstream.train import initialize_manifold, load_predictor, train
from climate_manifold.physical_information import digest


def dual_args(archive, root, bridge='guided', encoder='conv_gru', *extra):
    route = (['--guide-architecture', 'fusion', '--guide-fusion-depth', '1',
              '--guide-direct-information'] if bridge == 'guided' else [])
    return _args(
        archive, root/f'{bridge}-{encoder}.pt', '--model', 'mlp', '--bridge', bridge,
        '--raw-backend', 'matched', '--information', str(root/'pinn-information.npz'),
        '--constraint-pair', 'statistical', '--constraint-decoder',
        'separate_surface_and_information', '--statistical-loss', 'signed_measure',
        '--history-steps', '2', '--weather-depth', '1', '--transformer-heads', '2',
        '--tendency-weight', '0', '--dynamics-encoder', encoder,
        '--dynamics-hidden-dim', '4', '--dynamics-channels', '2', *route, *extra)


def _pipeline(prepared, root, bridge='guided', *, encoder='conv_gru', mode='learned', family='mlp'):
    _, _, _, archive = prepared
    args = dual_args(archive, root, bridge, encoder)
    manifold, _, data = initialize_manifold(args)
    config = PredictorConfig(
        model=family, bridge=bridge, training_mode='joint', latent_layout='spatial',
        raw_backend='matched', hidden_dim=8, weather_depth=1, transformer_heads=2,
        climode_step_hours=6.,
        guide_architecture='fusion' if bridge == 'guided' else 'joint',
        guide_fusion_depth=1, guide_direct_information=bridge == 'guided',
        dynamics_encoder=encoder, dynamics_mode=mode, dynamics_hidden_dim=4,
        dynamics_channels=2)
    pipeline = ForecastPipeline(manifold, config, schema=data['schema'],
                                separate_reconstruction_decoder=True)
    return pipeline, _batch(data, manifold.config), args


def _has_no_gradient(module):
    return all(p.grad is None or not bool(torch.count_nonzero(p.grad))
               for p in module.parameters())


def _objective_config(kind='signed_measure', *, reconstruction=.1, statistics=.1, tendency=0.):
    return {'version': 'climate_manifold.dynamics_auxiliary.v1',
            'statistical_loss_config': make_statistical_config(kind),
            'reconstruction_weight': reconstruction, 'statistical_weight': statistics,
            'tendency_weight': tendency}


@pytest.mark.parametrize('bridge', ['latent', 'guided'])
@pytest.mark.parametrize('encoder', ['conv_gru', 'temporal_conv'])
def test_dual_joint_train_reload_evaluate_and_future_label_isolation(
        pinn_prepared, tmp_path, monkeypatch, bridge, encoder):
    _, _, _, archive = pinn_prepared
    args = dual_args(archive, tmp_path, bridge, encoder)
    captured = _capture(monkeypatch)
    checkpoint = train(args)
    trained, initial = captured[0]
    _assert_finite_gradient(trained.dynamics_encoder)
    _assert_finite_gradient(trained.dynamics_auxiliary_head)
    assert any(not torch.equal(value, trained.state_dict()[name])
               for name, value in initial.items() if name.startswith('dynamics_encoder.'))
    # Independent encoders; each is shared by its forecast and auxiliary routes.
    spatial_parameters = {id(p) for p in trained.bridge.manifold.core.manifold.encoder.parameters()}
    assert not spatial_parameters.intersection(id(p) for p in trained.dynamics_encoder.parameters())
    restored, payload = load_predictor(checkpoint)
    assert payload['config']['dynamics_encoder'] == encoder
    assert payload['dynamics_contract'] and payload['dynamics_auxiliary_config']
    assert payload['forecast_parameters'] + payload['constraint_parameters'] == payload['trainable_parameters']
    metrics = json.loads(checkpoint.with_suffix('.metrics.json').read_text())[0]['train']
    assert np.isfinite(metrics['dynamics_regularization']) and metrics['dynamics_regularization'] > 0
    _, _, data = initialize_manifold(args)
    batch = _batch(data, restored.a_config)
    assert 'information_targets' not in batch
    trained.eval()
    with torch.no_grad():
        actual, expected = restored(*_inputs(batch)), trained(*_inputs(batch))
        for key in ('mean', 'history_latent', 'dynamics_history_latent'):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        assert actual['mean'].shape == batch['targets'][:, :2].shape
        assert all('auxiliary' not in key for key in actual)
        if bridge == 'latent':
            assert actual['forecast_history_latent'].shape == actual['history_latent'].shape
        else:
            assert actual['fused_history'].shape == batch['history'].shape
        # The standalone forecast accepts no future or reconstruction labels.
        for key in ('targets', 'constraint_states', 'constraint_information'):
            batch[key].fill_(float('nan'))
        torch.testing.assert_close(restored(*_inputs(batch))['mean'], actual['mean'], rtol=0, atol=0)
    report = evaluate(checkpoint, archive, tmp_path/'dual.validation.json',
                      information=tmp_path/'pinn-information.npz', max_cases=2)
    assert report['finite_forecast_fraction'] == 1. and not report['failed_origins']
    assert report['dynamics_contract'] == payload['dynamics_contract']
    assert report['dynamics_auxiliary_config'] == payload['dynamics_auxiliary_config']


@pytest.mark.parametrize('bridge', ['latent', 'guided'])
def test_off_checkpoint_without_new_config_fields_strictly_reloads(
        pinn_prepared, tmp_path, bridge):
    _, _, _, archive = pinn_prepared
    args = dual_args(archive, tmp_path, bridge, 'off')
    checkpoint = train(args)
    reference, payload = load_predictor(checkpoint)
    assert reference.dynamics_encoder is reference.dynamics_adapter is reference.dynamics_auxiliary_head is None
    assert not any(name.startswith('dynamics_') for name in reference.state_dict())
    if bridge == 'guided':
        assert not any('dynamics' in name for name in reference.guide_fusion.state_dict())
    historical = deepcopy(payload)
    for key in ('dynamics_encoder', 'dynamics_mode', 'dynamics_hidden_dim', 'dynamics_channels'):
        historical['config'].pop(key, None)
    historical.pop('dynamics_contract', None)
    historical.pop('dynamics_auxiliary_config', None)
    torch.save(historical, checkpoint)
    manifest_path = checkpoint.with_suffix('.manifest.json')
    manifest = json.loads(manifest_path.read_text())
    manifest['checkpoint_sha256'] = digest(checkpoint)
    manifest_path.write_text(json.dumps(manifest))
    restored, _ = load_predictor(checkpoint)
    _, _, data = initialize_manifold(args)
    batch = _batch(data, reference.a_config)
    with torch.no_grad():
        torch.testing.assert_close(restored(*_inputs(batch))['mean'],
                                   reference(*_inputs(batch))['mean'], rtol=0, atol=0)


@pytest.mark.parametrize('bridge', ['latent', 'guided'])
@pytest.mark.parametrize('mode', ['learned', 'zero'])
def test_forecast_dynamics_gradient_obeys_content_ablation(pinn_prepared, tmp_path, bridge, mode):
    pipeline, batch, _ = _pipeline(pinn_prepared, tmp_path, bridge, mode=mode)
    # Open the initially zero residual path. Otherwise the first step legitimately
    # has zero E_d forecast gradient even for the learned-content arm.
    projection = (pipeline.dynamics_adapter.projection if bridge == 'latent'
                  else pipeline.guide_fusion.head[-1])
    torch.nn.init.normal_(projection.weight, std=.05)
    prediction = pipeline(*_inputs(batch), reconstruct_origin=False)
    assert bool(torch.count_nonzero(prediction['dynamics_history_latent']))
    (prediction['mean'] - batch['targets'][:, :2]).square().mean().backward()
    _assert_finite_gradient(pipeline.predictor)
    assert _has_no_gradient(pipeline.dynamics_auxiliary_head)
    if mode == 'zero':
        assert _has_no_gradient(pipeline.dynamics_encoder)
    else:
        _assert_finite_gradient(pipeline.dynamics_encoder)


@pytest.mark.parametrize('loss', ['w2', 'kl_entropy', 'signed_measure'])
def test_future_statistics_train_shared_dynamics_even_with_zero_forecast_guide(
        pinn_prepared, tmp_path, loss):
    pipeline, batch, _ = _pipeline(pinn_prepared, tmp_path, mode='zero')
    prediction = pipeline(*_inputs(batch), reconstruct_origin=False)
    # Future labels remain detached, and observed auxiliary labels are unrelated
    # to this future-surface objective.
    batch['targets'].requires_grad_(True)
    batch['constraint_states'].fill_(float('nan'))
    batch['constraint_information'].fill_(float('nan'))
    values = dynamics_auxiliary_losses(
        pipeline, prediction, batch, _inputs(batch)[-1],
        _objective_config(loss, reconstruction=0., statistics=1.))
    assert values['dynamics_statistical'] > 0
    assert all(torch.isfinite(value).all() for value in values.values())
    values['dynamics_regularization'].backward()
    _assert_finite_gradient(pipeline.dynamics_encoder)
    _assert_finite_gradient(pipeline.dynamics_auxiliary_head)
    for module in (pipeline.predictor, pipeline.guide_fusion,
                   pipeline.bridge.manifold.core.manifold.encoder,
                   pipeline.reconstruction_decoder, pipeline.bridge.manifold.info_head):
        assert _has_no_gradient(module)
    assert batch['targets'].grad is None


def test_nonuniform_physical_leads_select_dense_labels_and_scale_tendency(
        pinn_prepared, tmp_path):
    pipeline, batch, _ = _pipeline(pinn_prepared, tmp_path, 'latent')
    batch_size, _, state_dim = batch['targets'].shape
    # Dense labels: value k at archive lead 6*k hours. Nonuniform requested
    # leads must select entries 0, 2, 5 rather than the first three entries.
    batch['targets'] = torch.arange(1., 7.)[None, :, None].expand(batch_size, 6, state_dim).clone()
    lead_hours = torch.tensor([6., 18., 36.])
    prediction = pipeline(batch['history'], batch['information'], batch['origin_time_ns'],
                          lead_hours, reconstruct_origin=False)

    class KnownLeadHead(torch.nn.Module):
        def __init__(self, error_per_hour):
            super().__init__()
            self.error_per_hour = error_per_hour

        def forward(self, history, leads):
            value = leads / pipeline.a_config.step_hours + self.error_per_hour * leads
            return value[None, :, None].expand(len(history), -1, state_dim)

    pipeline.dynamics_auxiliary_head = KnownLeadHead(0.)
    config = _objective_config(reconstruction=1., statistics=0., tendency=1.)
    exact = dynamics_auxiliary_losses(pipeline, prediction, batch, lead_hours, config)
    assert exact['dynamics_regularization'] == 0
    pipeline.dynamics_auxiliary_head = KnownLeadHead(.1)
    values = dynamics_auxiliary_losses(pipeline, prediction, batch, lead_hours, config)
    temporal = pipeline.bridge.manifold.temporal
    expected = ((.1 * temporal.scale / temporal.tendency_scale).square() * temporal.metric).sum()
    torch.testing.assert_close(values['dynamics_tendency'], expected)
    with pytest.raises(ValueError, match='aligned'):
        dynamics_auxiliary_losses(pipeline, prediction, batch, torch.tensor([7., 18.]), config)
    with pytest.raises(ValueError, match='cover'):
        dynamics_auxiliary_losses(pipeline, prediction, batch, torch.tensor([6., 42.]), config)


@pytest.mark.parametrize('bridge', ['latent', 'guided'])
@pytest.mark.parametrize('family', [
    'mlp', 'neural_ode', 'climode', 'convlstm', 'simvp', 'fourcastnet', 'climax', 'transformer'])
def test_all_forecast_families_accept_dynamics_and_backpropagate(
        pinn_prepared, tmp_path, bridge, family):
    pipeline, batch, _ = _pipeline(pinn_prepared, tmp_path, bridge, family=family)
    prediction = pipeline(*_inputs(batch), reconstruct_origin=False)
    assert prediction['mean'].shape == batch['targets'][:, :2].shape
    assert torch.isfinite(prediction['mean']).all()
    auxiliary = dynamics_auxiliary_losses(
        pipeline, prediction, batch, _inputs(batch)[-1], _objective_config())
    forecast_loss = (prediction['mean'] - batch['targets'][:, :2]).square().mean()
    (forecast_loss + auxiliary['dynamics_regularization']).backward()
    _assert_finite_gradient(pipeline.predictor)
    _assert_finite_gradient(pipeline.dynamics_encoder)
    _assert_finite_gradient(pipeline.dynamics_auxiliary_head)


@pytest.mark.parametrize('bridge', ['latent', 'guided'])
def test_optional_dynamics_preserves_common_initialization(pinn_prepared, tmp_path, bridge):
    old, batch, _ = _pipeline(pinn_prepared, tmp_path, bridge, encoder='off')
    dual, _, _ = _pipeline(pinn_prepared, tmp_path, bridge, encoder='conv_gru')
    for old_module, dual_module in (
            (old.predictor, dual.predictor),
            (old.bridge.manifold, dual.bridge.manifold),
            (old.reconstruction_decoder, dual.reconstruction_decoder)):
        old_state, dual_state = old_module.state_dict(), dual_module.state_dict()
        assert old_state.keys() == dual_state.keys()
        for name, value in old_state.items():
            torch.testing.assert_close(value, dual_state[name], rtol=0, atol=0)
    # Both integration routes start from the identical old forecast inputs.
    old.eval()
    dual.eval()
    with torch.no_grad():
        torch.testing.assert_close(old(*_inputs(batch))['mean'], dual(*_inputs(batch))['mean'],
                                   rtol=0, atol=0)
