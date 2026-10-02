"""End-to-end gates for raw forecasts conditioned by constrained latent guides."""
import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from test_pinn_training import pinn_prepared
from test_spatial_training import _args, _capture, _assert_finite_gradient
from climate_manifold.downstream.evaluate import evaluate
from climate_manifold.downstream.pipeline import ForecastPipeline, PredictorConfig
from climate_manifold.downstream.reconstruction_objective import reconstruction_constraint_losses
from climate_manifold.downstream.statistical_objective import make_statistical_config
from climate_manifold.downstream.train import (
    initialize_manifold, load_predictor, objective_weights, train, windows,
)


def guided_args(archive, root, *extra):
    return _args(archive, root/'guided.pt', '--model', 'transformer', '--bridge', 'guided',
                 '--information', str(root/'pinn-information.npz'),
                 '--constraint-pair', 'statistical', '--statistical-loss', 'signed_measure',
                 '--spatial-variable-conditioning', '--history-steps', '2',
                 '--weather-depth', '1', '--weather-patch-size', '2',
                 '--transformer-heads', '2', '--tendency-weight', '0', *extra)


def _batch(data, config):
    return next(iter(DataLoader(windows(data, config, 'validation', max_windows=2,
                                        reconstruction_constraints=True), batch_size=2)))


def _inputs(batch):
    return (batch['history'], batch['information'], batch['origin_time_ns'],
            torch.tensor([6., 12.]))


def test_guided_rejects_misleading_no_information_conditioning():
    with pytest.raises(ValueError, match='disabling information conditioning is unsupported'):
        PredictorConfig(model='transformer', bridge='guided', training_mode='joint',
                        latent_layout='spatial', condition_information=False)


def _guided_pipeline(prepared, tmp_path, *, guide_mode='learned'):
    _, _, _, archive = prepared
    args = guided_args(archive, tmp_path)
    manifold, _, data = initialize_manifold(args)
    config = PredictorConfig(model='transformer', bridge='guided', training_mode='joint',
                             latent_layout='spatial', hidden_dim=8, weather_depth=1,
                             weather_patch_size=2, transformer_heads=2, guide_mode=guide_mode)
    pipeline = ForecastPipeline(manifold, config, schema=data['schema'],
                                separate_reconstruction_decoder=True)
    return pipeline, _batch(data, manifold.config), args


def test_guided_joint_training_checkpoint_and_heldout_evaluation(pinn_prepared, tmp_path, monkeypatch):
    _, _, _, archive = pinn_prepared
    args = guided_args(archive, tmp_path)
    captured = _capture(monkeypatch)
    checkpoint = train(args)
    model, initial = captured[0]
    manifold = model.bridge.manifold
    assert len(model.reconstruction_decoder.decoder.variable_decoders) == manifold.config.grid[0]
    for component in (manifold.core.manifold.encoder, manifold.information,
                      manifold.latent_fusion, manifold.info_head,
                      model.reconstruction_decoder, model.predictor):
        _assert_finite_gradient(component)
    decoder_prefix = 'bridge.manifold.core.manifold.decoder.'
    assert all(not p.requires_grad and p.grad is None
               for p in manifold.core.manifold.decoder.parameters())
    for name, value in initial.items():
        if name.startswith(decoder_prefix):
            torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
    assert manifold.pinn is None and model.conditional_flow is None

    restored, payload = load_predictor(checkpoint)
    assert payload['config']['bridge'] == 'guided'
    assert payload['config']['guide_mode'] == 'learned'
    assert payload['a_metadata']['config']['spatial_variable_conditioning'] is True
    assert payload['constraint_pair'] == 'statistical'
    assert payload['statistical_loss'] == 'signed_measure'
    assert payload['constraint_decoder'] == 'separate_surface_and_information'
    assert payload['objective_weights']['pinn'] == payload['objective_weights']['static'] == 0
    assert payload['forecast_parameters'] + payload['constraint_parameters'] == payload['trainable_parameters']
    metrics = json.loads(checkpoint.with_suffix('.metrics.json').read_text())[0]['train']
    assert np.isfinite(metrics['statistical_signed_measure']) and metrics['statistical_signed_measure'] > 0
    assert metrics['pinn_total'] == 0

    _, _, data = initialize_manifold(args)
    batch = _batch(data, restored.a_config)
    assert 'information_targets' not in batch and 'statistical_flow_information_targets' not in batch
    model.eval()
    with torch.no_grad():
        actual, expected = restored(*_inputs(batch)), model(*_inputs(batch))
        torch.testing.assert_close(actual['mean'], expected['mean'], rtol=0, atol=0)
        torch.testing.assert_close(actual['history_latent'], expected['history_latent'], rtol=0, atol=0)
        assert actual['predicted_latent'] is None and actual['std'] is None
        # Auxiliary labels are not arguments to standalone forecasting.
        for key in ('targets', 'constraint_states', 'constraint_information'):
            batch[key].fill_(float('nan'))
        torch.testing.assert_close(restored(*_inputs(batch))['mean'], actual['mean'], rtol=0, atol=0)

    forecast_path = tmp_path/'guided.forecast.npz'
    report = evaluate(checkpoint, archive, tmp_path/'guided.evaluation.json',
                      information=tmp_path/'pinn-information.npz', max_cases=2,
                      forecast_output=forecast_path)
    assert report['finite_forecast_fraction'] == 1.
    assert not report['failed_origins'] and report['latent_diagnostics'] is None
    assert report['guide_contract'] == payload['guide_contract']
    with np.load(forecast_path) as values:
        assert 'guide_history_latent' in values and 'predicted_latent' not in values
        assert np.isfinite(values['mean']).all()


def test_guided_forecast_and_information_objectives_have_separate_gradients(pinn_prepared, tmp_path):
    pipe, batch, args = _guided_pipeline(pinn_prepared, tmp_path)
    manifold = pipe.bridge.manifold
    prediction = pipe(*_inputs(batch), reconstruct_origin=False)
    (prediction['mean'] - batch['targets'][:, :2]).square().mean().backward()
    for module in (manifold.core.manifold.encoder, manifold.information,
                   manifold.latent_fusion, pipe.predictor):
        _assert_finite_gradient(module)
    for module in (manifold.info_head, pipe.reconstruction_decoder,
                   manifold.core.manifold.decoder):
        assert all(p.grad is None for p in module.parameters())

    pipe.zero_grad(set_to_none=True)
    # Future targets are deliberately invalid: the information objective must
    # use only the observed, co-located pair to constrain the shared encoder.
    batch['targets'].fill_(float('nan'))
    losses = reconstruction_constraint_losses(
        pipe, batch, objective_weights(args, manifold), 'statistical',
        statistical_config=make_statistical_config('signed_measure'))
    losses['regularization'].backward()
    for module in (manifold.core.manifold.encoder, manifold.information,
                   manifold.latent_fusion, manifold.info_head, pipe.reconstruction_decoder):
        _assert_finite_gradient(module)
    for module in (pipe.predictor, manifold.core.manifold.decoder):
        assert all(p.grad is None for p in module.parameters())


def test_zero_guide_ablation_blocks_forecast_gradient_to_encoder(pinn_prepared, tmp_path):
    pipe, batch, _ = _guided_pipeline(pinn_prepared, tmp_path, guide_mode='zero')
    prediction = pipe(*_inputs(batch), reconstruct_origin=False)
    (prediction['mean'] - batch['targets'][:, :2]).square().mean().backward()
    _assert_finite_gradient(pipe.predictor)
    for module in (pipe.bridge.manifold.core.manifold.encoder,
                   pipe.bridge.manifold.information, pipe.bridge.manifold.latent_fusion):
        assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in module.parameters())


@pytest.mark.parametrize('bridge', ['raw', 'latent'])
def test_transformer_raw_and_latent_controls_train_and_reload(pinn_prepared, tmp_path, bridge):
    _, _, _, archive = pinn_prepared
    args = _args(archive, tmp_path/f'{bridge}.pt', '--model', 'transformer', '--bridge', bridge,
                 '--information', str(tmp_path/'pinn-information.npz'), '--regularization', 'none',
                 '--history-steps', '2', '--weather-depth', '1', '--weather-patch-size', '2',
                 '--transformer-heads', '2', '--reconstruction-weight', '0', '--tendency-weight', '0')
    checkpoint = train(args)
    model, payload = load_predictor(checkpoint)
    assert payload['config']['bridge'] == bridge
    _, _, data = initialize_manifold(args)
    batch = _batch(data, model.a_config)
    with torch.no_grad():
        output = model(*_inputs(batch))
    assert output['mean'].shape == batch['targets'][:, :2].shape
    assert torch.isfinite(output['mean']).all()
    assert (output['predicted_latent'] is not None) == (bridge == 'latent')


@pytest.mark.parametrize('guide_mode', ['learned', 'zero'])
def test_guided_direct_information_matches_raw_access(pinn_prepared, tmp_path, guide_mode):
    _, _, _, archive = pinn_prepared
    args = guided_args(archive, tmp_path, '--guide-mode', guide_mode, '--guide-direct-information')
    checkpoint = train(args)
    model, payload = load_predictor(checkpoint)
    assert payload['config']['guide_direct_information'] is True
    assert payload['conditioning']['direct_origin_information'] is True
    assert payload['guide_contract']['direct_origin_information'] is True
    assert model.predictor.information_channels > 0
    _, _, data = initialize_manifold(args)
    batch = _batch(data, model.a_config)
    model.eval()
    history, information, origin, leads = _inputs(batch)
    with torch.no_grad():
        guide = model.bridge.encode_history(history, information)
        base = model.predictor(history, leads, origin, information, guide_history=guide)[0]
        moved = model.predictor(history, leads, origin, information + 1., guide_history=guide)[0]
        torch.testing.assert_close(model(*_inputs(batch))['mean'], base, rtol=0, atol=0)
    # Information reaches the forecast directly, not only through the encoder.
    assert not torch.equal(base, moved)


def test_guided_direct_information_defaults_off(pinn_prepared, tmp_path):
    pipe, _, _ = _guided_pipeline(pinn_prepared, tmp_path)
    assert pipe.config.guide_direct_information is False
    assert pipe.predictor.information_channels == 0
    with pytest.raises(ValueError, match='Direct guide information requires bridge=guided'):
        PredictorConfig(model='transformer', bridge='latent', training_mode='joint',
                        latent_layout='spatial', guide_direct_information=True)
