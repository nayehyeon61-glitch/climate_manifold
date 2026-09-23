"""Train/save/evaluate contracts for spatial, jointly learned E--F--D models."""
import importlib
import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from test_pinn_training import pinn_prepared
from test_joint_training import _sealed_a_checkpoint
from climate_manifold.downstream.compare import compare
from climate_manifold.downstream.evaluate import evaluate
from climate_manifold.downstream.train import parser, train, initialize_manifold, load_predictor, windows


def _args(archive, output, *extra):
    # Leave --latent-layout absent: a fresh run must choose a spatial encoder.
    return parser().parse_args([
        '--archive', str(archive), '--output', str(output),
        '--epochs', '1', '--batch-size', '2', '--max-windows', '2',
        '--window-stride', '1', '--horizon-steps', '2', '--history-stride', '1',
        '--hidden-dim', '8', '--latent-channels', '3', '--spatial-downsample', '2',
        '--spatial-hidden-dim', '8', '--climode-step-hours', '6', '--device', 'cpu',
        *extra,
    ])


def _capture(monkeypatch):
    module = importlib.import_module('climate_manifold.downstream.train')
    constructor = module.ForecastPipeline
    captured = []
    def factory(*args, **kwargs):
        model = constructor(*args, **kwargs)
        captured.append((model, {key: value.clone() for key, value in model.state_dict().items()}))
        return model
    monkeypatch.setattr(module, 'ForecastPipeline', factory)
    return captured


def _assert_finite_gradient(module):
    gradients = [p.grad for p in module.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum() for g in gradients) > 0


@pytest.mark.parametrize('family', ['neural_ode', 'climode'])
def test_spatial_joint_train_checkpoint_evaluation_and_matched_controls(
        pinn_prepared, tmp_path, monkeypatch, family):
    _, _, _, archive = pinn_prepared
    captured = _capture(monkeypatch)
    reports, initial_representations, metadata = [], [], []
    for regularization in ('none', 'full'):
        output = tmp_path / f'{family}-{regularization}.pt'
        args = _args(archive, output, '--model', family, '--regularization', regularization,
                     '--reconstruction-weight', '0', '--tendency-weight', '0')
        assert args.a_checkpoint is None and args.training_mode == 'joint'
        count = len(captured)
        checkpoint = train(args)
        trained, initial = captured[count]
        manifold = trained.bridge.manifold
        assert manifold.config.representation_kind == 'spatial'
        assert manifold.config.latent_grid == (3, 2, 4)
        assert manifold.config.manifold_dim == 24
        for prefix, component in (
                ('bridge.manifold.core.manifold.encoder.', manifold.core.manifold.encoder),
                ('bridge.manifold.core.manifold.decoder.', manifold.core.manifold.decoder),
                ('predictor.', trained.predictor)):
            _assert_finite_gradient(component)
            assert any(not torch.equal(value, trained.state_dict()[key])
                       for key, value in initial.items() if key.startswith(prefix)), prefix
        # The none branch has no reconstruction, tendency, or physical loss;
        # future-field prediction alone must update all three components.
        initial_representations.append({key: value for key, value in initial.items()
                                        if key.startswith('bridge.manifold.')})
        restored, payload = load_predictor(checkpoint)
        metadata.append(payload)
        assert payload['latent_shape'] == [3, 2, 4]
        assert payload['constants'] is None and payload['constants_sha256'] is None
        assert payload['a_sha256'] is None and not payload['a_frozen']
        assert payload['experiment']['suite'] == 'primary'
        assert payload['experiment']['path'] == 'encoder -> predictor -> decoder'
        assert payload['representation_training'] == 'jointly_trained'
        assert not bool(restored.bridge.manifold.core.manifold_ready)
        _, _, data = initialize_manifold(args)
        batch = next(iter(DataLoader(windows(data, restored.a_config, 'validation', max_windows=2),
                                     batch_size=2)))
        inputs = (batch['history'], batch.get('information'), batch['origin_time_ns'],
                  torch.tensor([6., 12.]))
        trained.eval()
        with torch.no_grad():
            expected, actual = trained(*inputs), restored(*inputs)
            assert expected['std'] is None and actual['std'] is None
            for key in ('mean', 'predicted_latent', 'origin_latent'):
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                assert torch.isfinite(actual[key]).all()
        report_path = tmp_path / f'{family}-{regularization}.evaluation.json'
        forecast_path = tmp_path / f'{family}-{regularization}.npz'
        report = evaluate(checkpoint, archive, report_path, max_cases=2, forecast_output=forecast_path)
        assert report['finite_forecast_fraction'] == 1.
        assert not report['failed_origins'] and not report['latent_diagnostic_failures']
        assert report['latent_shape'] == [3, 2, 4]
        assert report['representation_config']['representation_kind'] == 'spatial'
        for scores in report['scores']['climode']['per_variable'].values():
            assert scores['aggregate']['crps'] is None
            assert scores['aggregate']['crps_valid_cases'] == 0
            assert np.isfinite(scores['aggregate']['rmse'])
        with np.load(forecast_path) as forecast:
            assert 'std' not in forecast
            np.testing.assert_array_equal(forecast['latent_shape'], [3, 2, 4])
            assert forecast['predicted_latent_spatial'].shape == (2, 3, 2, 4)
            np.testing.assert_array_equal(forecast['predicted_latent_spatial'].reshape(2, -1),
                                          forecast['predicted_latent'])
        reports.append(report_path)
    # Both ablations really start from equal E/D weights, not only equal widths.
    for key, value in initial_representations[0].items():
        torch.testing.assert_close(value, initial_representations[1][key], rtol=0, atol=0)
    assert metadata[0]['a_metadata']['config'] == metadata[1]['a_metadata']['config']
    assert metadata[0]['trainable_parameters'] == metadata[1]['trainable_parameters']
    result = compare(reports, tmp_path / f'{family}-comparison.json')
    assert result['experiment_suite'] == 'primary' and result['ranking_allowed']
    assert len(result['paired_effects']) == 1
    assert result['paired_effects'][0]['control'] == 'forecast_only'


def test_spatial_climode_pinn_reaches_forecast_pressure_and_closure(
        pinn_prepared, tmp_path, monkeypatch):
    _, _, _, archive = pinn_prepared
    args = _args(archive, tmp_path / 'spatial-pinn.pt', '--model', 'climode',
                 '--information', str(tmp_path / 'pinn-information.npz'),
                 '--pinn', '--pinn-weight', '.1', '--distribution-weight', '.1')
    captured = _capture(monkeypatch)
    checkpoint = train(args)
    trained, _ = captured[0]
    manifold = trained.bridge.manifold
    for component in (manifold.core.manifold.encoder, manifold.core.manifold.decoder,
                      manifold.information, manifold.info_head, manifold.pinn, trained.predictor):
        _assert_finite_gradient(component)
    restored, payload = load_predictor(checkpoint)
    assert restored.bridge.manifold.pinn is not None
    assert payload['objective_weights']['pinn'] == .1
    assert payload['objective_weights']['distribution'] == .1
    assert payload['a_metadata']['mode'] == 'enriched'
    metrics = json.loads(checkpoint.with_suffix('.metrics.json').read_text())[0]
    for key in ('information_future', 'information_spatial_quantile', 'pinn_total'):
        assert np.isfinite(metrics['train'][key]) and metrics['train'][key] > 0, key
    assert metrics['selection']['state_mse'] == payload['best_selection_state_mse']


def test_global_pretrained_weights_cannot_silently_become_spatial(pinn_prepared, tmp_path):
    original, _, data, archive = pinn_prepared
    information = tmp_path / 'pinn-information.npz'
    checkpoint = _sealed_a_checkpoint(original, data, archive, information, tmp_path / 'global-a.pt')
    args = _args(archive, tmp_path / 'unused.pt', '--a-checkpoint', str(checkpoint),
                 '--information', str(information), '--initialization', 'pretrained',
                 '--latent-layout', 'spatial')
    with pytest.raises(ValueError, match='Pretrained global A weights cannot initialize a spatial encoder'):
        initialize_manifold(args)
    # The same reference can supply data splits/statistics for a fresh spatial
    # model, provided no global weights are presented as transferred weights.
    args.initialization = 'fresh'
    spatial, _, fresh_data = initialize_manifold(args)
    assert spatial.config.representation_kind == 'spatial'
    assert spatial.config.latent_grid == (3, 2, 4)
    assert fresh_data['split'] == data['split']
    assert not bool(spatial.core.manifold_ready)
