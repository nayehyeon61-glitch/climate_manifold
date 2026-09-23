"""Raw forecast controls train and evaluate without constructing an encoder/decoder."""
import importlib

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from test_pinn_training import pinn_prepared
from climate_manifold.downstream.evaluate import evaluate
from climate_manifold.downstream.train import (
    initialize_manifold, load_predictor, parser, train, windows,
)


def _args(archive, information, output, family, *extra):
    return parser().parse_args([
        '--archive', str(archive), '--information', str(information),
        '--output', str(output), '--model', family, '--bridge', 'raw',
        '--regularization', 'none', '--epochs', '1', '--batch-size', '2',
        '--max-windows', '2', '--window-stride', '1', '--history-stride', '1',
        '--horizon-steps', '2', '--hidden-dim', '8', '--latent-channels', '3',
        '--spatial-downsample', '2', '--spatial-hidden-dim', '8',
        '--climode-step-hours', '6', '--device', 'cpu', *extra,
    ])


def _forbid_manifold(*args, **kwargs):
    raise AssertionError('Raw controls must not construct an unused manifold')


@pytest.mark.parametrize('family', ['neural_ode', 'climode'])
def test_raw_control_training_checkpoint_evaluation_without_encoder_decoder(
        pinn_prepared, tmp_path, monkeypatch, family):
    _, _, _, archive = pinn_prepared
    information = tmp_path / 'pinn-information.npz'
    args = _args(archive, information, tmp_path / f'{family}-raw.pt', family,
                 '--pinn', '--pinn-weight', '.1', '--distribution-weight', '.1')
    assert args.a_checkpoint is None and args.raw_backend is None
    module = importlib.import_module('climate_manifold.downstream.train')
    spatial_module = importlib.import_module('climate_manifold.spatial')
    pipeline_constructor = module.ForecastPipeline
    captured = []

    def capture(*constructor_args, **constructor_kwargs):
        result = pipeline_constructor(*constructor_args, **constructor_kwargs)
        captured.append((result, {k: v.clone() for k, v in result.state_dict().items()}))
        return result

    # This covers initialization, checkpoint restoration and evaluation. A raw
    # control should not allocate E/D temporarily and then discard their weights.
    with monkeypatch.context() as patch:
        patch.setattr(module, 'ClimateManifold', _forbid_manifold)
        patch.setattr(spatial_module, 'SpatialClimateManifold', _forbid_manifold)
        patch.setattr(module, 'ForecastPipeline', capture)
        checkpoint = train(args)
        trained, initial = captured[0]
        assert trained.bridge.manifold is None
        assert trained.config.raw_backend == 'matched'
        assert initial and all(key.startswith('predictor.') for key in initial)
        assert all(name.startswith('predictor.') for name, _ in trained.named_parameters())
        gradients = [p.grad for p in trained.predictor.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients) > 0
        assert any(not torch.equal(value, trained.state_dict()[key])
                   for key, value in initial.items())

        restored, payload = load_predictor(checkpoint)
        assert restored.bridge.manifold is None
        assert payload['a_sha256'] is None and payload['representation_sha256'] is None
        assert payload['representation_training'] == 'none'
        assert payload['latent_shape'] is None
        assert payload['constants'] is None and payload['constants_sha256'] is None
        assert all(value == 0 for value in payload['objective_weights'].values())
        assert payload['experiment']['suite'] == 'primary'
        assert payload['experiment']['path'] == 'observations -> field predictor'
        assert payload['conditioning']['direct_origin_information']
        assert not payload['conditioning']['manifold_origin_information']
        assert payload['conditioning']['observed_information_available']
        assert payload['forecast_state_grid'] == list(restored.a_config.grid)
        assert payload['trainable_parameters'] == payload['total_parameters']
        assert all(key.startswith('predictor.') for key in payload['model'])
        if family == 'climode':
            assert restored.predictor.spatial_factor == 1
            # Two original cells per one latent cell gives equal speed in the
            # source grid for the paired downsampled transport model.
            assert restored.predictor.max_speed == 4.
            assert payload['transport_contract']['velocity_bound_cells_per_day'] == 4.

        _, raw_metadata, data = initialize_manifold(args)
        batch = next(iter(DataLoader(windows(data, restored.a_config, 'validation', max_windows=2),
                                     batch_size=2)))
        leads = torch.tensor([6., 12.])
        inputs = (batch['history'], batch['information'], batch['origin_time_ns'], leads)
        trained.eval()
        with torch.no_grad():
            expected, actual = trained(*inputs), restored(*inputs)
            torch.testing.assert_close(actual['mean'], expected['mean'], rtol=0, atol=0)
            assert actual['mean'].shape == (2, 2, restored.a_config.state_dim)
            assert torch.isfinite(actual['mean']).all()
            assert actual['std'] is None
            for key in ('history_latent', 'predicted_latent', 'origin_latent'):
                assert actual[key] is None
            torch.testing.assert_close(actual['reconstructed_origin'], batch['origin'], rtol=0, atol=0)

        # Both pressure fields and terrain enter through causal origin context,
        # with a differentiable connection to the forecast itself.
        origin_information = batch['information'].clone().requires_grad_()
        field = restored(batch['history'], origin_information, batch['origin_time_ns'], leads)['mean']
        information_gradient, = torch.autograd.grad(field.square().mean(), origin_information)
        assert torch.isfinite(information_gradient).all()
        assert information_gradient.abs().sum() > 0

        report_path = tmp_path / f'{family}-raw.evaluation.json'
        forecast_path = tmp_path / f'{family}-raw.npz'
        report = evaluate(checkpoint, archive, report_path, information=information,
                          max_cases=2, forecast_output=forecast_path)
        assert report['finite_forecast_fraction'] == 1.
        assert not report['failed_origins']
        assert report['latent_diagnostics'] is None and report['latent_shape'] is None
        assert report['representation_sha256'] is None
        for scores in report['scores']['climode']['per_variable'].values():
            assert scores['aggregate']['crps'] is None
            assert scores['aggregate']['crps_valid_cases'] == 0
            assert np.isfinite(scores['aggregate']['rmse'])
        with np.load(forecast_path) as forecast:
            assert 'std' not in forecast
            assert not any('latent' in key for key in forecast.files)
            assert forecast['mean'].shape == forecast['truth'].shape

    # Encoding is the treatment; observations, source normalization and splits
    # must remain identical to those available to the latent treatment arm.
    args.bridge = 'latent'
    _, latent_metadata, latent_data = initialize_manifold(args)
    for key in ('archive_sha256', 'information_sha256', 'information_metadata', 'split'):
        assert raw_metadata[key] == latent_metadata[key], key
    for key in ('mean', 'scale', 'information_mean', 'information_scale'):
        np.testing.assert_array_equal(raw_metadata[key], latent_metadata[key])
    latent_batch = next(iter(DataLoader(windows(latent_data, restored.a_config, 'validation', max_windows=2),
                                       batch_size=2)))
    for key in ('history', 'information', 'origin_time_ns', 'targets'):
        torch.testing.assert_close(batch[key], latent_batch[key], rtol=0, atol=0)
