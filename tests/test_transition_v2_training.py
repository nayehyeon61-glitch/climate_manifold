"""Training/checkpoint gates for observed-transition E_d supervision (v2)."""
from copy import deepcopy
from dataclasses import replace
import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_pinn_training import pinn_prepared
from test_dual_dynamics_training import dual_args, _pipeline
from test_guided_training import _batch, _inputs
from test_spatial_training import _capture, _assert_finite_gradient
from climate_manifold.downstream.dynamics_objective import dynamics_contract
from climate_manifold.downstream.pipeline import ForecastPipeline, PredictorConfig
from climate_manifold.downstream.train import initialize_manifold, load_predictor, train, windows
from climate_manifold.downstream.transition_measure import make_transition_config
from climate_manifold.physical_information import digest


def transition_args(archive, root, bridge='latent', interaction='off', supervision='field'):
    # The synthetic 6h data contract retains a 120h auxiliary target span.
    # Space windows apart so overlap exclusion has genuine other histories.
    extras = ['--dynamics-version', 'v2', '--dynamics-information', 'off',
              '--max-windows', '5', '--window-stride', '16',
              '--transition-neighbors', '3', '--transition-bank-size', '8',
              '--transition-exclusion-steps', '0', '--transition-pool-lat', '2',
              '--transition-pool-lon', '2']
    if supervision != 'field':
        extras += ['--dynamics-auxiliary-architecture', 'latent_rollout',
                   '--dynamics-supervision', supervision]
    if interaction != 'off':
        extras += ['--guide-interaction', interaction, '--guide-interaction-depth', '1']
    return dual_args(archive, root, bridge, 'conv_gru', *extras)


@pytest.mark.parametrize('bridge,interaction,supervision', [
    ('latent', 'off', 'field'),
    ('guided', 'off', 'field_latent'),
    ('guided', 'transformer', 'vector_only'),
])
def test_v2_joint_training_retrieves_only_training_windows_and_loads_without_bank(
        pinn_prepared, tmp_path, monkeypatch, bridge, interaction, supervision):
    _, _, _, archive = pinn_prepared
    args = transition_args(archive, tmp_path, bridge, interaction, supervision)
    args.statistical_loss = {'field': 'w2', 'field_latent': 'kl_entropy',
                             'vector_only': 'signed_measure'}[supervision]
    module = importlib.import_module('climate_manifold.downstream.train')
    captured = _capture(monkeypatch)
    actual_loss, actual_bank = module.transition_measure_losses, module.TransitionBank
    calls, bank_starts = [], []

    def loss_spy(model, prediction, batch, bank, cfg):
        assert model.training, 'Calibration/selection must not retrieve training analogues'
        calls.append(batch['origin_time_ns'].clone())
        return actual_loss(model, prediction, batch, bank, cfg)

    def bank_spy(data, a_config, starts, cfg):
        expected = windows(data, a_config, 'train', args.window_stride, args.max_windows).starts
        assert list(starts) == expected
        bank_starts.extend(starts)
        return actual_bank(data, a_config, starts, cfg)

    monkeypatch.setattr(module, 'transition_measure_losses', loss_spy)
    monkeypatch.setattr(module, 'TransitionBank', bank_spy)
    checkpoint = train(args)
    trained, _ = captured[0]
    _assert_finite_gradient(trained.dynamics_encoder)
    assert len(calls) == 3 and len(bank_starts) == 5
    metrics = json.loads(checkpoint.with_suffix('.metrics.json').read_text())[0]
    assert all(np.isfinite(value) for value in metrics['train'].values())
    assert metrics['train']['transition_regularization'] >= 0
    assert metrics['train']['transition_neighbors'] >= 2
    assert metrics['train']['transition_effective_neighbors'] >= 1
    assert not any(key.startswith('transition_') for key in metrics['selection'])

    def forbidden(*args, **kwargs):
        raise AssertionError('Checkpoint loading/inference must not instantiate or query a bank')

    monkeypatch.setattr(module, 'TransitionBank', forbidden)
    monkeypatch.setattr(module, 'transition_measure_losses', forbidden)
    restored, payload = load_predictor(checkpoint)
    assert payload['config']['dynamics_version'] == 'v2'
    assert payload['statistical_loss'] == args.statistical_loss
    assert payload['dynamics_contract']['encoder_training_version'] == 'observed_transition_measure_v2'
    assert payload['dynamics_contract']['transition_inference_used'] is False
    assert payload['dynamics_transition_bank']['train_only'] is True
    assert payload['dynamics_transition_bank']['starts'] == bank_starts
    assert payload['dynamics_transition_config']['forecast_steps'] == args.horizon_steps
    assert payload['dynamics_transition_bank']['future_steps'] == args.horizon_steps
    assert payload['best_selection_state_mse'] == metrics['selection']['state_mse']
    assert not any('transition_bank' in key for key in payload['model'])
    assert 'no inverse PDE solver' in payload['objective_semantics']['dynamics_transition']
    _, _, data = initialize_manifold(args)
    batch = _batch(data, restored.a_config)
    trained.eval()
    with torch.no_grad():
        expected, actual = trained(*_inputs(batch)), restored(*_inputs(batch))
        torch.testing.assert_close(actual['mean'], expected['mean'], rtol=0, atol=0)
        # Only observation history is forecast input; changing labels does not
        # influence v2 forecasts, even though those labels trained E_d.
        for key in ('targets', 'constraint_states', 'constraint_information'):
            batch[key].fill_(float('nan'))
        torch.testing.assert_close(restored(*_inputs(batch))['mean'], actual['mean'], rtol=0, atol=0)


def test_v2_training_version_does_not_change_model_parameters_or_initial_forecast(pinn_prepared, tmp_path):
    original, batch, _ = _pipeline(pinn_prepared, tmp_path, 'guided')
    config = replace(original.config, dynamics_version='v2')
    torch.manual_seed(31)
    reference = ForecastPipeline(deepcopy(original.bridge.manifold), original.config,
                                 schema=pinn_prepared[2]['schema'],
                                 separate_reconstruction_decoder=True)
    torch.manual_seed(31)
    alternative = ForecastPipeline(deepcopy(original.bridge.manifold), config,
                                   schema=pinn_prepared[2]['schema'],
                                   separate_reconstruction_decoder=True)
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(value, alternative.state_dict()[key], rtol=0, atol=0)
    assert sum(p.numel() for p in alternative.parameters()) == sum(p.numel() for p in reference.parameters())
    reference.eval(); alternative.eval()
    with torch.no_grad():
        for key in ('mean', 'dynamics_history_latent'):
            torch.testing.assert_close(reference(*_inputs(batch))[key], alternative(*_inputs(batch))[key],
                                       rtol=0, atol=0)
    old = dynamics_contract(original.config)
    assert 'encoder_training_version' not in old
    assert old['version'] == 'climate_manifold.dual_dynamics.v1'


def _rewrite_checkpoint(path, payload):
    torch.save(payload, path)
    manifest_path = path.with_suffix('.manifest.json')
    manifest = json.loads(manifest_path.read_text())
    manifest['checkpoint_sha256'] = digest(path)
    manifest_path.write_text(json.dumps(manifest))


def test_v2_missing_or_invalid_bank_provenance_is_rejected(pinn_prepared, tmp_path):
    _, _, _, archive = pinn_prepared
    checkpoint = train(transition_args(archive, tmp_path))
    _, payload = load_predictor(checkpoint)
    variants = []
    missing = deepcopy(payload); missing.pop('dynamics_transition_config'); variants.append(missing)
    missing = deepcopy(payload); missing.pop('dynamics_transition_bank'); variants.append(missing)
    held_out = deepcopy(payload); held_out['dynamics_transition_bank']['train_only'] = False; variants.append(held_out)
    bad_version = deepcopy(payload); bad_version['config']['dynamics_version'] = 'v1'; variants.append(bad_version)
    for invalid in variants:
        _rewrite_checkpoint(checkpoint, invalid)
        with pytest.raises(ValueError):
            load_predictor(checkpoint)


def test_historical_v1_without_version_field_still_loads(pinn_prepared, tmp_path):
    _, _, _, archive = pinn_prepared
    args = dual_args(archive, tmp_path, 'latent')
    checkpoint = train(args)
    original, payload = load_predictor(checkpoint)
    assert 'dynamics_transition_config' not in payload and 'dynamics_transition_bank' not in payload
    payload['config'].pop('dynamics_version')
    _rewrite_checkpoint(checkpoint, payload)
    restored, historical = load_predictor(checkpoint)
    assert restored.config.dynamics_version == 'v1'
    assert historical['dynamics_contract'] == dynamics_contract(original.config)


def test_v2_requires_existing_joint_dynamics_contract_and_v1_rejects_transition_options():
    with pytest.raises(ValueError, match='requires an enabled dynamics encoder'):
        PredictorConfig(dynamics_version='v2')
    with pytest.raises(ValueError, match='Dynamics requires joint spatial'):
        PredictorConfig(dynamics_version='v2', dynamics_encoder='conv_gru')
    with pytest.raises(ValueError, match='version'):
        PredictorConfig(dynamics_version='v3')
    with pytest.raises(ValueError):
        make_transition_config(SimpleNamespace(transition_weight=.1), PredictorConfig())
