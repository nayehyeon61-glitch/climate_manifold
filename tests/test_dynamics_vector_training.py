"""Vector-only means only E_d/F_d latent targets in the dynamics auxiliary loss."""
import json
from types import SimpleNamespace

import pytest
import torch

from test_pinn_training import pinn_prepared
from test_dual_dynamics_training import dual_args
from test_spatial_training import _capture, _assert_finite_gradient
from test_dynamics_latent_core import _toy, _predictor_config, _config
from climate_manifold.downstream.dynamics_objective import (
    configure_dynamics_supervision, dynamics_auxiliary_config_from_args,
    dynamics_auxiliary_losses, validate_dynamics_auxiliary_config,
)
from climate_manifold.downstream.train import load_predictor, train
from climate_manifold.physical_information import digest


def vector_config(**extra):
    return dynamics_auxiliary_config_from_args(
        SimpleNamespace(dynamics_supervision='vector_only', **extra), _predictor_config())


def test_vector_defaults_leave_no_field_objective_and_keep_positive_latent_weight():
    config = vector_config()
    assert config['latent_weight'] == .1
    assert all(config[key] == 0 for key in (
        'reconstruction_weight', 'statistical_weight', 'tendency_weight'))
    for mode in ('field', 'field_latent'):
        old = dynamics_auxiliary_config_from_args(
            SimpleNamespace(dynamics_supervision=mode,
                            dynamics_reconstruction_weight=None,
                            dynamics_statistical_weight=None,
                            dynamics_tendency_weight=None), _predictor_config())
        assert old['reconstruction_weight'] == old['statistical_weight'] == .1
        assert old['tendency_weight'] == 0


@pytest.mark.parametrize('name', ['reconstruction_weight', 'statistical_weight', 'tendency_weight'])
def test_vector_rejects_explicit_field_supervision(name):
    with pytest.raises(ValueError, match='vector_only requires zero'):
        vector_config(**{'dynamics_'+name: .1})
    with pytest.raises(ValueError, match='vector_only requires zero'):
        validate_dynamics_auxiliary_config({**vector_config(), name: .1})


@pytest.mark.parametrize('weight', [0., -1., float('nan'), float('inf')])
def test_vector_requires_finite_positive_latent_weight(weight):
    with pytest.raises(ValueError, match='weight'):
        vector_config(dynamics_latent_weight=weight)


def test_vector_gradient_reaches_E_and_F_but_never_calls_or_updates_D():
    pipeline, batch, leads = _toy(stride=3)
    config = vector_config(dynamics_latent_weight=.25)
    contract = configure_dynamics_supervision(pipeline, config)
    assert not contract['trained'] and contract['trainable_parameters'] == 0
    assert contract['total_parameters'] > 0
    assert not any(p.requires_grad for p in pipeline.dynamics_auxiliary_head.decoder.parameters())
    def forbidden(*args):
        raise AssertionError('Vector-only must not invoke D_d')
    hook = pipeline.dynamics_auxiliary_head.decoder.register_forward_pre_hook(forbidden)
    batch['targets'].requires_grad_(True)
    batch['dynamics_target_sequence'].requires_grad_(True)
    batch.pop('origin')  # Field/tendency supervision data is unnecessary.
    prediction = {'dynamics_history_latent': pipeline.dynamics_encoder(batch['history'])}
    result = dynamics_auxiliary_losses(pipeline, prediction, batch, leads, config)
    for name in ('dynamics_reconstruction', 'dynamics_statistical', 'dynamics_tendency'):
        assert result[name] == 0
    torch.testing.assert_close(result['dynamics_regularization'], .25*result['dynamics_latent'])
    result['dynamics_regularization'].backward()
    _assert_finite_gradient(pipeline.dynamics_encoder)
    _assert_finite_gradient(pipeline.dynamics_auxiliary_head.transition)
    assert all(p.grad is None for p in pipeline.dynamics_auxiliary_head.decoder.parameters())
    assert batch['targets'].grad is batch['dynamics_target_sequence'].grad is None
    hook.remove()


def test_no_decode_rollout_preserves_identical_latent_transition():
    pipeline, batch, leads = _toy()
    history = pipeline.dynamics_encoder(batch['history'])
    full = pipeline.dynamics_auxiliary_head.rollout(history, leads)
    vector = pipeline.dynamics_auxiliary_head.rollout(history, leads, decode_fields=False)
    assert vector['fields'] is None
    torch.testing.assert_close(full['latent'], vector['latent'], rtol=0, atol=0)


@pytest.mark.parametrize('bridge', ['latent', 'guided'])
def test_joint_vector_training_preserves_M_and_Es_losses_freezes_D_and_reloads(
        pinn_prepared, tmp_path, monkeypatch, bridge):
    _, _, _, archive = pinn_prepared
    captured = _capture(monkeypatch)
    initial_states, payloads = [], []
    for mode in ('field_latent', 'vector_only'):
        args = dual_args(archive, tmp_path, bridge, 'conv_gru',
                         '--dynamics-auxiliary-architecture', 'latent_rollout',
                         '--dynamics-supervision', mode, '--history-stride', '2',
                         '--max-windows', '4')
        args.output = str(tmp_path / f'{bridge}-{mode}.pt')
        checkpoint = train(args)
        trained, initial = captured[-1]
        initial_states.append(initial)
        restored, payload = load_predictor(checkpoint)
        payloads.append(payload)
        for module in (trained.dynamics_encoder, trained.dynamics_auxiliary_head.transition,
                       trained.predictor, trained.bridge.manifold.core.manifold.encoder):
            _assert_finite_gradient(module)
        metrics = json.loads(checkpoint.with_suffix('.metrics.json').read_text())[0]
        assert metrics['selection']['state_mse'] == payload['best_selection_state_mse']
        assert metrics['train']['dynamics_latent'] > 0
        # The main forecast and observed E_s objectives still train as before.
        assert metrics['train']['state_mse'] > 0
        assert metrics['train']['regularization'] > 0
        decoder = trained.dynamics_auxiliary_head.decoder
        if mode == 'vector_only':
            for name in ('dynamics_reconstruction', 'dynamics_statistical', 'dynamics_tendency'):
                assert metrics['train'][name] == 0
            assert not any(p.requires_grad for p in decoder.parameters())
            assert not any(p.requires_grad for p in restored.dynamics_auxiliary_head.decoder.parameters())
            assert all(p.grad is None for p in decoder.parameters())
            for name, value in initial.items():
                if name.startswith('dynamics_auxiliary_head.decoder.'):
                    torch.testing.assert_close(value, trained.state_dict()[name], rtol=0, atol=0)
        else:
            _assert_finite_gradient(decoder)
    for key in initial_states[0]:
        torch.testing.assert_close(initial_states[0][key], initial_states[1][key], rtol=0, atol=0)
    hybrid, vector = payloads
    assert hybrid['config'] == vector['config']
    assert hybrid['total_parameters'] == vector['total_parameters']
    assert hybrid['forecast_parameters'] == vector['forecast_parameters']
    frozen = vector['dynamics_decoder_contract']['total_parameters']
    for key in ('constraint_parameters', 'trainable_parameters'):
        assert hybrid[key] - vector[key] == frozen
    assert hybrid['dynamics_decoder_contract']['trained'] is True
    assert vector['dynamics_decoder_contract']['trained'] is False
    assert vector['dynamics_decoder_contract']['trainable_parameters'] == 0
    # Loading cannot silently reinterpret an untrained D_d as a field decoder.
    vector.pop('dynamics_decoder_contract')
    torch.save(vector, checkpoint)
    manifest = checkpoint.with_suffix('.manifest.json')
    meta = json.loads(manifest.read_text())
    meta['checkpoint_sha256'] = digest(checkpoint)
    manifest.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='disabled dynamics_decoder_contract'):
        load_predictor(checkpoint)
