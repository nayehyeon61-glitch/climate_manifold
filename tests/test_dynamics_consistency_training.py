"""Joint training, causal labels and checkpoint gates for the paired experiment."""
import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from test_pinn_training import pinn_prepared
from test_dual_dynamics_training import dual_args
from test_spatial_training import _capture, _assert_finite_gradient
from climate_manifold.downstream.dynamics_objective import future_dynamics_targets
from climate_manifold.downstream.train import initialize_manifold, load_predictor, train, windows


@pytest.mark.parametrize('bridge', ['latent', 'guided'])
@pytest.mark.parametrize('encoder', ['conv_gru', 'temporal_conv'])
def test_field_and_hybrid_joint_train_from_identical_weights_and_roundtrip(
        pinn_prepared, tmp_path, monkeypatch, bridge, encoder):
    _, _, _, archive = pinn_prepared
    captured = _capture(monkeypatch)
    starts, payloads = [], []
    for mode in ('field', 'field_latent'):
        args = dual_args(archive, tmp_path, bridge, encoder,
                         '--dynamics-auxiliary-architecture', 'latent_rollout',
                         '--dynamics-supervision', mode, '--history-stride', '2',
                         '--max-windows', '4')
        args.output = str(tmp_path / f'{bridge}-{encoder}-{mode}.pt')
        checkpoint = train(args)
        trained, initial = captured[-1]
        starts.append(initial)
        for module in (trained.dynamics_encoder, trained.dynamics_auxiliary_head.transition,
                       trained.dynamics_auxiliary_head.decoder, trained.predictor,
                       trained.bridge.manifold.core.manifold.encoder):
            _assert_finite_gradient(module)
        restored, payload = load_predictor(checkpoint)
        payloads.append(payload)
        assert payload['implementation'].startswith('dynamics_rollout_v1+dual_dynamics_v1+')
        assert payload['dynamics_auxiliary_config']['supervision'] == mode
        assert payload['dynamics_auxiliary_config']['latent_weight'] == (.1 if mode == 'field_latent' else 0.)
        assert payload['forecast_parameters'] + payload['constraint_parameters'] == payload['trainable_parameters']
        metrics = json.loads(checkpoint.with_suffix('.metrics.json').read_text())[0]
        assert metrics['selection']['state_mse'] == payload['best_selection_state_mse']
        assert metrics['train']['dynamics_reconstruction'] > 0
        assert metrics['train']['dynamics_statistical'] > 0
        assert np.isfinite(metrics['train']['dynamics_regularization'])
        if mode == 'field_latent':
            assert metrics['train']['dynamics_latent'] > 0
            assert metrics['train']['dynamics_target_latent_variance'] > 0
        else:
            assert metrics['train']['dynamics_latent'] == 0
        _, _, data = initialize_manifold(args)
        dataset = windows(data, restored.a_config, 'validation', max_windows=2,
                          dynamics_target_sequence=True)
        batch = next(iter(DataLoader(dataset, batch_size=2)))
        leads = torch.tensor([6., 12.])
        inputs = (batch['history'], batch['information'], batch['origin_time_ns'], leads)
        trained.eval()
        with torch.no_grad():
            actual, expected = restored(*inputs), trained(*inputs)
            torch.testing.assert_close(actual['mean'], expected['mean'], rtol=0, atol=0)
            truth = future_dynamics_targets(restored, batch, leads)
            assert truth.shape[:2] == (2, 2)
            assert not truth.requires_grad
            future = restored.dynamics_auxiliary_head.rollout(actual['dynamics_history_latent'], leads)
            # Poisoning every future label cannot alter either inference path.
            batch['targets'].fill_(float('nan'))
            batch['dynamics_target_sequence'].fill_(float('nan'))
            repeated = restored(*inputs)
            repeated_future = restored.dynamics_auxiliary_head.rollout(
                repeated['dynamics_history_latent'], leads)
            torch.testing.assert_close(repeated['mean'], actual['mean'], rtol=0, atol=0)
            for key in ('latent', 'fields'):
                torch.testing.assert_close(repeated_future[key], future[key], rtol=0, atol=0)
    assert starts[0].keys() == starts[1].keys()
    for name in starts[0]:
        torch.testing.assert_close(starts[0][name], starts[1][name], rtol=0, atol=0)
    for key in ('config', 'training_contract', 'trainable_parameters', 'forecast_parameters',
                'constraint_parameters', 'dynamics_contract'):
        assert payloads[0][key] == payloads[1][key], key


def test_dense_teacher_span_is_opt_in_and_matches_strided_history(pinn_prepared, tmp_path):
    _, _, _, archive = pinn_prepared
    args = dual_args(archive, tmp_path, 'latent', 'conv_gru', '--history-stride', '3')
    manifold, _, data = initialize_manifold(args)
    plain = windows(data, manifold.config, 'train', max_windows=1)[0]
    dense = windows(data, manifold.config, 'train', max_windows=1,
                    dynamics_target_sequence=True)[0]
    assert 'dynamics_target_sequence' not in plain
    sequence = dense['dynamics_target_sequence']
    config = manifold.config
    assert sequence.shape[0] == config.history_span_steps + config.horizon_steps
    torch.testing.assert_close(sequence[:config.history_span_steps:config.history_stride],
                               dense['history'], rtol=0, atol=0)
    torch.testing.assert_close(sequence[config.history_span_steps:], dense['targets'], rtol=0, atol=0)
