"""Aligned geopotential history and fixed-terrain conditioning for E_d v2."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from test_pinn_training import pinn_prepared
from test_transition_v2_training import transition_args, _rewrite_checkpoint
from test_guided_training import _inputs
from climate_manifold.downstream.dynamics_guide import DynamicsGuide
from climate_manifold.downstream.dynamics_information import select_dynamics_information
from climate_manifold.downstream.dynamics_objective import future_dynamics_targets
from climate_manifold.downstream.pipeline import PredictorConfig
from climate_manifold.downstream.train import initialize_manifold, load_predictor, train, windows


@pytest.mark.parametrize('kind', ['conv_gru', 'temporal_conv'])
def test_geopotential_and_fixed_terrain_condition_encoder_without_terrain_derivative(kind):
    torch.manual_seed(21)
    encoder = DynamicsGuide((4, 2, 3), (2, 2, 3), 3, hidden=4, kind=kind,
                            history_dt_hours=48., information_mode='geopotential_terrain')
    raw = torch.randn(2, 3, 24)
    info = torch.randn(2, 3, 3, 2, 3)
    info[:, :, 2] = info[:, :1, 2]
    info = info.flatten(2)
    stem_inputs = []
    hook = encoder.stem.register_forward_pre_hook(lambda module, args: stem_inputs.append(args[0].detach().clone()))
    original = encoder(raw, info)
    hook.remove()
    assert encoder.information_fields == ('z500', 'z850', 'terrain_height')
    assert encoder.stem_channels == 14  # raw4 + d(raw)4 + dt1 + z2 + d(z)2 + terrain1
    actual = stem_inputs[0].reshape(2, 3, 14, 2, 3)
    fields = info.reshape(2, 3, 3, 2, 3)
    torch.testing.assert_close(actual[:, :, 9:11], fields[:, :, :2])
    torch.testing.assert_close(actual[:, 1:, 11:13], (fields[:, 1:, :2]-fields[:, :-1, :2])/2.)
    assert not actual[:, 0, 11:13].count_nonzero()
    torch.testing.assert_close(actual[:, :, 13], fields[:, :, 2])
    changed = info.clone().reshape(2, 3, 3, 2, 3)
    changed[:, -1, :2] += 3.
    future_changed = encoder(raw, changed.flatten(2))
    torch.testing.assert_close(future_changed[:, :-1], original[:, :-1], rtol=0, atol=0)
    assert not torch.equal(future_changed[:, -1], original[:, -1])
    changed = fields.clone(); changed[:, :, 2] += 3.
    assert not torch.equal(encoder(raw, changed.flatten(2)), original)
    changed[:, -1, 2] += 1.
    with pytest.raises(ValueError, match='fixed'):
        encoder(raw, changed.flatten(2))
    with pytest.raises(ValueError, match='actual observed'):
        encoder(raw)


def test_observed_information_windows_are_actual_strided_times_and_shard_compatible(pinn_prepared, tmp_path):
    _, _, _, archive = pinn_prepared
    args = transition_args(archive, tmp_path)
    args.dynamics_information = None
    args.history_steps, args.history_stride = 3, 2
    manifold, _, data = initialize_manifold(args)
    materialized = data['information']

    class ContiguousOnly:
        def __getitem__(self, key):
            assert isinstance(key, (int, slice))
            if isinstance(key, slice):
                assert key.step in (None, 1), 'Shards only support contiguous reads'
            return materialized[key]

    data['information'] = ContiguousOnly()
    dataset = windows(data, manifold.config, 'train', max_windows=1,
                      dynamics_information='geopotential_terrain', dynamics_target_sequence=True)
    row = dataset[0]
    start = dataset.starts[0]
    expected = select_dynamics_information(materialized[start:start+5][::2],
                    data['information_metadata'], 'geopotential_terrain', raw_grid=manifold.config.grid)
    np.testing.assert_array_equal(row['dynamics_information_history'], expected)
    assert not torch.equal(row['dynamics_information_history'][0], row['dynamics_information_history'][-1])
    torch.testing.assert_close(row['dynamics_information_target_sequence'][:5:2],
                               row['dynamics_information_history'], rtol=0, atol=0)
    assert row['dynamics_information_targets'].shape[0] == manifold.config.horizon_steps


@pytest.mark.parametrize('bridge,supervision', [('latent', 'vector_only'), ('guided', 'field_latent')])
def test_v2_default_information_joint_training_teacher_alignment_and_bank_free_reload(
        pinn_prepared, tmp_path, bridge, supervision):
    _, _, _, archive = pinn_prepared
    interaction = 'transformer' if bridge == 'guided' else 'off'
    args = transition_args(archive, tmp_path, bridge, interaction, supervision)
    args.dynamics_information = None  # CLI v2 default enables geopotential/terrain.
    checkpoint = train(args)
    model, payload = load_predictor(checkpoint)
    assert model.config.dynamics_information == 'geopotential_terrain'
    assert payload['dynamics_information_contract']['fields'] == ['z500', 'z850', 'terrain_height']
    assert payload['dynamics_transition_config']['information_mode'] == 'geopotential_terrain'
    assert payload['dynamics_contract']['future_information_input'] is False
    metrics = json.loads(checkpoint.with_suffix('.metrics.json').read_text())[0]
    assert np.isfinite(metrics['train']['transition_regularization'])
    assert not any(name.startswith('transition_') for name in metrics['selection'])
    _, _, data = initialize_manifold(args)
    batch = next(iter(DataLoader(windows(data, model.a_config, 'train', max_windows=2,
        dynamics_information='geopotential_terrain', dynamics_target_sequence=True), batch_size=2)))
    info = batch['dynamics_information_history']
    with torch.no_grad():
        output = model(*_inputs(batch), dynamics_information_history=info)
        teacher = future_dynamics_targets(model, batch, _inputs(batch)[-1])
        altered = {key: value.clone() for key, value in batch.items()}
        cells = model.a_config.grid[1]*model.a_config.grid[2]
        origin = model.a_config.history_span_steps-1
        altered['dynamics_information_target_sequence'][:, origin+1:, :2*cells] += 2.
        altered['dynamics_information_targets'][:, :, :2*cells] += 2.
        changed_teacher = future_dynamics_targets(model, altered, _inputs(batch)[-1])
        assert not torch.equal(teacher, changed_teacher)
        # These future labels change the auxiliary teacher but never the main forecast.
        torch.testing.assert_close(model(*_inputs(altered), dynamics_information_history=info)['mean'],
                                   output['mean'], rtol=0, atol=0)
        altered['dynamics_information_targets'].fill_(float('nan'))
        altered['dynamics_information_target_sequence'].fill_(float('nan'))
        torch.testing.assert_close(model(*_inputs(altered), dynamics_information_history=info)['mean'],
                                   output['mean'], rtol=0, atol=0)
    with pytest.raises(ValueError, match='actual observed'):
        model(*_inputs(batch))
    invalid = deepcopy(payload)
    invalid['dynamics_information_contract']['fields'][0] = 'z250'
    _rewrite_checkpoint(checkpoint, invalid)
    with pytest.raises(ValueError, match='dynamics_information_contract'):
        load_predictor(checkpoint)


def test_information_requires_v2_and_actual_information_archive(pinn_prepared, tmp_path):
    with pytest.raises(ValueError, match='enabled v2'):
        PredictorConfig(dynamics_information='geopotential_terrain')
    _, _, _, archive = pinn_prepared
    args = transition_args(archive, tmp_path)
    args.dynamics_information = None
    args.information = None
    args.constraint_pair = args.constraint_decoder = args.statistical_loss = None
    with pytest.raises(ValueError, match='information metadata'):
        train(args)
