"""E_d v2 with every pressure-level information field plus both terrain conditions."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from test_pinn_training import pinn_prepared
from test_transition_v2_training import transition_args
from test_guided_training import _inputs
from test_dynamics_consistency_runner import fits
from test_full_manifold_matrix_runner import ARMS, setup_full
from test_fusion_runner import arg, run_script
from climate_manifold.downstream.dynamics_guide import DynamicsGuide
from climate_manifold.downstream.dynamics_information import (
    dynamics_information_counts, dynamics_information_fields, resolve_dynamics_information)
from climate_manifold.downstream.evaluate import evaluate
from climate_manifold.downstream.train import initialize_manifold, load_predictor, parser, train, windows

PRESSURE = ['z850', 'z500', 'z250', 'u850', 'v850', 't850', 't500', 'terrain_height', 'terrain_slope']


def metadata():
    specs = [('t2m', 'K', 'dynamic'), ('z250', 'm', 'dynamic'), ('terrain_slope', '1', 'static'),
             ('z850', 'm', 'dynamic'), ('u850', 'm/s', 'dynamic'), ('terrain_height', 'm', 'static'),
             ('v850', 'm/s', 'dynamic'), ('t500', 'K', 'dynamic'), ('z500', 'm', 'dynamic'),
             ('t850', 'K', 'dynamic')]
    return {'shape': [len(specs), 2, 3], 'grid': {'lat': [-30., 30.], 'lon': [0., 120., 240.]},
            'variables': [{'name': n, 'unit': u, 'kind': k} for n, u, k in specs]}


def test_contract_orders_dynamic_then_static_fields_with_units():
    contract = resolve_dynamics_information(metadata(), 'pressure_terrain', (4, 2, 3))
    assert contract['fields'] == PRESSURE == dynamics_information_fields('pressure_terrain')
    assert contract['indices'] == [3, 8, 1, 4, 6, 9, 7, 5, 2]
    assert contract['grid'] == [9, 2, 3]
    assert contract['dynamic_indices'] == list(range(7)) and contract['static_indices'] == [7, 8]
    assert contract['units'] == ['m', 'm', 'm', 'm/s', 'm/s', 'K', 'K', 'm', '1']
    assert dynamics_information_counts('pressure_terrain') == (7, 2)
    # The original v2 mode keeps its exact contract.
    legacy = resolve_dynamics_information(metadata(), 'geopotential_terrain', (4, 2, 3))
    assert legacy['fields'] == ['z500', 'z850', 'terrain_height'] and legacy['grid'] == [3, 2, 3]
    assert legacy['dynamic_indices'] == [0, 1] and legacy['static_indices'] == [2]


@pytest.mark.parametrize('name,unit', [('u850', 'm'), ('t500', 'm'), ('terrain_slope', 'm')])
def test_contract_rejects_wrong_units(name, unit):
    meta = metadata()
    next(v for v in meta['variables'] if v['name'] == name)['unit'] = unit
    with pytest.raises(ValueError, match=name):
        resolve_dynamics_information(meta, 'pressure_terrain', (4, 2, 3))


def test_contract_rejects_missing_field():
    meta = metadata()
    meta['variables'][1]['name'] = 'q850'
    with pytest.raises(ValueError, match='z250'):
        resolve_dynamics_information(meta, 'pressure_terrain', (4, 2, 3))


@pytest.mark.parametrize('kind', ['conv_gru', 'temporal_conv'])
def test_encoder_uses_every_pressure_field_and_static_terrain(kind):
    torch.manual_seed(5)
    encoder = DynamicsGuide((4, 2, 3), (2, 2, 3), 3, hidden=4, kind=kind,
                            history_dt_hours=48., information_mode='pressure_terrain')
    assert encoder.information_fields == tuple(PRESSURE)
    assert encoder.stem_channels == 4+4+1+7+7+2
    raw = torch.randn(2, 3, 24)
    info = torch.randn(2, 3, 9, 2, 3)
    info[:, :, 7:] = info[:, :1, 7:]
    stem_inputs = []
    hook = encoder.stem.register_forward_pre_hook(lambda m, a: stem_inputs.append(a[0].detach().clone()))
    original = encoder(raw, info.flatten(2))
    hook.remove()
    actual = stem_inputs[0].reshape(2, 3, 25, 2, 3)
    torch.testing.assert_close(actual[:, :, 9:16], info[:, :, :7])
    torch.testing.assert_close(actual[:, 1:, 16:23], (info[:, 1:, :7]-info[:, :-1, :7])/2.)
    torch.testing.assert_close(actual[:, :, 23:], info[:, :, 7:])
    for channel in range(9):  # Each field reaches E_d.
        changed = info.clone()
        changed[:, :, channel] += 1.
        assert not torch.equal(encoder(raw, changed.flatten(2)), original)
    changed = info.clone()
    changed[:, -1, 8] += 1.
    with pytest.raises(ValueError, match='fixed'):
        encoder(raw, changed.flatten(2))


def test_pressure_v2_trains_with_transition_bank_and_evaluates(pinn_prepared, tmp_path):
    _, _, _, archive = pinn_prepared
    args = transition_args(archive, tmp_path, 'guided', 'off', 'field_latent')
    args.dynamics_information = 'pressure_terrain'
    checkpoint = train(args)
    model, payload = load_predictor(checkpoint)
    assert model.config.dynamics_information == 'pressure_terrain'
    assert payload['dynamics_information_contract']['fields'] == PRESSURE
    assert payload['dynamics_transition_config']['information_fields'] == PRESSURE
    assert payload['dynamics_contract']['information_dynamic_fields'] == PRESSURE[:7]
    assert payload['dynamics_contract']['future_information_input'] is False
    metrics = json.loads(checkpoint.with_suffix('.metrics.json').read_text())[0]
    assert np.isfinite(metrics['train']['transition_regularization'])
    _, _, data = initialize_manifold(args)
    batch = next(iter(DataLoader(windows(data, model.a_config, 'train', max_windows=2,
        dynamics_information='pressure_terrain', dynamics_target_sequence=True), batch_size=2)))
    info = batch['dynamics_information_history']
    assert info.shape[-1] == 9*model.a_config.grid[1]*model.a_config.grid[2]
    with torch.no_grad():
        output = model(*_inputs(batch), dynamics_information_history=info)
        assert torch.isfinite(output['mean']).all()
    report = evaluate(checkpoint, archive, tmp_path/'pressure.validation.json',
                      information=tmp_path/'pinn-information.npz', max_cases=2)
    assert report['finite_forecast_fraction'] == 1. and not report['failed_origins']
    assert report['dynamics_information_contract']['fields'] == PRESSURE


def test_runner_pressure_information_and_dynamics_arms_only(tmp_path):
    env, log = setup_full(tmp_path)
    result, calls = run_script(env, log, DYNAMICS_VERSION='v2', DYNAMICS_INFORMATION='pressure_terrain',
                               DYNAMICS_ARMS_ONLY='1')
    assert result.returncode == 0, result.stderr
    commands = fits(calls)
    assert len(commands) == len(ARMS)-3 and 'Fits: 9' in result.stdout
    for command in commands:
        config = parser().parse_args(command[2:])
        assert config.dynamics_encoder == 'conv_gru' and config.dynamics_version == 'v2'
        assert arg(command, '--dynamics-information') == 'pressure_terrain'
    manifest = json.loads((Path(env['RUN'])/'run-manifest.json').read_text())
    assert manifest['settings']['dynamics_arms_only'] == '1'
    assert manifest['settings']['dynamics_information'] == 'pressure_terrain'


def test_runner_without_arms_only_keeps_baselines_and_settings(tmp_path):
    env, log = setup_full(tmp_path)
    result, calls = run_script(env, log, DYNAMICS_VERSION='v2')
    assert result.returncode == 0, result.stderr
    assert len(fits(calls)) == len(ARMS)
    manifest = json.loads((Path(env['RUN'])/'run-manifest.json').read_text())
    assert 'dynamics_arms_only' not in manifest['settings']
