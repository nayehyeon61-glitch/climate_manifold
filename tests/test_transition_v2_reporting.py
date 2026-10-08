"""Observed-transition training is explicit in identities and held-out audits."""
from copy import deepcopy
import hashlib
import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from test_dynamics_reporting import dynamics_report, raw_report
from test_dynamics_vector_reporting import vector_report
from test_raw_comparison import run_compare
from test_pinn_training import pinn_prepared
from test_dual_dynamics_training import dual_args
from climate_manifold.downstream.compare import _arm, _validate_weather_provenance
from climate_manifold.downstream.dynamics_objective import dynamics_contract
from climate_manifold.downstream.evaluate import evaluate
from climate_manifold.downstream.pipeline import PredictorConfig
from climate_manifold.downstream.plot_comparison import summarize_comparison
from climate_manifold.downstream.protocol import dynamics_version_identity, experiment_contract
from climate_manifold.downstream.train import train, load_predictor, guide_contract
from climate_manifold.downstream.transition_measure import (
    make_transition_config, validate_transition_bank_manifest)
from climate_manifold.downstream.dynamics_information import resolve_dynamics_information


def _hash(value):
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def _data_contract(row):
    row = deepcopy(row)
    row['representation_config']['horizon_steps'] = 2
    row['training_contract']['train_starts_sha256'] = _hash(list(range(60)))
    return row


def transition_report(row=None, information='off'):
    row = _data_contract(vector_report() if row is None else row)
    row['config']['dynamics_version'] = 'v2'
    row['config']['dynamics_information'] = information
    cfg = PredictorConfig(**row['config'])
    row['dynamics_contract'] = dynamics_contract(cfg)
    row['guide_contract'] = guide_contract(cfg)
    objective = make_transition_config(SimpleNamespace(horizon_steps=2), cfg)
    row['dynamics_transition_config'] = objective
    a = row['representation_config']
    span = (a['history_steps']-1)*a['history_stride']+1
    prefix = 59+span+a['horizon_steps']
    starts = list(range(0, 60, 6))
    row['dynamics_transition_bank'] = {
        'version': 'climate_manifold.transition_bank.v1', 'train_only': True,
        'count': len(starts), 'starts': starts, 'starts_sha256': _hash(starts),
        'training_starts_count': 60, 'training_starts_sha256': _hash(list(range(60))),
        'train_prefix_end': prefix, 'history_span_steps': span, 'horizon_steps': a['horizon_steps'],
        'future_steps': objective['forecast_steps'],
        'information_mode': information, 'information_fields': objective['information_fields'],
        'train_information_sha256': None if information == 'off' else 'c'*64,
        'step_hours': a['step_hours'], 'exclusion_steps': objective['exclusion_steps'],
        'train_first_time_ns': 0, 'train_last_time_ns': (prefix-1)*a['step_hours']*3600*10**9,
        'effective_pool_lat': 4, 'effective_pool_lon': 8,
        'train_states_sha256': 'a'*64, 'normalization_sha256': 'b'*64,
        'transition_config_sha256': hashlib.sha256(json.dumps(objective, sort_keys=True).encode()).hexdigest(),
    }
    row['a_metadata'] = {'config': deepcopy(a), 'split': {'train': list(range(60))}}
    if information != 'off':
        height, width = a['grid'][-2:]
        metadata = {
            'shape': [3, height, width],
            'variables': [{'name': name, 'unit': 'm', 'kind': kind} for name, kind in (
                ('z500', 'dynamic'), ('z850', 'dynamic'), ('terrain_height', 'static'))],
            'grid': {'lat': list(range(height)), 'lon': list(range(width))},
        }
        row['a_metadata']['information_metadata'] = metadata
        row['dynamics_information_contract'] = resolve_dynamics_information(metadata, information, a['grid'])
    row['implementation'] = 'observed_transition_measure_v2+'+row['implementation']
    return row


def test_v1_identity_and_reports_remain_unannotated(tmp_path):
    old = vector_report('field')
    explicit = deepcopy(old)
    explicit['config']['dynamics_version'] = 'v1'
    assert _arm(old) == _arm(explicit)
    assert dynamics_version_identity(old) == 'v1'
    assert 'dynamics_version' not in experiment_contract(old['config'])
    result = run_compare(tmp_path, [old])
    assert ':dynamics_version=' not in result['rows'][0]['candidate_arm']
    assert not any('Observed-transition' in row['label'] for row in summarize_comparison(result))


@pytest.mark.parametrize('information', ['off', 'geopotential_terrain'])
def test_v2_provenance_keeps_raw_routes_field_vector_and_t1_effects(tmp_path, information):
    reports = []
    for seed in (7, 19):
        reports.append(_data_contract(raw_report(seed)))
        for bridge in ('latent', 'guided'):
            reports.append(_data_contract(dynamics_report(bridge, seed=seed, encoder='off')))
            for supervision in ('field', 'field_latent', 'vector_only'):
                reports.append(transition_report(vector_report(supervision, bridge=bridge, seed=seed), information))
                if bridge == 'guided':
                    reports.append(transition_report(vector_report(
                        supervision, bridge=bridge, seed=seed, interaction='transformer'), information))
    result = run_compare(tmp_path, reports)
    assert len(result['seed_summary']) == 12
    assert all(group['seeds'] == [7, 19] for group in result['seed_summary'].values())
    assert result['route_effects'] and result['guide_interaction_effects']
    assert result['dynamics_consistency_effects'] and result['dynamics_vector_effects']
    dynamic = [row for row in result['rows'] if row['dynamics_encoder'] != 'off']
    assert all(':dynamics_version=v2' in row['candidate_arm'] for row in dynamic)
    assert all(row['dynamics_transition_bank']['train_only'] for row in dynamic)
    assert {_arm(row) for row in result['rows']} == {
        _arm(row) for row in result['climode_benchmark']['rows']}
    labels = summarize_comparison(result)
    assert any('Observed-transition v2' in row['label'] for row in labels)
    assert all(row['valid_seeds'] == 2 for row in labels)
    if information != 'off':
        assert all(':dynamics_information=geopotential_terrain' in row['candidate_arm'] for row in dynamic)
        assert any('Z500/Z850/terrain history' in row['label'] for row in labels)


def test_v2_information_ablation_requires_separate_suites(tmp_path):
    first = transition_report(vector_report(seed=7), 'off')
    second = transition_report(vector_report(seed=19), 'geopotential_terrain')
    with pytest.raises(ValueError, match='mixed dynamics_information'):
        run_compare(tmp_path, [first, second])


def test_v2_rejects_tampered_information_channel_contract(tmp_path):
    row = transition_report(information='geopotential_terrain')
    row['dynamics_information_contract']['indices'] = [1, 0, 2]
    with pytest.raises(ValueError, match='dynamics_information_contract'):
        run_compare(tmp_path, [row])


@pytest.mark.parametrize('mutation', ['mixed_version', 'config', 'bank', 'missing_bank',
                                    'not_train_only', 'training_hash', 'split', 'time_contract'])
def test_v2_mismatched_or_unverifiable_provenance_is_rejected(tmp_path, mutation):
    first = transition_report(vector_report(seed=7))
    second = transition_report(vector_report(seed=19))
    if mutation == 'mixed_version':
        second = _data_contract(vector_report(seed=19))
    elif mutation == 'config':
        second['dynamics_transition_config']['weight'] = .2
        second['dynamics_transition_bank']['transition_config_sha256'] = hashlib.sha256(
            json.dumps(second['dynamics_transition_config'], sort_keys=True).encode()).hexdigest()
    elif mutation == 'bank':
        bank = second['dynamics_transition_bank']
        bank['starts'] = [value+1 for value in bank['starts']]
        bank['starts_sha256'] = _hash(bank['starts'])
        validate_transition_bank_manifest(bank, second['dynamics_transition_config'])
    elif mutation == 'missing_bank':
        second.pop('dynamics_transition_bank')
    elif mutation == 'not_train_only':
        second['dynamics_transition_bank']['train_only'] = False
    elif mutation == 'training_hash':
        second['training_contract']['train_starts_sha256'] = 'f'*64
    elif mutation == 'split':
        second.pop('a_metadata')
    else:
        second['a_metadata']['config']['history_stride'] = 1
    with pytest.raises(ValueError, match='[Tt]ransition|dynamics_version|dynamics_transition'):
        run_compare(tmp_path, [first, second])


def test_v1_rejects_smuggled_transition_metadata(tmp_path):
    row = transition_report()
    row['config']['dynamics_version'] = 'v1'
    with pytest.raises(ValueError, match='v1 checkpoint'):
        run_compare(tmp_path, [row])


@pytest.mark.parametrize('family', ['fourcastnet', 'climax'])
@pytest.mark.parametrize('interaction', ['off', 'transformer'])
def test_v2_weather_core_identity_preserves_optional_wrappers(family, interaction):
    row = transition_report(vector_report('field', interaction=interaction))
    row['config']['model'] = family
    core = family+'_pinned_core'
    row['implementation'] = 'observed_transition_measure_v2+' + (
        'two_stage_guide_v1+' if interaction != 'off' else '') + (
        'dynamics_rollout_v1+dual_dynamics_v1+guide_fusion_v1+'+core)
    row['predictor_provenance'] = dict(
        family=family, implementation=core, upstream_source='https://example.test/official',
        upstream_commit='a'*40, implementation_variant='adapted', pretrained=False,
        depth=row['config']['weather_depth'], patch_size=row['config']['weather_patch_size'],
        hidden_dim=row['config']['hidden_dim'])
    _validate_weather_provenance([row])
    row['implementation'] = row['implementation'].removeprefix('observed_transition_measure_v2+')
    with pytest.raises(ValueError, match='observed_transition_measure_v2'):
        _validate_weather_provenance([row])


@pytest.mark.parametrize('bridge', ['latent', 'guided'])
def test_train_evaluate_exports_v2_audit_without_retrieving_on_heldout(
        pinn_prepared, tmp_path, monkeypatch, bridge):
    archive = pinn_prepared[-1]
    args = dual_args(archive, tmp_path, bridge, 'conv_gru',
                     '--dynamics-version', 'v2', '--max-windows', '0', '--window-stride', '30',
                     '--transition-neighbors', '2', '--transition-bank-size', '32')
    checkpoint = train(args)
    _, payload = load_predictor(checkpoint)
    module = importlib.import_module('climate_manifold.downstream.transition_measure')

    def forbidden(*args, **kwargs):
        raise AssertionError('Held-out evaluation attempted transition retrieval')

    monkeypatch.setattr(module.TransitionBank, '__init__', forbidden)
    monkeypatch.setattr(module.TransitionBank, 'retrieve', forbidden)
    forecast = tmp_path/'v2-forecast.npz'
    result = evaluate(checkpoint, archive, tmp_path/'v2-evaluation.json',
                      information=tmp_path/'pinn-information.npz', max_cases=2,
                      forecast_output=forecast)
    assert result['finite_forecast_fraction'] == 1.
    for key in ('dynamics_transition_config', 'dynamics_transition_bank'):
        assert result[key] == payload[key]
    assert result['config']['dynamics_information'] == 'geopotential_terrain'
    assert set(result['a_metadata']) == {'config', 'split', 'information_metadata'}
    assert set(result['a_metadata']['split']) == {'train'}
    with np.load(forecast, allow_pickle=False) as saved:
        for key in ('dynamics_transition_config', 'dynamics_transition_bank'):
            assert json.loads(str(saved[key+'_json'])) == result[key]
        assert json.loads(str(saved['dynamics_transition_training_data_json'])) == json.loads(
            json.dumps(result['a_metadata']))
        assert saved['dynamics_information_history'].shape == (2, 3*4*8)
        assert str(saved['dynamics_information_history_space']) == 'checkpoint_train_normalized'
        assert str(saved['dynamics_information_mode']) == 'geopotential_terrain'
        assert json.loads(str(saved['dynamics_information_fields_json'])) == ['z500', 'z850', 'terrain_height']
        assert json.loads(str(saved['dynamics_information_contract_json'])) == result['dynamics_information_contract']
        assert np.isfinite(saved['mean']).all()
    comparison_dir = tmp_path/'compare'
    comparison_dir.mkdir()
    comparison = run_compare(comparison_dir, [result])
    assert comparison['rows'][0]['dynamics_version'] == 'v2'
    assert ':dynamics_version=v2' in comparison['rows'][0]['candidate_arm']


def test_future_geopotential_changes_only_teacher_audit_not_main_forecast(
        pinn_prepared, tmp_path, monkeypatch):
    archive = pinn_prepared[-1]
    args = dual_args(archive, tmp_path, 'guided', 'conv_gru',
                    '--dynamics-version', 'v2', '--max-windows', '0', '--window-stride', '30',
                    '--transition-neighbors', '2', '--transition-bank-size', '32',
                    '--dynamics-auxiliary-architecture', 'latent_rollout',
                    '--dynamics-supervision', 'field_latent')
    checkpoint = train(args)
    module = importlib.import_module('climate_manifold.downstream.evaluate')
    original = module.dynamics_evaluation_windows

    def altered_future_windows(*args, **kwargs):
        assert kwargs['dynamics_information'] == 'geopotential_terrain'
        dataset, metadata = original(*args, **kwargs)
        span = dataset.config.history_span_steps
        cells = np.prod(dataset.config.grid[-2:])

        class ChangedFuture:
            def __len__(self):
                return len(dataset)

            def __getitem__(self, index):
                sample = dataset[index]
                # Keep all observed information and the static terrain fixed;
                # only future geopotential labels used by the teacher change.
                for key in ('dynamics_information_targets', 'dynamics_information_target_sequence'):
                    sample[key] = sample[key].clone()
                sample['dynamics_information_targets'][:, :2*cells] += 5.
                sample['dynamics_information_target_sequence'][span:, :2*cells] += 5.
                return sample

        return ChangedFuture(), metadata

    reports = []
    for index in range(2):
        if index:
            monkeypatch.setattr(module, 'dynamics_evaluation_windows', altered_future_windows)
        reports.append(evaluate(checkpoint, archive, tmp_path/f'geopotential-{index}.json',
                                information=tmp_path/'pinn-information.npz', max_cases=2,
                                dynamics_diagnostics=True, dynamics_rollout_steps=4,
                                forecast_output=tmp_path/f'geopotential-{index}.npz'))
    assert reports[0]['scores'] == reports[1]['scores']
    assert reports[0]['main_forecast_signed_diagnostics'] == reports[1]['main_forecast_signed_diagnostics']
    with np.load(tmp_path/'geopotential-0.npz') as first, np.load(tmp_path/'geopotential-1.npz') as second:
        for key in ('mean', 'dynamics_history_latent', 'dynamics_information_history',
                    'auxiliary_dynamics_predicted_latent'):
            np.testing.assert_array_equal(first[key], second[key])
        assert not np.array_equal(first['auxiliary_dynamics_teacher_latent'],
                                  second['auxiliary_dynamics_teacher_latent'])
        assert first['mean'].shape[-1] == 4*4*8  # Four raw forecast variables, not seven.
