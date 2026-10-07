"""Auxiliary latent consistency is isolated from architecture and field forecasting."""
from copy import deepcopy

import pytest

from test_dynamics_reporting import dynamics_report, raw_report
from test_raw_comparison import run_compare
from climate_manifold.downstream.compare import _arm, _validate_weather_provenance
from climate_manifold.downstream.dynamics_objective import dynamics_contract
from climate_manifold.downstream.pipeline import PredictorConfig
from climate_manifold.downstream.plot_comparison import summarize_comparison
from climate_manifold.downstream.protocol import (
    dynamics_identity, dynamics_supervision_identity, guide_interaction_contract, validate_experiment)
from climate_manifold.downstream.train import guide_contract


def rollout_report(bridge='guided', supervision='field', seed=7, weight=.1,
                   mode='learned', interaction='off'):
    row = dynamics_report(bridge, mode=mode, seed=seed)
    row['config'].update(dynamics_auxiliary_architecture='latent_rollout',
                         guide_interaction=interaction, guide_interaction_depth=1)
    cfg = PredictorConfig(**row['config'])
    row['guide_contract'] = guide_contract(cfg)
    row['dynamics_contract'] = dynamics_contract(cfg)
    row['guide_interaction_contract'] = guide_interaction_contract(cfg)
    row['dynamics_auxiliary_config'].update(
        version='climate_manifold.dynamics_auxiliary.v2', supervision=supervision,
        latent_weight=weight if supervision == 'field_latent' else 0.,
        target_encoder='online_stop_gradient', target_semantics='same_length_strided_truth_window')
    for name in ('constraint_parameters', 'total_parameters', 'trainable_parameters'):
        row[name] += 20  # Same F_d/D_d capacity for field and field+latent.
    row['implementation'] = 'dynamics_rollout_v1+dual_dynamics_v1+' + row['implementation']
    if interaction != 'off':
        row['implementation'] = 'two_stage_guide_v1+' + row['implementation']
        row['guide_interaction_evaluation'] = dict(
            requested='checkpoint', trained=interaction, effective=interaction, bypassed=False)
        for name in ('total_parameters', 'trainable_parameters', 'forecast_parameters'):
            row[name] += 20
    row['dynamics_evaluation_contract'] = dict(rollout_steps=4, step_hours=6,
                                              target_semantics='same_length_strided_truth_window')
    row['dynamics_auxiliary_evaluation'] = {'latent_mse': 10 if supervision == 'field' else 0.}
    return row


def test_old_missing_architecture_is_direct_and_encoder_identity_stays_compatible():
    direct = dynamics_report()
    assert dynamics_supervision_identity(direct) == ('direct', 'field', 0.)
    new = rollout_report(supervision='field_latent')
    assert dynamics_identity(new) == dynamics_identity(direct) == ('conv_gru', 'learned')
    assert dynamics_supervision_identity(new) == ('latent_rollout', 'field_latent', .1)
    assert ':auxiliary=' not in _arm(direct)
    assert _arm(new).endswith(':auxiliary=latent_rollout:supervision=field_latent:latent_weight=0.1')
    assert dynamics_supervision_identity(raw_report(7)) == ('off', None, None)
    with pytest.raises(ValueError, match='requires dynamics encoder'):
        validate_experiment({**new['config'], 'dynamics_encoder': 'off'}, 'primary')


def test_new_and_legacy_arms_are_never_pooled_and_physical_consistency_pairs_are_exported(tmp_path):
    reports = []
    for seed in (7, 19):
        reports.append(raw_report(seed))
        for bridge in ('latent', 'guided'):
            reports.extend([dynamics_report(bridge, seed=seed),
                            rollout_report(bridge, seed=seed),
                            rollout_report(bridge, 'field_latent', seed)])
    result = run_compare(tmp_path, reports)
    assert len(result['seed_summary']) == 7
    assert all(row['seeds'] == [7, 19] for row in result['seed_summary'].values())
    effects = result['dynamics_consistency_effects']
    assert len(effects) == 8  # Two routes x two seeds x two leads.
    assert all(e['baseline_parameters'] == e['candidate_parameters'] for e in effects)
    assert all(e['scored_prediction'] == 'downstream_M_physical_forecast' for e in effects)
    assert not result['dynamics_effects']
    assert (tmp_path/'comparison.dynamics-consistency-effects.csv').is_file()
    assert {_arm(r) for r in result['rows']} == {_arm(r) for r in result['climode_benchmark']['rows']}
    summary = summarize_comparison(result)
    assert any(r['metric'] == 'dynamics_consistency_gain_percent' for r in summary)
    assert all(r['valid_seeds'] == 2 for r in summary)
    assert any('Field + latent (0.1)' in r['label'] for r in summary)
    assert result['rows'][2]['dynamics_auxiliary_evaluation'] is not None


def test_consistency_metric_measures_downstream_fields_even_when_latent_error_improves(tmp_path):
    field, hybrid = rollout_report(), rollout_report(supervision='field_latent')
    for report, rmse in ((field, 2.), (hybrid, 3.)):
        for lead in report['scores']['climode']['per_variable']['t2m']['by_lead']:
            lead['rmse'] = rmse
    result = run_compare(tmp_path, [field, hybrid])
    assert hybrid['dynamics_auxiliary_evaluation']['latent_mse'] < field['dynamics_auxiliary_evaluation']['latent_mse']
    assert all(r['rmse_skill_vs_baseline'] == -.5 for r in result['dynamics_consistency_effects'])
    summary = summarize_comparison(result)
    assert all(r['mean'] == -50 for r in summary if r['metric'] == 'dynamics_consistency_gain_percent')


def test_latent_weights_define_separate_experiments(tmp_path):
    reports = [rollout_report(), rollout_report(supervision='field_latent'),
               rollout_report(supervision='field_latent', weight=.2)]
    result = run_compare(tmp_path, reports)
    assert len(result['seed_summary']) == 3
    assert {r['candidate_latent_weight'] for r in result['dynamics_consistency_effects']} == {.1, .2}
    assert all(r['valid_seeds'] == 1 for r in summarize_comparison(result))


@pytest.mark.parametrize('mutation', ['capacity', 'evaluation_contract'])
def test_mismatched_architecture_or_audit_scope_is_not_a_consistency_effect(tmp_path, mutation):
    field, hybrid = rollout_report(), rollout_report(supervision='field_latent')
    if mutation == 'capacity':
        hybrid['total_parameters'] += 1
    else:
        hybrid['dynamics_evaluation_contract']['rollout_steps'] = 8
    result = run_compare(tmp_path, [field, hybrid])
    assert not result['dynamics_consistency_effects']


@pytest.mark.parametrize('mutation,match', [
    ('v1_for_rollout', 'version.*architecture'), ('direct_for_v2', 'version.*architecture'),
    ('field_positive_weight', 'latent weight'), ('hybrid_zero_weight', 'latent weight'),
    ('target_semantics', 'target semantics'), ('field_weight', 'dynamics_auxiliary_config')])
def test_reject_mislabelled_objectives_and_unmatched_field_supervision(tmp_path, mutation, match):
    field, hybrid = rollout_report(), rollout_report(supervision='field_latent')
    if mutation == 'v1_for_rollout':
        hybrid['dynamics_auxiliary_config'] = dynamics_report()['dynamics_auxiliary_config']
    elif mutation == 'direct_for_v2':
        hybrid['config']['dynamics_auxiliary_architecture'] = 'direct'
        hybrid['dynamics_contract'] = dynamics_contract(PredictorConfig(**hybrid['config']))
        hybrid['guide_contract'] = guide_contract(PredictorConfig(**hybrid['config']))
    elif mutation == 'field_positive_weight':
        field['dynamics_auxiliary_config']['latent_weight'] = .1
    elif mutation == 'hybrid_zero_weight':
        hybrid['dynamics_auxiliary_config']['latent_weight'] = 0.
    elif mutation == 'target_semantics':
        hybrid['dynamics_auxiliary_config']['target_semantics'] = 'future_input_to_forecast'
    else:
        hybrid['dynamics_auxiliary_config']['reconstruction_weight'] = .2
    with pytest.raises(ValueError, match=match):
        run_compare(tmp_path, [field, hybrid])


def test_zero_and_attention_effects_do_not_pair_different_supervision(tmp_path):
    result = run_compare(tmp_path, [
        rollout_report(mode='zero'), rollout_report(supervision='field_latent'),
        rollout_report(interaction='transformer')])
    assert not result['dynamics_effects']
    assert not result['guide_interaction_effects']
    assert not result['dynamics_consistency_effects']


def test_plotter_rejects_mislabeled_auxiliary_scores(tmp_path):
    result = run_compare(tmp_path, [rollout_report(), rollout_report(supervision='field_latent')])
    result['dynamics_consistency_effects'][0]['scored_prediction'] = 'auxiliary_D_d'
    with pytest.raises(ValueError, match='matched field'):
        summarize_comparison(result)


@pytest.mark.parametrize('family', ['fourcastnet', 'climax'])
@pytest.mark.parametrize('interaction', ['off', 'transformer'])
def test_weather_provenance_strips_only_declared_latent_rollout_prefix(family, interaction):
    row = rollout_report(interaction=interaction)
    row['config']['model'] = family
    core = family+'_pinned_core'
    row['implementation'] = ('two_stage_guide_v1+' if interaction != 'off' else '') + (
        'dynamics_rollout_v1+dual_dynamics_v1+guide_fusion_v1+'+core)
    row['predictor_provenance'] = dict(
        family=family, implementation=core, upstream_source='https://example.test/official',
        upstream_commit='a'*40, implementation_variant='adapted', pretrained=False,
        depth=row['config']['weather_depth'], patch_size=row['config']['weather_patch_size'],
        hidden_dim=row['config']['hidden_dim'])
    _validate_weather_provenance([row])
    wrong = deepcopy(row)
    wrong['implementation'] = wrong['implementation'].replace('dynamics_rollout_v1+', '')
    with pytest.raises(ValueError, match='dynamics_rollout_v1'):
        _validate_weather_provenance([wrong])
    wrong = deepcopy(row)
    wrong['implementation'] = wrong['implementation'].replace(core, 'unknown_core')
    with pytest.raises(ValueError, match='implementation differs'):
        _validate_weather_provenance([wrong])
