"""Vector-only is a different objective, with an explicitly unused decoder."""
from copy import deepcopy

import pytest

from test_dynamics_consistency_reporting import rollout_report
from test_raw_comparison import run_compare
from climate_manifold.downstream.plot_comparison import summarize_comparison, plot_comparison
from climate_manifold.downstream.protocol import dynamics_supervision_identity


def vector_report(supervision='vector_only', **kwargs):
    row = rollout_report(supervision='field_latent' if supervision == 'vector_only' else supervision, **kwargs)
    row['dynamics_auxiliary_config']['supervision'] = supervision
    trained = supervision != 'vector_only'
    row['dynamics_decoder_contract'] = dict(
        version='climate_manifold.dynamics_decoder.v1', trained=trained,
        total_parameters=8, trainable_parameters=8 if trained else 0)
    if not trained:
        for key in ('reconstruction_weight', 'statistical_weight', 'tendency_weight'):
            row['dynamics_auxiliary_config'][key] = 0.
        for key in ('trainable_parameters', 'constraint_parameters'):
            row[key] -= 8
        row['dynamics_auxiliary_evaluation'] = dict(
            scores=None, signed_diagnostics=None,
            decoder_status='untrained_frozen_not_evaluated', evaluated_outputs=['latent'])
    return row


def test_three_modes_remain_separate_and_keep_both_vector_controls(tmp_path):
    result = run_compare(tmp_path, [vector_report(mode, bridge=bridge, seed=seed)
        for seed in (7, 19) for bridge in ('latent', 'guided')
        for mode in ('field', 'field_latent', 'vector_only')])
    assert len(result['seed_summary']) == 6
    assert all(group['seeds'] == [7, 19] for group in result['seed_summary'].values())
    assert len(result['dynamics_consistency_effects']) == 8
    effects = result['dynamics_vector_effects']
    assert len(effects) == 16  # Two routes x two controls x two seeds x two leads.
    assert {e['baseline_supervision'] for e in effects} == {'field', 'field_latent'}
    assert {e['interpretation'] for e in effects} == {
        'replace_dynamics_field_with_vector_supervision', 'remove_dynamics_field_supervision'}
    assert all(e['baseline_parameters'] == e['candidate_parameters'] for e in effects)
    assert all(e['baseline_trainable_parameters'] - e['candidate_trainable_parameters'] == 8 for e in effects)
    assert all(e['scored_prediction'] == 'downstream_M_physical_forecast' for e in effects)
    assert (tmp_path/'comparison.dynamics-vector-effects.csv').is_file()
    summary = summarize_comparison(result)
    vector = [r for r in summary if r['metric'] == 'dynamics_vector_gain_percent']
    assert len(vector) == 8 and all(r['valid_seeds'] == 2 for r in vector)
    assert len({r['control'] for r in vector}) == 4
    assert all('Vector only (0.1)' in r['label'] and ' / vs ' in r['label'] for r in vector)
    assert len({r['arm'] for r in summary if r['metric'] == 'rmse'}) == 6
    assert dynamics_supervision_identity(vector_report()) == ('latent_rollout', 'vector_only', .1)


def test_vector_physical_effect_does_not_rank_auxiliary_latent_scores(tmp_path):
    field, vector = vector_report('field'), vector_report()
    field['dynamics_auxiliary_evaluation']['latent_mse'] = 100.
    vector['dynamics_auxiliary_evaluation']['latent_mse'] = 0.
    for report, rmse in ((field, 2.), (vector, 3.)):
        for lead in report['scores']['climode']['per_variable']['t2m']['by_lead']:
            lead['rmse'] = rmse
    result = run_compare(tmp_path, [field, vector])
    assert all(e['rmse_skill_vs_baseline'] == -.5 for e in result['dynamics_vector_effects'])
    assert all(e['mean'] == -50. for e in summarize_comparison(result)
               if e['metric'] == 'dynamics_vector_gain_percent')


@pytest.mark.parametrize('mutation', ['trainable', 'constraint', 'total', 'forecast', 'audit_scope', 'weight'])
def test_unmatched_capacity_scope_or_latent_weight_has_no_hybrid_vector_effect(tmp_path, mutation):
    hybrid, vector = vector_report('field_latent'), vector_report()
    if mutation in ('trainable', 'constraint', 'total', 'forecast'):
        vector[mutation+'_parameters'] += 1
    elif mutation == 'audit_scope':
        vector['dynamics_evaluation_contract']['rollout_steps'] = 10
    else:
        vector['dynamics_auxiliary_config']['latent_weight'] = .2
    result = run_compare(tmp_path, [hybrid, vector])
    assert not result['dynamics_vector_effects']
    assert not result['dynamics_consistency_effects']


@pytest.mark.parametrize('mutation', ['missing_decoder', 'trained_decoder', 'decoder_count', 'auxiliary_field_scores'])
def test_vector_metadata_must_exclude_untrained_field_forecasts(tmp_path, mutation):
    vector = vector_report()
    if mutation == 'missing_decoder':
        vector.pop('dynamics_decoder_contract')
    elif mutation == 'trained_decoder':
        vector['dynamics_decoder_contract']['trained'] = True
    elif mutation == 'decoder_count':
        vector['dynamics_decoder_contract']['trainable_parameters'] = 8
    else:
        vector['dynamics_auxiliary_evaluation']['scores'] = {'rmse': 0}
    with pytest.raises(ValueError, match='decoder|latent-only'):
        run_compare(tmp_path, [vector])


def test_vector_first_does_not_allow_mismatched_field_weights(tmp_path):
    vector, field, hybrid = vector_report(), vector_report('field'), vector_report('field_latent')
    hybrid['dynamics_auxiliary_config']['statistical_weight'] = .2
    with pytest.raises(ValueError, match='dynamics_auxiliary_config'):
        run_compare(tmp_path, [vector, field, hybrid])


def test_legacy_field_reports_without_decoder_metadata_remain_valid(tmp_path):
    result = run_compare(tmp_path, [rollout_report(), rollout_report(supervision='field_latent'), vector_report()])
    assert result['dynamics_consistency_effects']
    assert not result['dynamics_vector_effects']


@pytest.mark.parametrize('mutation', ['score', 'weight', 'capacity'])
def test_plotter_rejects_mislabelled_vector_effects(tmp_path, mutation):
    result = run_compare(tmp_path, [vector_report('field_latent'), vector_report()])
    result = deepcopy(result)
    if mutation == 'score':
        result['dynamics_vector_effects'][0]['scored_prediction'] = 'auxiliary_D_d'
    elif mutation == 'weight':
        result['dynamics_vector_effects'][0]['baseline_latent_weight'] = .2
    else:
        result['rows'][1]['trainable_parameters'] += 1
    with pytest.raises(ValueError, match='Dynamics vector'):
        summarize_comparison(result)


def test_vector_plot_renders_both_controls_without_duplicate_series(tmp_path):
    run_compare(tmp_path, [vector_report(m) for m in ('field', 'field_latent', 'vector_only')])
    manifest = plot_comparison(tmp_path/'comparison.json', tmp_path/'figures')
    assert 'dynamics_vector_gain_percent_by_lead_t2m.png' in manifest['figures']
    assert (tmp_path/'figures'/'plot_summary.csv').is_file()
