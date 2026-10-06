"""First-stage attention and evaluation bypass are separately identified experiments."""
from copy import deepcopy

import pytest

from test_dynamics_reporting import dynamics_report, raw_report
from test_raw_comparison import run_compare
from test_pinn_training import pinn_prepared
from test_spatial_training import _args
from climate_manifold.downstream.compare import _arm, _validate_weather_provenance, compare
from climate_manifold.downstream.evaluate import evaluate
from climate_manifold.downstream.pipeline import PredictorConfig
from climate_manifold.downstream.plot_comparison import summarize_comparison
from climate_manifold.downstream.protocol import guide_interaction_contract, validate_experiment
from climate_manifold.downstream.train import guide_contract, train


def interaction_report(kind='transformer', seed=7, depth=1, bypass=False):
    row = dynamics_report(seed=seed)
    row['config'].update(guide_interaction=kind, guide_interaction_depth=depth)
    row['guide_interaction_contract'] = guide_interaction_contract(row['config'])
    row['guide_contract'] = guide_contract(PredictorConfig(**row['config']))
    row['checkpoint_sha256'] = f'{kind}-{depth}-{seed}'
    row['representation_sha256'] = row['checkpoint_sha256']
    if kind != 'off':
        row['guide_interaction_evaluation'] = dict(
            requested='off' if bypass else 'checkpoint', trained=kind,
            effective='off' if bypass else kind, bypassed=bypass)
        for name in ('total_parameters', 'trainable_parameters', 'forecast_parameters'):
            row[name] += 20 * depth
        row['implementation'] = 'two_stage_guide_v1+'+row['implementation']
    return row


def test_protocol_describes_guide_only_first_stage_and_rejects_wrong_routes():
    cfg = interaction_report()['config']
    contract = validate_experiment(cfg, 'primary')
    assert contract['guide_interaction'] == 'transformer'
    assert 'guide interaction transformer(spatial, dynamics)' in contract['path']
    assert 'fusion transformer(raw, refined guides)' in contract['path']
    for changes in ({'bridge': 'latent'}, {'dynamics_encoder': 'off'},
                    {'guide_architecture': 'joint'}, {'guide_interaction_depth': 0}):
        with pytest.raises(ValueError, match='guide_interaction'):
            validate_experiment({**cfg, **changes}, 'primary')


def test_single_two_stage_and_bypass_are_distinct_same_seed_arms(tmp_path):
    reports = [row for seed in (7, 19) for row in (
        raw_report(seed), interaction_report('off', seed), interaction_report(seed=seed),
        interaction_report(seed=seed, bypass=True))]
    result = run_compare(tmp_path, reports)
    assert len(result['seed_summary']) == 4
    assert all(row['seeds'] == [7, 19] for row in result['seed_summary'].values())
    assert len(result['guide_interaction_effects']) == 8  # two effects, two seeds, two leads
    assert not result['guide_effects']
    assert not result['dynamics_effects']
    effects = result['guide_interaction_effects']
    assert {r['interpretation'] for r in effects} == {
        'add_guide_interaction', 'bypass_trained_guide_interaction'}
    assert all(r['same_checkpoint'] is r['inference_intervention'] for r in effects)
    assert (tmp_path/'comparison.guide-interaction-effects.csv').exists()
    assert {_arm(r) for r in result['rows']} == {
        _arm(r) for r in result['climode_benchmark']['rows']}
    summary = summarize_comparison(result)
    assert {r['metric'] for r in summary} >= {
        'two_stage_gain_percent', 'guide_interaction_input_gain_percent'}
    assert any('Inference bypass' in r['label'] for r in summary)
    assert all(r['valid_seeds'] == 2 for r in summary)


def test_different_depths_are_not_pooled_as_training_seeds(tmp_path):
    rows = [interaction_report('off'), interaction_report(depth=1), interaction_report(depth=2)]
    result = run_compare(tmp_path, rows)
    assert len(result['seed_summary']) == 3
    assert len(result['guide_interaction_effects']) == 4


def test_bypass_requires_identical_checkpoint_for_intervention_pair(tmp_path):
    active, bypass = interaction_report(), interaction_report(bypass=True)
    bypass['checkpoint_sha256'] = 'different-weights'
    result = run_compare(tmp_path, [active, bypass])
    assert not result['guide_interaction_effects']


@pytest.mark.parametrize('mutation,match', [
    ('raw_input', 'guide_interaction_contract'),
    ('missing_evaluation', 'evaluation metadata'),
    ('mislabel_bypass', 'guide_interaction_evaluation'),
    ('no_checkpoint', 'checkpoint_sha256'),
    ('archive', 'archive_sha256'),
    ('history', 'representation_config|input history_steps'),
])
def test_false_contracts_and_unfair_data_comparisons_are_rejected(tmp_path, mutation, match):
    baseline, row = interaction_report('off'), interaction_report(bypass=mutation == 'no_checkpoint')
    if mutation == 'raw_input':
        row['guide_interaction_contract']['raw_input'] = True
    elif mutation == 'missing_evaluation':
        del row['guide_interaction_evaluation']
    elif mutation == 'mislabel_bypass':
        row['guide_interaction_evaluation']['effective'] = 'off'
    elif mutation == 'no_checkpoint':
        del row['checkpoint_sha256']
    elif mutation == 'archive':
        row['archive_sha256'] = 'different-data'
    else:
        row['representation_config']['history_steps'] += 1
    with pytest.raises(ValueError, match=match):
        run_compare(tmp_path, [baseline, row])


@pytest.mark.parametrize('family', ['fourcastnet', 'climax'])
def test_weather_core_provenance_ignores_only_declared_front_end_wrappers(family):
    row = interaction_report()
    row['config']['model'] = family
    core = family+'_pinned_core'
    row['implementation'] = 'two_stage_guide_v1+dual_dynamics_v1+guide_fusion_v1+'+core
    row['predictor_provenance'] = dict(
        family=family, implementation=core, upstream_source='https://example.test/official',
        upstream_commit='a'*40, implementation_variant='adapted', pretrained=False,
        depth=row['config']['weather_depth'], patch_size=row['config']['weather_patch_size'],
        hidden_dim=row['config']['hidden_dim'])
    _validate_weather_provenance([row])
    wrong = deepcopy(row)
    wrong['implementation'] = wrong['implementation'].replace(core, 'different_core')
    with pytest.raises(ValueError, match='implementation differs'):
        _validate_weather_provenance([wrong])


@pytest.mark.parametrize('family', ['fourcastnet', 'climax'])
def test_trained_weather_reports_compare_single_two_stage_and_bypass(
        pinn_prepared, tmp_path, family):
    _, _, _, archive = pinn_prepared
    information = tmp_path/'pinn-information.npz'
    paths = []
    for kind in ('off', 'transformer'):
        name = family+'-'+kind
        args = _args(archive, tmp_path/(name+'.pt'),
            '--model', family, '--bridge', 'guided', '--raw-backend', 'matched',
            '--guide-architecture', 'fusion', '--guide-fusion-depth', '1',
            '--transformer-heads', '2', '--weather-depth', '1', '--history-steps', '2',
            '--information', str(information), '--mode', 'enriched',
            '--constraint-pair', 'statistical', '--regularization', 'full',
            '--constraint-decoder', 'separate_surface_and_information',
            '--dynamics-encoder', 'conv_gru', '--dynamics-hidden-dim', '4',
            '--dynamics-channels', '2', '--guide-interaction', kind)
        checkpoint = train(args)
        for override in (('checkpoint', 'off') if kind == 'transformer' else ('checkpoint',)):
            path = tmp_path/(name+'-'+override+'.json')
            report = evaluate(checkpoint, archive, path, information=information,
                              max_cases=2, guide_interaction_override=override)
            assert report['finite_forecast_fraction'] == 1
            paths.append(path)
    result = compare(paths, tmp_path/'comparison.json')
    assert result['ranking_allowed'] and len(result['seed_summary']) == 3
    assert {r['interpretation'] for r in result['guide_interaction_effects']} == {
        'add_guide_interaction', 'bypass_trained_guide_interaction'}
    assert {r['metric'] for r in summarize_comparison(result)} >= {
        'two_stage_gain_percent', 'guide_interaction_input_gain_percent'}
