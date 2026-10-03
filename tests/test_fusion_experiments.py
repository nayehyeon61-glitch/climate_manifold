"""Train the manifold front-end and its controls through physical comparison.

Small synthetic examples verify routing and reporting, not weather skill.
"""
from copy import deepcopy

import pytest

from test_pinn_training import pinn_prepared
from test_spatial_training import _args
from test_guided_comparison import guide_report
from climate_manifold.downstream.compare import compare, _arm, _validate_guides
from climate_manifold.downstream.evaluate import evaluate
from climate_manifold.downstream.train import train


@pytest.mark.parametrize('family', ['mlp', 'climode'])
@pytest.mark.parametrize('loss', ['w2', 'kl_entropy', 'signed_measure'])
def test_manifold_and_controls_keep_identity_through_field_metrics(
        pinn_prepared, tmp_path, family, loss):
    _, _, _, archive = pinn_prepared
    info = tmp_path/'pinn-information.npz'
    common = ['--model', family, '--training-mode', 'joint', '--initialization', 'fresh',
              '--raw-backend', 'matched', '--information', str(info),
              '--mode', 'enriched', '--history-steps', '2', '--weather-depth', '1',
              '--transformer-heads', '2', '--reconstruction-weight', '0.1']
    statistics = ['--constraint-pair', 'statistical', '--regularization', 'full',
                  '--constraint-decoder', 'separate_surface_and_information',
                  '--statistical-loss', loss, '--statistical-weight', '0.1']
    guide = ['--bridge', 'guided', '--guide-architecture', 'fusion',
             '--guide-fusion-depth', '1', '--guide-direct-information']
    arms = {
        'raw': ['--bridge', 'raw', '--regularization', 'none'],
        'latent': ['--bridge', 'latent', *statistics],
        'manifold': [*guide, *statistics],
        'zero_guide': [*guide, '--guide-mode', 'zero', *statistics],
        'no_observed_constraints': [*guide, '--regularization', 'none'],
    }
    paths, reports = [], {}
    for name, route in arms.items():
        checkpoint = train(_args(archive, tmp_path/(name+'.pt'), *common, *route))
        path = tmp_path/(name+'.validation.json')
        reports[name] = evaluate(checkpoint, archive, path, information=info, max_cases=2)
        assert reports[name]['finite_forecast_fraction'] == 1.
        paths.append(path)
    result = compare(paths, tmp_path/'comparison.json')
    assert result['ranking_allowed'] and result['same_successful_origins']
    assert len(result['seed_summary']) == 5
    identities = {(row['model'], row['bridge'], _arm(row)) for row in result['rows']}
    physical_identities = {(row['model'], row['bridge'], _arm(row))
                           for row in result['climode_benchmark']['rows']}
    assert identities == physical_identities
    assert {effect['interpretation'] for effect in result['guide_effects']} == {
        'change_guide_input', 'add_observed_encoder_constraints'}
    assert not result['statistical_loss_effects']  # One chosen metric, not a metric ablation.
    for name in ('zero_guide', 'no_observed_constraints'):
        assert reports[name]['forecast_parameters'] == reports['manifold']['forecast_parameters']
    # Forecast-only means the complete observed constraint objective is absent.
    control = reports['no_observed_constraints']
    # The historical information head remains in the checkpoint but is unused;
    # zero loss weights, not its mere allocation, define this ablation.
    assert control['constraint_pair'] is None and control['constraint_contract'] is None
    assert all(weight == 0 for weight in control['objective_weights'].values())


def test_cannot_label_different_fusion_depths_as_a_guide_ablation():
    first = guide_report()
    first['config'].update(guide_architecture='fusion', guide_fusion_depth=1)
    second = deepcopy(first)
    second['config'].update(guide_fusion_depth=2, guide_mode='zero')
    with pytest.raises(ValueError, match='guide_fusion_depth'):
        _validate_guides([first, second])


def test_cannot_mix_fusion_frontend_and_legacy_transformer_in_one_suite():
    first = guide_report()
    second = deepcopy(first)
    second['config']['guide_architecture'] = 'fusion'
    with pytest.raises(ValueError, match='guide_architecture'):
        _validate_guides([first, second])
