"""Zero-guide capacity controls for every fusion route, and their reporting."""
import json
from pathlib import Path

import pytest
import torch

from test_pinn_training import pinn_prepared
from test_dual_dynamics_training import dual_args
from test_dynamics_consistency_reporting import rollout_report
from test_dynamics_consistency_runner import fits, evaluations
from test_dynamics_reporting import dynamics_report, raw_report
from test_dynamics_vector_runner import setup_vector
from test_fusion_runner import run_script
from test_raw_comparison import run_compare
from climate_manifold.downstream.compare import _arm
from climate_manifold.downstream.dynamics_objective import dynamics_contract
from climate_manifold.downstream.evaluate import evaluate
from climate_manifold.downstream.pipeline import PredictorConfig
from climate_manifold.downstream.protocol import guide_interaction_contract
from climate_manifold.downstream.train import guide_contract, load_predictor, parser, train


def _zero_guide(row):
    row['config']['guide_mode'] = 'zero'
    cfg = PredictorConfig(**row['config'])
    row['guide_contract'] = guide_contract(cfg)
    if row.get('dynamics_contract') is not None:
        row['dynamics_contract'] = dynamics_contract(cfg)
    if 'guide_interaction_contract' in row:
        row['guide_interaction_contract'] = guide_interaction_contract(cfg)
    row['representation_sha256'] += '-zero-guide'
    return row


def test_runner_zero_guide_suite_trains_the_three_fusion_controls(tmp_path):
    env, log = setup_vector(tmp_path)
    result, rows = run_script(env, log, INCLUDE_ZERO_GUIDE_CONTROLS='1', ZERO_GUIDE_SUITE='1',
                              INCLUDE_TWO_STAGE='1', INCLUDE_ZERO_GUIDE='0')
    assert result.returncode == 0, result.stderr
    configs = [parser().parse_args(r[2:]) for r in fits(rows)]
    assert len(configs) == 3 and 'Fits: 3' in result.stdout
    assert all(a.bridge == 'guided' and a.guide_mode == 'zero' for a in configs)
    spatial, joint, two_stage = configs
    assert spatial.dynamics_encoder == 'off'
    for a in (joint, two_stage):
        assert a.dynamics_encoder == 'conv_gru' and a.dynamics_mode == 'zero'
        assert a.dynamics_auxiliary_architecture == 'latent_rollout' and a.dynamics_supervision == 'field'
    assert joint.guide_interaction == 'off' and two_stage.guide_interaction == 'transformer'
    # Only the dynamics routes carry auxiliary diagnostics.
    assert sum('--dynamics-diagnostics' in r for r in evaluations(rows)) == 4
    settings = json.loads((Path(env['RUN'])/'run-manifest.json').read_text())['settings']
    assert settings['include_zero_guide_controls'] == '1' and settings['zero_guide_suite'] == '1'


def test_runner_rejects_two_supplement_suites_at_once(tmp_path):
    env, log = setup_vector(tmp_path)
    result, rows = run_script(env, log, INCLUDE_ZERO_GUIDE_CONTROLS='1', ZERO_GUIDE_SUITE='1',
                              INCLUDE_DYNAMICS_ONLY='1', DYNAMICS_ONLY_SUITE='1')
    assert result.returncode != 0 and 'Choose one supplement suite' in result.stderr
    assert not fits(rows)


def test_combined_comparison_accepts_main_dynamics_only_and_zero_guide_arms(tmp_path):
    reports = []
    for seed in (7, 19):
        reports += [raw_report(seed),
                    dynamics_report('guided', seed=seed, encoder='off'),
                    _zero_guide(dynamics_report('guided', seed=seed, encoder='off')),
                    rollout_report('guided', seed=seed),
                    rollout_report('guided', seed=seed, interaction='transformer'),
                    rollout_report('guided', seed=seed, mode='only'),
                    _zero_guide(rollout_report('guided', seed=seed, mode='zero')),
                    _zero_guide(rollout_report('guided', seed=seed, mode='zero',
                                               interaction='transformer'))]
    for i, row in enumerate(reports):
        if row.get('representation_sha256'):
            row['representation_sha256'] += f'-{i}'
    result = run_compare(tmp_path, reports)
    arms = {_arm(r) for r in result['rows']}
    assert len(arms) == 8 and len(result['rows']) == 16
    assert all(row['seeds'] == [7, 19] for row in result['seed_summary'].values())
    assert sum(':zero:' in arm for arm in arms) == 3


@pytest.mark.parametrize('interaction', [[], ['--guide-interaction', 'transformer']])
def test_all_zero_fusion_trains_evaluates_and_ignores_guides(pinn_prepared, tmp_path, interaction):
    _, _, _, archive = pinn_prepared
    args = dual_args(archive, tmp_path, 'guided', 'conv_gru', '--guide-mode', 'zero',
                     '--dynamics-mode', 'zero', '--dynamics-auxiliary-architecture', 'latent_rollout',
                     '--dynamics-supervision', 'field', *interaction)
    checkpoint = train(args)
    restored, payload = load_predictor(checkpoint)
    assert payload['config']['guide_mode'] == payload['config']['dynamics_mode'] == 'zero'
    assert payload['guide_contract']['forecast_gradient_to_encoder'] is False
    assert payload['dynamics_contract']['forecast_gradient_to_dynamics_encoder'] is False
    fusion = restored.guide_fusion
    torch.nn.init.normal_(fusion.head[-1].weight)
    history = torch.randn(2, fusion.history_steps, fusion.dimension)
    math_prod = fusion.dynamics_grid[0]*fusion.dynamics_grid[1]*fusion.dynamics_grid[2]
    with torch.no_grad():
        first = fusion(history, torch.randn(2, fusion.history_steps, fusion.guide_dimension),
                       dynamics_history=torch.randn(2, fusion.history_steps, math_prod))
        second = fusion(history, torch.randn(2, fusion.history_steps, fusion.guide_dimension),
                        dynamics_history=torch.randn(2, fusion.history_steps, math_prod))
    torch.testing.assert_close(first, second)
    report = evaluate(checkpoint, archive, tmp_path/'zero.validation.json',
                      information=tmp_path/'pinn-information.npz', max_cases=2)
    assert report['finite_forecast_fraction'] == 1. and not report['failed_origins']
