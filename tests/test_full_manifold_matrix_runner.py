"""Full raw/spatial/dynamics matrix selects matched rollout arms, not legacy fits."""
import json
from pathlib import Path
import subprocess

import pytest

from climate_manifold.downstream.train import parser
from test_dynamics_consistency_runner import fits, evaluations
from test_dynamics_vector_runner import setup_vector
from test_fusion_runner import ROOT, arg, fixture_inputs, run_script, setup_runner


ARMS = (
    'raw', 'latent', 'guided',
    'latent_dynamics_field', 'latent_dynamics_field_latent',
    'guided_dynamics_field', 'guided_dynamics_field_latent',
    'guided_dynamics_two_stage_field', 'guided_dynamics_two_stage_field_latent',
    'latent_dynamics_vector_only', 'guided_dynamics_vector_only',
    'guided_dynamics_two_stage_vector_only',
)
MODELS = 'transformer mlp neural_ode climode convlstm simvp fourcastnet climax'


def setup_full(tmp_path):
    env, log = setup_vector(tmp_path)
    env.update(CONSISTENCY_ONLY='0', INCLUDE_LEGACY_DYNAMICS='0', INCLUDE_ZERO_GUIDE='0',
               INCLUDE_TWO_STAGE='1', INCLUDE_TWO_STAGE_BYPASS='0', INCLUDE_ZERO_DYNAMICS='0')
    return env, log


def test_twelve_arms_for_each_of_eight_forecasters(tmp_path):
    env, log = setup_full(tmp_path)
    result, rows = run_script(env, log, MODELS=MODELS, STATISTICAL_LOSS='signed_measure')
    assert result.returncode == 0, result.stderr
    assert len(fits(rows)) == 96 and len(evaluations(rows)) == 192
    assert 'Fits: 96' in result.stdout
    for family in MODELS.split():
        group = [r for r in fits(rows) if arg(r, '--model') == family]
        assert [Path(arg(r, '--output')).stem for r in group] == [
            f'{family}-{arm}-seed7' for arm in ARMS]
        configs = [parser().parse_args(r[2:]) for r in group]
        raw, spatial, guided = configs[:3]
        assert [a.bridge for a in configs[:3]] == ['raw', 'latent', 'guided']
        assert all(a.dynamics_encoder == 'off' for a in configs[:3])
        assert raw.regularization == 'none'
        assert spatial.regularization == guided.regularization == 'full'
        assert guided.guide_interaction == 'off' and guided.guide_architecture == 'fusion'
        for arm, a in zip(ARMS[3:], configs[3:]):
            assert a.dynamics_encoder == 'conv_gru' and a.dynamics_mode == 'learned'
            assert a.dynamics_auxiliary_architecture == 'latent_rollout'
            assert a.training_mode == 'joint' and a.regularization == 'full'
            assert a.constraint_pair == 'statistical'
            assert a.reconstruction_weight == a.statistical_weight == 0.1
            assert a.statistical_loss == a.dynamics_statistical_loss == 'signed_measure'
            assert a.guide_interaction == ('transformer' if '_two_stage_' in arm else 'off')
            if a.bridge == 'guided':
                assert a.guide_architecture == 'fusion' and a.guide_mode == 'learned'
            if arm.endswith('vector_only'):
                assert a.dynamics_supervision == 'vector_only' and a.dynamics_latent_weight == 0.1
                assert a.dynamics_reconstruction_weight == a.dynamics_statistical_weight == a.dynamics_tendency_weight == 0
            else:
                assert a.dynamics_reconstruction_weight == a.dynamics_statistical_weight == 0.1
        assert all(a.batch_size == 16 for a in configs)
    settings = json.loads((Path(env['RUN'])/'run-manifest.json').read_text())['settings']
    assert settings['include_legacy_dynamics'] == '0'
    assert settings['arms'].split() == list(ARMS)
    # New latent diagnostics belong to the nine rollout arms; baseline evals
    # remain ordinary main-forecast evaluations on the same origins.
    assert sum('--dynamics-diagnostics' in r for r in evaluations(rows)) == 9*8*2


@pytest.mark.parametrize('loss', ['w2', 'kl_entropy', 'signed_measure'])
def test_statistical_choice_routes_to_both_encoders_without_touching_raw(tmp_path, loss):
    env, log = setup_full(tmp_path)
    result, rows = run_script(env, log, STATISTICAL_LOSS=loss, EVALUATE_TEST='0')
    assert result.returncode == 0, result.stderr
    configs = [parser().parse_args(r[2:]) for r in fits(rows)]
    assert len(configs) == 12
    assert configs[0].regularization == 'none'
    for a in configs[1:]:
        assert a.statistical_loss == loss and a.statistical_weight > 0
        if a.dynamics_encoder != 'off':
            assert a.dynamics_statistical_loss == loss
            # The Dynamics statistical selector is metadata only for vector-only.
            assert (a.dynamics_statistical_weight == 0) == (a.dynamics_supervision == 'vector_only')


@pytest.mark.parametrize('settings,message', [
    ({'INCLUDE_LEGACY_DYNAMICS': '2'}, 'INCLUDE_LEGACY_DYNAMICS must be 0 or 1'),
    ({'INCLUDE_ZERO_DYNAMICS': '1'}, 'requires INCLUDE_ZERO_DYNAMICS=0'),
    ({'INCLUDE_TWO_STAGE_BYPASS': '1'}, 'requires INCLUDE_LEGACY_DYNAMICS=1'),
    ({'INCLUDE_DYNAMICS_CONSISTENCY': '0', 'INCLUDE_VECTOR_ONLY': '0'},
     'requires INCLUDE_DYNAMICS_CONSISTENCY=1 when Dynamics is enabled'),
])
def test_conflicting_legacy_controls_fail_before_training(tmp_path, settings, message):
    env, log = setup_full(tmp_path)
    result, rows = run_script(env, log, **settings)
    assert result.returncode != 0 and message in result.stderr
    assert not fits(rows)


def test_resume_rejects_switching_legacy_arm_membership(tmp_path):
    env, log = setup_full(tmp_path)
    result, _ = run_script(env, log)
    assert result.returncode == 0, result.stderr
    result, rows = run_script(env, log, RESUME='1')
    assert result.returncode == 0, result.stderr
    assert not fits(rows) and not evaluations(rows)
    result, rows = run_script(env, log, RESUME='1', INCLUDE_LEGACY_DYNAMICS='1')
    assert result.returncode != 0 and 'differs' in result.stderr
    assert not fits(rows) and not evaluations(rows)


def test_inactive_legacy_switch_does_not_change_effective_manifest(tmp_path):
    env, log = setup_vector(tmp_path)  # consistency-only omits all legacy arms
    result, _ = run_script(env, log)
    assert result.returncode == 0, result.stderr
    settings = json.loads((Path(env['RUN'])/'run-manifest.json').read_text())['settings']
    assert settings['include_legacy_dynamics'] == '0'
    result, rows = run_script(env, log, RESUME='1', INCLUDE_LEGACY_DYNAMICS='0')
    assert result.returncode == 0, result.stderr
    assert not fits(rows) and not evaluations(rows)


def test_daily_preflight_resets_legacy_toggle_but_real_matrix_retains_it(tmp_path):
    env, log = setup_full(tmp_path)
    daily = tmp_path/'daily'
    (daily/'prepared').mkdir(parents=True)
    fixture_inputs(daily/'prepared')
    env.pop('ARCHIVE')
    env.pop('INFO')
    env.update(DAILY_WORK=str(daily), ERA5_ROOT=str(tmp_path/'raw'), DEVICE='cpu',
               RUN=str(daily/'runs'/'full'), RUN_PREFLIGHT_TESTS='1', EVALUATE_TEST='0',
               PYTEST_ENV_LOG=str(tmp_path/'pytest-env.json'))
    with Path(env['PYTHON']).open('a') as stream:
        stream.write('''
if '-m' in args and value('-m') == 'pytest':
    Path(os.environ['PYTEST_ENV_LOG']).write_text(json.dumps({
        'legacy': os.environ['INCLUDE_LEGACY_DYNAMICS'], 'arguments': args}))
''')
    result = subprocess.run(['bash', str(ROOT/'scripts/run_daily_manifold_fusion.sh')],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    preflight = json.loads(Path(env['PYTEST_ENV_LOG']).read_text())
    assert preflight['legacy'] == '1'
    assert 'tests/test_full_manifold_matrix_runner.py' in preflight['arguments']
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(fits(rows)) == 12


def test_fixture_isolates_inherited_legacy_toggle(tmp_path, monkeypatch):
    monkeypatch.setenv('INCLUDE_LEGACY_DYNAMICS', '0')
    env, _ = setup_runner(tmp_path)
    assert env['INCLUDE_LEGACY_DYNAMICS'] == '1'
