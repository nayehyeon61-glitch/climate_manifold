"""Vector-only routing is optional, scoped to Dynamics, and resume-auditable."""
import json
from pathlib import Path
import subprocess

import pytest

from climate_manifold.downstream.train import parser
from test_dynamics_consistency_runner import setup_consistency, fits, evaluations
from test_fusion_runner import ROOT, arg, fixture_inputs, run_script, setup_runner


def setup_vector(tmp_path):
    env, log = setup_consistency(tmp_path)
    env['INCLUDE_VECTOR_ONLY'] = '1'
    with Path(env['PYTHON']).open('a') as stream:
        stream.write('''
if '-m' in args and value('-m').endswith('.evaluate') and '_vector_only-' in value('--checkpoint'):
    path = Path(value('--output'))
    report = json.loads(path.read_text())
    report['dynamics_auxiliary_config']['supervision'] = 'vector_only'
    path.write_text(json.dumps(report))
''')
    return env, log


def test_vector_arms_for_all_forecasters_preserve_primary_and_observed_losses(tmp_path):
    env, log = setup_vector(tmp_path)
    families = 'transformer mlp neural_ode climode convlstm simvp fourcastnet climax'
    result, rows = run_script(env, log, MODELS=families, DYNAMICS_LATENT_WEIGHT='0.3',
                              DYNAMICS_RECONSTRUCTION_WEIGHT='0.2',
                              DYNAMICS_STATISTICAL_WEIGHT='0.4', DYNAMICS_TENDENCY_WEIGHT='0.6',
                              STATISTICAL_LOSS='signed_measure')
    assert result.returncode == 0, result.stderr
    assert len(fits(rows)) == 48 and len(evaluations(rows)) == 96
    assert 'Fits: 48' in result.stdout
    for family in families.split():
        configs = [parser().parse_args(r[2:]) for r in fits(rows) if arg(r, '--model') == family]
        assert [a.dynamics_supervision for a in configs] == [
            'field', 'field_latent', 'field', 'field_latent', 'vector_only', 'vector_only']
        for a in configs:
            assert a.batch_size == 16 and a.dynamics_auxiliary_architecture == 'latent_rollout'
            assert a.regularization == 'full' and a.constraint_pair == 'statistical'
            assert a.statistical_weight == a.reconstruction_weight == 0.1
            assert a.statistical_loss == a.dynamics_statistical_loss == 'signed_measure'
        for a in configs[-2:]:
            assert a.dynamics_latent_weight == 0.3
            assert a.dynamics_reconstruction_weight == a.dynamics_statistical_weight == a.dynamics_tendency_weight == 0
        for a in configs[:4]:
            assert a.dynamics_reconstruction_weight == 0.2
            assert a.dynamics_statistical_weight == 0.4 and a.dynamics_tendency_weight == 0.6
        assert [a.bridge for a in configs[-2:]] == ['latent', 'guided']
    assert all('--dynamics-diagnostics' in r for r in evaluations(rows))
    settings = json.loads((Path(env['RUN'])/'run-manifest.json').read_text())['settings']
    assert settings['include_vector_only'] == '1'


def test_two_stage_adds_vector_only_with_same_fusion_configuration(tmp_path):
    env, log = setup_vector(tmp_path)
    result, rows = run_script(env, log, INCLUDE_TWO_STAGE='1', GUIDE_INTERACTION_DEPTH='2',
                              AUX_ROLLOUT_STEPS='10')
    assert result.returncode == 0, result.stderr
    assert len(fits(rows)) == 9 and len(evaluations(rows)) == 18
    configs = [parser().parse_args(r[2:]) for r in fits(rows)]
    assert sum(a.dynamics_supervision == 'vector_only' for a in configs) == 3
    a = configs[-1]
    assert a.bridge == 'guided' and a.guide_interaction == 'transformer'
    assert a.guide_interaction_depth == 2 and a.dynamics_supervision == 'vector_only'
    assert all(arg(r, '--dynamics-rollout-steps') == '10' for r in evaluations(rows))


@pytest.mark.parametrize('settings,message', [
    ({'INCLUDE_VECTOR_ONLY': '2'}, 'INCLUDE_VECTOR_ONLY must be 0 or 1'),
    ({'INCLUDE_DYNAMICS_CONSISTENCY': '0', 'CONSISTENCY_ONLY': '0'},
     'INCLUDE_VECTOR_ONLY=1 requires INCLUDE_DYNAMICS_CONSISTENCY=1'),
])
def test_invalid_vector_option_never_starts_fit(tmp_path, settings, message):
    env, log = setup_vector(tmp_path)
    result, rows = run_script(env, log, **settings)
    assert result.returncode != 0 and message in result.stderr
    assert not fits(rows)


def test_vector_resume_repairs_missing_report_and_rejects_wrong_supervision(tmp_path):
    env, log = setup_vector(tmp_path)
    result, _ = run_script(env, log)
    assert result.returncode == 0, result.stderr
    result, rows = run_script(env, log, RESUME='1')
    assert result.returncode == 0, result.stderr
    assert not fits(rows) and not evaluations(rows)
    path = Path(env['RUN'])/'mlp-guided_dynamics_vector_only-seed7.test.json'
    path.unlink()
    result, rows = run_script(env, log, RESUME='1')
    assert result.returncode == 0, result.stderr
    assert not fits(rows) and len(evaluations(rows)) == 1
    report = json.loads(path.read_text())
    report['dynamics_auxiliary_config']['supervision'] = 'field_latent'
    path.write_text(json.dumps(report))
    result, rows = run_script(env, log, RESUME='1')
    assert result.returncode != 0 and 'consistency report/configuration mismatch' in result.stderr
    assert not fits(rows) and not evaluations(rows)


def test_resume_rejects_changing_vector_matrix(tmp_path):
    env, log = setup_vector(tmp_path)
    result, _ = run_script(env, log)
    assert result.returncode == 0, result.stderr
    result, rows = run_script(env, log, RESUME='1', INCLUDE_VECTOR_ONLY='0')
    assert result.returncode != 0 and 'differs' in result.stderr
    assert not fits(rows) and not evaluations(rows)


def test_resume_repairs_missing_populated_vector_effect_csv(tmp_path):
    env, log = setup_vector(tmp_path)
    result, _ = run_script(env, log, EVALUATE_TEST='0')
    assert result.returncode == 0, result.stderr
    path = Path(env['RUN'])/'comparison.json'
    report = json.loads(path.read_text())
    report['dynamics_vector_effects'] = [{'model': 'mlp'}]
    path.write_text(json.dumps(report))
    result, rows = run_script(env, log, RESUME='1', EVALUATE_TEST='0')
    assert result.returncode == 0, result.stderr
    assert not fits(rows) and not evaluations(rows)
    assert any('climate_manifold.downstream.compare' in r for r in rows)


def test_daily_preflight_isolates_vector_flag_but_preserves_real_experiment(tmp_path):
    env, log = setup_vector(tmp_path)
    daily = tmp_path/'daily'
    (daily/'prepared').mkdir(parents=True)
    fixture_inputs(daily/'prepared')
    env.pop('ARCHIVE')
    env.pop('INFO')
    env.update(DAILY_WORK=str(daily), ERA5_ROOT=str(tmp_path/'raw'), DEVICE='cpu',
               RUN=str(daily/'runs'/'vector'), RUN_PREFLIGHT_TESTS='1', EVALUATE_TEST='0',
               PYTEST_ENV_LOG=str(tmp_path/'pytest-env.json'))
    with Path(env['PYTHON']).open('a') as stream:
        stream.write('''
if '-m' in args and value('-m') == 'pytest':
    Path(os.environ['PYTEST_ENV_LOG']).write_text(json.dumps({
        'vector_flag': os.environ['INCLUDE_VECTOR_ONLY'], 'arguments': args}))
''')
    result = subprocess.run(['bash', str(ROOT/'scripts/run_dynamics_consistency_comparison.sh')],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    preflight = json.loads(Path(env['PYTEST_ENV_LOG']).read_text())
    assert preflight['vector_flag'] == '0'
    for name in ('training', 'evaluation', 'reporting', 'runner'):
        assert f'tests/test_dynamics_vector_{name}.py' in preflight['arguments']
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(fits(rows)) == 6


def test_fixture_ignores_inherited_vector_flag(tmp_path, monkeypatch):
    monkeypatch.setenv('INCLUDE_VECTOR_ONLY', '1')
    env, _ = setup_runner(tmp_path)
    assert env['INCLUDE_VECTOR_ONLY'] == '0'
