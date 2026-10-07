"""Matched field/hybrid fits, diagnostic coverage, and resume provenance."""
import json
from pathlib import Path
import subprocess

import pytest

from test_fusion_runner import ROOT, arg, fixture_inputs, run_script, setup_runner
from climate_manifold.downstream.train import parser


def fits(rows):
    return [r for r in rows if 'climate_manifold.downstream.train' in r]


def evaluations(rows):
    return [r for r in rows if 'climate_manifold.downstream.evaluate' in r]


def setup_consistency(tmp_path, *, step=24):
    env, log = setup_runner(tmp_path, step=step)
    # Simulate only artifact creation; production config/rollout checks are
    # exercised by the integration tests, while this suite audits shell routing.
    with Path(env['PYTHON']).open('a') as stream:
        stream.write('''
if '-m' in args and value('-m').endswith('.evaluate') and '--dynamics-diagnostics' in args:
    path = Path(value('--output'))
    report = json.loads(path.read_text())
    manifest = json.loads((path.parent/'run-manifest.json').read_text())
    settings = manifest['settings']
    report['config'] = {'dynamics_auxiliary_architecture': 'latent_rollout'}
    report['dynamics_auxiliary_config'] = {
        'supervision': 'field_latent' if '_field_latent-' in value('--checkpoint') else 'field'}
    report['dynamics_evaluation_contract'] = {
        'version': 'climate_manifold.dynamics_evaluation.v1',
        'rollout_steps': int(settings['consistency_rollout_steps']),
        'step_hours': float(settings['consistency_step_hours']),
        'target_semantics': 'same_length_strided_truth_window',
        'main_forecast_horizon_steps': int(settings['consistency_train_horizon'])}
    path.write_text(json.dumps(report))
''')
    env.update(INCLUDE_DYNAMICS='1', INCLUDE_DYNAMICS_CONSISTENCY='1', CONSISTENCY_ONLY='1',
               INCLUDE_ZERO_DYNAMICS='0', MODELS='mlp', SEEDS='7', MAKE_PLOTS='0')
    return env, log


def test_four_matched_arms_with_identical_field_objectives_and_diagnostics(tmp_path):
    env, log = setup_consistency(tmp_path)
    result, rows = run_script(env, log, MODELS='mlp neural_ode', DYNAMICS_LATENT_WEIGHT='0.2',
                              AUX_ROLLOUT_STEPS='10', STATISTICAL_LOSS='signed_measure')
    assert result.returncode == 0, result.stderr
    assert len(fits(rows)) == 8 and len(evaluations(rows)) == 16
    assert 'Fits: 8' in result.stdout
    for family in ('mlp', 'neural_ode'):
        configs = [parser().parse_args(r[2:]) for r in fits(rows) if arg(r, '--model') == family]
        assert [a.bridge for a in configs] == ['latent', 'latent', 'guided', 'guided']
        assert [a.dynamics_supervision for a in configs] == ['field', 'field_latent']*2
        assert [a.dynamics_latent_weight for a in configs] == [None, 0.2]*2
        for a in configs:
            assert a.dynamics_auxiliary_architecture == 'latent_rollout'
            assert a.dynamics_reconstruction_weight == a.dynamics_statistical_weight == 0.1
            assert a.dynamics_encoder == 'conv_gru' and a.dynamics_mode == 'learned'
            assert a.guide_interaction == 'off' and a.batch_size == 16
            assert a.history_stride == 1 and a.horizon_steps == 5
            assert a.dynamics_statistical_loss == 'signed_measure'
    assert all('--dynamics-diagnostics' in r and arg(r, '--dynamics-rollout-steps') == '10'
               for r in evaluations(rows))
    last_fit = max(i for i, r in enumerate(rows) if r in fits(rows))
    assert min(i for i, r in enumerate(rows) if '--split' in r and arg(r, '--split') == 'test') > last_fit
    manifest = json.loads((Path(env['RUN'])/'run-manifest.json').read_text())
    assert manifest['settings']['consistency_rollout_steps'] == '10'
    assert 'scripts/run_dynamics_consistency_comparison.sh' in manifest['source']
    assert '--dynamics-latent-weight' in manifest['arguments']


def test_optional_two_stage_adds_matched_pair_without_legacy_bypass(tmp_path):
    env, log = setup_consistency(tmp_path)
    result, rows = run_script(env, log, INCLUDE_TWO_STAGE='1')
    assert result.returncode == 0, result.stderr
    assert len(fits(rows)) == 6 and len(evaluations(rows)) == 12
    configs = [parser().parse_args(r[2:]) for r in fits(rows)]
    assert [a.guide_interaction for a in configs] == ['off']*4+['transformer']*2
    assert all(a.bridge == 'guided' for a in configs[-2:])
    assert not any('--guide-interaction-override' in r for r in rows)


def test_optional_consistency_arms_coexist_with_historical_direct_arms(tmp_path):
    env, log = setup_consistency(tmp_path)
    result, rows = run_script(env, log, CONSISTENCY_ONLY='0', INCLUDE_ZERO_DYNAMICS='1',
                              EVALUATE_TEST='0')
    assert result.returncode == 0, result.stderr
    assert len(fits(rows)) == 12
    configs = [parser().parse_args(r[2:]) for r in fits(rows)]
    assert [a.dynamics_auxiliary_architecture for a in configs] == ['direct']*8+['latent_rollout']*4
    assert sum('--dynamics-diagnostics' in r for r in evaluations(rows)) == 4


def test_six_hour_archive_uses_matching_step_and_horizon_contract(tmp_path):
    env, log = setup_consistency(tmp_path, step=6)
    result, rows = run_script(env, log, EVALUATE_TEST='0')
    assert result.returncode == 0, result.stderr
    settings = json.loads((Path(env['RUN'])/'run-manifest.json').read_text())['settings']
    assert settings['consistency_train_horizon'] == settings['consistency_rollout_steps'] == '20'
    assert settings['consistency_step_hours'] == '6'
    assert all(arg(r, '--history-stride') == '4' for r in fits(rows))


@pytest.mark.parametrize('settings,message', [
    ({'INCLUDE_DYNAMICS_CONSISTENCY': '2'}, 'must be 0 or 1'),
    ({'CONSISTENCY_ONLY': '2'}, 'must be 0 or 1'),
    ({'INCLUDE_DYNAMICS': '0'}, 'requires INCLUDE_DYNAMICS=1'),
    ({'INCLUDE_DYNAMICS_CONSISTENCY': '0'}, 'requires INCLUDE_DYNAMICS_CONSISTENCY=1'),
    ({'INCLUDE_TWO_STAGE': '1', 'INCLUDE_TWO_STAGE_BYPASS': '1'}, 'requires INCLUDE_TWO_STAGE_BYPASS=0'),
    ({'DYNAMICS_LATENT_WEIGHT': '0'}, 'finite and positive'),
    ({'DYNAMICS_LATENT_WEIGHT': 'nan'}, 'finite and positive'),
    ({'DYNAMICS_RECONSTRUCTION_WEIGHT': '0'}, 'finite and positive'),
    ({'DYNAMICS_STATISTICAL_WEIGHT': '-1'}, 'finite and positive'),
    ({'AUX_ROLLOUT_STEPS': '4'}, 'integer >= HORIZON_STEPS'),
    ({'AUX_ROLLOUT_STEPS': '5.5'}, 'integer >= HORIZON_STEPS'),
    ({'AUX_ROLLOUT_STEPS': '10', 'CONSISTENCY_ONLY': '0'}, 'requires CONSISTENCY_ONLY=1'),
])
def test_invalid_comparison_never_fits(tmp_path, settings, message):
    env, log = setup_consistency(tmp_path)
    result, rows = run_script(env, log, **settings)
    assert result.returncode != 0 and message in result.stderr
    assert not fits(rows)


@pytest.mark.parametrize('settings', [
    {'DYNAMICS_LATENT_WEIGHT': '0.2'}, {'AUX_ROLLOUT_STEPS': '10'},
    {'CONSISTENCY_ONLY': '0'}, {'INCLUDE_TWO_STAGE': '1'},
])
def test_resume_refuses_changed_loss_architecture_or_diagnostic_coverage(tmp_path, settings):
    env, log = setup_consistency(tmp_path)
    result, _ = run_script(env, log)
    assert result.returncode == 0, result.stderr
    result, rows = run_script(env, log, RESUME='1', **settings)
    assert result.returncode != 0 and 'differs' in result.stderr
    assert not fits(rows) and not evaluations(rows)


def test_resume_repairs_only_missing_eval_and_checks_diagnostic_contract(tmp_path):
    env, log = setup_consistency(tmp_path)
    result, _ = run_script(env, log)
    assert result.returncode == 0, result.stderr
    result, rows = run_script(env, log, RESUME='1')
    assert result.returncode == 0, result.stderr
    assert not fits(rows) and not evaluations(rows)
    missing = Path(env['RUN'])/'mlp-guided_dynamics_field_latent-seed7.test.json'
    missing.unlink()
    result, rows = run_script(env, log, RESUME='1')
    assert result.returncode == 0, result.stderr
    assert not fits(rows) and len(evaluations(rows)) == 1
    report = json.loads(missing.read_text())
    report['dynamics_evaluation_contract']['rollout_steps'] = 999
    missing.write_text(json.dumps(report))
    result, rows = run_script(env, log, RESUME='1')
    assert result.returncode != 0 and 'consistency report/configuration mismatch' in result.stderr
    assert not fits(rows) and not evaluations(rows)


def test_wrapper_defaults_select_only_four_arms_signed_measure_batch16(tmp_path):
    env, log = setup_consistency(tmp_path)
    for name in ('INCLUDE_DYNAMICS', 'INCLUDE_DYNAMICS_CONSISTENCY', 'CONSISTENCY_ONLY',
                 'STATISTICAL_LOSS', 'DYNAMICS_STATISTICAL_LOSS'):
        env.pop(name, None)
    result = subprocess.run(['bash', str(ROOT/'scripts/run_dynamics_consistency_comparison.sh')],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(fits(rows)) == 4
    assert all(arg(r, '--batch-size') == '16' and arg(r, '--statistical-loss') == 'signed_measure'
               for r in fits(rows))


def test_fixture_discards_inherited_consistency_options(tmp_path, monkeypatch):
    for name, value in {'INCLUDE_DYNAMICS_CONSISTENCY': '1', 'CONSISTENCY_ONLY': '1',
                        'DYNAMICS_LATENT_WEIGHT': '0.9', 'AUX_ROLLOUT_STEPS': '20'}.items():
        monkeypatch.setenv(name, value)
    env, _ = setup_runner(tmp_path)
    assert env['INCLUDE_DYNAMICS_CONSISTENCY'] == env['CONSISTENCY_ONLY'] == '0'
    assert 'DYNAMICS_LATENT_WEIGHT' not in env and 'AUX_ROLLOUT_STEPS' not in env


def test_daily_preflight_preserves_real_consistency_matrix(tmp_path):
    env, log = setup_consistency(tmp_path)
    daily = tmp_path/'daily'
    (daily/'prepared').mkdir(parents=True)
    fixture_inputs(daily/'prepared')
    env.pop('ARCHIVE')
    env.pop('INFO')
    env.update(DAILY_WORK=str(daily), ERA5_ROOT=str(tmp_path/'raw'), DEVICE='cpu',
               RUN=str(daily/'runs'/'consistency'), RUN_PREFLIGHT_TESTS='1', EVALUATE_TEST='0',
               PYTEST_ENV_LOG=str(tmp_path/'pytest-env.json'))
    with Path(env['PYTHON']).open('a') as stream:
        stream.write('''
if '-m' in args and value('-m') == 'pytest':
    Path(os.environ['PYTEST_ENV_LOG']).write_text(json.dumps({
        'environment': {name: os.environ[name] for name in ('INCLUDE_DYNAMICS_CONSISTENCY', 'CONSISTENCY_ONLY')},
        'arguments': args}))
''')
    result = subprocess.run(['bash', str(ROOT/'scripts/run_dynamics_consistency_comparison.sh')],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    preflight = json.loads(Path(env['PYTEST_ENV_LOG']).read_text())
    assert preflight['environment'] == {'INCLUDE_DYNAMICS_CONSISTENCY': '0', 'CONSISTENCY_ONLY': '0'}
    assert 'tests/test_dynamics_consistency_runner.py' in preflight['arguments']
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(fits(rows)) == 4


def test_daily_wrapper_preserves_write_boundary(tmp_path):
    env, log = setup_consistency(tmp_path)
    env.pop('ARCHIVE')
    env.pop('INFO')
    env.update(DAILY_WORK=str(tmp_path/'daily'), ERA5_ROOT=str(tmp_path/'raw'), RUN=str(tmp_path/'outside'))
    result = subprocess.run(['bash', str(ROOT/'scripts/run_dynamics_consistency_comparison.sh')],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode != 0 and 'RUN must be a subdirectory' in result.stderr
