"""Observed-transition v2 is opt-in, Dynamics-only, and separately auditable."""
import csv
import json
from pathlib import Path
import subprocess

import pytest

from climate_manifold.downstream.train import parser
from test_dynamics_consistency_runner import fits, evaluations
from test_full_manifold_matrix_runner import ARMS, setup_full
from test_full_manifold_matrix_orchestration import setup_orchestration, run_orchestration, LOSSES
from test_fusion_runner import ROOT, arg, fixture_inputs, run_script, setup_runner


def test_v2_applies_to_nine_dynamics_arms_and_preserves_three_baselines(tmp_path):
    env, log = setup_full(tmp_path)
    result, calls = run_script(env, log, DYNAMICS_VERSION='v2', TRANSITION_WEIGHT='0.25',
                              TRANSITION_NEIGHBORS='4', TRANSITION_BANK_SIZE='256',
                              TRANSITION_EXCLUSION_STEPS='12')
    assert result.returncode == 0, result.stderr
    commands = fits(calls)
    assert len(commands) == 12 and len(evaluations(calls)) == 24
    configs = [parser().parse_args(r[2:]) for r in commands]
    for command, config in zip(commands[:3], configs[:3]):
        assert '--dynamics-version' not in command
        assert '--dynamics-information' not in command
        assert not any(x.startswith('--transition-') for x in command)
        assert config.dynamics_encoder == 'off' and config.dynamics_version == 'v1'
    for arm, command, config in zip(ARMS[3:], commands[3:], configs[3:]):
        assert arg(command, '--dynamics-version') == config.dynamics_version == 'v2'
        assert arg(command, '--dynamics-information') == config.dynamics_information == 'geopotential_terrain'
        assert config.transition_weight == 0.25 and config.transition_neighbors == 4
        assert config.transition_bank_size == 256 and config.transition_exclusion_steps == 12
        assert config.transition_endpoint_temperature == config.transition_relation_temperature == 1
        assert config.transition_latent_temperature == 1
        assert config.transition_history_weight == config.transition_future_weight == 1
        assert config.transition_geometry_weight == 0.1
        assert (config.transition_pool_lat, config.transition_pool_lon) == (4, 8)
        assert config.guide_interaction == ('transformer' if '_two_stage_' in arm else 'off')
        assert (config.dynamics_statistical_weight == 0) == arm.endswith('vector_only')
    manifest = json.loads((Path(env['RUN'])/'run-manifest.json').read_text())
    assert manifest['settings']['dynamics_version'] == 'v2'
    assert manifest['settings']['dynamics_information'] == 'geopotential_terrain'
    assert '--transition-weight' in manifest['arguments']


def test_v1_ignores_transition_knobs_without_changing_training_or_resume(tmp_path):
    env, log = setup_full(tmp_path)
    result, calls = run_script(env, log, TRANSITION_WEIGHT='not-active', TRANSITION_NEIGHBORS='999')
    assert result.returncode == 0, result.stderr
    assert len(fits(calls)) == 12
    assert all('--dynamics-version' not in r and not any(x.startswith('--transition-') for x in r)
               for r in fits(calls))
    result, calls = run_script(env, log, RESUME='1', TRANSITION_WEIGHT='55', TRANSITION_NEIGHBORS='3')
    assert result.returncode == 0, result.stderr
    assert not fits(calls) and not evaluations(calls)


@pytest.mark.parametrize('settings,message', [
    ({'DYNAMICS_VERSION': 'v3'}, 'DYNAMICS_VERSION must be v1 or v2'),
    ({'DYNAMICS_INFORMATION': 'unknown'}, 'Unknown DYNAMICS_INFORMATION'),
    ({'DYNAMICS_VERSION': 'v1', 'DYNAMICS_INFORMATION': 'geopotential_terrain'},
     'requires DYNAMICS_VERSION=v2'),
    ({'DYNAMICS_VERSION': 'v2', 'TRANSITION_WEIGHT': '-1'}, '--transition-weight has an invalid value'),
    ({'DYNAMICS_VERSION': 'v2', 'TRANSITION_LATENT_TEMPERATURE': 'nan'}, '--transition-latent-temperature'),
    ({'DYNAMICS_VERSION': 'v2', 'TRANSITION_BANK_SIZE': '3'}, 'TRANSITION_NEIGHBORS must not exceed'),
    ({'DYNAMICS_VERSION': 'v2', 'TRANSITION_POOL_LAT': '0'}, '--transition-pool-lat'),
    ({'DYNAMICS_VERSION': 'v2', 'TRANSITION_NEIGHBORS': '1'}, '--transition-neighbors'),
    ({'DYNAMICS_VERSION': 'v2', 'TRANSITION_HISTORY_WEIGHT': '0', 'TRANSITION_FUTURE_WEIGHT': '0'},
     'At least one of TRANSITION_HISTORY_WEIGHT / TRANSITION_FUTURE_WEIGHT must be positive'),
])
def test_invalid_v2_configuration_fails_before_fitting_any_baseline(tmp_path, settings, message):
    env, log = setup_full(tmp_path)
    result, calls = run_script(env, log, **settings)
    assert result.returncode != 0 and message in result.stderr
    assert not fits(calls)


def test_v2_requires_dynamics_enabled(tmp_path):
    env, log = setup_runner(tmp_path)
    result, calls = run_script(env, log, DYNAMICS_VERSION='v2')
    assert result.returncode != 0 and 'requires INCLUDE_DYNAMICS=1' in result.stderr
    assert not fits(calls)


def test_resume_rejects_version_and_transition_distribution_changes(tmp_path):
    env, log = setup_full(tmp_path)
    result, _ = run_script(env, log, DYNAMICS_VERSION='v2')
    assert result.returncode == 0, result.stderr
    result, calls = run_script(env, log, DYNAMICS_VERSION='v2', RESUME='1')
    assert result.returncode == 0, result.stderr
    assert not fits(calls) and not evaluations(calls)
    for changes in ({'DYNAMICS_VERSION': 'v1'}, {'TRANSITION_FUTURE_WEIGHT': '0.25'},
                    {'DYNAMICS_INFORMATION': 'off'}):
        result, calls = run_script(env, log, **{'DYNAMICS_VERSION': 'v2', 'RESUME': '1', **changes})
        assert result.returncode != 0 and 'differs' in result.stderr
        assert not fits(calls) and not evaluations(calls)


def test_daily_preflight_isolates_v2_options_while_real_experiment_keeps_them(tmp_path):
    env, log = setup_full(tmp_path)
    daily = tmp_path/'daily'
    (daily/'prepared').mkdir(parents=True)
    fixture_inputs(daily/'prepared')
    env.pop('ARCHIVE'); env.pop('INFO')
    env.update(DAILY_WORK=str(daily), ERA5_ROOT=str(tmp_path/'raw'), DEVICE='cpu',
               RUN=str(daily/'runs'/'v2'), RUN_PREFLIGHT_TESTS='1', EVALUATE_TEST='0',
               DYNAMICS_VERSION='v2', DYNAMICS_INFORMATION='geopotential_terrain',
               TRANSITION_WEIGHT='0.2', TRANSITION_EXCLUSION_STEPS='5',
               PYTEST_ENV_LOG=str(tmp_path/'pytest-env.json'))
    with Path(env['PYTHON']).open('a') as stream:
        stream.write('''
if '-m' in args and value('-m') == 'pytest':
    Path(os.environ['PYTEST_ENV_LOG']).write_text(json.dumps({
        'version': os.environ['DYNAMICS_VERSION'], 'information': os.environ['DYNAMICS_INFORMATION'], 'arguments': args,
        'transition': {k:v for k,v in os.environ.items() if k.startswith('TRANSITION_')}}))
''')
    result = subprocess.run(['bash', str(ROOT/'scripts/run_daily_manifold_fusion.sh')],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    preflight = json.loads(Path(env['PYTEST_ENV_LOG']).read_text())
    assert preflight['version'] == 'v1' and preflight['transition'] == {}
    assert preflight['information'] == 'auto'
    for name in ('test_transition_measure.py', 'test_transition_v2_training.py',
                 'test_transition_v2_reporting.py', 'test_transition_v2_runner.py',
                 'test_dynamics_information.py', 'test_transition_v2_information_training.py'):
        assert 'tests/'+name in preflight['arguments']
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(fits(calls)) == 12
    assert all(arg(row, '--transition-weight') == '0.2' for row in fits(calls)[3:])


@pytest.mark.parametrize('suite,enabled', [('guided_transformer', '1'), ('manifold_fusion', '0')])
def test_daily_v2_configuration_guard_runs_before_workspace_or_data_preparation(tmp_path, suite, enabled):
    env, log = setup_runner(tmp_path)
    env.update(DAILY_WORK=str(tmp_path/'daily'), ERA5_ROOT=str(tmp_path/'raw'),
               SUITE=suite, INCLUDE_DYNAMICS=enabled, DYNAMICS_VERSION='v2')
    log.write_text('')
    result = subprocess.run(['bash', str(ROOT/'scripts/run_daily_guided_transformer.sh')],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert 'requires SUITE=manifold_fusion and INCLUDE_DYNAMICS=1' in result.stderr
    assert log.read_text() == '' and not (tmp_path/'daily').exists()


def test_fixture_clears_inherited_transition_configuration(tmp_path, monkeypatch):
    monkeypatch.setenv('DYNAMICS_VERSION', 'v2')
    monkeypatch.setenv('DYNAMICS_INFORMATION', 'geopotential_terrain')
    monkeypatch.setenv('TRANSITION_WEIGHT', '0.9')
    monkeypatch.setenv('TRANSITION_FUTURE_SETTING', 'unexpected')
    env, _ = setup_runner(tmp_path)
    assert env['DYNAMICS_VERSION'] == 'v1'
    assert env['DYNAMICS_INFORMATION'] == 'auto'
    assert not any(key.startswith('TRANSITION_') for key in env)


@pytest.mark.parametrize('information,suite_prefix,summary_name', [
    ('auto', 'full-v2-', 'full-matrix-v2-summary.csv'),
    ('off', 'full-v2-surface-', 'full-matrix-v2-surface-summary.csv'),
])
def test_full_v2_suite_paths_and_summary_do_not_overwrite_v1(tmp_path, information, suite_prefix, summary_name):
    env, log, daily = setup_orchestration(tmp_path)
    # Preserve representative older outputs while choosing the same DAILY_WORK.
    previous = daily/'runs'/'full-w2'
    previous.mkdir(parents=True)
    sentinel = previous/'comparison.json'
    sentinel.write_text('previous-v1-result')
    summary_v1 = daily/'full-matrix-summary.csv'
    summary_v1.write_text('previous-v1-summary')
    other_v2 = daily/'runs'/('full-v2-w2' if information == 'off' else 'full-v2-surface-w2')
    other_v2.mkdir(parents=True)
    (other_v2/'comparison.json').write_text('other-v2-information-mode')
    result, calls = run_orchestration(env, log, DYNAMICS_VERSION='v2', DYNAMICS_INFORMATION=information)
    assert result.returncode == 0, result.stderr
    assert len(fits(calls)) == 36 and len(evaluations(calls)) == 72
    assert sentinel.read_text() == 'previous-v1-result'
    assert summary_v1.read_text() == 'previous-v1-summary'
    assert (other_v2/'comparison.json').read_text() == 'other-v2-information-mode'
    assert {Path(arg(r, '--output')).parent.name for r in fits(calls)} == {
        suite_prefix+loss for loss in LOSSES}
    expected = 'geopotential_terrain' if information == 'auto' else 'off'
    dynamics = [r for r in fits(calls) if arg(r, '--dynamics-encoder') != 'off']
    assert len(dynamics) == 27
    assert all(arg(r, '--dynamics-information') == expected for r in dynamics)
    with (daily/summary_name).open(newline='') as stream:
        summary = list(csv.DictReader(stream))
    assert len(summary) == 72
    assert sum(row['dynamics_version'] == 'v2' for row in summary) == 54
    assert sum(row['dynamics_version'] == 'v1' for row in summary) == 18
    assert sum(row['dynamics_information'] == 'geopotential_terrain' for row in summary) == (
        54 if information == 'auto' else 0)


def test_daily_rejects_v1_information_before_data_work(tmp_path):
    env, log = setup_runner(tmp_path)
    env.update(DAILY_WORK=str(tmp_path/'daily'), ERA5_ROOT=str(tmp_path/'raw'),
               SUITE='manifold_fusion', INCLUDE_DYNAMICS='1', DYNAMICS_VERSION='v1',
               DYNAMICS_INFORMATION='geopotential_terrain')
    log.write_text('')
    result = subprocess.run(['bash', str(ROOT/'scripts/run_daily_guided_transformer.sh')],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode != 0 and 'requires DYNAMICS_VERSION=v2' in result.stderr
    assert log.read_text() == '' and not (tmp_path/'daily').exists()
