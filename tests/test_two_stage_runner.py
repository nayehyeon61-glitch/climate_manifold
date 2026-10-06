"""Two-stage fits and checkpoint bypasses must remain distinct and resumable."""
import json
from pathlib import Path
import subprocess

import pytest

from test_fusion_runner import ROOT, arg, fixture_inputs, run_script, setup_runner
from climate_manifold.downstream.train import parser


def setup_two_stage(tmp_path):
    env, log = setup_runner(tmp_path)
    shim = Path(env['PYTHON'])
    # Extend the recording shim with production evaluation provenance, without
    # requiring ERA5 or invoking the actual training/evaluation network.
    with shim.open('a') as stream:
        stream.write('''
if '-m' in args and value('-m').endswith('.evaluate'):
    if 'guided_dynamics_two_stage-' in value('--checkpoint'):
        path = Path(value('--output'))
        report = json.loads(path.read_text())
        bypassed = '--guide-interaction-override' in args
        report['guide_interaction_evaluation'] = {
            'requested': 'off' if bypassed else 'checkpoint',
            'trained': 'transformer', 'effective': 'off' if bypassed else 'transformer',
            'bypassed': bypassed}
        path.write_text(json.dumps(report))
''')
    env.update(MODELS='mlp', SEEDS='7', INCLUDE_DYNAMICS='1', INCLUDE_TWO_STAGE='1',
               INCLUDE_TWO_STAGE_BYPASS='1', GUIDE_INTERACTION_DEPTH='1', MAKE_PLOTS='0')
    return env, log


def fits(rows):
    return [r for r in rows if 'climate_manifold.downstream.train' in r]


def evaluations(rows):
    return [r for r in rows if 'climate_manifold.downstream.evaluate' in r]


def test_two_stage_is_one_extra_fit_and_bypass_is_evaluation_only(tmp_path):
    env, log = setup_two_stage(tmp_path)
    result, rows = run_script(env, log, MODELS='mlp neural_ode', GUIDE_INTERACTION_DEPTH='2')
    assert result.returncode == 0, result.stderr
    assert len(fits(rows)) == 18
    assert len(evaluations(rows)) == 40  # 9 fits + 1 bypass, both splits, 2 models.
    assert 'Fits: 18' in result.stdout
    for family in ('mlp', 'neural_ode'):
        group = [r for r in fits(rows) if arg(r, '--model') == family]
        configs = [parser().parse_args(r[2:]) for r in group]
        assert [a.guide_interaction for a in configs] == ['off']*8+['transformer']
        extra = configs[-1]
        assert extra.guide_interaction_depth == 2
        assert extra.bridge == 'guided' and extra.guide_architecture == 'fusion'
        assert extra.dynamics_encoder == 'conv_gru' and extra.dynamics_mode == 'learned'
        assert extra.batch_size == 16
    bypasses = [r for r in evaluations(rows) if '--guide-interaction-override' in r]
    assert len(bypasses) == 4
    for row in bypasses:
        assert arg(row, '--guide-interaction-override') == 'off'
        assert '-guided_dynamics_two_stage-seed7.pt' in arg(row, '--checkpoint')
        assert '-guided_dynamics_two_stage_bypass-seed7.' in arg(row, '--output')
        assert '-guided_dynamics_two_stage_bypass-seed7.' in arg(row, '--forecast-output')
    test_start = min(i for i, r in enumerate(rows) if '--split' in r and arg(r, '--split') == 'test')
    assert test_start > max(i for i, r in enumerate(rows) if r in fits(rows))
    comparisons = [r for r in rows if 'climate_manifold.downstream.compare' in r]
    assert all(len(r[r.index('--reports')+1:r.index('--output')]) == 20 for r in comparisons)
    manifest = json.loads((Path(env['RUN'])/'run-manifest.json').read_text())
    assert manifest['settings']['arms'].endswith('guided_dynamics_two_stage')
    assert manifest['settings']['include_two_stage_bypass'] == '1'
    assert '--guide-interaction-depth' in manifest['arguments']
    assert 'scripts/run_two_stage_manifold_comparison.sh' in manifest['source']


def test_bypass_is_optional_and_old_dynamics_arms_keep_off(tmp_path):
    env, log = setup_two_stage(tmp_path)
    result, rows = run_script(env, log, INCLUDE_TWO_STAGE_BYPASS='0', EVALUATE_TEST='0')
    assert result.returncode == 0, result.stderr
    assert len(fits(rows)) == len(evaluations(rows)) == 9
    assert not any('--guide-interaction-override' in r for r in rows)


@pytest.mark.parametrize('changed', [
    {'GUIDE_INTERACTION_DEPTH': '2'},
    {'INCLUDE_TWO_STAGE_BYPASS': '0'},
    {'INCLUDE_TWO_STAGE': '0', 'INCLUDE_TWO_STAGE_BYPASS': '0'},
])
def test_resume_refuses_two_stage_experiment_changes(tmp_path, changed):
    env, log = setup_two_stage(tmp_path)
    result, _ = run_script(env, log)
    assert result.returncode == 0, result.stderr
    before = {p.name: p.read_bytes() for p in Path(env['RUN']).glob('*.pt')}
    result, rows = run_script(env, log, RESUME='1', **changed)
    assert result.returncode != 0 and 'differs' in result.stderr
    assert not fits(rows) and not evaluations(rows)
    assert before == {p.name: p.read_bytes() for p in Path(env['RUN']).glob('*.pt')}


def test_resume_reuses_fits_and_only_repairs_missing_bypass(tmp_path):
    env, log = setup_two_stage(tmp_path)
    result, _ = run_script(env, log)
    assert result.returncode == 0, result.stderr
    result, rows = run_script(env, log, RESUME='1')
    assert result.returncode == 0, result.stderr
    assert not fits(rows) and not evaluations(rows)
    missing = Path(env['RUN'])/'mlp-guided_dynamics_two_stage_bypass-seed7.test.json'
    missing.unlink()
    result, rows = run_script(env, log, RESUME='1')
    assert result.returncode == 0, result.stderr
    assert not fits(rows) and len(evaluations(rows)) == 1
    assert arg(evaluations(rows)[0], '--output') == str(missing)
    assert '--guide-interaction-override' in evaluations(rows)[0]


@pytest.mark.parametrize('corruption', ['checkpoint', 'evaluation'])
def test_resume_checks_bypass_checkpoint_and_execution_contract(tmp_path, corruption):
    env, log = setup_two_stage(tmp_path)
    result, _ = run_script(env, log)
    assert result.returncode == 0, result.stderr
    path = Path(env['RUN'])/'mlp-guided_dynamics_two_stage_bypass-seed7.test.json'
    data = json.loads(path.read_text())
    if corruption == 'checkpoint':
        data['checkpoint_sha256'] = 'wrong-checkpoint'
    else:
        data['guide_interaction_evaluation']['effective'] = 'transformer'
    path.write_text(json.dumps(data))
    result, rows = run_script(env, log, RESUME='1')
    assert result.returncode != 0 and 'mismatch' in result.stderr
    assert not fits(rows) and not evaluations(rows)


@pytest.mark.parametrize('settings,message', [
    ({'INCLUDE_TWO_STAGE': '2'}, 'INCLUDE_TWO_STAGE must be'),
    ({'INCLUDE_TWO_STAGE_BYPASS': '2'}, 'INCLUDE_TWO_STAGE_BYPASS must be'),
    ({'INCLUDE_DYNAMICS': '0'}, 'requires INCLUDE_DYNAMICS=1'),
    ({'INCLUDE_TWO_STAGE': '0'}, 'requires INCLUDE_TWO_STAGE=1'),
    ({'GUIDE_INTERACTION_DEPTH': '0'}, 'positive integer'),
    ({'GUIDE_INTERACTION_DEPTH': '1.5'}, 'positive integer'),
])
def test_invalid_two_stage_settings_never_train(tmp_path, settings, message):
    env, log = setup_two_stage(tmp_path)
    result, rows = run_script(env, log, **settings)
    assert result.returncode != 0 and message in result.stderr
    assert not fits(rows)


@pytest.mark.parametrize('enabled,count', [('1', 9), ('0', 8)])
def test_two_stage_wrapper_defaults_and_disable(tmp_path, enabled, count):
    env, log = setup_two_stage(tmp_path)
    for name in ('INCLUDE_DYNAMICS', 'INCLUDE_TWO_STAGE_BYPASS', 'STATISTICAL_LOSS',
                 'DYNAMICS_STATISTICAL_LOSS'):
        env.pop(name, None)
    env.update(INCLUDE_TWO_STAGE=enabled, EVALUATE_TEST='0')
    result = subprocess.run(['bash', str(ROOT/'scripts/run_two_stage_manifold_comparison.sh')],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(fits(rows)) == count
    assert len(evaluations(rows)) == count+(enabled == '1')
    assert arg(fits(rows)[-1], '--statistical-loss') == 'signed_measure'


def test_two_stage_daily_wrapper_retains_server_write_boundary(tmp_path):
    env, log = setup_two_stage(tmp_path)
    env.pop('ARCHIVE')
    env.pop('INFO')
    env.update(DAILY_WORK=str(tmp_path/'daily'), ERA5_ROOT=str(tmp_path/'raw'),
               RUN=str(tmp_path/'outside'))
    result = subprocess.run(['bash', str(ROOT/'scripts/run_two_stage_manifold_comparison.sh')],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode != 0 and 'RUN must be a subdirectory' in result.stderr
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert not fits(rows)


def test_daily_preflight_isolates_tests_without_disabling_real_experiment(tmp_path):
    env, log = setup_two_stage(tmp_path)
    daily = tmp_path/'daily'
    (daily/'prepared').mkdir(parents=True)
    fixture_inputs(daily/'prepared')
    env.pop('ARCHIVE')
    env.pop('INFO')
    env.update(DAILY_WORK=str(daily), ERA5_ROOT=str(tmp_path/'raw'), DEVICE='cpu',
               RUN=str(daily/'runs'/'two-stage'), RUN_PREFLIGHT_TESTS='1',
               EVALUATE_TEST='0', PYTEST_ENV_LOG=str(tmp_path/'pytest-env.json'))
    with Path(env['PYTHON']).open('a') as stream:
        stream.write('''
if '-m' in args and value('-m') == 'pytest':
    names = ('INCLUDE_DYNAMICS', 'INCLUDE_TWO_STAGE', 'INCLUDE_TWO_STAGE_BYPASS',
             'GUIDE_INTERACTION_DEPTH')
    Path(os.environ['PYTEST_ENV_LOG']).write_text(json.dumps({
        'environment': {name: os.environ[name] for name in names}, 'arguments': args}))
''')
    result = subprocess.run(['bash', str(ROOT/'scripts/run_two_stage_manifold_comparison.sh')],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    preflight = json.loads(Path(env['PYTEST_ENV_LOG']).read_text())
    assert preflight['environment'] == {
        'INCLUDE_DYNAMICS': '0', 'INCLUDE_TWO_STAGE': '0',
        'INCLUDE_TWO_STAGE_BYPASS': '0', 'GUIDE_INTERACTION_DEPTH': '1'}
    assert 'tests/test_two_stage_runner.py' in preflight['arguments']
    assert 'tests/test_guide_interaction.py' in preflight['arguments']
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(fits(rows)) == 9 and len(evaluations(rows)) == 10
    assert arg(fits(rows)[-1], '--guide-interaction') == 'transformer'


def test_runner_fixture_ignores_exported_optional_architecture(tmp_path, monkeypatch):
    for name, value in {'INCLUDE_DYNAMICS': '1', 'INCLUDE_TWO_STAGE': '1',
                        'INCLUDE_TWO_STAGE_BYPASS': '1', 'GUIDE_INTERACTION_DEPTH': '3',
                        'DYNAMICS_ENCODER': 'temporal_conv',
                        'DYNAMICS_STATISTICAL_LOSS': 'signed_measure',
                        'DYNAMICS_CHANNELS': '64'}.items():
        monkeypatch.setenv(name, value)
    env, _ = setup_runner(tmp_path)
    assert env['INCLUDE_DYNAMICS'] == env['INCLUDE_TWO_STAGE'] == '0'
    assert env['INCLUDE_TWO_STAGE_BYPASS'] == '0' and env['GUIDE_INTERACTION_DEPTH'] == '1'
    assert all(name not in env for name in ('DYNAMICS_ENCODER', 'DYNAMICS_STATISTICAL_LOSS',
                                          'DYNAMICS_CHANNELS'))
