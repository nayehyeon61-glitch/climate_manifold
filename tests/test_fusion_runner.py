"""Exercise the Fusion matrix and its resume contract without an ERA5/GPU mount."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from climate_manifold.fusion_run import initialize_run
from climate_manifold.downstream.train import parser


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT/'scripts/run_guided_fusion_comparison.sh'


def fixture_inputs(tmp_path, step=24):
    archive = tmp_path/'surface.npz'
    archive.write_bytes(b'prepared-surface')
    archive.with_suffix('.schema.json').write_text(json.dumps({'forecast_step_hours': step}))
    info = tmp_path/'information.npz'
    info.write_bytes(b'prepared-information')
    return archive, info


def setup_runner(tmp_path, *, step=24):
    archive, info = fixture_inputs(tmp_path, step)
    log = tmp_path/'commands.jsonl'
    shim = tmp_path/'record-python'
    shim.write_text(f'''#!{sys.executable}
import hashlib, json, os, runpy, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ['COMMAND_LOG'], 'a') as stream:
    stream.write(json.dumps(args)+'\\n')
def value(key): return args[args.index(key)+1]
def checksum(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
if '-m' in args:
    module = value('-m')
    if module == 'climate_manifold.fusion_run':
        sys.argv = [module]+args[args.index('-m')+2:]
        runpy.run_module(module, run_name='__main__')
    elif module.endswith('.train'):
        path = Path(value('--output')); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'completed-checkpoint')
        path.with_suffix('.manifest.json').write_text(json.dumps({{'checkpoint_sha256': checksum(path)}}))
    elif module.endswith('.evaluate'):
        Path(value('--output')).write_text(json.dumps({{'checkpoint_sha256': checksum(value('--checkpoint')), 'split': value('--split')}}))
    elif module.endswith('.compare'):
        path = Path(value('--output'))
        path.write_text(json.dumps({{'direct_comparison': {{'effects': [{{'model': 'mlp'}}]}},
                                    'constraint_pair_effects': [], 'statistical_flow_effects': []}}))
        # Production deliberately omits empty effect-table CSVs.
        for suffix in ('.csv','.raw-effects.csv'):
            path.with_suffix(suffix).write_text('header\\n')
    elif module.endswith('.plot_comparison'):
        path = Path(value('--output')); path.mkdir(parents=True, exist_ok=True)
        (path/'manifest.json').write_text('{{}}')
elif args == ['-B', '-']:
    exec(sys.stdin.read(), {{'__name__': '__main__'}})
elif args and args[0] == '-':
    sys.argv = args
    exec(sys.stdin.read(), {{'__name__': '__main__'}})
''')
    shim.chmod(0o755)
    env = {**os.environ, 'PYTHON': str(shim), 'COMMAND_LOG': str(log),
           'ARCHIVE': str(archive), 'INFO': str(info), 'RUN': str(tmp_path/'run'),
           'DEVICE': 'cpu', 'GPU_GUARD': '0', 'SEEDS': '7 19 43',
           'MODELS': 'transformer mlp neural_ode climode convlstm simvp fourcastnet climax',
           'A_CHECKPOINT': '', 'STATISTICAL_FLOW_WEIGHT': '0', 'CONDITIONAL_FLOW_WEIGHT': '0',
           'STATISTICAL_LOSS': 'w2', 'RESUME': '0', 'INCLUDE_ZERO_GUIDE': '1',
           'INCLUDE_GUIDED_FORECAST_ONLY': '0', 'VARIABLE_CONDITIONING': '0',
           'GUIDE_DIRECT_INFORMATION': '1', 'MAKE_PLOTS': '1', 'EVALUATE_TEST': '1'}
    return env, log


def run_script(env, log, **settings):
    log.write_text('')
    result = subprocess.run(['bash', str(SCRIPT)], cwd=ROOT,
                            env={**env, **settings}, text=True, capture_output=True)
    return result, [json.loads(line) for line in log.read_text().splitlines()]


def arg(row, key):
    return row[row.index(key)+1]


def test_default_four_route_matrix_and_daily_time(tmp_path):
    env, log = setup_runner(tmp_path)
    result, rows = run_script(env, log)
    assert result.returncode == 0, result.stderr
    train = [r for r in rows if 'climate_manifold.downstream.train' in r]
    evaluations = [r for r in rows if 'climate_manifold.downstream.evaluate' in r]
    assert len(train) == 96 and len(evaluations) == 192
    assert 'Fits: 96' in result.stdout
    for model in env['MODELS'].split():
        for seed in ('7', '19', '43'):
            group = [r for r in train if arg(r, '--model') == model and arg(r, '--seed') == seed]
            assert [arg(r, '--bridge') for r in group] == ['raw', 'latent', 'guided', 'guided']
            assert [arg(r, '--guide-mode') for r in group[2:]] == ['learned', 'zero']
            assert all(arg(r, '--guide-architecture') == 'fusion' for r in group[2:])
    for row in train:
        parser().parse_args(row[2:])
        assert arg(row, '--batch-size') == '16'
        assert arg(row, '--history-stride') == '1'
        assert arg(row, '--horizon-steps') == '5'
        assert arg(row, '--window-stride') == '1'
    last_fit = max(i for i, row in enumerate(rows) if 'climate_manifold.downstream.train' in row)
    test_indices = [i for i, row in enumerate(rows) if '--split' in row and arg(row, '--split') == 'test']
    assert min(test_indices) > last_fit
    plots = [row for row in rows if 'climate_manifold.downstream.plot_comparison' in row]
    assert [Path(arg(row, '--output')).name.split('-')[0] for row in plots] == ['.validation', '.test']
    assert all((Path(env['RUN'])/'plots'/split/'manifest.json').is_file() for split in ('validation', 'test'))
    contract = json.loads((Path(env['RUN'])/'run-manifest.json').read_text())
    assert contract['settings']['arms'] == 'raw latent guided guided_zero'
    assert contract['inputs']['archive']['files']['surface.npz']['sha256']


def test_optional_forecast_only_arm_and_six_hour_defaults(tmp_path):
    env, log = setup_runner(tmp_path, step=6)
    result, rows = run_script(env, log, MODELS='neural_ode', SEEDS='7',
                              INCLUDE_GUIDED_FORECAST_ONLY='1', STATISTICAL_LOSS='kl_entropy',
                              EVALUATE_TEST='0')
    assert result.returncode == 0, result.stderr
    train = [r for r in rows if 'climate_manifold.downstream.train' in r]
    assert len(train) == 5
    extra = train[-1]
    args = parser().parse_args(extra[2:])
    assert args.bridge == 'guided' and args.guide_mode == 'learned' and args.guide_architecture == 'fusion'
    assert args.regularization == 'none' and args.constraint_pair is None
    for row in train:
        assert arg(row, '--history-stride') == arg(row, '--window-stride') == '4'
        assert arg(row, '--horizon-steps') == '20'
    assert not any('--split' in r and arg(r, '--split') == 'test' for r in rows)


@pytest.mark.parametrize('settings,message', [
    ({'MODELS': 'mlp mlp'}, 'Duplicate model'),
    ({'MODELS': 'graphcast'}, 'Unknown model'),
    ({'STATISTICAL_FLOW_WEIGHT': '1'}, 'must remain 0'),
    ({'INCLUDE_GUIDED_FORECAST_ONLY': '2'}, 'must be 0 or 1'),
    ({'STATISTICAL_LOSS': 'other'}, 'Unknown STATISTICAL_LOSS'),
])
def test_invalid_matrix_has_no_fits(tmp_path, settings, message):
    env, log = setup_runner(tmp_path)
    result, rows = run_script(env, log, **settings)
    assert result.returncode != 0 and message in result.stderr
    assert not any('climate_manifold.downstream.train' in row for row in rows)


def fake_source(tmp_path):
    source = tmp_path/'source'
    (source/'src').mkdir(parents=True)
    (source/'src'/'model.py').write_text('value = 1\n')
    (source/'scripts').mkdir()
    for name in ('run_guided_fusion_comparison.sh', 'run_daily_manifold_fusion.sh', 'run_daily_guided_transformer.sh'):
        (source/'scripts'/name).write_text('#!/bin/bash\n')
    (source/'pyproject.toml').write_text('[project]\nname="test"\n')
    return source


def test_resume_skips_completed_fits_without_removing_outputs(tmp_path):
    env, log = setup_runner(tmp_path)
    settings = {'MODELS': 'mlp', 'SEEDS': '7'}
    result, _ = run_script(env, log, **settings)
    assert result.returncode == 0, result.stderr
    run = Path(env['RUN'])
    before = {p.relative_to(run): p.read_bytes() for p in run.rglob('*') if p.is_file()}
    assert not (run/'comparison.flow-effects.csv').exists()
    result, rows = run_script(env, log, **settings, RESUME='1')
    assert result.returncode == 0, result.stderr
    assert not any('climate_manifold.downstream.train' in r or 'climate_manifold.downstream.evaluate' in r for r in rows)
    assert not any('climate_manifold.downstream.compare' in r for r in rows)
    assert before == {p.relative_to(run): p.read_bytes() for p in run.rglob('*') if p.is_file()}


def test_resume_repairs_partial_analysis_without_refitting(tmp_path):
    env, log = setup_runner(tmp_path)
    settings = {'MODELS': 'mlp', 'SEEDS': '7'}
    result, _ = run_script(env, log, **settings)
    assert result.returncode == 0, result.stderr
    run = Path(env['RUN'])
    checkpoint = run/'mlp-guided-seed7.pt'
    before = checkpoint.read_bytes()
    (run/'comparison.raw-effects.csv').unlink()
    (run/'plots'/'validation'/'manifest.json').unlink()
    (run/'plots'/'validation'/'partial.png').write_bytes(b'preserve-partial-image')
    result, rows = run_script(env, log, **settings, RESUME='1')
    assert result.returncode == 0, result.stderr
    assert not any('climate_manifold.downstream.train' in r or 'climate_manifold.downstream.evaluate' in r for r in rows)
    assert len([r for r in rows if 'climate_manifold.downstream.compare' in r]) == 1
    assert len([r for r in rows if 'climate_manifold.downstream.plot_comparison' in r]) == 1
    assert (run/'comparison.raw-effects.csv').is_file()
    assert (run/'plots'/'validation'/'manifest.json').is_file()
    partials = list((run/'plots').glob('validation.partial-*/partial.png'))
    assert len(partials) == 1 and partials[0].read_bytes() == b'preserve-partial-image'
    assert checkpoint.read_bytes() == before


@pytest.mark.parametrize('changed', ['settings', 'inputs', 'source', 'checkpoint', 'report', 'partial'])
def test_resume_rejects_changed_experiment_without_deleting_checkpoint(tmp_path, changed):
    archive, info = fixture_inputs(tmp_path)
    source = fake_source(tmp_path)
    run = tmp_path/'run'
    settings = {'models': 'mlp', 'seeds': '7', 'arms': 'guided'}
    kwargs = dict(run=run, archive=archive, information=info, source_root=source,
                  settings=settings, arguments=['--epochs', '20'])
    initialize_run(**kwargs, resume=False)
    checkpoint = run/'mlp-guided-seed7.pt'
    checkpoint.write_bytes(b'keep-me')
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model_manifest = checkpoint.with_suffix('.manifest.json')
    model_manifest.write_text(json.dumps({'checkpoint_sha256': checksum}))
    if changed == 'settings':
        kwargs['arguments'] = ['--epochs', '21']
    elif changed == 'inputs':
        info.write_bytes(b'changed-info')
    elif changed == 'source':
        (source/'src'/'model.py').write_text('value = 2\n')
    elif changed == 'checkpoint':
        checkpoint.write_bytes(b'changed-weights')
    elif changed == 'report':
        checkpoint.with_suffix('.validation.json').write_text(json.dumps({'checkpoint_sha256': 'wrong'}))
    else:
        model_manifest.unlink()
    before = checkpoint.read_bytes()
    with pytest.raises(ValueError, match='differs|mismatch|Incomplete'):
        initialize_run(**kwargs, resume=True)
    assert checkpoint.read_bytes() == before


def test_daily_fusion_refuses_run_outside_daily_work_before_preprocessing(tmp_path):
    env, log = setup_runner(tmp_path)
    env.update(DAILY_WORK=str(tmp_path/'daily'), ERA5_ROOT=str(tmp_path/'raw'),
               SUITE='manifold_fusion', RUN=str(tmp_path/'outside'))
    result = subprocess.run(['bash', str(ROOT/'scripts/run_daily_manifold_fusion.sh')],
                            env=env, text=True, capture_output=True)
    assert result.returncode != 0 and 'RUN must be a subdirectory' in result.stderr
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert not any('climate_manifold.daily_era5' in row for row in rows)
    assert not Path(env['RUN']).exists()
