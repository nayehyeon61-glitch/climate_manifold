"""Audit loss-suite orchestration through the daily runner without ERA5/GPU I/O."""
import csv
import json
from pathlib import Path
import subprocess

import pytest

from climate_manifold.downstream.train import parser
from test_dynamics_consistency_runner import evaluations, fits
from test_full_manifold_matrix_runner import ARMS, setup_full
from test_fusion_runner import ROOT, arg, fixture_inputs


SCRIPT = ROOT/'scripts/run_full_manifold_matrix.sh'
LOSSES = ('w2', 'kl_entropy', 'signed_measure')


def setup_orchestration(tmp_path):
    env, log = setup_full(tmp_path)
    daily = tmp_path/'daily'
    (daily/'prepared').mkdir(parents=True)
    fixture_inputs(daily/'prepared')
    env.update(DAILY_WORK=str(daily), ERA5_ROOT=str(tmp_path/'raw'), DEVICE='cpu',
               MODELS='mlp', SEEDS='7', MAKE_PLOTS='1', EVALUATE_TEST='1',
               FULL_MATRIX_LOSSES=' '.join(LOSSES), RUN_PREFLIGHT_TESTS='1')
    # The shared shim already records all real shell routing and runs production
    # manifest checks. Supply representative comparison rows so the final CSV
    # assertion verifies suite/split provenance, not merely an empty header.
    with Path(env['PYTHON']).open('a') as stream:
        stream.write('''
if '-m' in args and value('-m').endswith('.train'):
    Path(value('--output')).with_suffix('.route.json').write_text(json.dumps(args))
if '-m' in args and value('-m').endswith('.compare'):
    report_paths = args[args.index('--reports')+1:args.index('--output')]
    rows = []
    for name in report_paths:
        origin = Path(name)
        suffix = '.test.json' if name.endswith('.test.json') else '.validation.json'
        prefix = name[:-len(suffix)]
        training = json.loads(Path(prefix+'.route.json').read_text())
        def option(key, default=None):
            return training[training.index(key)+1] if key in training else default
        supervision = option('--dynamics-supervision')
        rows.append({
            'model': option('--model'), 'bridge': option('--bridge'),
            'candidate_arm': Path(prefix).name.removeprefix('mlp-').removesuffix('-seed7'),
            'seed': int(option('--seed')), 'dynamics_encoder': option('--dynamics-encoder'),
            'dynamics_auxiliary_config': {'supervision': supervision} if supervision else None,
            'guide_interaction': option('--guide-interaction', 'off'),
            'normalized_rmse': 0.5, 'finite_forecast_fraction': 1.0,
        })
    path = Path(value('--output'))
    report = json.loads(path.read_text()); report['rows'] = rows
    path.write_text(json.dumps(report))
''')
    return env, log, daily


def run_orchestration(env, log, **settings):
    log.write_text('')
    result = subprocess.run(['bash', str(SCRIPT)], cwd=ROOT,
                            env={**env, **settings}, text=True, capture_output=True)
    return result, [json.loads(line) for line in log.read_text().splitlines()]


def test_full_loss_matrix_preserves_routing_and_separate_evaluation_outputs(tmp_path):
    env, log, daily = setup_orchestration(tmp_path)
    result, calls = run_orchestration(env, log)
    assert result.returncode == 0, result.stderr
    assert len(fits(calls)) == 36 and len(evaluations(calls)) == 72
    assert sum('-m' in r and arg(r, '-m') == 'pytest' for r in calls) == 1
    preparations = [r for r in calls if 'climate_manifold.daily_era5' in r]
    assert len(preparations) == 3
    assert {arg(r, '--output') for r in preparations} == {str(daily/'prepared')}
    comparisons = [r for r in calls if 'climate_manifold.downstream.compare' in r]
    plots = [r for r in calls if 'climate_manifold.downstream.plot_comparison' in r]
    assert len(comparisons) == len(plots) == 6

    for loss in LOSSES:
        run = daily/'runs'/f'full-{loss}'
        group = [r for r in fits(calls) if Path(arg(r, '--output')).parent == run]
        assert [Path(arg(r, '--output')).stem for r in group] == [
            f'mlp-{arm}-seed7' for arm in ARMS]
        configs = [parser().parse_args(r[2:]) for r in group]
        assert configs[0].bridge == 'raw' and configs[0].regularization == 'none'
        for config in configs[1:]:
            assert config.statistical_loss == loss and config.statistical_weight > 0
            if config.dynamics_encoder == 'off':
                continue
            assert config.dynamics_statistical_loss == loss
            assert (config.dynamics_statistical_weight == 0) == (
                config.dynamics_supervision == 'vector_only')
            if config.dynamics_supervision == 'vector_only':
                assert config.dynamics_reconstruction_weight == config.dynamics_tendency_weight == 0
                assert config.dynamics_latent_weight > 0
        scored = [r for r in evaluations(calls) if Path(arg(r, '--output')).parent == run]
        assert len(scored) == 24
        assert {arg(r, '--split') for r in scored} == {'validation', 'test'}
        fit_indices = [i for i, r in enumerate(calls) if r in group]
        test_indices = [i for i, r in enumerate(calls) if r in scored and arg(r, '--split') == 'test']
        assert min(test_indices) > max(fit_indices)
        for name in ('comparison.json', 'comparison.test.json'):
            report = json.loads((run/name).read_text())
            assert len(report['rows']) == 12
        assert all((run/'plots'/split/'manifest.json').is_file()
                   for split in ('validation', 'test'))

    with (daily/'full-matrix-summary.csv').open(newline='') as stream:
        summary = list(csv.DictReader(stream))
    assert len(summary) == 72
    assert {(r['statistical_suite'], r['split']) for r in summary} == {
        (loss, split) for loss in LOSSES for split in ('validation', 'test')}
    assert sum(r['dynamics_supervision'] == 'vector_only' for r in summary) == 18
    assert all(r['normalized_rmse'] == '0.5' for r in summary)


@pytest.mark.parametrize('losses,message', [
    ('w2 w2', 'Duplicate statistical loss: w2'),
    ('w2 unknown', 'Unknown statistical loss: unknown'),
])
def test_invalid_loss_selection_fails_before_any_daily_work(tmp_path, losses, message):
    env, log, daily = setup_orchestration(tmp_path)
    result, calls = run_orchestration(env, log, FULL_MATRIX_LOSSES=losses)
    assert result.returncode != 0 and message in result.stderr
    assert not fits(calls) and not evaluations(calls)
    assert not any('climate_manifold.daily_era5' in r for r in calls)
    assert not (daily/'runs').exists()
    assert not (daily/'full-matrix-summary.csv').exists()
