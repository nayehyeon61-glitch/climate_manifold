"""The pairwise runner includes one matched raw control per family and seed."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def run_runner(tmp_path, **settings):
    stub = tmp_path / 'record-python'
    stub.write_text('#!' + sys.executable + '\nimport json,os,sys\n'
        'with open(os.environ["CALLS"], "a") as f: '
        'f.write(json.dumps(sys.argv[1:]) + "\\n")\n')
    stub.chmod(0o755)
    calls = tmp_path / 'calls.jsonl'
    env = dict(PATH=os.environ['PATH'], PYTHON=str(stub), CALLS=str(calls),
        ARCHIVE='surface archive.npz', INFO='physical information',
        RUN=str(tmp_path / 'run'))
    env.update(settings)
    script = Path(__file__).resolve().parents[1] / 'scripts/run_pairwise_manifold_comparison.sh'
    result = subprocess.run(['bash', str(script)], env=env, text=True, capture_output=True)
    commands = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
    return result, commands


def option(command, name):
    assert command.count(name) == 1
    return command[command.index(name) + 1]


def trains(commands):
    return [row for row in commands if row[1] == 'climate_manifold.downstream.train']


def test_one_seed_has_eight_fits_with_shared_training_settings_and_batch_16(tmp_path):
    result, commands = run_runner(tmp_path, SEEDS='7', EPOCHS='12', MAX_WINDOWS='25')
    assert result.returncode == 0, result.stderr
    training = trains(commands)
    evaluation = [row for row in commands if row[1] == 'climate_manifold.downstream.evaluate']
    assert len(training) == len(evaluation) == 8
    assert len({option(row, '--output') for row in training}) == 8
    raw = [row for row in training if option(row, '--bridge') == 'raw']
    latent = [row for row in training if option(row, '--bridge') == 'latent']
    assert len(raw) == 2 and len(latent) == 6
    for row in training:
        assert option(row, '--batch-size') == '16'
        assert option(row, '--archive') == 'surface archive.npz'
        assert option(row, '--information') == 'physical information'
        assert option(row, '--seed') == '7'
        assert option(row, '--epochs') == '12'
        assert option(row, '--max-windows') == '25'
        for name in ('--horizon-steps', '--history-steps', '--history-stride',
                     '--learning-rate', '--tendency-weight', '--reconstruction-weight',
                     '--window-stride', '--training-mode', '--initialization'):
            assert option(row, name) == option(training[0], name)
    for row in raw:
        assert option(row, '--raw-backend') == 'matched'
        assert option(row, '--regularization') == 'none'
        assert not {'--constraint-pair', '--pinn', '--pinn-levels', '--pinn-weight',
                    '--statistical-weight', '--static-weight'} & set(row)
        assert option(row, '--output').endswith(f"{option(row, '--model')}-raw-seed7.pt")
    for row in latent:
        assert option(row, '--regularization') == 'full'
        pair = option(row, '--constraint-pair')
        assert ('--pinn' in row) == pair.startswith('pinn_')
    comparison = commands[-1]
    assert comparison[1] == 'climate_manifold.downstream.compare'
    reports = comparison[comparison.index('--reports') + 1:comparison.index('--output')]
    assert len(reports) == 8
    assert set(reports) == {option(row, '--output') for row in evaluation}


def test_default_three_seeds_have_24_fits_and_six_unique_raw_controls(tmp_path):
    result, commands = run_runner(tmp_path)
    assert result.returncode == 0, result.stderr
    training = trains(commands)
    assert len(training) == 24
    raw = [row for row in training if option(row, '--bridge') == 'raw']
    assert len(raw) == 6
    assert {(option(row, '--model'), option(row, '--seed')) for row in raw} == {
        (family, seed) for family in ('neural_ode', 'climode') for seed in ('7', '19', '43')}


def test_single_pinn_statistical_pair_has_four_fits_including_controls(tmp_path):
    result, commands = run_runner(tmp_path, SEEDS='7', PAIRS='pinn_statistical')
    assert result.returncode == 0, result.stderr
    training = trains(commands)
    assert len(training) == 4
    assert sum(option(row, '--bridge') == 'raw' for row in training) == 2


def test_raw_controls_can_be_disabled_and_batch_size_overridden(tmp_path):
    result, commands = run_runner(tmp_path, SEEDS='7', INCLUDE_RAW='0', BATCH_SIZE='8')
    assert result.returncode == 0, result.stderr
    training = trains(commands)
    assert len(training) == 6
    assert all(option(row, '--bridge') == 'latent' for row in training)
    assert all(option(row, '--batch-size') == '8' for row in training)


@pytest.mark.parametrize('value', ['yes', '2', '-1'])
def test_invalid_raw_switch_fails_before_training(tmp_path, value):
    result, commands = run_runner(tmp_path, INCLUDE_RAW=value)
    assert result.returncode != 0 and not commands
    assert 'INCLUDE_RAW must be 0 or 1' in result.stderr
