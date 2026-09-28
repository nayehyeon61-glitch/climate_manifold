"""The pairwise runner includes one matched raw control per family and seed."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def run_runner(tmp_path, script_name='run_pairwise_manifold_comparison.sh', **settings):
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
    script = Path(__file__).resolve().parents[1] / 'scripts' / script_name
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
        assert not {'--constraint-pair', '--constraint-decoder', '--pinn', '--pinn-levels', '--pinn-weight',
                    '--statistical-weight', '--static-weight'} & set(row)
        assert option(row, '--output').endswith(f"{option(row, '--model')}-raw-seed7.pt")
    for row in latent:
        assert option(row, '--regularization') == 'full'
        assert option(row, '--constraint-decoder') == 'separate_surface_and_information'
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


@pytest.mark.parametrize('mode', [
    'separate_surface_and_information', 'information_only', 'surface_and_information',
])
def test_constraint_decoder_override_only_applies_to_latent_routes(tmp_path, mode):
    result, commands = run_runner(tmp_path, SEEDS='7', PAIRS='pinn_statistical',
        CONSTRAINT_DECODER=mode)
    assert result.returncode == 0, result.stderr
    for row in trains(commands):
        if option(row, '--bridge') == 'raw':
            assert '--constraint-decoder' not in row
        else:
            assert option(row, '--constraint-decoder') == mode


@pytest.mark.parametrize('value', ['both', 'surface_only', 'information'])
def test_invalid_constraint_decoder_fails_before_training(tmp_path, value):
    result, commands = run_runner(tmp_path, CONSTRAINT_DECODER=value)
    assert result.returncode != 0 and not commands
    assert 'CONSTRAINT_DECODER must be separate_surface_and_information, information_only or surface_and_information' in result.stderr


@pytest.mark.parametrize('value', ['yes', '2', '-1'])
def test_invalid_raw_switch_fails_before_training(tmp_path, value):
    result, commands = run_runner(tmp_path, INCLUDE_RAW=value)
    assert result.returncode != 0 and not commands
    assert 'INCLUDE_RAW must be 0 or 1' in result.stderr


def test_five_predictors_with_one_pair_and_raw_have_ten_fits(tmp_path):
    families = 'mlp neural_ode climode convlstm simvp'
    result, commands = run_runner(tmp_path, MODELS=families, PAIRS='pinn_statistical', SEEDS='7')
    assert result.returncode == 0, result.stderr
    training = trains(commands)
    assert len(training) == 10
    assert {(option(row, '--model'), option(row, '--bridge')) for row in training} == {
        (family, bridge) for family in families.split() for bridge in ('raw', 'latent')}
    assert all(option(row, '--batch-size') == '16' for row in training)


def test_sequence_predictors_alone_with_one_pair_have_four_fits(tmp_path):
    result, commands = run_runner(tmp_path, MODELS='convlstm simvp', PAIRS='pinn_statistical', SEEDS='7')
    assert result.returncode == 0, result.stderr
    training = trains(commands)
    assert len(training) == 4
    for row in training:
        assert option(row, '--training-mode') == 'joint'
        assert option(row, '--latent-layout') == 'spatial'
        assert option(row, '--history-steps') == '6'
        assert option(row, '--history-stride') == '4'
        assert option(row, '--horizon-steps') == '20'
        if option(row, '--bridge') == 'raw':
            assert option(row, '--raw-backend') == 'matched'


def test_primary_runner_accepts_sequence_predictors_as_spatial_joint(tmp_path):
    result, commands = run_runner(tmp_path, script_name='run_model_comparison.sh',
        MODELS='convlstm simvp', SEEDS='7', BATCH_SIZE='16')
    assert result.returncode == 0, result.stderr
    training = trains(commands)
    assert len(training) == 6
    for row in training:
        assert option(row, '--training-mode') == 'joint'
        assert option(row, '--latent-layout') == 'spatial'
        assert option(row, '--batch-size') == '16'


@pytest.mark.parametrize('settings,message', [
    ({'TRAINING_MODE': 'frozen'}, 'TRAINING_MODE=joint'),
    ({'LATENT_LAYOUT': 'global'}, 'LATENT_LAYOUT=spatial'),
    ({'RAW_BACKEND': 'legacy'}, 'RAW_BACKEND=matched'),
])
def test_primary_sequence_predictors_reject_unsupported_routes_before_training(tmp_path, settings, message):
    result, commands = run_runner(tmp_path, script_name='run_model_comparison.sh',
        MODELS='convlstm simvp', SEEDS='7', **settings)
    assert result.returncode != 0 and not commands
    assert message in result.stderr


def test_flow_and_statistical_sweeps_share_raw_controls_across_five_models(tmp_path):
    result, commands = run_runner(tmp_path, SEEDS='7', PAIRS='pinn_statistical',
        MODELS='mlp neural_ode climode convlstm simvp', STATISTICAL_LOSSES='w2 kl_entropy',
        STATISTICAL_FLOW_WEIGHTS='0 .100', STATISTICAL_FLOW_QUANTILES='17')
    assert result.returncode == 0, result.stderr
    training = trains(commands)
    assert len(training) == 25
    assert len({option(row, '--output') for row in training}) == 25
    raw = [row for row in training if option(row, '--bridge') == 'raw']
    assert len(raw) == 5
    for row in raw:
        assert not {'--statistical-flow-weight', '--statistical-flow-quantiles',
                    '--statistical-loss'} & set(row)
    latent = [row for row in training if option(row, '--bridge') == 'latent']
    assert len(latent) == 20
    enabled = [row for row in latent if '--statistical-flow-weight' in row]
    assert len(enabled) == 10
    assert {option(row, '--statistical-loss') for row in enabled} == {'w2', 'kl_entropy'}
    for row in enabled:
        assert option(row, '--statistical-flow-weight') == '0.1'
        assert option(row, '--statistical-flow-quantiles') == '17'
        assert option(row, '--output').endswith('-flow0.1-seed7.pt')
    for row in latent:
        assert option(row, '--statistical-weight') == '0.1'
        assert option(row, '--batch-size') == '16'
        assert ('--kl-bins' in row) == (option(row, '--statistical-loss') == 'kl_entropy')
        if '--statistical-flow-weight' not in row:
            assert '--statistical-flow-quantiles' not in row
            assert '-flow' not in Path(option(row, '--output')).name
    assert len(commands[-1][commands[-1].index('--reports') + 1:commands[-1].index('--output')]) == 25


def test_all_pair_flow_sweep_does_not_duplicate_pinn_static(tmp_path):
    result, commands = run_runner(tmp_path, MODELS='neural_ode', SEEDS='7',
        STATISTICAL_LOSSES='w2 kl_entropy', STATISTICAL_FLOW_WEIGHTS='0 0.2')
    assert result.returncode == 0, result.stderr
    training = trains(commands)
    assert len(training) == 10  # 2 statistical pairs x 2 losses x 2 weights + PINN/Static + Raw.
    inactive = [row for row in training if '--statistical-loss' not in row]
    assert len(inactive) == 2
    assert all('--statistical-flow-weight' not in row for row in inactive)


def test_single_flow_option_applies_to_both_statistical_losses(tmp_path):
    result, commands = run_runner(tmp_path, MODELS='climode', SEEDS='7',
        PAIRS='pinn_statistical', STATISTICAL_LOSSES='w2 kl_entropy', STATISTICAL_FLOW_WEIGHT='0.05')
    assert result.returncode == 0, result.stderr
    training = trains(commands)
    assert len(training) == 3
    for row in training:
        if option(row, '--bridge') == 'latent':
            assert option(row, '--statistical-flow-weight') == '0.05'
            assert option(row, '--statistical-flow-quantiles') == '32'


@pytest.mark.parametrize('settings,message', [
    ({'STATISTICAL_FLOW_WEIGHTS': '0 0.0'}, 'Duplicate statistical flow weight'),
    ({'STATISTICAL_FLOW_WEIGHTS': '0.1 .100'}, 'Duplicate statistical flow weight'),
    ({'STATISTICAL_FLOW_WEIGHT': '-1'}, 'finite nonnegative decimal'),
    ({'STATISTICAL_FLOW_WEIGHT': 'NaN'}, 'finite nonnegative decimal'),
    ({'STATISTICAL_FLOW_WEIGHT': 'inf'}, 'finite nonnegative decimal'),
    ({'STATISTICAL_FLOW_WEIGHT': '1e-3'}, 'finite nonnegative decimal'),
    ({'STATISTICAL_FLOW_WEIGHT': '9' * 400}, 'finite nonnegative decimal'),
    ({'STATISTICAL_FLOW_WEIGHT': '0.1', 'PAIRS': 'pinn_static'}, 'pair containing statistical'),
    ({'STATISTICAL_FLOW_QUANTILES': '16'}, 'requires a positive statistical flow weight'),
    ({'STATISTICAL_FLOW_WEIGHT': '0.1', 'STATISTICAL_FLOW_QUANTILES': '0'}, 'positive integer'),
    ({'STATISTICAL_FLOW_WEIGHT': '0.1', 'STATISTICAL_FLOW_QUANTILES': '1.5'}, 'positive integer'),
    ({'STATISTICAL_FLOW_WEIGHT': '0.1', 'STATISTICAL_FLOW_QUANTILES': '513'}, 'between 1 and 512'),
])
def test_invalid_flow_configuration_fails_before_any_training(tmp_path, settings, message):
    result, commands = run_runner(tmp_path, **settings)
    assert result.returncode != 0 and not commands
    assert message in result.stderr


def test_observed_conditional_flow_sweep_shares_raw_and_preserves_base_arms(tmp_path):
    result, commands = run_runner(tmp_path, SEEDS='7', PAIRS='pinn_statistical',
        MODELS='mlp neural_ode climode convlstm simvp', STATISTICAL_LOSSES='w2 kl_entropy',
        CONDITIONAL_FLOW_WEIGHTS='0 .100', CONDITIONAL_FLOW_QUANTILES='17',
        CONDITIONAL_FLOW_HIDDEN_DIM='48', CONDITIONAL_FLOW_NOISE_SCALE='.3')
    assert result.returncode == 0, result.stderr
    training = trains(commands)
    assert len(training) == 25
    assert len({option(row, '--output') for row in training}) == 25
    raw = [row for row in training if option(row, '--bridge') == 'raw']
    assert len(raw) == 5
    enabled = [row for row in training if '--conditional-flow-weight' in row]
    assert len(enabled) == 10
    assert {option(row, '--statistical-loss') for row in enabled} == {'w2', 'kl_entropy'}
    for row in enabled:
        assert option(row, '--conditional-flow-weight') == '0.1'
        assert option(row, '--conditional-flow-quantiles') == '17'
        assert option(row, '--conditional-flow-hidden-dim') == '48'
        assert option(row, '--conditional-flow-noise-scale') == '0.3'
        assert option(row, '--constraint-decoder') == 'separate_surface_and_information'
        assert option(row, '--output').endswith('-cfm0.1-seed7.pt')
    for row in training:
        assert '--statistical-flow-weight' not in row
        assert option(row, '--batch-size') == '16'
        if '--conditional-flow-weight' not in row:
            assert not {'--conditional-flow-quantiles', '--conditional-flow-hidden-dim',
                        '--conditional-flow-noise-scale'} & set(row)
            assert '-cfm' not in Path(option(row, '--output')).name
    for row in raw:
        assert '--statistical-loss' not in row
    comparison = commands[-1]
    assert len(comparison[comparison.index('--reports') + 1:comparison.index('--output')]) == 25


def test_all_pair_conditional_sweep_keeps_nonstatistical_and_raw_single(tmp_path):
    result, commands = run_runner(tmp_path, MODELS='neural_ode', SEEDS='7',
        STATISTICAL_LOSSES='w2 kl_entropy', CONDITIONAL_FLOW_WEIGHTS='0 0.2')
    assert result.returncode == 0, result.stderr
    training = trains(commands)
    assert len(training) == 10
    inactive = [row for row in training if '--statistical-loss' not in row]
    assert len(inactive) == 2
    assert all('--conditional-flow-weight' not in row for row in inactive)


@pytest.mark.parametrize('decoder', ['separate_surface_and_information', 'information_only'])
def test_single_conditional_weight_uses_defaults_and_supported_decoder(tmp_path, decoder):
    result, commands = run_runner(tmp_path, MODELS='climode', SEEDS='7',
        PAIRS='pinn_statistical', STATISTICAL_LOSS='kl_entropy',
        CONDITIONAL_FLOW_WEIGHT='0.05', CONSTRAINT_DECODER=decoder)
    assert result.returncode == 0, result.stderr
    training = trains(commands)
    assert len(training) == 2
    latent = next(row for row in training if option(row, '--bridge') == 'latent')
    assert option(latent, '--conditional-flow-weight') == '0.05'
    assert option(latent, '--conditional-flow-quantiles') == '32'
    assert option(latent, '--conditional-flow-hidden-dim') == '128'
    assert option(latent, '--conditional-flow-noise-scale') == '0.2'


@pytest.mark.parametrize('settings,message', [
    ({'CONDITIONAL_FLOW_WEIGHTS': '0 0.0'}, 'Duplicate conditional flow weight'),
    ({'CONDITIONAL_FLOW_WEIGHTS': '0.1 .100'}, 'Duplicate conditional flow weight'),
    ({'CONDITIONAL_FLOW_WEIGHT': '-1'}, 'finite nonnegative decimal'),
    ({'CONDITIONAL_FLOW_WEIGHT': 'NaN'}, 'finite nonnegative decimal'),
    ({'CONDITIONAL_FLOW_WEIGHT': 'inf'}, 'finite nonnegative decimal'),
    ({'CONDITIONAL_FLOW_WEIGHT': '1e-3'}, 'finite nonnegative decimal'),
    ({'CONDITIONAL_FLOW_WEIGHT': '9' * 400}, 'finite nonnegative decimal'),
    ({'CONDITIONAL_FLOW_WEIGHT': '0.1', 'PAIRS': 'pinn_static'}, 'pair containing statistical'),
    ({'CONDITIONAL_FLOW_WEIGHT': '0.1', 'CONSTRAINT_DECODER': 'surface_and_information'},
     'independent or information-only'),
    ({'CONDITIONAL_FLOW_WEIGHTS': '0 .1', 'STATISTICAL_FLOW_WEIGHTS': '0 .1'},
     'cannot both have positive weights'),
    ({'CONDITIONAL_FLOW_QUANTILES': '16'}, 'requires a positive conditional flow weight'),
    ({'CONDITIONAL_FLOW_HIDDEN_DIM': '16'}, 'requires a positive conditional flow weight'),
    ({'CONDITIONAL_FLOW_NOISE_SCALE': '.2'}, 'requires a positive conditional flow weight'),
    ({'CONDITIONAL_FLOW_WEIGHT': '.1', 'CONDITIONAL_FLOW_QUANTILES': '0'}, 'between 1 and 512'),
    ({'CONDITIONAL_FLOW_WEIGHT': '.1', 'CONDITIONAL_FLOW_QUANTILES': '513'}, 'between 1 and 512'),
    ({'CONDITIONAL_FLOW_WEIGHT': '.1', 'CONDITIONAL_FLOW_HIDDEN_DIM': '1.5'}, 'between 1 and 4096'),
    ({'CONDITIONAL_FLOW_WEIGHT': '.1', 'CONDITIONAL_FLOW_HIDDEN_DIM': '0'}, 'between 1 and 4096'),
    ({'CONDITIONAL_FLOW_WEIGHT': '.1', 'CONDITIONAL_FLOW_HIDDEN_DIM': '4097'}, 'between 1 and 4096'),
    ({'CONDITIONAL_FLOW_WEIGHT': '.1', 'CONDITIONAL_FLOW_NOISE_SCALE': '0'}, 'finite and positive'),
    ({'CONDITIONAL_FLOW_WEIGHT': '.1', 'CONDITIONAL_FLOW_NOISE_SCALE': '-0.2'}, 'finite and positive'),
    ({'CONDITIONAL_FLOW_WEIGHT': '.1', 'CONDITIONAL_FLOW_NOISE_SCALE': 'NaN'}, 'finite and positive'),
    ({'CONDITIONAL_FLOW_WEIGHT': '.1', 'CONDITIONAL_FLOW_NOISE_SCALE': 'inf'}, 'finite and positive'),
    ({'CONDITIONAL_FLOW_WEIGHT': '.1', 'CONDITIONAL_FLOW_NOISE_SCALE': '1e999'}, 'finite and positive'),
])
def test_invalid_conditional_configuration_fails_before_any_training(tmp_path, settings, message):
    result, commands = run_runner(tmp_path, **settings)
    assert result.returncode != 0 and not commands
    assert message in result.stderr
