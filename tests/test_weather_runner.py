"""Weather model comparisons keep raw controls shared and Flow disabled."""
import pytest

from test_pairwise_runner_controls import run_runner, option, trains


FAMILIES = 'mlp neural_ode climode convlstm simvp fourcastnet climax'


@pytest.mark.parametrize('losses, expected', [('w2', 14), ('w2 kl_entropy', 21)])
def test_seven_model_pairwise_comparison_shares_one_raw_control(tmp_path, losses, expected):
    result, commands = run_runner(tmp_path, MODELS=FAMILIES, SEEDS='7',
        PAIRS='pinn_statistical', STATISTICAL_LOSSES=losses,
        BATCH_SIZE='16', WEATHER_DEPTH='2', WEATHER_PATCH_SIZE='2')
    assert result.returncode == 0, result.stderr
    training = trains(commands)
    evaluation = [row for row in commands if row[1] == 'climate_manifold.downstream.evaluate']
    assert len(training) == len(evaluation) == expected
    assert len({option(row, '--output') for row in training}) == expected
    raw = [row for row in training if option(row, '--bridge') == 'raw']
    assert len(raw) == 7
    assert {option(row, '--model') for row in raw} == set(FAMILIES.split())
    for row in training:
        assert option(row, '--batch-size') == '16'
        assert option(row, '--training-mode') == 'joint'
        assert option(row, '--latent-layout') == 'spatial'
        assert '--statistical-flow-weight' not in row and '--conditional-flow-weight' not in row
        if option(row, '--model') in ('fourcastnet', 'climax'):
            assert option(row, '--weather-depth') == '2'
            assert option(row, '--weather-patch-size') == '2'
        if option(row, '--bridge') == 'raw':
            assert option(row, '--raw-backend') == 'matched'
            assert '--constraint-pair' not in row and '--statistical-loss' not in row
        else:
            assert option(row, '--constraint-pair') == 'pinn_statistical'
            assert option(row, '--statistical-loss') in losses.split()
    comparison = commands[-1]
    assert comparison[1] == 'climate_manifold.downstream.compare'
    assert len(comparison[comparison.index('--reports') + 1:comparison.index('--output')]) == expected


def test_primary_runner_supports_weather_raw_and_joint_control_arms(tmp_path):
    result, commands = run_runner(tmp_path, script_name='run_model_comparison.sh',
        MODELS='fourcastnet climax', SEEDS='7', BATCH_SIZE='16',
        WEATHER_DEPTH='2', WEATHER_PATCH_SIZE='2')
    assert result.returncode == 0, result.stderr
    training = trains(commands)
    assert len(training) == 6
    for row in training:
        assert option(row, '--batch-size') == '16'
        assert option(row, '--training-mode') == 'joint'
        assert option(row, '--latent-layout') == 'spatial'
        assert option(row, '--weather-depth') == '2'
        assert option(row, '--weather-patch-size') == '2'


@pytest.mark.parametrize('script', ['run_pairwise_manifold_comparison.sh', 'run_model_comparison.sh'])
def test_graphcast_not_silently_substituted_in_pytorch_runner(tmp_path, script):
    result, commands = run_runner(tmp_path, script_name=script,
        MODELS='graphcast', PAIRS='pinn_statistical', SEEDS='7')
    assert result.returncode != 0 and not commands
    assert 'graphcast' in result.stderr.lower()
    assert any(word in result.stderr.lower() for word in ('official', 'jax', 'external', 'standalone'))
