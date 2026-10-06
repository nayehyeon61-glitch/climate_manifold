"""Optional temporal branches retain paired controls and exact-resume identity."""
import json
from pathlib import Path
import subprocess

import pytest

from test_fusion_runner import ROOT, arg, run_script, setup_runner
from climate_manifold.downstream.train import parser


def train_rows(rows):
    return [r for r in rows if 'climate_manifold.downstream.train' in r]


@pytest.mark.parametrize('encoder', ['conv_gru', 'temporal_conv'])
def test_optional_dynamics_pairs_share_hyperparameters_and_hold_raw_fixed(tmp_path, encoder):
    env, log = setup_runner(tmp_path)
    result, rows = run_script(env, log, MODELS='mlp neural_ode', SEEDS='7',
                              INCLUDE_DYNAMICS='1', INCLUDE_ZERO_DYNAMICS='1',
                              DYNAMICS_ENCODER=encoder, DYNAMICS_STATISTICAL_LOSS='signed_measure',
                              EVALUATE_TEST='1', MAKE_PLOTS='0')
    assert result.returncode == 0, result.stderr
    fits = train_rows(rows)
    assert len(fits) == 16
    for model in ('mlp', 'neural_ode'):
        group = [r for r in fits if arg(r, '--model') == model]
        args = [parser().parse_args(r[2:]) for r in group]
        assert [a.bridge for a in args] == ['raw', 'latent', 'guided', 'guided',
                                          'latent', 'guided', 'latent', 'guided']
        assert [a.dynamics_encoder for a in args] == ['off']*4+[encoder]*4
        assert [a.dynamics_mode for a in args[4:]] == ['learned', 'learned', 'zero', 'zero']
        for a in args[4:]:
            assert a.dynamics_statistical_loss == 'signed_measure'
            assert a.dynamics_hidden_dim == 32 and a.dynamics_channels == 8
            assert a.dynamics_reconstruction_weight == a.dynamics_statistical_weight == 0.1
            assert a.dynamics_tendency_weight == 0.0
            assert a.statistical_loss == 'w2'  # Independent spatial and temporal objectives.
            assert a.batch_size == 16 and a.history_stride == 1 and a.horizon_steps == 5
    last_fit = max(i for i, r in enumerate(rows) if 'climate_manifold.downstream.train' in r)
    first_test = min(i for i, r in enumerate(rows) if '--split' in r and arg(r, '--split') == 'test')
    assert first_test > last_fit
    manifest = json.loads((Path(env['RUN'])/'run-manifest.json').read_text())
    assert manifest['settings']['arms'].endswith('latent_dynamics_zero guided_dynamics_zero')


def test_dynamics_zero_controls_are_optional_and_loss_inherits(tmp_path):
    env, log = setup_runner(tmp_path)
    result, rows = run_script(env, log, MODELS='climode', SEEDS='7',
                              INCLUDE_DYNAMICS='1', INCLUDE_ZERO_DYNAMICS='0',
                              STATISTICAL_LOSS='kl_entropy', EVALUATE_TEST='0', MAKE_PLOTS='0')
    assert result.returncode == 0, result.stderr
    assert len(train_rows(rows)) == 6
    for row in train_rows(rows)[-2:]:
        assert arg(row, '--dynamics-statistical-loss') == 'kl_entropy'
        assert arg(row, '--dynamics-mode') == 'learned'


def test_resume_refuses_changed_dynamics_but_preserves_completed_checkpoints(tmp_path):
    env, log = setup_runner(tmp_path)
    settings = dict(MODELS='mlp', SEEDS='7', INCLUDE_DYNAMICS='1',
                    DYNAMICS_ENCODER='conv_gru', MAKE_PLOTS='0', EVALUATE_TEST='0')
    result, _ = run_script(env, log, **settings)
    assert result.returncode == 0, result.stderr
    run = Path(env['RUN'])
    before = {p.name: p.read_bytes() for p in run.glob('*.pt')}
    result, rows = run_script(env, log, **{**settings, 'DYNAMICS_ENCODER': 'temporal_conv'}, RESUME='1')
    assert result.returncode != 0 and 'differs' in result.stderr
    assert not train_rows(rows)
    assert before == {p.name: p.read_bytes() for p in run.glob('*.pt')}
    result, rows = run_script(env, log, **settings, RESUME='1')
    assert result.returncode == 0, result.stderr
    assert not train_rows(rows)


@pytest.mark.parametrize('settings, message', [
    ({'INCLUDE_DYNAMICS': '2'}, 'INCLUDE_DYNAMICS must be'),
    ({'INCLUDE_ZERO_DYNAMICS': '2'}, 'INCLUDE_ZERO_DYNAMICS must be'),
    ({'INCLUDE_DYNAMICS': '1', 'DYNAMICS_ENCODER': 'rnn'}, 'DYNAMICS_ENCODER must be'),
    ({'DYNAMICS_STATISTICAL_LOSS': 'other'}, 'Unknown DYNAMICS_STATISTICAL_LOSS'),
])
def test_invalid_dynamics_matrix_never_launches_training(tmp_path, settings, message):
    env, log = setup_runner(tmp_path)
    result, rows = run_script(env, log, **settings)
    assert result.returncode != 0 and message in result.stderr
    assert not train_rows(rows)


def test_dynamics_wrapper_prepared_mode_and_defaults(tmp_path):
    env, log = setup_runner(tmp_path)
    env.update(MODELS='mlp', SEEDS='7', MAKE_PLOTS='0', EVALUATE_TEST='0')
    env.pop('STATISTICAL_LOSS')
    for name in ('INCLUDE_DYNAMICS', 'INCLUDE_ZERO_DYNAMICS', 'DYNAMICS_ENCODER',
                 'DYNAMICS_STATISTICAL_LOSS'):
        env.pop(name, None)
    result = subprocess.run(['bash', str(ROOT/'scripts/run_dynamics_manifold_comparison.sh')],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(row) for row in log.read_text().splitlines()]
    fits = train_rows(rows)
    assert len(fits) == 8
    assert arg(fits[-1], '--dynamics-encoder') == 'conv_gru'
    assert arg(fits[-1], '--statistical-loss') == 'signed_measure'
    assert arg(fits[-1], '--dynamics-statistical-loss') == 'signed_measure'


def test_dynamics_daily_wrapper_keeps_write_boundary(tmp_path):
    env, log = setup_runner(tmp_path)
    env.pop('ARCHIVE')
    env.pop('INFO')
    env.update(DAILY_WORK=str(tmp_path/'daily'), ERA5_ROOT=str(tmp_path/'raw'),
               RUN=str(tmp_path/'outside'))
    result = subprocess.run(['bash', str(ROOT/'scripts/run_dynamics_manifold_comparison.sh')],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode != 0 and 'RUN must be a subdirectory' in result.stderr
    rows = [json.loads(row) for row in log.read_text().splitlines()]
    assert not train_rows(rows)
