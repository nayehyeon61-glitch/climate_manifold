"""Optional signed measure (A+-, G+-) on predicted future fields, any bridge."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from test_pinn_training import pinn_prepared
from test_spatial_training import _args
from test_guided_fusion import fusion_args
from climate_manifold.downstream.evaluate import evaluate
from climate_manifold.downstream.train import forecast_signed_measure_loss, load_predictor, train


def _temporal(variables=2, cells=4):
    return SimpleNamespace(grid=(variables, 2, cells//2), area=torch.ones(2, cells//2))


def test_forecast_signed_is_zero_on_truth_and_sees_location_and_strength():
    temporal = _temporal()
    target = torch.tensor([[[1., 2., -1., -2., 0., 3., -3., 1.]]])  # [B, T, V*cells]
    batch = {'targets': target}
    for value in forecast_signed_measure_loss({'mean': target.clone()}, batch, temporal, .1).values():
        assert value == 0
    # Swap positive cells inside variable 0: same histogram, different location.
    relocated = target.clone()
    relocated[..., [0, 1]] = relocated[..., [1, 0]]
    scores = forecast_signed_measure_loss({'mean': relocated}, batch, temporal, .1)
    assert scores['forecast_signed_spatial_js'] > 0 and scores['forecast_signed_mass_mse'] == 0
    stronger = forecast_signed_measure_loss({'mean': 2*target}, batch, temporal, .1)
    assert stronger['forecast_signed_mass_mse'] > 0
    torch.testing.assert_close(stronger['forecast_signed_regularization'],
                               .1*stronger['forecast_signed_measure'])


def test_forecast_signed_uses_only_scored_leads_and_has_gradient():
    temporal = _temporal()
    prediction = torch.randn(2, 2, 8, requires_grad=True)
    target = torch.randn(2, 3, 8)
    scores = forecast_signed_measure_loss({'mean': prediction}, {'targets': target}, temporal, 1.)
    scores['forecast_signed_measure'].backward()
    assert torch.isfinite(prediction.grad).all() and prediction.grad.abs().sum() > 0


@pytest.mark.parametrize('bridge', ['raw', 'guided'])
def test_flag_trains_records_and_default_is_off(pinn_prepared, tmp_path, bridge):
    _, _, _, archive = pinn_prepared
    if bridge == 'raw':
        make = lambda name, *extra: _args(archive, tmp_path/name, '--model', 'mlp', '--bridge', 'raw',
            '--raw-backend', 'matched', '--regularization', 'none',
            '--information', str(tmp_path/'pinn-information.npz'), *extra)
    else:
        make = lambda name, *extra: fusion_args(archive, tmp_path, 'mlp', '--output', str(tmp_path/name), *extra)
    off = train(make('off.pt'))
    on = train(make('on.pt', '--forecast-signed-weight', '0.1'))
    assert load_predictor(off)[1]['forecast_signed_measure'] is None
    payload = load_predictor(on)[1]
    assert payload['forecast_signed_measure']['weight'] == .1
    assert payload['forecast_signed_measure']['config']['kind'] == 'signed_measure'
    rows = json.loads(on.with_suffix('.metrics.json').read_text())
    assert 'forecast_signed_measure' in rows[0]['train']
    assert 'forecast_signed_measure' not in json.loads(off.with_suffix('.metrics.json').read_text())[0]['train']
    report = evaluate(on, archive, tmp_path/'on.evaluation.json',
                      information=tmp_path/'pinn-information.npz', max_cases=1)
    assert report['forecast_signed_measure']['weight'] == .1


def test_negative_weight_is_rejected(pinn_prepared, tmp_path):
    _, _, _, archive = pinn_prepared
    with pytest.raises(ValueError, match='forecast_signed_weight'):
        train(_args(archive, tmp_path/'bad.pt', '--model', 'mlp', '--bridge', 'raw', '--raw-backend', 'matched',
                    '--regularization', 'none', '--information', str(tmp_path/'pinn-information.npz'),
                    '--forecast-signed-weight', '-1'))


def _runner(tmp_path, **settings):
    stub = tmp_path/'record-python'
    stub.write_text('#!'+sys.executable+'\nimport json,os,sys\n'
        'with open(os.environ["CALLS"],"a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n')
    stub.chmod(0o755)
    calls = tmp_path/'calls.jsonl'
    env = {key: value for key, value in os.environ.items()
           if key not in ('SEEDS', 'MODELS', 'FORECAST_SIGNED_WEIGHT', 'COMPARE', 'EVALUATE_TEST', 'GPU_GUARD')}
    env.update(PYTHON=str(stub), CALLS=str(calls), ARCHIVE='a.npz', INFO='i.npz',
               RUN=str(tmp_path/'run'), SEEDS='7', MODELS='mlp', **settings)
    script = Path(__file__).resolve().parents[1]/'scripts/run_guided_fusion_comparison.sh'
    outcome = subprocess.run(['bash', str(script)], env=env, text=True, capture_output=True)
    commands = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
    return outcome, commands


def test_runner_applies_weight_to_every_arm_and_compare_toggle(tmp_path):
    result, commands = _runner(tmp_path, FORECAST_SIGNED_WEIGHT='0.1', COMPARE='0')
    assert result.returncode == 0, result.stderr
    fits = [c for c in commands if c[1] == 'climate_manifold.downstream.train']
    assert {c[c.index('--bridge')+1] for c in fits} == {'raw', 'latent', 'guided'}
    assert all(c[c.index('--forecast-signed-weight')+1] == '0.1' for c in fits)
    assert not any(c[1] == 'climate_manifold.downstream.compare' for c in commands)
    (tmp_path/'off').mkdir(); (tmp_path/'bad').mkdir()
    result, commands = _runner(tmp_path/'off')
    assert result.returncode == 0, result.stderr
    assert not any('--forecast-signed-weight' in c for c in commands)
    assert any(c[1] == 'climate_manifold.downstream.compare' for c in commands)
    result, _ = _runner(tmp_path/'bad', FORECAST_SIGNED_WEIGHT='-1')
    assert result.returncode == 2
