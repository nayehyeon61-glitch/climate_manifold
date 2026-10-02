"""Check the server command matrix without a GPU or access to the ERA5 mount."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from climate_manifold.downstream.train import parser


SCRIPT = Path(__file__).resolve().parents[1]/'scripts/run_raw_latent_comparison.sh'


def run_matrix(tmp_path, *, step=24, **settings):
    work = tmp_path/'work'; work.mkdir()
    archive = work/'surface.npz'; archive.touch()
    archive.with_suffix('.schema.json').write_text(json.dumps({'forecast_step_hours': step}))
    info = work/'information.npz'
    variables = [f'{v}{p}' for v in 'uvtzw' for p in (500, 850)] + ['sp','terrain_height','terrain_slope']
    np.savez(info, metadata_json=json.dumps({'variables': [{'name': v} for v in variables]}))
    commands = work/'commands.jsonl'
    shim = work/'record-python'
    shim.write_text(f'''#!{sys.executable}
import json, os, sys
from pathlib import Path
with open(os.environ['COMMAND_LOG'], 'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')
if sys.argv[1:] == ['-']:
    exec(sys.stdin.read(), {{'__name__':'__main__'}})
''')
    shim.chmod(0o755)
    env = {**os.environ, 'DAILY_WORK': str(work), 'ERA5_ROOT': str(tmp_path),
           'PYTHON': str(shim), 'COMMAND_LOG': str(commands), 'DEVICE': 'cpu',
           'ARCHIVE': str(archive), 'INFO': str(info), 'SEEDS': '7 19 43',
           'MODELS': 'transformer mlp neural_ode climode convlstm simvp fourcastnet climax',
           'STATISTICAL_LOSSES': 'w2 signed_measure', 'CONSTRAINT_PAIRS': 'statistical',
           'EVALUATE_TEST': '1', 'RUN_PREFLIGHT_TESTS': '0', 'BATCH_SIZE': '16',
           'A_CHECKPOINT': '', 'STATISTICAL_FLOW_WEIGHT': '0', 'CONDITIONAL_FLOW_WEIGHT': '0',
           **settings}
    result = subprocess.run(['bash', str(SCRIPT)], env=env, text=True, capture_output=True)
    rows = [json.loads(line) for line in commands.read_text().splitlines()]
    return result, rows


def arg(row, option):
    return row[row.index(option)+1]


def test_default_two_route_matrix_has_no_added_transformer(tmp_path):
    result, rows = run_matrix(tmp_path)
    assert result.returncode == 0, result.stderr
    train = [r for r in rows if 'climate_manifold.downstream.train' in r]
    evaluation = [r for r in rows if 'climate_manifold.downstream.evaluate' in r]
    assert len(train) == 72 and len(evaluation) == 144
    for family in ('transformer','mlp','neural_ode','climode','convlstm','simvp','fourcastnet','climax'):
        for seed in ('7','19','43'):
            group = [r for r in train if arg(r,'--model') == family and arg(r,'--seed') == seed]
            assert len(group) == 3
            assert [arg(r,'--bridge') for r in group] == ['raw','latent','latent']
            assert arg(group[0],'--regularization') == 'none'
            assert [arg(r,'--statistical-loss') for r in group[1:]] == ['w2','signed_measure']
    for row in train:
        parsed = parser().parse_args(row[2:])
        assert parsed.bridge in ('raw', 'latent')
        assert arg(row,'--batch-size') == '16'
        assert arg(row,'--training-mode') == 'joint'
        assert arg(row,'--history-stride') == '1' and arg(row,'--horizon-steps') == '5'
        assert arg(row,'--statistical-flow-weight') == arg(row,'--conditional-flow-weight') == '0'
        assert '--history-transformer' not in row
        assert Path(arg(row,'--output')).is_relative_to(tmp_path/'work'/'runs')
    last_fit = max(i for i,r in enumerate(rows) if 'climate_manifold.downstream.train' in r)
    test_indices = [i for i,r in enumerate(rows) if '--split' in r and arg(r,'--split') == 'test']
    assert len(test_indices) == 72 and min(test_indices) > last_fit


@pytest.mark.parametrize('settings,message', [
    ({'MODELS':'transformer transformer'}, 'Duplicate model'),
    ({'MODELS':'transformer graphcast'}, 'Unsupported downstream M'),
    ({'STATISTICAL_FLOW_WEIGHT':'0.1'}, 'must remain zero'),
    ({'CONSTRAINT_PAIRS':'pinn_statistical'}, 'Daily PINN is unsupported'),
    ({'INFO':''}, 'Supply both ARCHIVE and INFO'),
])
def test_invalid_matrix_stops_before_training(tmp_path, settings, message):
    result, rows = run_matrix(tmp_path, **settings)
    assert result.returncode != 0 and message in result.stderr
    assert not any('climate_manifold.downstream.train' in r for r in rows)


def test_six_hour_pinn_and_kl_are_opt_in_and_keep_the_same_two_routes(tmp_path):
    result, rows = run_matrix(tmp_path, step=6, MODELS='neural_ode', SEEDS='7',
                              STATISTICAL_LOSSES='kl_entropy',
                              CONSTRAINT_PAIRS='statistical pinn_statistical', EVALUATE_TEST='0')
    assert result.returncode == 0, result.stderr
    train = [r for r in rows if 'climate_manifold.downstream.train' in r]
    assert len(train) == 3
    for row in train:
        parser().parse_args(row[2:])
    assert [arg(r,'--bridge') for r in train] == ['raw','latent','latent']
    assert '--pinn' not in train[1] and '--pinn' in train[2]
    assert all(arg(r,'--history-stride') == '4' and arg(r,'--horizon-steps') == '20' for r in train)
    assert not any('--split' in r and arg(r,'--split') == 'test' for r in rows)
