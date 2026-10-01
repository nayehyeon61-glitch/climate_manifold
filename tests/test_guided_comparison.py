"""Guidance routes remain field forecasts with distinct comparison identities."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from climate_manifold.downstream.protocol import validate_experiment
from climate_manifold.downstream.statistical_objective import make_statistical_config
from climate_manifold.downstream.constraint_protocol import make_constraint_contract
from test_pairwise_comparison import pair_report
from test_raw_pair_comparison import raw_report
from test_raw_comparison import run_compare


def guide_report(bridge='guided', mode='learned', seed=7, loss='w2'):
    row = pair_report('statistical_static', seed, 'transformer')
    row['config'].update(bridge=bridge, transformer_heads=4, weather_depth=2,
                         weather_patch_size=2, guide_mode=mode)
    row.update(constraint_pair='statistical', constraint_decoder='separate_surface_and_information',
        constraint_contract=make_constraint_contract('statistical', {'statistical'}, 'separate_surface_and_information'),
        statistical_loss=loss, statistical_loss_config=make_statistical_config(loss))
    row['split_objective_weights']['static'] = 0.
    row['objective_weights'] = deepcopy(row['split_objective_weights'])
    if bridge == 'guided':
        row.update(guide_contract={
            'version':'climate_manifold.statistical_guide.v1', 'mode':mode,
            'raw_input':True, 'guide_input':'observed_history_encoded_with_origin_information',
            'fusion':'joint_raw_guide_token_self_attention', 'forecast_output':'physical_fields',
            'forecast_decoder_used':False, 'forecast_gradient_to_encoder':mode=='learned',
            'stochastic_sampling':False}, forecast_parameters=90,
            total_parameters=110, trainable_parameters=110, implementation='guided_transformer')
    return row


def transformer_raw(seed=7):
    row = raw_report(seed, 'transformer')
    row['config'].update(transformer_heads=4, weather_depth=2, weather_patch_size=2)
    return row


def test_protocol_guided_is_primary_field_forecasting():
    cfg = guide_report()['config']
    result = validate_experiment(cfg, 'primary')
    assert result['prediction_space'] == 'field'
    assert result['predictor_variant'] == 'guided_transformer'
    assert 'raw observations' in result['path']
    for name, value in [('model','climode'),('training_mode','frozen'),('latent_layout','global')]:
        with pytest.raises(ValueError):
            validate_experiment({**cfg, name:value}, 'primary')


def test_three_routes_and_zero_guide_have_distinct_arms_and_raw_effects(tmp_path):
    rows = [row for seed in (7,19) for row in (
        transformer_raw(seed), guide_report('latent',seed=seed),
        guide_report(seed=seed), guide_report(mode='zero',seed=seed))]
    result = run_compare(tmp_path, rows)
    assert len(result['seed_summary']) == 4
    assert {row['candidate_arm'] for row in result['rows']} == {
        'raw','statistical','guided:learned:statistical','guided:zero:statistical'}
    assert len(result['direct_comparison']['effects']) == 12
    guided = [row for row in result['rows'] if row['bridge']=='guided']
    assert all(row['prediction_space']=='field' for row in guided)
    assert len([row for row in result['paired_effects'] if row['interpretation']=='change_guide_input']) == 2
    assert len(result['guide_effects']) == 4
    assert len(result['route_effects']) == 8
    assert all(r['baseline_guide_mode']=='zero' and r['candidate_guide_mode']=='learned' for r in result['guide_effects'])
    assert not result['statistical_loss_effects']
    assert all(len(summary['seeds'])==2 for summary in result['seed_summary'].values())


@pytest.mark.parametrize('key,value', [
    ('transformer_heads',8), ('weather_depth',3), ('weather_patch_size',1), ('hidden_dim',64)])
def test_guides_reject_changed_transformer_capacity(tmp_path,key,value):
    row = guide_report()
    row['config'][key] = value
    with pytest.raises(ValueError,match=key):
        run_compare(tmp_path,[transformer_raw(),row])


@pytest.mark.parametrize('mutation', ['missing','sampling','future','decoder','wrong_mode'])
def test_guidance_contract_rejects_mislabelled_scientific_scope(tmp_path, mutation):
    row = guide_report()
    if mutation=='missing':
        del row['guide_contract']
    else:
        key,value = {'sampling':('stochastic_sampling',True),
            'future':('guide_input','future_information'),
            'decoder':('forecast_decoder_used',True), 'wrong_mode':('mode','zero')}[mutation]
        row['guide_contract'][key]=value
    with pytest.raises(ValueError,match='guide_contract'):
        run_compare(tmp_path,[transformer_raw(),row])


def test_signed_statistical_guide_is_separate_from_w2(tmp_path):
    result = run_compare(tmp_path,[transformer_raw(),guide_report(),guide_report(loss='signed_measure')])
    assert len(result['seed_summary']) == 3
    assert 'guided:learned:statistical:signed_measure' in {r['candidate_arm'] for r in result['rows']}


def _runner(tmp_path, **settings):
    stub=tmp_path/'record-python'
    stub.write_text('#!'+sys.executable+'\nimport json,os,sys\n'
        'with open(os.environ["CALLS"],"a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n')
    stub.chmod(0o755)
    calls=tmp_path/'calls.jsonl'
    env={key:value for key,value in os.environ.items() if key not in (
        'SEEDS','A_CHECKPOINT','BATCH_SIZE','CONSTRAINT_DECODER','STATISTICAL_LOSS',
        'INCLUDE_ZERO_GUIDE','VARIABLE_CONDITIONING','STATISTICAL_FLOW_WEIGHT','CONDITIONAL_FLOW_WEIGHT')}
    env.update(PYTHON=str(stub),CALLS=str(calls),ARCHIVE='surface archive.npz',
               INFO='physical information',RUN=str(tmp_path/'run'),**settings)
    script=Path(__file__).resolve().parents[1]/'scripts/run_guided_transformer_comparison.sh'
    outcome=subprocess.run(['bash',str(script)],env=env,text=True,capture_output=True)
    commands=[json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
    return outcome,commands


def test_runner_nine_fits_default_batch16_and_no_flow(tmp_path):
    result, commands = _runner(tmp_path)
    assert result.returncode == 0, result.stderr
    train=[r for r in commands if r[1]=='climate_manifold.downstream.train']
    assert len(train)==9
    assert {r[r.index('--bridge')+1] for r in train}=={'raw','latent','guided'}
    for args in train:
        assert args[args.index('--batch-size')+1]=='16'
        assert args[args.index('--archive')+1]=='surface archive.npz'
        assert '--pinn' not in args and '--static-weight' not in args
        if args[args.index('--bridge')+1]!='raw':
            assert args[args.index('--constraint-pair')+1]=='statistical'
            assert args[args.index('--statistical-flow-weight')+1]=='0'
            assert args[args.index('--conditional-flow-weight')+1]=='0'
    assert commands[-1][1]=='climate_manifold.downstream.compare'


def test_runner_zero_guide_keeps_matching_supervision(tmp_path):
    result, commands = _runner(tmp_path, SEEDS='7', INCLUDE_ZERO_GUIDE='1', STATISTICAL_LOSS='signed_measure')
    assert result.returncode == 0, result.stderr
    train=[r for r in commands if r[1]=='climate_manifold.downstream.train']
    assert len(train)==4
    guides=[r for r in train if r[r.index('--bridge')+1]=='guided']
    assert {r[r.index('--guide-mode')+1] for r in guides}=={'learned','zero'}
    assert all(r[r.index('--statistical-loss')+1]=='signed_measure' for r in guides)


@pytest.mark.parametrize('settings',[
    {'SEEDS':'7 7'}, {'STATISTICAL_LOSS':'unknown'}, {'STATISTICAL_FLOW_WEIGHT':'0.1'},
    {'CONDITIONAL_FLOW_WEIGHT':'1'}, {'CONSTRAINT_DECODER':'surface_and_information'}])
def test_runner_rejects_bad_contract_before_training(tmp_path,settings):
    result, commands = _runner(tmp_path, **settings)
    assert result.returncode != 0 and not commands


def forecast_only_guide(seed=7):
    row = guide_report(seed=seed)
    row.update(regularization='none', constraint_pair=None, constraint_path=None,
               constraint_contract=None, constraint_decoder=None, split_objective_weights=None,
               statistical_loss=None, statistical_loss_config=None,
               # The information head can remain present but unsupervised; the
               # independent reconstruction decoder is absent in this arm.
               total_parameters=100, trainable_parameters=100, constraint_parameters=10)
    row['objective_weights'] = {key:0. for key in row['objective_weights']}
    row['training_contract'].pop('constraint_path')
    return row


def test_guided_forecast_only_control_isolates_added_observed_constraints(tmp_path):
    rows=[row for seed in (7,19) for row in (
        forecast_only_guide(seed), guide_report(seed=seed))]
    result=run_compare(tmp_path,rows)
    assert len(result['seed_summary'])==2
    assert len(result['paired_effects'])==2
    assert {r['interpretation'] for r in result['paired_effects']}=={'add_observed_encoder_constraints'}
    assert len(result['guide_effects'])==4
    assert all(r['control']=='guided:learned:none' for r in result['guide_effects'])
    assert all(r['forecast_parameters']==90 for r in result['guide_effects'])
    assert all(r['baseline_parameters']==100 and r['candidate_parameters']==110 for r in result['guide_effects'])
    assert not result['direct_comparison']['available']
    assert not result['statistical_loss_effects']


@pytest.mark.parametrize('path,value,message',[
    (('objective_weights','reconstruction'),.1,'zero effective objective_weights'),
    (('constraint_path',),'observed_reconstruction','constraint objective'),
    (('training_contract','constraint_path'),'observed_reconstruction','training_contract.constraint_path'),
    (('training_contract','epochs'),999,'training_contract'),
    (('forecast_parameters',),91,'forecast_parameters'),
    (('conditioning','manifold_origin_information'),False,'conditioning'),
])
def test_guided_constraint_ablation_does_not_normalize_away_confounds(tmp_path,path,value,message):
    control=forecast_only_guide()
    target=control
    for key in path[:-1]:
        target=target[key]
    target[path[-1]]=value
    with pytest.raises(ValueError,match=message):
        run_compare(tmp_path,[control,guide_report()])
