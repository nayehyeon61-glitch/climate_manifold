"""Pair identities, constraint routes and common weights cannot be pooled away."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from climate_manifold.downstream.climode_benchmark import benchmark
from climate_manifold.downstream.compare import CONSTRAINT_GROUPS
from climate_manifold.downstream.constraint_protocol import CONSTRAINT_DECODERS, make_constraint_contract
from test_raw_comparison import report, run_compare


def pair_report(pair, seed=7, family='neural_ode'):
    row = report('climate_manifold', seed, family)
    groups = CONSTRAINT_GROUPS[pair]
    row.update(constraint_pair=pair, constraint_path='observed_reconstruction',
        constraint_contract={'version':'climate_manifold.reconstruction_constraints.v1',
            'observed_pair':'origin-6h,origin','groups':sorted(groups),
            'pinn_tendency_supervision':False},
        split_objective_weights={'reconstruction':.1,'pinn':.1 if 'pinn' in groups else 0.,
            'statistical':.1 if 'statistical' in groups else 0.,'static':.05 if 'static' in groups else 0.},
        forecast_parameters=80, constraint_parameters=30 if 'pinn' in groups else 20)
    row['total_parameters'] = row['trainable_parameters'] = 80+row['constraint_parameters']
    row['training_contract']['constraint_path'] = 'observed_reconstruction'
    row['objective_weights'] = dict(row['split_objective_weights'])
    return row


def test_pairs_are_distinct_arms_and_seed_groups_with_physical_effects(tmp_path):
    data = [pair_report(pair,seed) for seed in (7,19) for pair in CONSTRAINT_GROUPS]
    result = run_compare(tmp_path,data)
    assert {row['candidate_arm'] for row in result['rows']} == set(CONSTRAINT_GROUPS)
    assert len(result['seed_summary']) == 3
    assert all(row['seeds']==[7,19] for row in result['seed_summary'].values())
    assert len(result['paired_effects']) == 6
    assert len(result['paired_summary']) == 3
    assert len(result['constraint_pair_effects']) == 12  # 3 pairs x 2 seeds x 2 leads
    assert (tmp_path/'comparison.constraint-effects.csv').exists()
    assert {row['interpretation'] for row in result['paired_effects']} == {
        'replace_one_constraint_group_with_one_fixed'}
    for row in result['paired_effects']:
        baseline = CONSTRAINT_GROUPS[row['control']]
        candidate = CONSTRAINT_GROUPS[row['candidate_arm']]
        assert {row['fixed_group']} == baseline & candidate
        assert {row['removed_group']} == baseline-candidate
        assert {row['added_group']} == candidate-baseline
    assert not result['direct_comparison']['available']


def test_pair_field_effects_do_not_depend_on_mixed_unit_aggregate(tmp_path):
    rows=[pair_report(pair) for pair in CONSTRAINT_GROUPS]
    for row in rows:row['scores']['aggregate']=None
    result=run_compare(tmp_path,rows)
    assert result['paired_effects']==[]
    assert len(result['constraint_pair_effects'])==6


def scoped_pair_report(pair, seed=7, decoder='information_only'):
    row = pair_report(pair, seed)
    row['constraint_decoder'] = decoder
    row['constraint_contract'] = make_constraint_contract(pair, CONSTRAINT_GROUPS[pair], decoder)
    return row


@pytest.mark.parametrize('decoder', CONSTRAINT_DECODERS)
def test_scoped_pairs_accept_matched_raw_and_export_scope(tmp_path, decoder):
    data = [scoped_pair_report(pair, decoder=decoder) for pair in CONSTRAINT_GROUPS]
    result = run_compare(tmp_path, [report('raw'), *data])
    assert result['direct_comparison']['ranking_allowed']
    assert len(result['direct_comparison']['effects']) == 6
    assert result['rows'][0]['constraint_decoder'] is None
    assert all(row['constraint_decoder'] == decoder for row in result['rows'][1:])
    assert len(result['constraint_pair_effects']) == 6


def test_legacy_scope_is_inferred_and_matches_explicit_both_mode(tmp_path):
    data = [pair_report('pinn_statistical'),
            scoped_pair_report('pinn_statistical', seed=19, decoder='surface_and_information')]
    result = run_compare(tmp_path, data)
    assert len(result['seed_summary']) == 1
    assert all(row['constraint_decoder'] == 'surface_and_information' for row in result['rows'])
    assert result['rows'][0]['constraint_contract']['version'].endswith('.v1')


@pytest.mark.parametrize('second_pair,second_seed', [('pinn_static', 7), ('pinn_statistical', 19)])
@pytest.mark.parametrize('first_decoder,second_decoder', [
    ('information_only', 'surface_and_information'),
    ('information_only', 'separate_surface_and_information'),
    ('surface_and_information', 'separate_surface_and_information'),
])
def test_mixed_decoder_scopes_cannot_be_compared_or_pooled(
        tmp_path, second_pair, second_seed, first_decoder, second_decoder):
    data = [scoped_pair_report('pinn_statistical', decoder=first_decoder),
            scoped_pair_report(second_pair, second_seed, decoder=second_decoder)]
    with pytest.raises(ValueError, match='constraint_decoder scope'):
        run_compare(tmp_path, data)


def test_explicit_decoder_metadata_cannot_contradict_contract(tmp_path):
    row = scoped_pair_report('pinn_statistical')
    row['constraint_decoder'] = 'surface_and_information'
    with pytest.raises(ValueError, match='constraint_decoder disagrees'):
        run_compare(tmp_path, [row])


def test_raw_cannot_declare_a_decoder_constraint(tmp_path):
    raw = report('raw')
    raw['constraint_decoder'] = 'information_only'
    with pytest.raises(ValueError, match='Raw controls must not declare'):
        run_compare(tmp_path, [raw, scoped_pair_report('pinn_statistical')])


@pytest.mark.parametrize('field,value,message',[
    ('constraint_path','forecast_trajectory','constraint_path'),
    ('constraint_pair','all_three','constraint_pair'),
    ('forecast_parameters',90,'forecast_parameters'),
    ('constraint_contract',None,'constraint_contract'),
    ('split_objective_weights',None,'split_objective_weights'),
])
def test_pair_metadata_mismatch_is_rejected(tmp_path,field,value,message):
    first=pair_report('pinn_statistical');second=pair_report('pinn_static')
    second[field]=value
    with pytest.raises(ValueError,match=message):run_compare(tmp_path,[first,second])


def test_pairs_cannot_mix_routes_or_change_fixed_group_weights(tmp_path):
    first=pair_report('pinn_statistical');second=pair_report('pinn_static')
    second['split_objective_weights']['pinn']=.2
    with pytest.raises(ValueError,match='shared split_objective_weights.pinn'):
        run_compare(tmp_path,[first,second])
    with pytest.raises(ValueError,match='mixed constraint_path'):
        run_compare(tmp_path,[first,report('climate_manifold',19)])


def test_pinn_internal_weights_cannot_be_pooled_or_replaced_silently(tmp_path):
    first=pair_report('pinn_statistical');second=pair_report('pinn_static')
    first['pinn_config']={'continuity_weight':.1,'levels_hpa':[500,850]}
    second['pinn_config']={'continuity_weight':.2,'levels_hpa':[500,850]}
    with pytest.raises(ValueError,match='pinn_config'):run_compare(tmp_path,[first,second])


@pytest.mark.parametrize('key,value',[
    ('observed_pair','origin,origin+6h'),('pinn_tendency_supervision',True),
])
def test_constraint_contract_cannot_change_observations_or_double_tendencies(tmp_path,key,value):
    first=pair_report('pinn_statistical');second=pair_report('pinn_static')
    second['constraint_contract'][key]=value
    with pytest.raises(ValueError,match='constraint_contract'):run_compare(tmp_path,[first,second])


@pytest.mark.parametrize('decoder', [None, *CONSTRAINT_DECODERS])
def test_pairs_preserved_in_climode_benchmark_identity(decoder):
    data=[pair_report(pair) if decoder is None else scoped_pair_report(pair, decoder=decoder)
          for pair in CONSTRAINT_GROUPS]
    reference=report('raw',family='climode')
    reference['config']['raw_backend']='legacy'
    result=benchmark(data,[reference])
    assert {row['constraint_pair'] for row in result['effects']} == set(CONSTRAINT_GROUPS)
    assert {row['constraint_path'] for row in result['rows']} == {'observed_reconstruction'}
    expected_decoder = decoder or 'surface_and_information'
    for name in ('rows', 'effects'):
        assert {row['constraint_decoder'] for row in result[name]} == {expected_decoder}


@pytest.mark.parametrize('field,value',[
    ('total_parameters',120),('constraint_parameters',40),('implementation','other_impl'),
])
def test_same_pair_seed_pool_cannot_hide_capacity_or_implementation_changes(tmp_path,field,value):
    first=pair_report('pinn_statistical');second=pair_report('pinn_statistical',19)
    second[field]=value
    with pytest.raises(ValueError,match=field):run_compare(tmp_path,[first,second])


def _runner(tmp_path, **settings):
    stub=tmp_path/'record-python'
    stub.write_text('#!'+sys.executable+'\nimport json,os,sys\n'
        'with open(os.environ["CALLS"],"a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n')
    stub.chmod(0o755)
    calls=tmp_path/'calls.jsonl'
    env={key:value for key,value in os.environ.items() if key not in (
        'MODELS','SEEDS','PAIRS','A_CHECKPOINT','TRAINING_MODE','INITIALIZATION',
        'LATENT_LAYOUT','ANCHOR','CLIMODE_REFERENCE_DIR','INCLUDE_RAW','BATCH_SIZE','CONSTRAINT_DECODER')}
    env.update(PYTHON=str(stub),CALLS=str(calls),ARCHIVE='surface archive.npz',
               INFO='physical information',RUN=str(tmp_path/'run'),**settings)
    script=Path(__file__).resolve().parents[1]/'scripts/run_pairwise_manifold_comparison.sh'
    outcome=subprocess.run(['bash',str(script)],env=env,text=True,capture_output=True)
    commands=[json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
    return outcome,commands


def test_runner_dispatches_18_distinct_fits_and_keeps_legacy_losses_out(tmp_path):
    outcome,commands=_runner(tmp_path,INCLUDE_RAW='0')
    assert outcome.returncode==0,outcome.stderr
    train=[row for row in commands if row[1]=='climate_manifold.downstream.train']
    evaluate=[row for row in commands if row[1]=='climate_manifold.downstream.evaluate']
    assert len(train)==len(evaluate)==18
    assert len({row[row.index('--output')+1] for row in train})==18
    for row in train:
        pair=row[row.index('--constraint-pair')+1]
        assert ('--pinn' in row)==('pinn' in CONSTRAINT_GROUPS[pair])
        assert ('--pinn-weight' in row)==('pinn' in CONSTRAINT_GROUPS[pair])
        assert row[row.index('--archive')+1]=='surface archive.npz'
        assert row[row.index('--information')+1]=='physical information'
        assert '--statistical-weight' in row
        assert not {'--physics-weight','--distribution-weight','--information-weight'} & set(row)
    compare=commands[-1]
    assert compare[1]=='climate_manifold.downstream.compare'
    assert len(compare[compare.index('--reports')+1:compare.index('--output')])==18


@pytest.mark.parametrize('settings',[
    {'PAIRS':'pinn_static pinn_static'}, {'INITIALIZATION':'pretrained'},
    {'TRAINING_MODE':'frozen'}, {'LATENT_LAYOUT':'global'}, {'PAIRS':'all_three'},
])
def test_runner_invalid_comparisons_fail_before_training(tmp_path,settings):
    outcome,commands=_runner(tmp_path,**settings)
    assert outcome.returncode!=0 and not commands


def test_runner_requires_new_output_directory(tmp_path):
    (tmp_path/'run').mkdir()
    outcome,commands=_runner(tmp_path)
    assert outcome.returncode!=0 and not commands
    assert 'Choose a new RUN' in outcome.stderr


@pytest.mark.parametrize('decoder', ['information_only', 'surface_and_information'])
def test_v2_reports_pool_only_with_equivalent_v3_scope(tmp_path, decoder):
    first = scoped_pair_report('pinn_statistical', decoder=decoder)
    first['constraint_contract']['version'] = 'climate_manifold.reconstruction_constraints.v2'
    first['constraint_contract'].pop('forecast_decoder_in_constraints')
    first['constraint_contract'].pop('separate_reconstruction_decoder')
    second = scoped_pair_report('pinn_statistical', seed=19, decoder=decoder)
    result = run_compare(tmp_path, [first, second])
    assert len(result['seed_summary']) == 1
    assert all(row['constraint_decoder'] == decoder for row in result['rows'])
