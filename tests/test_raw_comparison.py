"""Three-arm forecasts compare physical outputs without confusing their controls."""
from copy import deepcopy
import json
import pytest

from climate_manifold.downstream.compare import compare
from climate_manifold.downstream.climode_benchmark import benchmark


def report(arm, seed=7, family='neural_ode', physical=True):
    raw = arm=='raw'
    error = {'raw':4.,'forecast_only':3.,'climate_manifold':2.}[arm]
    row = {
        'format':'climate_manifold.downstream_evaluation.v1',
        'a_sha256':None,'archive_sha256':'archive','information_sha256':'information',
        'split':'validation','origin_times':['t'],'successful_origin_times':['t'],
        'lead_hours':[6.,12.],'seed':seed,
        'config':{'model':family,'bridge':'raw' if raw else 'latent','anchor':'none',
                  'training_mode':'joint','latent_layout':'spatial','raw_backend':'matched',
                  'hidden_dim':32,'ode_substeps':2,'condition_information':True},
        'representation_config':{'representation_kind':'spatial','latent_channels':32,
            'grid':[4,5,9],'state_dim':180,'history_steps':6,'history_stride':4,
            'step_hours':6,'spatial_downsample':2},
        'regularization':'full' if arm=='climate_manifold' else 'none',
        'initialization':'fresh','representation_training':'none' if raw else 'jointly_trained',
        'representation_sha256':None if raw else f'{arm}-{seed}',
        'objective_weights':{'reconstruction':0. if raw else .1,
                             'physics':.1 if arm=='climate_manifold' else 0.},
        'training_contract':{'epochs':3,'horizon_steps':2,'reconstruction_weight':.1},
        'scores':{'aggregate':{'normalized_rmse':error}},
        'finite_forecast_fraction':1.,'trainable_parameters':60 if raw else 100,
        'total_parameters':60 if raw else 100,'inference_seconds':1.,'training_seconds':2.,
        'implementation':('raw_matched_' if raw else 'latent_')+family,
        'conditioning':{'direct_origin_information':raw,'manifold_origin_information':not raw,
                        'observed_information_available':True},
    }
    if family=='climode':
        row['transport_contract']={
            'velocity_bound_cells_per_day':4. if raw else 2.,
            'raw_velocity_rate_bound_per_day':1.,'reference_spatial_downsample':2,
            'speed_scaling':'source_grid_factor' if raw else 'latent_grid',
            'uncertainty':'deterministic'}
    if physical:
        row['scores']['climode']={
            'protocol':{'name':'physical','climatology':'train_only'},'case_count':1,
            'per_variable':{'t2m':{'units':'K','by_lead':[
                {'lead_hours':lead,'rmse':error,'rmse_valid_cases':1,
                 'acc':1-error/10,'acc_valid_cases':1,'crps':None,'crps_valid_cases':0,'case_count':1}
                for lead in (6.,12.)]}}}
    return row


def run_compare(tmp_path, reports):
    paths=[]
    for index, row in enumerate(reports):
        path=tmp_path/f'report-{index}.json'
        path.write_text(json.dumps(row));paths.append(path)
    return compare(paths,tmp_path/'comparison.json')


@pytest.mark.parametrize('family',['neural_ode','climode'])
def test_three_arms_pair_both_raw_comparisons_and_regularization(tmp_path,family):
    data=[report(arm,seed,family) for seed in (7,19)
          for arm in ('raw','forecast_only','climate_manifold')]
    result=run_compare(tmp_path,data)
    assert result['experiment_suite']=='primary'
    assert result['ranking_allowed']
    assert len(result['seed_summary'])==3
    assert len(result['paired_effects'])==6
    assert set(result['paired_summary'])=={
        family+'/forecast_only/vs_raw',family+'/climate_manifold/vs_raw',
        family+'/climate_manifold/vs_forecast_only'}
    assert all(group['seeds']==[7,19] for group in result['paired_summary'].values())
    direct=result['direct_comparison']
    assert direct['available'] and direct['ranking_allowed']
    assert len(direct['effects'])==8
    for row in direct['effects']:
        full=row['candidate_arm']=='climate_manifold'
        assert row['rmse_skill_vs_raw']==pytest.approx(.5 if full else .25)
        assert row['acc_difference']==pytest.approx(.2 if full else .1)
        assert row['crps_skill_vs_raw'] is None
        assert row['raw_parameters']==60 and row['candidate_parameters']==100
    assert (tmp_path/'comparison.raw-effects.csv').exists()
    assert result['climode_benchmark']['reference_seeds']==[]
    assert not result['climode_benchmark']['ranking_allowed']


def test_raw_comparison_metrics_optional_for_old_reports(tmp_path):
    result=run_compare(tmp_path,[report('raw',physical=False),report('forecast_only',physical=False)])
    assert len(result['paired_effects'])==1
    assert not result['direct_comparison']['available']
    assert len(result['direct_comparison']['missing_metric_pairs'])==1
    assert not (tmp_path/'comparison.raw-effects.csv').exists()


def test_physical_comparison_does_not_require_mixed_unit_scalar(tmp_path):
    data=[report('raw'),report('forecast_only')]
    for row in data:row['scores']['aggregate']=None
    result=run_compare(tmp_path,data)
    assert result['paired_effects']==[]
    assert len(result['direct_comparison']['effects'])==2


def test_undefined_acc_and_zero_reference_are_not_skill(tmp_path):
    raw=report('raw');candidate=report('forecast_only')
    raw['scores']['climode']['per_variable']['t2m']['by_lead'][0]['rmse']=0.
    candidate['scores']['climode']['per_variable']['t2m']['by_lead'][0].update(acc=None,acc_valid_cases=0)
    effect=run_compare(tmp_path,[raw,candidate])['direct_comparison']['effects'][0]
    assert effect['rmse_skill_vs_raw'] is None
    assert effect['acc_difference'] is None


def test_raw_information_availability_must_match_latent_arm(tmp_path):
    raw=report('raw');candidate=report('forecast_only')
    raw['conditioning']['observed_information_available']=False
    with pytest.raises(ValueError,match='observed_information_available'):
        run_compare(tmp_path,[raw,candidate])


def test_raw_physical_protocol_must_match(tmp_path):
    raw=report('raw');candidate=report('forecast_only')
    candidate['scores']['climode']['protocol']['climatology']='test_year'
    with pytest.raises(ValueError,match='metric protocol'):
        run_compare(tmp_path,[raw,candidate])


def test_matched_raw_transport_cannot_be_original_climode_reference():
    raw=report('raw',family='climode');candidate=report('forecast_only',family='climode')
    with pytest.raises(ValueError,match='raw_backend=legacy'):
        benchmark([candidate],[raw])
    legacy=deepcopy(raw);legacy['config']['raw_backend']='legacy'
    legacy['implementation']='official_climode_custom_data_adaptation'
    result=benchmark([candidate],[legacy])
    assert result['ranking_allowed']
    assert 'raw_backend=legacy' in result['reference']


def test_failed_cases_suppress_raw_ranking(tmp_path):
    raw=report('raw',physical=False);candidate=report('forecast_only',physical=False)
    candidate['finite_forecast_fraction']=0.
    candidate['successful_origin_times']=[]
    result=run_compare(tmp_path,[raw,candidate])
    assert not result['ranking_allowed']
    assert result['paired_effects']==[]
    assert not result['direct_comparison']['ranking_allowed']


def test_raw_export_path_is_protected_from_overwrite(tmp_path):
    (tmp_path/'comparison.raw-effects.csv').write_text('existing result')
    with pytest.raises(FileExistsError):run_compare(tmp_path,[report('raw')])


@pytest.mark.parametrize('field',['raw_backend','implementation'])
def test_raw_seed_pool_rejects_backend_or_implementation_mix(tmp_path,field):
    first=report('raw',seed=7);second=report('raw',seed=19)
    if field=='raw_backend':
        second['config']['raw_backend']='legacy'
    else:
        second[field]='a_different_neural_ode'
    with pytest.raises(ValueError,match=field):
        run_compare(tmp_path,[first,second])


@pytest.mark.parametrize('field,value',[
    ('grid',[4,9,5]),('state_dim',360),('history_steps',3),
    ('history_stride',2),('step_hours',12),
])
def test_raw_and_latent_require_common_observed_input_contract(tmp_path,field,value):
    raw=report('raw');candidate=report('forecast_only')
    raw['representation_config'][field]=value
    with pytest.raises(ValueError,match='input '+field):
        run_compare(tmp_path,[raw,candidate])


def test_input_contract_supports_all_missing_legacy_but_not_partial_metadata(tmp_path):
    raw=report('raw');candidate=report('forecast_only')
    del raw['representation_config']['history_steps']
    with pytest.raises(ValueError,match='input history_steps'):
        run_compare(tmp_path,[raw,candidate])
    del candidate['representation_config']['history_steps']
    result=run_compare(tmp_path,[raw,candidate])
    assert result['ranking_allowed']


@pytest.mark.parametrize('field,value',[
    ('reference_spatial_downsample',4),('velocity_bound_cells_per_day',2.),
    ('raw_velocity_rate_bound_per_day',2.),('speed_scaling','latent_grid'),
])
def test_raw_transport_scaling_contract_is_validated(tmp_path,field,value):
    raw=report('raw',family='climode');candidate=report('forecast_only',family='climode')
    raw['transport_contract'][field]=value
    with pytest.raises(ValueError,match=field):
        run_compare(tmp_path,[raw,candidate])


def test_transport_reference_factor_must_match_representation_config(tmp_path):
    raw=report('raw',family='climode');candidate=report('forecast_only',family='climode')
    raw['representation_config']['spatial_downsample']=4
    with pytest.raises(ValueError,match='reference_spatial_downsample'):
        run_compare(tmp_path,[raw,candidate])


@pytest.mark.parametrize('field,value',[
    ('latent_max_speed',3.),('latent_max_acceleration',2.),
])
def test_transport_effective_bounds_match_declared_predictor_options(tmp_path,field,value):
    raw=report('raw',family='climode');candidate=report('forecast_only',family='climode')
    for row in (raw,candidate):row['config'][field]=value
    with pytest.raises(ValueError,match=field):
        run_compare(tmp_path,[raw,candidate])


def test_transport_contract_supports_old_reports_but_rejects_partial_metadata(tmp_path):
    raw=report('raw',family='climode');candidate=report('forecast_only',family='climode')
    del raw['transport_contract']
    with pytest.raises(ValueError,match='missing transport_contract'):
        run_compare(tmp_path,[raw,candidate])
    del candidate['transport_contract']
    result=run_compare(tmp_path,[raw,candidate])
    assert result['ranking_allowed']
