"""Joint controls must compare learned representations without conflating seeds."""
from copy import deepcopy
import json
import pytest
from climate_manifold.downstream.compare import compare
from climate_manifold.downstream.climode_benchmark import benchmark


def _report(regularization='full', seed=7, error=1.):
    return {
        'format':'climate_manifold.downstream_evaluation.v1',
        'a_sha256':None, 'archive_sha256':'data', 'information_sha256':'info',
        'split':'validation', 'origin_times':['t'], 'successful_origin_times':['t'],
        'lead_hours':[6.], 'seed':seed,
        'config':{'model':'neural_ode', 'bridge':'latent', 'anchor':'none',
                  'representation':'climate_manifold', 'training_mode':'joint',
                  'hidden_dim':32, 'ode_substeps':2},
        'representation_config':{'manifold_dim':64, 'hidden_dim':512},
        'regularization':regularization, 'initialization':'fresh',
        'representation_training':'jointly_trained',
        'representation_sha256':f'learned-{regularization}-{seed}',
        'objective_weights':{'reconstruction':.1, 'physics':.1 if regularization=='full' else 0.},
        'training_contract':{'epochs':3, 'horizon_steps':1, 'reconstruction_weight':.1},
        'scores':{'aggregate':{'normalized_rmse':error}},
        'finite_forecast_fraction':1., 'trainable_parameters':100,
        'total_parameters':100, 'inference_seconds':1., 'training_seconds':2.,
        'conditioning':{'manifold_origin_information':True},
    }


def _compare(tmp_path, reports, name='comparison'):
    paths=[]
    for index, report in enumerate(reports):
        path=tmp_path/f'{name}-{index}.json'
        path.write_text(json.dumps(report)); paths.append(path)
    return compare(paths, tmp_path/f'{name}.json')


def test_joint_regularization_ablations_pair_seed_without_collision(tmp_path):
    reports=[_report('none',7,2.),_report('full',7,1.),
             _report('none',19,3.),_report('full',19,2.)]
    result=_compare(tmp_path,reports)
    assert result['ranking_allowed']
    assert len(result['seed_summary'])==2
    assert all(group['seeds']==[7,19] for group in result['seed_summary'].values())
    assert len(result['paired_effects'])==2
    assert {row['control'] for row in result['paired_effects']}=={'forecast_only'}
    assert all(row['rmse_reduction']==1. for row in result['paired_effects'])
    assert {row['representation_sha256'] for row in result['rows']}=={
        report['representation_sha256'] for report in reports}
    assert 'encoder, predictor and decoder per seed' in ' '.join(result['notes'])


@pytest.mark.parametrize('field,value,message',[
    ('representation_config',{'manifold_dim':16,'hidden_dim':512},'representation_config'),
    ('initialization','pretrained','initialization'),
    ('training_contract',{'epochs':4},'training_contract'),
    ('conditioning',{'manifold_origin_information':False},'conditioning'),
    ('total_parameters',200,'total_parameters'),
])
def test_joint_comparison_rejects_confounded_controls(tmp_path,field,value,message):
    baseline=_report('none'); candidate=_report()
    candidate[field]=value
    with pytest.raises(ValueError,match=message):_compare(tmp_path,[baseline,candidate])


def test_joint_comparison_rejects_mixed_freezing_and_undeclared_capacity(tmp_path):
    candidate=_report(); baseline=_report('none')
    baseline['config']['training_mode']='frozen'
    with pytest.raises(ValueError,match='training_mode'):
        _compare(tmp_path,[baseline,candidate],'mixed')
    baseline=_report('none'); del baseline['representation_config']
    with pytest.raises(ValueError,match='representation_config'):
        _compare(tmp_path,[baseline,candidate],'unknown')


def test_duplicate_joint_variant_seed_still_rejected(tmp_path):
    with pytest.raises(ValueError,match='Duplicate seed'):
        _compare(tmp_path,[_report(),_report()])


def _physical_report(report, error):
    report=deepcopy(report)
    report['scores']['climode']={
        'protocol':{'name':'same','climatology':'training_only'}, 'case_count':1,
        'per_variable':{'t2m':{'units':'K','by_lead':[
            {'lead_hours':6.,'rmse':error,'rmse_valid_cases':1,
             'acc':.8,'acc_valid_cases':1,'crps':None,'crps_valid_cases':0,'case_count':1}]}}}
    return report


def test_climode_reference_preserves_joint_variant_identity_and_matches_data():
    reports=[_physical_report(_report(reg),1.) for reg in ('none','full')]
    reference=_physical_report(_report(),2.)
    reference['config'].update(model='climode',bridge='raw',training_mode='frozen')
    reference['a_sha256']='legacy-baseline-a'
    result=benchmark(reports,[reference])
    assert result['ranking_allowed']
    assert {row['regularization'] for row in result['effects']}=={'none','full'}
    assert all(row['rmse_skill_vs_climode']==.5 for row in result['effects'])
    reference['archive_sha256']='different-data'
    with pytest.raises(ValueError,match='archive_sha256'):benchmark(reports,[reference])


def test_seed_summary_cannot_pool_different_regularization_weights(tmp_path):
    first=_report(seed=7);second=_report(seed=19)
    second['objective_weights']['physics']=.2
    with pytest.raises(ValueError,match='objective_weights'):
        _compare(tmp_path,[first,second])
