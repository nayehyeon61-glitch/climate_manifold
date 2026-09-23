from copy import deepcopy
import json
import numpy as np
import pytest
import torch
from climate_manifold.downstream.metrics import ForecastMetrics
from climate_manifold.downstream.climode_benchmark import benchmark, main


def schema():
    return {'state_dim':4,'variables':[{'name':'t2m','shape':[2,2], 'dims':['lat','lon'], 'slice':[0,4],
                         'coords':{'lat':[0,60],'lon':[0,90]}, 'attrs':{'units':'K'}}]}


def scores(error=1., gaussian=False, batch_size=2):
    metrics=ForecastMetrics(schema(),np.arange(4)+270.,np.ones(4)*2,[6,12])
    truth=torch.tensor([[-1.,1.,-2.,2.]]).expand(2,2,4).clone()
    prediction=truth+error/2
    for start in range(0,2,batch_size):
        y=truth[start:start+batch_size];p=prediction[start:start+batch_size]
        metrics.update(p,y,y[:,0],std=torch.ones_like(p) if gaussian else None)
    return metrics.result()


def report(model='mlp',bridge='latent',error=1.,seed=7):
    return {'format':'climate_manifold.downstream_evaluation.v1',
            'a_sha256':'a','archive_sha256':'archive','information_sha256':'info',
            'split':'validation','origin_times':['a','b'],'successful_origin_times':['a','b'],
            'lead_hours':[6.,12.],'config':{'model':model,'bridge':bridge,'anchor':'none'},
            'seed':seed,'scores':scores(error,model=='climode'), 'finite_forecast_fraction':1.}


def test_casewise_rmse_is_not_pooled_and_uses_physical_latitude_weights():
    metrics=ForecastMetrics(schema(),np.zeros(4),np.ones(4)*2,[6])
    truth=torch.zeros(2,1,4)
    prediction=torch.tensor([[[.5,.5,1.5,1.5]],[[1.5,1.5,4.5,4.5]]])
    metrics.update(prediction,truth,truth[:,0])
    result=metrics.result();row=result['climode']['per_variable']['t2m']['by_lead'][0]
    # Latitude 0/60 weights are 2/3 and 1/3; physical case RMSEs are r and 3r.
    r=np.sqrt(11/3)
    assert row['rmse']==pytest.approx(2*r)
    assert row['rmse_std']==pytest.approx(r)
    assert result['per_variable']['t2m']['by_lead'][0]['rmse']==pytest.approx(np.sqrt(5)*r)
    assert row['acc'] is None and row['acc_valid_cases']==0
    assert row['crps'] is None


def test_acc_centers_anomalies_and_batching_does_not_change_scores():
    batch=scores(20.,True,2)['climode']
    separate=scores(20.,True,1)['climode']
    assert batch==separate
    row=batch['per_variable']['t2m']['by_lead'][0]
    # A spatially constant bias changes RMSE, but centered anomaly correlation is 1.
    assert row['acc']==pytest.approx(1.)
    assert row['rmse']==pytest.approx(20.)
    assert row['acc_valid_cases']==2


def test_gaussian_crps_physical_units_and_empty_report():
    result=scores(0.,True)['climode']
    row=result['per_variable']['t2m']['aggregate']
    assert row['crps']==pytest.approx(2*(np.sqrt(2)-1)/np.sqrt(np.pi))
    empty=ForecastMetrics(schema(),np.zeros(4),np.ones(4),[6]).result()['climode']
    assert empty['case_count']==0
    assert empty['per_variable']['t2m']['aggregate']['rmse'] is None


def test_reference_skill_and_incompatible_protocol_rejected():
    candidate=report();reference=report('climode','raw',2.)
    result=benchmark([candidate],[reference]);effect=result['effects'][0]
    assert result['ranking_allowed']
    assert effect['rmse_skill_vs_climode']==pytest.approx(.5)
    assert effect['acc_difference']==pytest.approx(0.)
    assert effect['crps_skill_vs_climode'] is None
    for key in ('origin_times','lead_hours','archive_sha256','a_sha256'):
        bad=deepcopy(reference);bad[key]=['different'] if isinstance(bad[key],list) else 'different'
        with pytest.raises(ValueError,match=key):benchmark([candidate],[bad])
    bad=deepcopy(reference);bad['scores']['climode']['protocol']['climatology']='test mean'
    with pytest.raises(ValueError,match='climatology'):benchmark([candidate],[bad])


def test_failures_and_undefined_acc_cannot_claim_improvement():
    candidate=report();reference=report('climode','raw',2.)
    candidate['finite_forecast_fraction']=.5;candidate['successful_origin_times']=['a']
    candidate['scores']['climode']['case_count']=1
    result=benchmark([candidate],[reference])
    assert not result['ranking_allowed']
    assert all(row['rmse_skill_vs_climode'] is None for row in result['effects'])
    candidate=report()
    candidate['scores']['climode']['per_variable']['t2m']['by_lead'][0]['acc_valid_cases']=1
    result=benchmark([candidate],[reference])
    assert result['effects'][0]['acc_difference'] is None
    assert result['effects'][1]['acc_difference']==pytest.approx(0.)
    assert not benchmark([report(seed=19)],[reference])['ranking_allowed']


def test_pure_a_report_and_cli_export(tmp_path):
    candidate=report()
    candidate['format']='climate_manifold.dynamics_evaluation.v1'
    candidate['checkpoint_sha256']=candidate.pop('a_sha256')
    candidate['config']={};candidate.pop('seed')
    reference=report('climode','raw',2.)
    paths=[tmp_path/'a.json',tmp_path/'reference.json']
    for path,data in zip(paths,[candidate,reference]):path.write_text(json.dumps(data))
    output=tmp_path/'comparison.json'
    assert main(['--reports',str(paths[0]),'--climode-reference-reports',str(paths[1]),'--output',str(output)])==0
    result=json.loads(output.read_text())
    assert result['effects'][0]['model']=='a_drift'
    assert result['effects'][0]['seed'] is None
    assert output.with_suffix('.effects.csv').exists()
    with pytest.raises(FileExistsError):main(['--reports',str(paths[0]),'--climode-reference-reports',str(paths[1]),'--output',str(output)])


def test_reference_must_be_raw_and_seeds_unique():
    with pytest.raises(ValueError,match='raw ClimODE'):
        benchmark([report()],[report('climode','decoded')])
    ref=report('climode','raw')
    with pytest.raises(ValueError,match='Duplicate'):
        benchmark([report()],[ref,ref])
