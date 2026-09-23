from copy import deepcopy
from dataclasses import asdict
import json
import numpy as np
import pytest
import torch
from test_pinn_training import pinn_prepared
from climate_manifold.downstream.bridge import ManifoldBridge
from climate_manifold.downstream.pipeline import ForecastPipeline,PredictorConfig
from climate_manifold.downstream.climode import ClimODEPredictor,validate_constants,advection,fit_initial_velocity
from climate_manifold.downstream.metrics import ForecastMetrics,gaussian_crps
from climate_manifold.downstream.train import CausalWindows,forecast_loss,load_predictor
from climate_manifold.downstream.compare import compare


def seal(model,data):
    model.seal(torch.tensor((data['states'][:data['train_end']]-data['mean'])/data['scale']),
               torch.tensor(data['information'][:data['train_end']]))
    return model


@pytest.mark.parametrize('family,mode',[('mlp','raw'),('mlp','latent'),('mlp','decoded'),('neural_ode','raw'),('neural_ode','latent')])
def test_gradients_train_predictor_through_frozen_decoder(pinn_prepared,family,mode):
    a,batch,data,_=pinn_prepared;seal(a,data)
    before={k:v.clone() for k,v in a.state_dict().items()}
    pipe=ForecastPipeline(a,PredictorConfig(model=family,bridge=mode,hidden_dim=24))
    pipe.train();leads=torch.tensor([6.,12.])
    if mode!='raw':assert pipe.predictor.information_dim==0
    out=pipe(batch['history'],batch['information'],batch['origin_time_ns'],leads)
    assert out['mean'].shape==(2,2,128)
    values=forecast_loss(out,batch,a.temporal,leads,.1)
    values['loss'].backward()
    params=[p for p in pipe.parameters() if p.requires_grad]
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in params)
    torch.optim.Adam(params,lr=1e-3).step()
    assert all(torch.equal(v,a.state_dict()[k]) for k,v in before.items())
    assert all(p.grad is None for p in a.parameters())
    if mode!='raw':assert not pipe.bridge.manifold.training


def test_bridge_coordinate_contract_and_anchor(pinn_prepared):
    a,batch,data,_=pinn_prepared;seal(a,data)
    bridge=ManifoldBridge(a,'latent','none')
    q=bridge.encode_history(batch['history'],batch['information'])
    assert q.shape==(2,6,4)
    expected=a.raw_encode(batch['origin'],batch['information'])
    torch.testing.assert_close(q[:,-1],(expected-a.core.latent_mean)/a.core.latent_scale)
    bridge.anchor='origin'
    pred=bridge.to_fields(q[:,-1,None].expand(-1,3,-1),batch['origin'],q[:,-1])
    torch.testing.assert_close(pred,batch['origin'][:,None].expand(-1,3,-1))
    with pytest.raises(ValueError,match='not a reshaped global latent'):
        PredictorConfig(model='climode',bridge='latent')


def test_causal_loader_never_reads_future_information(pinn_prepared):
    a,_,data,_=pinn_prepared
    origin=a.config.history_span_steps-1
    class OriginOnly:
        def __getitem__(self,index):
            assert index==origin,'Future information was accessed'
            return data['information'][index]
    ds=CausalWindows(data['states'],data['times'],a.config,[0],data['mean'],data['scale'],data['schema'],information=OriginOnly())
    row=ds[0]
    assert 'information_targets' not in row
    np.testing.assert_array_equal(row['information'].numpy(),data['information'][origin])


def test_pooled_metrics_known_errors_units_and_gaussian(pinn_prepared):
    a,_,data,_=pinn_prepared
    # Unit scale exposes analytically known physical errors of 1 and 3.
    metrics=ForecastMetrics(data['schema'],np.zeros(128),np.ones(128),[6,12])
    truth=torch.zeros(2,2,128);origin=torch.zeros(2,128)
    mean=torch.stack((torch.ones(2,128),3*torch.ones(2,128)))
    metrics.update(mean,truth,origin,std=torch.ones_like(mean))
    result=metrics.result()
    for row in result['per_variable'].values():
        assert row['aggregate']['rmse']==pytest.approx(np.sqrt(5))
        assert row['aggregate']['mae']==pytest.approx(2)
        assert row['aggregate']['coverage80']==pytest.approx(.5)
        assert row['aggregate']['spread_skill_ratio']==pytest.approx(1/np.sqrt(5))
        assert row['aggregate']['acc_train_mean'] is None
    score=gaussian_crps(torch.tensor(0.),torch.tensor(1.),torch.tensor(0.))
    assert float(score)==pytest.approx((np.sqrt(2)-1)/np.sqrt(np.pi),rel=1e-6)


def test_constants_are_real_aligned_and_unit_checked(pinn_prepared):
    _,_,data,_=pinn_prepared
    schema=data['schema'];coords=schema['variables'][0]['coords']
    constants={**coords,'orography':np.ones((4,8)),'lsm':np.ones((4,8)),'orography_units':'m'}
    validate_constants(constants,schema)
    with pytest.raises(ValueError,match='units'):validate_constants({**constants,'orography_units':'unknown'},schema)
    with pytest.raises(ValueError,match='land|Land'):validate_constants({**constants,'lsm':np.full((4,8),2)},schema)
    with pytest.raises(ValueError,match='coordinates'):validate_constants({**constants,'lat':[0,1,2,3]},schema)


@pytest.mark.parametrize('mode',['raw','decoded'])
def test_real_climode_backend_forward_backward_and_frozen_a(pinn_prepared,mode):
    pytest.importorskip('torchdiffeq')
    a,batch,data,_=pinn_prepared;seal(a,data)
    constants={**data['schema']['variables'][0]['coords'],'orography':np.ones((4,8)),'lsm':np.zeros((4,8)),'orography_units':'m'}
    pipe=ForecastPipeline(a,PredictorConfig(model='climode',bridge=mode,climode_attention=False,
        velocity_iterations=2,climode_step_hours=6),constants,data['schema'])
    out=pipe(batch['history'],batch['information'],batch['origin_time_ns'],torch.tensor([6.,12.]))
    assert out['mean'].shape==out['std'].shape==(2,2,128)
    assert torch.isfinite(out['mean']).all() and (out['std']>0).all()
    values=forecast_loss(out,batch,a.temporal,torch.tensor([6.,12.]),.1)
    values['loss'].backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in pipe.predictor.parameters())
    assert all(p.grad is None for p in a.parameters())
    pipe.eval()
    with torch.no_grad():
        first=pipe(batch['history'],batch['information'],batch['origin_time_ns'],torch.tensor([6.]))
        second=pipe(batch['history'],batch['information'],batch['origin_time_ns'],torch.tensor([6.]))
    torch.testing.assert_close(first['mean'],second['mean'],rtol=0,atol=0)


def test_climode_attention_on_supported_grid():
    pytest.importorskip('torchdiffeq');torch.set_num_threads(1)
    constants={'lat':np.linspace(-80,80,16),'lon':np.arange(16)*22.5,
               'orography':np.ones((16,16)),'lsm':np.ones((16,16)),'orography_units':'m'}
    model=ClimODEPredictor((4,16,16),constants,attention=True,velocity_iterations=1,step_hours=6)
    model.eval()
    with torch.no_grad():
        mean,std=model(torch.randn(1,2,1024),torch.tensor([6.]),torch.tensor([978307200000000000]))
    assert mean.shape==std.shape==(1,1,1024)
    assert torch.isfinite(mean).all()


def test_velocity_fit_reduces_observed_backward_tendency_error():
    torch.set_num_threads(1)
    field=torch.linspace(0,1,8).repeat(1,4,4,1)
    history=torch.stack((field-.02,field),1)
    target=(history[:,-1]-history[:,-2])/.24
    velocity=fit_initial_velocity(history,24,iterations=30)
    assert (advection(field,velocity)-target).square().mean()<target.square().mean()


def test_comparison_rejects_unmatched_origins_and_flags_failed_subsets(tmp_path):
    base={'format':'climate_manifold.downstream_evaluation.v1','a_sha256':'a','archive_sha256':'b',
          'information_sha256':'c','split':'validation','origin_times':['x','y'],'successful_origin_times':['x','y'],
          'lead_hours':[6],'config':{'model':'mlp','bridge':'raw','anchor':'none'},'seed':7,
          'scores':{'aggregate':{'normalized_rmse':1}},'finite_forecast_fraction':1,
          'trainable_parameters':10,'total_parameters':10,'inference_seconds':1,'training_seconds':1,'conditioning':{}}
    paths=[tmp_path/'a.json',tmp_path/'b.json']
    paths[0].write_text(json.dumps(base));other=deepcopy(base);other['config']['bridge']='latent'
    other['origin_times']=['z'];paths[1].write_text(json.dumps(other))
    with pytest.raises(ValueError,match='origin_times'):compare(paths,tmp_path/'bad.json')
    other['origin_times']=base['origin_times'];other['successful_origin_times']=['x'];other['finite_forecast_fraction']=.5
    paths[1].write_text(json.dumps(other))
    result=compare(paths,tmp_path/'fairness.json')
    assert not result['ranking_allowed']


def test_prepare_real_constants_grid_and_geopotential_units(pinn_prepared,tmp_path):
    import xarray as xr
    from prepare_climode_constants import prepare
    _,_,data,archive=pinn_prepared
    coords=data['schema']['variables'][0]['coords']
    ds=xr.Dataset({'z':(('lat','lon'),np.full((4,8),500*9.80665)),
                   'lsm':(('lat','lon'),np.full((4,8),.4))},coords=coords)
    ds.z.attrs['units']='m**2 s**-2'
    path=tmp_path/'constants.nc';ds.to_netcdf(path)
    output=prepare(archive,path,tmp_path/'constants.npz')
    with np.load(output) as f:
        np.testing.assert_allclose(f['orography'],500)
        np.testing.assert_allclose(f['lsm'],.4)
        assert str(f['orography_units'])=='m'
