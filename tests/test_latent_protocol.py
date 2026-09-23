"""Primary latent forecasting and comparable representation controls."""
from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
import pytest
import torch

from test_pinn_training import pinn_prepared
from test_downstream import seal
from climate_manifold.downstream.pipeline import PredictorConfig, ForecastPipeline
from climate_manifold.downstream.protocol import validate_experiment
from climate_manifold.downstream.plain_ae import new_plain_ae
from climate_manifold.downstream.metrics import LatentDiagnostics
from climate_manifold.downstream.compare import compare


def test_primary_contract_rejects_decoder_bypass_and_grid_forecasting():
    config=PredictorConfig()
    assert config.model=='neural_ode' and config.bridge=='latent'
    assert validate_experiment(config,'primary')['path']=='encoder -> predictor -> decoder'
    for cfg in (PredictorConfig(anchor='origin'),PredictorConfig(bridge='decoded'),
                PredictorConfig(model='climode',bridge='raw')):
        with pytest.raises(ValueError,match='Primary experiments'):
            validate_experiment(cfg,'primary')
        assert validate_experiment(cfg,'auxiliary')['suite']=='auxiliary'


@pytest.mark.parametrize('representation',['climate_manifold','plain_ae'])
def test_predictor_operates_only_in_frozen_latent_space(pinn_prepared,representation):
    a,batch,data,_=pinn_prepared;seal(a,data)
    ae=None
    if representation=='plain_ae':
        ae=new_plain_ae({'config':asdict(a.config),'information_metadata':a.info_metadata})
        ae.seal(batch['origin'],batch['information'])
    model=ForecastPipeline(a,PredictorConfig(representation=representation,hidden_dim=24),representation=ae)
    frozen={key:value.clone() for key,value in model.bridge.manifold.state_dict().items()}
    def check_inputs(module,args):
        features,_,_,information=args
        assert features.shape==(2,a.config.history_steps,a.config.manifold_dim)
        assert information is None,'Origin information bypassed the encoder'
    handle=model.predictor.register_forward_pre_hook(check_inputs)
    out=model(batch['history'],batch['information'],batch['origin_time_ns'],torch.tensor([6.,12.]))
    handle.remove()
    assert out['predicted_latent'].shape==(2,2,a.config.manifold_dim)
    torch.testing.assert_close(out['mean'],model.bridge.decode(out['predicted_latent']),rtol=0,atol=0)
    out['predicted_latent'].retain_grad()
    (out['mean']-batch['targets'][:,:2]).square().mean().backward()
    assert out['predicted_latent'].grad.abs().sum()>0
    params=list(model.predictor.parameters())
    torch.optim.Adam(params,lr=.001).step()
    assert all(torch.equal(value,model.bridge.manifold.state_dict()[key]) for key,value in frozen.items())
    assert all(parameter.grad is None for parameter in model.bridge.manifold.parameters())


def test_latent_metrics_identify_static_rollout_and_do_not_hide_projection_error(pinn_prepared):
    _,_,data,_=pinn_prepared
    metrics=LatentDiagnostics(data['schema'],[6.,12.])
    target=torch.tensor([[[1.,1.],[2.,2.]]])
    prediction={'predicted_latent':torch.zeros_like(target),'origin_latent':torch.zeros(1,2),
                'mean':torch.zeros(1,2,128),'reconstructed_origin':torch.zeros(1,128)}
    metrics.update(prediction,target,torch.ones(1,2,128),torch.zeros_like(target),torch.full((1,2,128),2.))
    result=metrics.result()['aggregate']
    assert result['latent_rmse']==pytest.approx(np.sqrt(2.5))
    assert result['latent_tendency_amplitude_ratio']==0
    assert result['latent_tendency_rmse_per_hour']==pytest.approx(1/6)
    assert result['future_reconstruction_normalized_rmse']==pytest.approx(1)
    assert result['projected_persistence_normalized_rmse']==pytest.approx(2)
    assert result['latent_cycle_rmse']==0


def test_three_representation_comparison_pairs_forecast_seeds_and_separates_auxiliary(tmp_path):
    base={'format':'climate_manifold.downstream_evaluation.v1','a_sha256':'a','archive_sha256':'b',
          'information_sha256':'c','split':'validation','origin_times':['x'],'successful_origin_times':['x'],
          'lead_hours':[6],'config':asdict(PredictorConfig(bridge='raw')),'seed':7,
          'scores':{'aggregate':{'normalized_rmse':2.}},'finite_forecast_fraction':1,
          'trainable_parameters':10,'total_parameters':10,'inference_seconds':1,'training_seconds':1,'conditioning':{}}
    reports=[]
    for index,(rep,bridge,error) in enumerate((('climate_manifold','raw',2.),('climate_manifold','latent',1.),('plain_ae','latent',1.5))):
        row=deepcopy(base);row['config'].update(representation=rep,bridge=bridge)
        row['scores']['aggregate']['normalized_rmse']=error
        path=tmp_path/f'{index}.json';path.write_text(json.dumps(row));reports.append(path)
    result=compare(reports,tmp_path/'compare.json')
    assert result['ranking_allowed']
    assert len(result['seed_summary'])==3
    assert {r['control']:r['rmse_reduction'] for r in result['paired_effects']}=={'raw':1.,'plain_ae':.5}
    other=deepcopy(base);other['config']['bridge']='decoded'
    path=tmp_path/'aux.json';path.write_text(json.dumps(other))
    with pytest.raises(ValueError,match='Do not mix'):
        compare([reports[0],path],tmp_path/'bad.json')
