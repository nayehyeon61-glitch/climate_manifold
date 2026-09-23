"""End-to-end gates for jointly learned manifold forecasting checkpoints."""
from dataclasses import asdict
import importlib
import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from test_pinn_training import pinn_prepared
from test_downstream import seal
from climate_manifold.model import FORMAT as A_FORMAT
from climate_manifold.physical_information import digest, information_digest
from climate_manifold.train import write_json
from climate_manifold.downstream.train import (
    parser, train, initialize_manifold, load_predictor, windows, objective_weights,
)


def _args(archive, output, *extra):
    return parser().parse_args([
        '--archive',str(archive),'--output',str(output),
        '--epochs','1','--batch-size','2','--max-windows','2','--window-stride','1',
        '--horizon-steps','2','--hidden-dim','24','--manifold-dim','4',
        '--manifold-hidden-dim','24','--context-dim','8','--history-stride','1',
        '--device','cpu',*extra,
    ])


def _capture_pipelines(monkeypatch):
    module=importlib.import_module('climate_manifold.downstream.train')
    cls=module.ForecastPipeline
    captured=[]
    def factory(*args,**kwargs):
        model=cls(*args,**kwargs)
        captured.append((model,{key:value.clone() for key,value in model.state_dict().items()}))
        return model
    monkeypatch.setattr(module,'ForecastPipeline',factory)
    return captured


def _has_gradient(module):
    grads=[p.grad for p in module.parameters() if p.grad is not None]
    return bool(grads) and all(torch.isfinite(g).all() for g in grads) and sum(g.abs().sum() for g in grads)>0


def _predict(model,batch):
    return model(batch['history'],batch.get('information'),batch['origin_time_ns'],torch.tensor([6.,12.]))['mean']


@pytest.mark.parametrize('family',['mlp','neural_ode'])
def test_forecast_only_joint_training_updates_all_three_and_roundtrips(
        pinn_prepared,tmp_path,monkeypatch,family):
    _,_,_,archive=pinn_prepared
    args=_args(archive,tmp_path/f'{family}.pt','--model',family,
               '--regularization','none','--reconstruction-weight','0','--tendency-weight','0')
    assert args.training_mode=='joint' and args.a_checkpoint is None
    captured=_capture_pipelines(monkeypatch)
    path=train(args)
    trained,initial=captured[0]
    for prefix,module in (
        ('bridge.manifold.core.manifold.encoder.',trained.bridge.manifold.core.manifold.encoder),
        ('bridge.manifold.core.manifold.decoder.',trained.bridge.manifold.core.manifold.decoder),
        ('predictor.',trained.predictor),
    ):
        assert _has_gradient(module),prefix
        assert any(not torch.equal(value,trained.state_dict()[key])
                   for key,value in initial.items() if key.startswith(prefix)),prefix
    assert all(p.grad is None for p in trained.bridge.manifold.core.manifold.latent_drift.parameters())
    assert all(p.grad is None for p in trained.bridge.manifold.a_sampler.parameters())
    restored,payload=load_predictor(path)
    assert payload['a_sha256'] is None and not payload['a_frozen']
    assert payload['representation_training']=='jointly_trained'
    assert all(value==0 for value in payload['objective_weights'].values())
    core=restored.bridge.manifold.core
    assert not bool(core.manifold_ready)
    torch.testing.assert_close(core.latent_mean,torch.zeros(4),rtol=0,atol=0)
    torch.testing.assert_close(core.latent_scale,torch.ones(4),rtol=0,atol=0)
    _,_,data=initialize_manifold(args)
    batch=next(iter(DataLoader(windows(data,restored.a_config,'validation',max_windows=2),batch_size=2)))
    trained.eval()
    with torch.no_grad():
        expected=_predict(trained,batch)
        actual=_predict(restored,batch)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        assert torch.isfinite(actual).all()
        # Future supervision is allowed only in losses; poisoning it cannot change inference.
        batch['targets'].fill_(float('nan'))
        batch['information_targets']=torch.full((2,20,1),float('nan'))
        torch.testing.assert_close(_predict(restored,batch),actual,rtol=0,atol=0)


def test_enriched_joint_pinn_trains_forecast_and_pressure_decoders(
        pinn_prepared,tmp_path,monkeypatch):
    _,_,_,archive=pinn_prepared
    args=_args(archive,tmp_path/'joint-pinn.pt','--model','neural_ode',
               '--information',str(tmp_path/'pinn-information.npz'),'--pinn',
               '--pinn-weight','0.1','--distribution-weight','0.1')
    captured=_capture_pipelines(monkeypatch)
    path=train(args)
    model,_=captured[0]
    a=model.bridge.manifold
    for module in (a.core.manifold.encoder,a.core.manifold.decoder,a.information,
                   a.info_head,a.pinn,model.predictor):
        assert _has_gradient(module)
    _,payload=load_predictor(path)
    assert payload['a_metadata']['mode']=='enriched'
    assert payload['objective_weights']['pinn']==.1
    assert payload['objective_weights']['distribution']==.1
    assert payload['initialization']=='fresh'
    metrics=json.loads(path.with_suffix('.metrics.json').read_text())[0]
    for key in ('information_future','information_spatial_quantile','pinn_total'):
        assert np.isfinite(metrics['train'][key]) and metrics['train'][key]>0,key
    assert metrics['selection']['state_mse']==payload['best_selection_state_mse']


def test_zero_pinn_weight_and_raw_control_preserve_explicit_ablation(pinn_prepared,tmp_path):
    _,_,_,archive=pinn_prepared
    args=_args(archive,tmp_path/'unused-zero.pt','--information',str(tmp_path/'pinn-information.npz'),
               '--pinn','--pinn-weight','0')
    manifold,_,_=initialize_manifold(args)
    assert manifold.pinn is not None and manifold.pinn.config.weight>0
    assert objective_weights(args,manifold).pinn==0
    args.pinn_weight=.1
    args.bridge='raw';args.regularization='none'
    assert all(value==0 for value in asdict(objective_weights(args,manifold)).values())


def _sealed_a_checkpoint(model,data,archive,information,path):
    seal(model,data)
    metadata={key:(value.tolist() if isinstance(value,np.ndarray) else value)
              for key,value in data.items() if key not in ('states','times','information')}
    payload={**metadata,'format':A_FORMAT,'stage':'A','mode':'enriched',
             'config':asdict(model.config),'pinn_config':asdict(model.pinn.config),
             'model':model.state_dict(),'archive_sha256':digest(archive),
             'information_sha256':information_digest(information)}
    torch.save(payload,path)
    write_json(path.with_suffix('.manifest.json'),{'checkpoint_sha256':digest(path)})
    return path


def test_pretrained_and_fresh_initialization_keep_same_data_contract(pinn_prepared,tmp_path):
    original,_,data,archive=pinn_prepared
    information=tmp_path/'pinn-information.npz'
    # Give this saved A distinct learned weights instead of the fixture's same-seed initialization.
    with torch.no_grad():next(original.core.manifold.encoder.parameters()).add_(.125)
    checkpoint=_sealed_a_checkpoint(original,data,archive,information,tmp_path/'a.pt')
    args=_args(archive,tmp_path/'unused.pt','--a-checkpoint',str(checkpoint),
               '--information',str(information),'--initialization','pretrained')
    warm,metadata,warm_data=initialize_manifold(args)
    assert bool(warm.core.manifold_ready)
    assert all(torch.equal(value,warm.state_dict()[key]) for key,value in original.state_dict().items())
    args.initialization='fresh'
    fresh,fresh_metadata,fresh_data=initialize_manifold(args)
    assert not bool(fresh.core.manifold_ready)
    assert asdict(fresh.config)==asdict(warm.config)
    assert fresh_metadata['archive_sha256']==metadata['archive_sha256']
    assert fresh_data['split']==warm_data['split']
    assert any(not torch.equal(value,fresh.core.manifold.encoder.state_dict()[key])
               for key,value in original.core.manifold.encoder.state_dict().items())
    # New training leaves a frozen-checkpoint path available, and v1 omission of
    # training_mode must continue to restore frozen semantics.
    args.training_mode='frozen';args.output=str(tmp_path/'frozen.pt')
    path=train(args)
    frozen,payload=load_predictor(path)
    assert payload['a_frozen'] and payload['initialization']=='pretrained'
    assert all(torch.equal(value,frozen.bridge.manifold.state_dict()[key])
               for key,value in original.state_dict().items())
    legacy=dict(payload);legacy['format']='climate_manifold.downstream.v1'
    legacy['config']={key:value for key,value in payload['config'].items() if key!='training_mode'}
    old=tmp_path/'legacy.pt';torch.save(legacy,old)
    write_json(old.with_suffix('.manifest.json'),{'checkpoint_sha256':digest(old)})
    restored,_=load_predictor(old)
    assert restored.config.training_mode=='frozen'
    assert all(not p.requires_grad for p in restored.bridge.manifold.parameters())
    batch=next(iter(DataLoader(windows(data,original.config,'validation',max_windows=2),batch_size=2)))
    with torch.no_grad():torch.testing.assert_close(_predict(restored,batch),_predict(frozen,batch),rtol=0,atol=0)
