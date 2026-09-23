"""Extraction contracts that must hold without any Hydra installation."""
from dataclasses import fields
import json
import numpy as np
import pytest
import torch
from test_pinn_training import pinn_prepared
from climate_manifold.architecture import ManifoldConfig
from climate_manifold.model import ClimateManifold
from climate_manifold.train import load_checkpoint


def test_default_dimensions_and_no_hydra_parameters(pinn_prepared):
    model,_,_,_=pinn_prepared
    assert ManifoldConfig.manifold_dim==64 and ManifoldConfig.hidden_dim==512
    assert not any('expert' in f.name or 'gate' in f.name for f in fields(ManifoldConfig))
    assert not any(any(s in k for s in ('expert','gate','reference','history_encoder')) for k in model.state_dict())
    with pytest.raises(ValueError,match='A only'):model.set_phase('B')
    with pytest.raises(ValueError,match='A only'):model.set_phase('C')


def test_sealing_changes_coordinates_not_physical_forecasts(pinn_prepared):
    model,batch,data,_=pinn_prepared
    noise=torch.randn(2,3,model.config.manifold_dim)
    options=dict(members=3,tau_steps=2,noise=noise,return_q=True)
    with torch.no_grad():
        before,_=model.rollout(batch['history'],batch['information'],**options)
        model.seal(torch.tensor((data['states'][:data['train_end']]-data['mean'])/data['scale']),
                   torch.tensor(data['information'][:data['train_end']]))
        after,q=model.rollout(batch['history'],batch['information'],**options)
    torch.testing.assert_close(before,after,rtol=2e-5,atol=2e-6)
    torch.testing.assert_close(after[:,:,0],batch['origin'][:,None].expand(-1,3,-1),rtol=0,atol=0)
    assert q.shape==(2,3,21,4)
    with pytest.raises(ValueError,match='Seal only once'):model.seal(batch['origin'],batch['information'])


def test_drift_ablation_has_zero_residual_and_duplicate_members(pinn_prepared):
    model,batch,_,_=pinn_prepared
    trace=[]
    prediction=model.rollout(batch['history'],batch['information'],members=2,tau_steps=1,
                             steps=3,drift_only=True,trace=trace)
    torch.testing.assert_close(prediction[:,0],prediction[:,1],rtol=0,atol=0)
    assert all(torch.count_nonzero(row['residual_per_day'])==0 for row in trace)
    for row in trace:
        torch.testing.assert_close(row['output']-row['input'],.25*row['drift_per_day'])


def test_jacobian_valid_in_inference_mode(pinn_prepared):
    model,batch,_,_=pinn_prepared
    with torch.inference_mode():
        q=model.encode(batch['origin'],batch['information'])
        jac=model.core.jacobian(q)
    assert jac.shape==(2,model.config.state_dim,model.config.manifold_dim)
    assert torch.isfinite(jac).all()


def test_surface_only_model_needs_no_information(pinn_prepared):
    previous,batch,data,_=pinn_prepared
    model=ClimateManifold(previous.config,data['schema'],data['mean'],data['scale'],data['statistics'])
    prediction=model.rollout(batch['history'],None,members=2,tau_steps=1,steps=2)
    assert prediction.shape==(2,2,3,model.config.state_dim)
    assert torch.isfinite(prediction).all()
    with pytest.raises(ValueError,match='Surface-only'):model.raw_encode(batch['origin'],batch['information'])


def test_manifest_mismatch_is_rejected_before_model_loading(tmp_path):
    path=tmp_path/'checkpoint.pt'
    torch.save({'format':'climate_manifold.a.v1'},path)
    path.with_suffix('.manifest.json').write_text(json.dumps({'checkpoint_sha256':'mismatch'}))
    with pytest.raises(ValueError,match='manifest/hash'):load_checkpoint(path)


def test_original_temporal_partitions_keep_disjoint_future_targets(pinn_prepared):
    model,_,data,_=pinn_prepared
    names=['train','expert_validation','calibration','validation','test']
    for left,right in zip(names,names[1:]):
        last_left=data['split'][left][-1]+model.config.history_span_steps+model.config.horizon_steps-1
        first_right=data['split'][right][0]+model.config.history_span_steps
        assert last_left<first_right
    end=data['train_end']
    np.testing.assert_array_equal(data['mean'],data['states'][:end].mean(0))
