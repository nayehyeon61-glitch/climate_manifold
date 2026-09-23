"""A dynamics curriculum, fixed validation and legacy objective contracts."""
from types import SimpleNamespace

import pytest
import torch

from test_pinn_training import pinn_prepared
from climate_manifold.train import batch_loss,parser,train,minimum_dynamics_epochs


def options(data,steps=4):
    return SimpleNamespace(members=2,tau_steps=1,curriculum_interval=1,profile='process',
        loss_weights=None,info_scale=data['information_scale'],
        info_tendency_scale=data['information_tendency_scale'],dynamics_max_steps=steps)


def streams():
    return {key:torch.Generator().manual_seed(seed) for key,seed in (('fm',11),('ensemble',29))}


def test_horizon_curriculum_and_fixed_full_validation_selection(pinn_prepared):
    model,batch,data,_=pinn_prepared
    args=options(data)
    model.train()
    lengths=[]
    for epoch in range(2,8):
        values=batch_loss(model,batch,args,epoch,streams())
        lengths.append(int(values['dynamics_rollout_steps']))
        weighted=sum(value for key,value in values.items() if key.startswith('weighted_'))
        torch.testing.assert_close(values['loss'],weighted)
    assert lengths==[1,1,2,4,4,4]
    model.eval()
    first=batch_loss(model,batch,args,2,streams())
    last=batch_loss(model,batch,args,7,streams())
    assert int(first['dynamics_rollout_steps'])==int(last['dynamics_rollout_steps'])==4
    # The epoch/curriculum changes the training objective, not validation selection.
    torch.testing.assert_close(first['selection'],last['selection'],rtol=0,atol=0)
    assert first['weighted_direct_state']==0
    assert last['weighted_direct_state']>0
    assert last['weighted_state_crps']>0 and last['weighted_transition_crps']>0
    assert last['weighted_loss_trajectory']>0 and last['weighted_info_distribution']>0


@pytest.mark.parametrize('mode',['disabled','warmup','information_profile'])
def test_legacy_and_closure_warmup_do_not_enter_new_rollout(pinn_prepared,monkeypatch,mode):
    model,batch,data,_=pinn_prepared
    args=options(data,steps=0 if mode=='disabled' else 4)
    def forbidden(*args,**kwargs):
        raise AssertionError('New dynamics path entered a legacy/warmup profile')
    monkeypatch.setattr(model,'dynamics_losses',forbidden)
    if mode=='information_profile':args.profile='information'
    warmup=mode=='warmup';model.set_pinn_warmup(warmup)
    values=batch_loss(model,batch,args,1 if warmup else 7,streams())
    assert 'direct_state' not in values
    assert 'dynamics_rollout_steps' not in values
    if warmup:
        values['loss'].backward()
        assert all(parameter.grad is None for name,parameter in model.named_parameters() if not name.startswith('pinn.'))
    else:
        expected=(values['state_crps']+values['transition_crps']+.1*values['loss_trajectory']
                  +.1*values['mean_state']+values['reconstruction']
                  +.05*(values['ae_delta']+values['decoded_drift'])+model.pinn.config.weight*values['pinn_total'])
        torch.testing.assert_close(values['selection'],expected,rtol=0,atol=0)


def test_cli_defaults_and_full_horizon_gate(tmp_path):
    args=parser().parse_args(['--archive','unused.npz','--output',str(tmp_path/'a.pt'),
                             '--profile','dynamics','--epochs','1'])
    assert args.dynamics_max_steps==4
    assert minimum_dynamics_epochs(4,4)==13
    assert minimum_dynamics_epochs(20,4)==25
    with pytest.raises(ValueError,match='full dynamics horizon'):
        train(args)
    for invalid in (-1,21,True,1.5):
        args.dynamics_max_steps=invalid
        with pytest.raises(ValueError,match='dynamics_max_steps'):
            train(args)
