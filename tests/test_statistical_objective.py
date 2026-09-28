"""Distribution choice preserves W2 and makes KL trainable and identifiable."""
import pytest
import torch

from climate_manifold.downstream.statistical_objective import (
    make_statistical_config, spatial_statistical_loss, statistical_config_from_payload,
)
from climate_manifold.downstream.joint_objective import spatial_quantile_loss
from test_pairwise_comparison import pair_report
from test_raw_comparison import report, run_compare
from test_pairwise_runner_controls import run_runner, trains, option


def test_w2_dispatch_is_unchanged():
    torch.manual_seed(2)
    predicted, target = torch.randn(2,3,12), torch.randn(2,3,12)
    area = torch.arange(1.,13.)
    expected = spatial_quantile_loss(predicted,target,area,quantiles=32)
    actual = spatial_statistical_loss(predicted,target,area)['loss']
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_kl_entropy_identity_area_weighting_and_gradient():
    target = torch.tensor([[-1.,0.,1.,2.]],dtype=torch.float64,requires_grad=True)
    predicted = (target.detach()+.4).requires_grad_()
    area = torch.tensor([1.,2.,3.,4.],dtype=torch.float64)
    cfg = make_statistical_config('kl_entropy',bins=12)
    scores = spatial_statistical_loss(predicted,target,area,cfg)
    assert scores['loss'] > 0
    torch.testing.assert_close(scores['loss'],scores['cross_entropy']-scores['target_entropy'])
    scores['loss'].backward()
    assert target.grad is None
    assert torch.isfinite(predicted.grad).all() and predicted.grad.abs().sum()>0
    assert torch.autograd.gradcheck(
        lambda x: spatial_statistical_loss(x,target,area,cfg)['loss'], (predicted,))
    equal = spatial_statistical_loss(target,target,area,cfg)
    assert equal['loss']==0
    torch.testing.assert_close(equal['target_entropy'],equal['reconstructed_entropy'])
    scaled = spatial_statistical_loss(predicted,target,10*area,cfg)
    torch.testing.assert_close(scores['loss'],scaled['loss'])
    uniform = spatial_statistical_loss(predicted,target,torch.ones_like(area),cfg)
    assert not torch.isclose(scores['loss'],uniform['loss'])


def test_kl_tails_are_finite_and_entropy_alone_is_not_the_objective():
    cfg = make_statistical_config('kl_entropy',bins=16)
    area = torch.ones(4)
    target = torch.full((1,4),-100.)
    predicted = torch.full((1,4),100.,requires_grad=True)
    scores = spatial_statistical_loss(predicted,target,area,cfg)
    assert all(torch.isfinite(value) for value in scores.values())
    torch.testing.assert_close(scores['target_entropy'],scores['reconstructed_entropy'])
    assert scores['loss'] > 1  # Same entropy does not mean matching distributions.


def kl_report(pair='pinn_statistical',seed=7,bandwidth=.2):
    row = pair_report(pair,seed)
    row.update(statistical_loss='kl_entropy',
               statistical_loss_config=make_statistical_config('kl_entropy',bandwidth=bandwidth))
    return row


def test_comparison_separates_loss_arms_and_exports_same_pair_effect(tmp_path):
    result = run_compare(tmp_path,[report('raw'),pair_report('pinn_statistical'),kl_report()])
    assert len(result['seed_summary'])==3
    assert {r['candidate_arm'] for r in result['rows']}=={
        'raw','pinn_statistical','pinn_statistical:kl_entropy'}
    assert len(result['direct_comparison']['effects'])==4
    assert len(result['statistical_loss_effects'])==2
    assert not result['constraint_pair_effects']
    assert (tmp_path/'comparison.statistical-effects.csv').exists()
    assert result['rows'][1]['statistical_loss']=='w2'


def test_compare_skips_confounded_pair_plus_metric_changes(tmp_path):
    result = run_compare(tmp_path,[pair_report('pinn_statistical'),kl_report('statistical_static')])
    assert not result['constraint_pair_effects'] and not result['statistical_loss_effects']


def test_compare_rejects_different_histograms_as_seeds(tmp_path):
    with pytest.raises(ValueError,match='statistical_loss_config'):
        run_compare(tmp_path,[kl_report(),kl_report(seed=19,bandwidth=.3)])


def test_checkpoint_legacy_and_inconsistent_metadata():
    assert statistical_config_from_payload(pair_report('pinn_statistical'))['kind']=='w2'
    row=kl_report()
    row['statistical_loss']='w2'
    with pytest.raises(ValueError,match='disagrees'):statistical_config_from_payload(row)
    row=kl_report()
    del row['statistical_loss_config']
    with pytest.raises(ValueError,match='Missing'):statistical_config_from_payload(row)
    row=report('raw')
    row['statistical_loss']='w2'
    with pytest.raises(ValueError,match='requires'):statistical_config_from_payload(row)


def test_runner_sweep_trains_raw_once_and_uses_distinct_paths(tmp_path):
    result,commands=run_runner(tmp_path,MODELS='neural_ode',SEEDS='7',
        PAIRS='pinn_statistical',STATISTICAL_LOSSES='w2 kl_entropy',KL_BINS='32')
    assert result.returncode==0,result.stderr
    fits=trains(commands)
    assert len(fits)==3
    assert len({option(r,'--output') for r in fits})==3
    raw=[r for r in fits if option(r,'--bridge')=='raw']
    assert len(raw)==1 and '--statistical-loss' not in raw[0]
    kl=[r for r in fits if '--kl-bins' in r]
    assert len(kl)==1 and option(kl[0],'--kl-bins')=='32'
    assert option(kl[0],'--constraint-pair')=='pinn_statistical'
    assert 'kl_entropy' in option(kl[0],'--output')


def test_runner_all_pairs_does_not_repeat_nonstatistical_fit(tmp_path):
    result,commands=run_runner(tmp_path,MODELS='neural_ode',SEEDS='7',
        STATISTICAL_LOSSES='w2 kl_entropy')
    assert result.returncode==0,result.stderr
    assert len(trains(commands))==6


@pytest.mark.parametrize('losses',['w2 w2','kl','kl_entropy kl_entropy'])
def test_runner_invalid_sweep_fails_before_training(tmp_path,losses):
    result,commands=run_runner(tmp_path,STATISTICAL_LOSSES=losses)
    assert result.returncode!=0 and not commands
