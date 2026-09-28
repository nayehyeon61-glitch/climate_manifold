"""Observed CFM is a distinct auxiliary architecture, not a repeated seed."""
import pytest

from climate_manifold.downstream.conditional_flow import make_conditional_flow_config
from climate_manifold.downstream.statistical_flow import make_statistical_flow_config
from climate_manifold.downstream.statistical_objective import make_statistical_config
from test_pairwise_comparison import scoped_pair_report
from test_raw_comparison import report, run_compare


def cfm_report(kind='w2',weight=0.,seed=7,pair='pinn_statistical'):
    row=scoped_pair_report(pair,seed,decoder='separate_surface_and_information')
    row['statistical_loss']=kind
    row['statistical_loss_config']=make_statistical_config(kind)
    row['conditional_flow_config']=make_conditional_flow_config(weight)
    if weight:
        row['constraint_parameters']+=25
        row['total_parameters']+=25
        row['trainable_parameters']+=25
    return row


def test_cfm_is_separate_arm_and_capacity_change_is_explicit(tmp_path):
    data=[report('raw')]+[cfm_report(kind,w) for kind in ('w2','kl_entropy') for w in (0.,.1)]
    result=run_compare(tmp_path,data)
    assert len(result['seed_summary'])==5
    assert {r['candidate_arm'] for r in result['rows']}=={
        'raw','pinn_statistical','pinn_statistical:cfm=0.1',
        'pinn_statistical:kl_entropy','pinn_statistical:kl_entropy:cfm=0.1'}
    assert len(result['conditional_flow_effects'])==4
    assert len(result['statistical_loss_effects'])==4
    assert not result['statistical_flow_effects']
    assert len(result['direct_comparison']['effects'])==8
    for effect in result['conditional_flow_effects']:
        assert effect['candidate_parameters']==effect['baseline_parameters']+25
        assert effect['baseline_conditional_flow_weight']==0
        assert effect['candidate_conditional_flow_weight']==.1
    assert (tmp_path/'comparison.conditional-flow-effects.csv').exists()


def test_cfm_seeds_cannot_pool_with_off(tmp_path):
    result=run_compare(tmp_path,[cfm_report(weight=w,seed=seed)
        for seed in (7,19) for w in (0,.1)])
    assert len(result['seed_summary'])==2
    assert all(r['seeds']==[7,19] for r in result['seed_summary'].values())


def test_cfm_architecture_or_source_noise_cannot_change_across_seeds(tmp_path):
    rows=[cfm_report(weight=.1),cfm_report(weight=.1,seed=19)]
    rows[1]['conditional_flow_config']=make_conditional_flow_config(.1,noise_scale=.3)
    with pytest.raises(ValueError,match='conditional_flow_config'):
        run_compare(tmp_path,rows)


def test_cfm_and_forecast_flow_are_not_a_single_factor_effect(tmp_path):
    forecast=cfm_report()
    forecast['statistical_flow_config']=make_statistical_flow_config(.1)
    result=run_compare(tmp_path,[forecast,cfm_report(weight=.1)])
    assert not result['paired_effects']
    assert not result['statistical_flow_effects']
    assert not result['conditional_flow_effects']


def test_both_flow_mechanisms_in_one_record_rejected(tmp_path):
    row=cfm_report(weight=.1)
    row['statistical_flow_config']=make_statistical_flow_config(.1)
    with pytest.raises(ValueError):
        run_compare(tmp_path,[row])


def test_changing_cfm_and_base_metric_is_confounded(tmp_path):
    result=run_compare(tmp_path,[cfm_report(),cfm_report('kl_entropy',.1)])
    assert not result['paired_effects']
