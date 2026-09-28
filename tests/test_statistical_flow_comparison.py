"""Temporal loss variants must remain distinct from base distribution choices."""
import pytest

from climate_manifold.downstream.statistical_flow import make_statistical_flow_config
from test_statistical_objective import kl_report
from test_pairwise_comparison import pair_report
from test_raw_comparison import report, run_compare


def flow_report(kind='w2',weight=0.,seed=7,pair='pinn_statistical'):
    row=kl_report(pair,seed) if kind=='kl_entropy' else pair_report(pair,seed)
    row['statistical_flow_config']=make_statistical_flow_config(weight)
    return row


def test_four_distribution_variants_share_raw_but_have_distinct_paired_effects(tmp_path):
    rows=[report('raw')]
    rows += [flow_report(kind,weight) for kind in ('w2','kl_entropy') for weight in (0.,.1)]
    result=run_compare(tmp_path,rows)
    assert len(result['seed_summary'])==5
    assert {row['candidate_arm'] for row in result['rows']}=={
        'raw','pinn_statistical','pinn_statistical:flow=0.1',
        'pinn_statistical:kl_entropy','pinn_statistical:kl_entropy:flow=0.1'}
    assert len(result['direct_comparison']['effects'])==8  # 4 candidates x 2 leads
    assert len(result['statistical_flow_effects'])==4      # W2/KL flow on vs off
    assert len(result['statistical_loss_effects'])==4      # W2 vs KL under each flow setting
    assert not result['constraint_pair_effects']
    assert (tmp_path/'comparison.flow-effects.csv').exists()
    for row in result['statistical_flow_effects']:
        assert row['baseline_flow_weight']==0
        assert row['candidate_flow_weight']==.1
        assert row['interpretation']=='change_statistical_flow_weight'


def test_flow_variants_do_not_pool_as_seeds(tmp_path):
    rows=[flow_report(weight=weight,seed=seed) for seed in (7,19) for weight in (0.,.1)]
    result=run_compare(tmp_path,rows)
    assert len(result['seed_summary'])==2
    assert all(s['seeds']==[7,19] for s in result['seed_summary'].values())
    assert len(result['statistical_flow_effects'])==4


def test_joint_change_of_loss_and_flow_has_no_single_factor_effect(tmp_path):
    result=run_compare(tmp_path,[flow_report(),flow_report('kl_entropy',.1)])
    assert not result['statistical_flow_effects']
    assert not result['statistical_loss_effects']
    assert not result['paired_effects']


def test_pair_change_cannot_also_change_shared_flow_setting(tmp_path):
    result=run_compare(tmp_path,[flow_report(weight=.1),
        flow_report(weight=0,pair='statistical_static')])
    assert not result['constraint_pair_effects']


def test_different_flow_quantiles_are_not_pooled(tmp_path):
    a,b=flow_report(weight=.1),flow_report(weight=.1,seed=19)
    b['statistical_flow_config']=make_statistical_flow_config(.1,16)
    with pytest.raises(ValueError,match='statistical_flow_config'):
        run_compare(tmp_path,[a,b])


def test_raw_report_cannot_claim_temporal_distribution_supervision(tmp_path):
    raw=report('raw')
    raw['statistical_flow_config']=make_statistical_flow_config(.1)
    with pytest.raises(ValueError,match='statistical constraint'):
        run_compare(tmp_path,[raw,flow_report()])
