"""Direct controls and reconstruction pairs retain their distinct experimental roles."""
from copy import deepcopy
import csv

import pytest

from climate_manifold.downstream.compare import CONSTRAINT_GROUPS
from test_pairwise_comparison import pair_report
from test_raw_comparison import report, run_compare


def raw_report(seed=7, family='neural_ode'):
    row = report('raw', seed, family)
    row.update(constraint_pair=None, constraint_path=None, constraint_contract=None,
               split_objective_weights=None, forecast_parameters=60, constraint_parameters=0)
    return row


def test_two_predictors_raw_plus_three_pairs_are_eight_distinct_arms(tmp_path):
    reports = [row for family in ('neural_ode','climode') for seed in (7,19)
               for row in [raw_report(seed,family), *[
                   pair_report(pair,seed,family) for pair in CONSTRAINT_GROUPS]]]
    result = run_compare(tmp_path,reports)
    assert len(result['rows']) == 16
    assert len(result['seed_summary']) == 8
    assert all(row['seeds']==[7,19] for row in result['seed_summary'].values())
    direct = result['direct_comparison']
    assert direct['ranking_allowed']
    assert len(direct['effects']) == 24  # 2 predictors x 2 seeds x 3 pairs x 2 leads
    assert len(result['constraint_pair_effects']) == 24
    assert len(result['paired_summary']) == 12  # raw contrasts + group replacements
    assert {row['candidate_arm'] for row in direct['effects']} == set(CONSTRAINT_GROUPS)
    assert {row['interpretation'] for row in direct['effects']} == {'whole_model_comparison'}
    for row in direct['effects']:
        assert row['constraint_pair'] == row['candidate_arm']
        assert row['constraint_path'] == 'observed_reconstruction'
        assert row['rmse_skill_vs_raw'] == pytest.approx(.5)
        assert row['raw_parameters'] == 60
        assert row['candidate_parameters'] in (100,110)
    with (tmp_path/'comparison.raw-effects.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 24
    assert {row['constraint_pair'] for row in rows} == set(CONSTRAINT_GROUPS)
    # Matched raw transport remains a same-family control, not the legacy reference.
    assert result['climode_benchmark']['reference_seeds'] == []


def test_single_pair_with_raw_controls_is_four_fits_and_no_replacement_effect(tmp_path):
    reports = [row for family in ('neural_ode','climode')
               for row in (raw_report(family=family),pair_report('pinn_statistical',family=family))]
    result = run_compare(tmp_path,reports)
    assert len(result['seed_summary']) == 4
    assert len(result['paired_effects']) == 2
    assert len(result['direct_comparison']['effects']) == 4
    assert not result['constraint_pair_effects']


@pytest.mark.parametrize('path,value,message',[
    (('config','raw_backend'),'legacy','matched raw'),
    (('config','training_mode'),'frozen','joint'),
    (('initialization',),'pretrained','fresh'),
    (('regularization',),'full','matched raw'),
    (('constraint_path',),'forecast_trajectory','constraint_path'),
    (('constraint_contract',),{},'constraint objective'),
    (('split_objective_weights',),{'pinn':0.},'constraint objective'),
    (('objective_weights','physics'),.1,'zero effective'),
    (('objective_weights','physics'),float('nan'),'zero effective'),
    (('objective_weights',),{},'zero effective'),
    (('constraint_parameters',),10,'constraint_parameters'),
    (('training_contract','constraint_path'),'observed_reconstruction','training_contract.constraint_path'),
    (('training_contract','epochs'),4,'training_contract'),
    (('conditioning','observed_information_available'),False,'observed_information_available'),
])
def test_raw_control_contract_cannot_hide_extra_supervision_or_changed_budget(tmp_path,path,value,message):
    raw = raw_report()
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError,match=message):
        run_compare(tmp_path,[raw,pair_report('pinn_statistical')])


def test_split_training_route_must_be_recorded_even_when_top_level_is_correct(tmp_path):
    candidate = pair_report('pinn_statistical')
    del candidate['training_contract']['constraint_path']
    with pytest.raises(ValueError,match='training_contract.constraint_path'):
        run_compare(tmp_path,[raw_report(),candidate])


def test_raw_training_contract_explicit_none_equivalent_to_absent_route(tmp_path):
    raw = raw_report()
    raw['training_contract']['constraint_path'] = None
    result = run_compare(tmp_path,[pair_report('pinn_statistical'),raw])
    assert result['direct_comparison']['ranking_allowed']


def test_legacy_latent_arm_is_rejected_in_raw_and_split_experiment(tmp_path):
    with pytest.raises(ValueError,match='mixed constraint_path'):
        run_compare(tmp_path,[raw_report(),pair_report('pinn_statistical'),report('forecast_only')])


def test_raw_pair_physical_effects_survive_missing_aggregate_but_not_failed_origins(tmp_path):
    rows = [raw_report(),pair_report('pinn_statistical')]
    for row in rows:
        row['scores']['aggregate'] = None
    result = run_compare(tmp_path,rows)
    assert not result['paired_effects']
    assert len(result['direct_comparison']['effects']) == 2


def test_raw_pair_metric_protocol_cannot_change(tmp_path):
    candidate = pair_report('pinn_statistical')
    candidate['scores']['climode']['protocol']['climatology'] = 'validation'
    with pytest.raises(ValueError,match='metric protocol'):
        run_compare(tmp_path,[raw_report(),candidate])


def test_raw_pair_duplicate_control_seed_is_rejected(tmp_path):
    raw = raw_report()
    with pytest.raises(ValueError,match='Duplicate seed'):
        run_compare(tmp_path,[raw,deepcopy(raw),pair_report('pinn_statistical')])


def test_failures_disable_raw_and_constraint_ranking(tmp_path):
    candidate = pair_report('pinn_statistical')
    candidate.update(finite_forecast_fraction=0.,successful_origin_times=[])
    candidate['scores'].pop('climode')
    result = run_compare(tmp_path,[raw_report(),candidate])
    assert not result['ranking_allowed']
    assert not result['direct_comparison']['available']
    assert not result['paired_effects']
