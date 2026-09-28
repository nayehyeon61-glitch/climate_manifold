"""Decoder scope is explicit for new runs and historically correct for v1."""
from copy import deepcopy

import pytest

from climate_manifold.downstream.constraint_protocol import (
    CONSTRAINT_CONTRACT_VERSION,
    constraint_decoder_from_payload,
    make_constraint_contract,
    normalize_constraint_contract,
)


def _payload(decoder='information_only'):
    return {'constraint_pair': 'pinn_statistical', 'constraint_decoder': decoder,
            'constraint_contract': make_constraint_contract(
                'pinn_statistical', {'pinn', 'statistical'}, decoder)}


@pytest.mark.parametrize('decoder', ['information_only', 'surface_and_information'])
def test_new_scope_round_trip(decoder):
    payload = _payload(decoder)
    contract = payload['constraint_contract']
    assert contract['version'] == CONSTRAINT_CONTRACT_VERSION
    assert contract['surface_decoder_in_constraints'] == (decoder == 'surface_and_information')
    assert normalize_constraint_contract(contract) == contract
    assert constraint_decoder_from_payload(payload) == decoder


def test_legacy_contract_maps_to_both_without_mutating_source_or_losing_extras():
    legacy = {'version': 'climate_manifold.reconstruction_constraints.v1',
              'groups': ['pinn', 'statistical'], 'observed_pair': 'origin-6h,origin',
              'pinn_tendency_supervision': False, 'extra_budget': 123}
    original = deepcopy(legacy)
    normalized = normalize_constraint_contract(legacy)
    assert legacy == original
    assert normalized['decoder'] == 'surface_and_information'
    assert normalized['surface_decoder_in_constraints'] is True
    assert normalized['extra_budget'] == 123
    assert constraint_decoder_from_payload(
        {'constraint_pair': 'pinn_statistical', 'constraint_contract': legacy}) == 'surface_and_information'


@pytest.mark.parametrize('changes', [
    {'decoder': 'information_only'}, {'surface_decoder_in_constraints': False},
    {'surface_decoder_in_constraints': 1},
])
def test_legacy_scope_cannot_be_rewritten(changes):
    legacy = {'version': 'climate_manifold.reconstruction_constraints.v1', **changes}
    with pytest.raises(ValueError, match='constraint_contract'):
        normalize_constraint_contract(legacy)


@pytest.mark.parametrize('changes', [
    {'version': 'unknown'}, {'decoder': 'unknown'}, {'decoder': None},
    {'surface_decoder_in_constraints': True}, {'surface_decoder_in_constraints': 0},
    {'reconstruction': 'surface and information'},
])
def test_new_contract_rejects_contradictory_or_missing_scope(changes):
    contract = _payload()['constraint_contract']
    contract.update(changes)
    with pytest.raises(ValueError, match='constraint_contract'):
        normalize_constraint_contract(contract)


@pytest.mark.parametrize('changes', [
    {'constraint_decoder': 'surface_and_information'}, {'constraint_pair': 'pinn_static'},
    {'constraint_contract': None},
])
def test_payload_scope_and_pair_must_agree(changes):
    payload = _payload()
    payload.update(changes)
    with pytest.raises(ValueError, match='constraint_(decoder|contract)'):
        constraint_decoder_from_payload(payload)


def test_unpaired_runs_have_no_decoder_scope():
    assert constraint_decoder_from_payload({}) is None
    assert constraint_decoder_from_payload({'constraint_pair': None, 'constraint_decoder': None,
                                           'constraint_contract': None}) is None
    for changes in ({'constraint_decoder': 'information_only'},
                    {'constraint_contract': _payload()['constraint_contract']}):
        with pytest.raises(ValueError, match='require a constraint_pair'):
            constraint_decoder_from_payload(changes)
