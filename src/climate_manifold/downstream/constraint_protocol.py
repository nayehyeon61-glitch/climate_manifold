"""Versioned decoder scope for observed reconstruction constraints.

Historical v1/v2 surface reconstruction shared the forecast decoder. The v3
contract distinguishes that route from an independent reconstruction decoder
without relabeling old checkpoints as the new experiment.
"""

CONSTRAINT_DECODERS = ('information_only', 'surface_and_information',
                       'separate_surface_and_information')
CONSTRAINT_CONTRACT_VERSION = 'climate_manifold.reconstruction_constraints.v3'
PREVIOUS_CONSTRAINT_CONTRACT_VERSION = 'climate_manifold.reconstruction_constraints.v2'
LEGACY_CONSTRAINT_CONTRACT_VERSION = 'climate_manifold.reconstruction_constraints.v1'

_RECONSTRUCTION = {
    'information_only': 'dynamic-information pointwise reconstruction only',
    'surface_and_information': 'common surface and dynamic-information pointwise reconstruction',
    'separate_surface_and_information':
        'mean of independent surface and dynamic-information pointwise reconstruction',
}


def _validate_decoder(decoder):
    if decoder not in CONSTRAINT_DECODERS:
        raise ValueError('Invalid reconstruction constraint_contract decoder: '+str(decoder))
    return decoder


def _decoder_scope(decoder):
    return {
        'surface_decoder_in_constraints': decoder != 'information_only',
        'forecast_decoder_in_constraints': decoder == 'surface_and_information',
        'separate_reconstruction_decoder': decoder == 'separate_surface_and_information',
    }


def _validate_scope_flags(contract, decoder, *, require_all):
    for name, expected in _decoder_scope(decoder).items():
        if (require_all or name in contract) and contract.get(name) is not expected:
            raise ValueError('Invalid reconstruction constraint_contract: inconsistent '+name)


def make_constraint_contract(pair, groups, decoder, *, step_hours=6):
    """Build the current observed-pair contract with an explicit decoder scope."""
    decoder = _validate_decoder(decoder)
    return {
        'version': CONSTRAINT_CONTRACT_VERSION,
        'pair': pair,
        'groups': sorted(groups) if isinstance(groups, (set, frozenset)) else list(groups),
        'observed_pair': f'origin-{step_hours}h,origin',
        'decoder': decoder,
        **_decoder_scope(decoder),
        'reconstruction': _RECONSTRUCTION[decoder],
        'pinn_tendency_supervision': False,
    }


def normalize_constraint_contract(contract):
    """Return a comparable v3 contract without changing its scientific scope.

    Extra metadata is preserved, so experiments with different additional
    declarations cannot silently be pooled. Historical reconstruction prose is
    normalized only for v1, whose implementation had a fixed decoder scope.
    """
    if not isinstance(contract, dict):
        raise ValueError('Invalid reconstruction constraint_contract')
    version = contract.get('version')
    if version == LEGACY_CONSTRAINT_CONTRACT_VERSION:
        decoder = 'surface_and_information'
        if contract.get('decoder', decoder) != decoder:
            raise ValueError('Invalid reconstruction constraint_contract: v1 used both decoders')
        _validate_scope_flags(contract, decoder, require_all=False)
    elif version in (PREVIOUS_CONSTRAINT_CONTRACT_VERSION, CONSTRAINT_CONTRACT_VERSION):
        decoder = _validate_decoder(contract.get('decoder'))
        if version == PREVIOUS_CONSTRAINT_CONTRACT_VERSION:
            if decoder == 'separate_surface_and_information':
                raise ValueError('Invalid reconstruction constraint_contract: v2 had no separate decoder')
            # v2 required surface scope; the two newer flags were absent, but
            # any supplied flag must still match its historical implementation.
            if contract.get('surface_decoder_in_constraints') is not _decoder_scope(decoder)['surface_decoder_in_constraints']:
                raise ValueError('Invalid reconstruction constraint_contract: inconsistent surface decoder scope')
            _validate_scope_flags(contract, decoder, require_all=False)
        else:
            _validate_scope_flags(contract, decoder, require_all=True)
        if contract.get('reconstruction') != _RECONSTRUCTION[decoder]:
            raise ValueError('Invalid reconstruction constraint_contract: inconsistent reconstruction scope')
    else:
        raise ValueError('Invalid reconstruction constraint_contract version: '+str(version))
    return {**contract, 'version': CONSTRAINT_CONTRACT_VERSION, 'decoder': decoder,
            **_decoder_scope(decoder), 'reconstruction': _RECONSTRUCTION[decoder]}


def constraint_decoder_from_payload(payload):
    """Resolve saved scope, rejecting contradictory or orphaned declarations."""
    explicit = payload.get('constraint_decoder')
    contract = payload.get('constraint_contract')
    pair = payload.get('constraint_pair')
    if pair is None:
        if explicit is not None or contract is not None:
            raise ValueError('constraint_decoder and constraint_contract require a constraint_pair')
        return None
    normalized = normalize_constraint_contract(contract)
    if normalized.get('pair', pair) != pair:
        raise ValueError('Invalid reconstruction constraint_contract: mismatched constraint_pair')
    decoder = normalized['decoder']
    if explicit is not None and explicit != decoder:
        raise ValueError('constraint_decoder disagrees with constraint_contract')
    return decoder
