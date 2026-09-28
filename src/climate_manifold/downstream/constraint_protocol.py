"""Versioned decoder scope for observed reconstruction constraints.

The original contract always constrained both the surface and information
decoders.  Missing decoder metadata in those v1 records must retain that meaning
when they are loaded alongside the newer information-only experiment.
"""

CONSTRAINT_DECODERS = ('information_only', 'surface_and_information')
CONSTRAINT_CONTRACT_VERSION = 'climate_manifold.reconstruction_constraints.v2'
LEGACY_CONSTRAINT_CONTRACT_VERSION = 'climate_manifold.reconstruction_constraints.v1'

_RECONSTRUCTION = {
    'information_only': 'dynamic-information pointwise reconstruction only',
    'surface_and_information': 'common surface and dynamic-information pointwise reconstruction',
}


def _validate_decoder(decoder):
    if decoder not in CONSTRAINT_DECODERS:
        raise ValueError('Invalid reconstruction constraint_contract decoder: '+str(decoder))
    return decoder


def make_constraint_contract(pair, groups, decoder):
    """Build the current observed-pair contract with an explicit decoder scope."""
    decoder = _validate_decoder(decoder)
    return {
        'version': CONSTRAINT_CONTRACT_VERSION,
        'pair': pair,
        'groups': sorted(groups) if isinstance(groups, (set, frozenset)) else list(groups),
        'observed_pair': 'origin-6h,origin',
        'decoder': decoder,
        'surface_decoder_in_constraints': decoder == 'surface_and_information',
        'reconstruction': _RECONSTRUCTION[decoder],
        'pinn_tendency_supervision': False,
    }


def normalize_constraint_contract(contract):
    """Return a comparable v2 contract without changing its scientific scope.

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
        if contract.get('surface_decoder_in_constraints', True) is not True:
            raise ValueError('Invalid reconstruction constraint_contract: v1 constrained the surface decoder')
    elif version == CONSTRAINT_CONTRACT_VERSION:
        decoder = _validate_decoder(contract.get('decoder'))
        expected_surface = decoder == 'surface_and_information'
        if contract.get('surface_decoder_in_constraints') is not expected_surface:
            raise ValueError('Invalid reconstruction constraint_contract: inconsistent surface decoder scope')
        if contract.get('reconstruction') != _RECONSTRUCTION[decoder]:
            raise ValueError('Invalid reconstruction constraint_contract: inconsistent reconstruction scope')
    else:
        raise ValueError('Invalid reconstruction constraint_contract version: '+str(version))
    return {**contract, 'version': CONSTRAINT_CONTRACT_VERSION, 'decoder': decoder,
            'surface_decoder_in_constraints': decoder == 'surface_and_information',
            'reconstruction': _RECONSTRUCTION[decoder]}


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
