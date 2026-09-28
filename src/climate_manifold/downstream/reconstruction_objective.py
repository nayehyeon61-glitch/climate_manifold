"""Pairwise manifold constraints on observed information reconstructions.

The forecasting branch trains E--F--D separately. This objective never invokes
F, consumes its predictions, or reads future targets. Its two observed frames
are origin minus six hours and origin, each encoded with co-located information.
PINN temporal derivatives therefore describe reconstructed observations, not a
forecast trajectory. All pairs retain a common pointwise reconstruction anchor;
the three optional constraint families are selected exactly two at a time.
By default only the information decoder is used. The former shared surface
decoder reconstruction is retained behind ``surface_and_information`` mode.
"""
from __future__ import annotations

import math

import torch

from .joint_objective import JointObjectiveWeights, spatial_quantile_loss


PAIRS = ('pinn_statistical', 'pinn_static', 'statistical_static')
DECODER_MODES = ('information_only', 'surface_and_information')


def _validate_weights(weights, constraint_pair):
    if not isinstance(weights, JointObjectiveWeights):
        raise TypeError('weights must be JointObjectiveWeights')
    if constraint_pair not in PAIRS:
        raise ValueError(f'constraint_pair must be one of {PAIRS}')
    if weights.reconstruction <= 0:
        raise ValueError('Split constraints require a positive common reconstruction weight')
    if weights.physics or weights.information:
        raise ValueError('Split constraints require physics=information=0 to avoid duplicate supervision')
    active = set(constraint_pair.split('_'))
    for family, field in (('pinn', 'pinn'), ('statistical', 'distribution'), ('static', 'static')):
        enabled = family in active
        if (getattr(weights, field) > 0) != enabled:
            expected = 'positive' if enabled else 'zero'
            raise ValueError(f'{constraint_pair} requires {field} weight to be {expected}')


def _observed_pair(batch, manifold):
    states = batch.get('constraint_states')
    information = batch.get('constraint_information')
    dt = batch.get('constraint_dt_hours')
    if (not isinstance(states, torch.Tensor) or states.ndim != 3
            or states.shape[0] < 1 or states.shape[1:] != (2, manifold.config.state_dim)
            or not states.is_floating_point() or not torch.isfinite(states).all()):
        raise ValueError('constraint_states must be finite floating [batch, 2, state_dim] observed frames')
    info_dim = math.prod(manifold.info_metadata['shape'])
    if (not isinstance(information, torch.Tensor)
            or information.shape != (states.shape[0], 2, info_dim)
            or not information.is_floating_point() or not torch.isfinite(information).all()
            or information.device != states.device or information.dtype != states.dtype):
        raise ValueError('constraint_information must be matching finite floating [batch, 2, information_dim]')
    if (not isinstance(dt, torch.Tensor) or dt.shape != (states.shape[0], 1)
            or not torch.isfinite(dt).all() or not bool((dt == 6).all())):
        raise ValueError('constraint_dt_hours must be [batch, 1] consecutive 6h observed intervals')
    return states.detach(), information.detach(), dt.detach().to(states)[:, 0]


def reconstruction_constraint_losses(pipeline, batch, weights, constraint_pair, *,
                                     decoder_mode='information_only'):
    """Return manifold-only losses and their weighted ``regularization`` sum.

    In ``information_only`` mode, ``reconstruction`` and
    ``information_spatial_quantile`` use dynamic information alone. The surface
    decoder is not invoked and its reported losses are zero; its forecasting
    path remains trainable. ``surface_and_information`` restores the legacy
    equal average of surface and dynamic-information terms. Quantile losses are
    spatial marginals, neither ensemble CRPS nor temporal distribution losses.
    Both modes exclude static information from these terms. ``static`` compares
    both reconstructed endpoints to origin terrain. PINN contributes physical
    residuals and closure regularization without duplicated tendency loss.
    """
    if decoder_mode not in DECODER_MODES:
        raise ValueError(f'decoder_mode must be one of {DECODER_MODES}')
    _validate_weights(weights, constraint_pair)
    bridge = getattr(pipeline, 'bridge', None)
    manifold = getattr(bridge, 'manifold', None)
    if manifold is None or not hasattr(manifold, 'temporal'):
        raise ValueError('Split constraints require a trainable ClimateManifold representation')
    if bridge.mode != 'latent' or bridge.anchor != 'none':
        raise ValueError('Split constraints require an unanchored latent bridge')
    if manifold.info_head is None or manifold.info_metadata is None:
        raise ValueError('Split constraints require enriched inputs and an information decoder')
    if weights.pinn and manifold.pinn is None:
        raise ValueError('Positive PINN weight requires an enabled Hybrid PINN')

    states, information, dt = _observed_pair(batch, manifold)
    grid = manifold.config.grid
    info_shape = manifold.info_metadata['shape']
    if tuple(info_shape[1:]) != tuple(grid[1:]):
        raise ValueError('Surface and information constraints must share the same geographic grid')
    cells = math.prod(grid[1:])
    kinds = [variable['kind'] for variable in manifold.info_metadata['variables']]
    if len(kinds) != info_shape[0] or any(kind not in ('static', 'dynamic') for kind in kinds):
        raise ValueError('Information variables must declare matching dynamic/static kinds')
    static = torch.tensor([kind == 'static' for kind in kinds], device=states.device)
    if not bool((~static).any()):
        raise ValueError('Common reconstruction requires dynamic information fields')
    if weights.static and not bool(static.any()):
        raise ValueError('Static constraints require static terrain information fields')
    area = manifold.temporal.area.flatten().to(states)
    if area.shape != (cells,) or not torch.isfinite(area).all() or bool((area <= 0).any()):
        raise ValueError('Constraint losses require positive finite geographic areas')

    # Neither F nor an independent A dynamics/sampling network is involved.
    # The default auxiliary path bypasses D entirely, without freezing it for
    # E--F--D forecasting. D_I and E still receive all active constraint losses.
    raw_pair = manifold.raw_encode(states, information)
    decoded_information = manifold.info_head(raw_pair)
    if decoded_information.shape != information.shape:
        raise ValueError('Constraint information decoder must preserve observed endpoint shapes')
    decoded = decoded_information.reshape(len(states), 2, info_shape[0], cells)
    info_target = information.reshape_as(decoded)
    surface = surface_target = None
    if decoder_mode == 'surface_and_information':
        reconstructed = manifold.core.manifold.decode(raw_pair)
        if reconstructed.shape != states.shape:
            raise ValueError('Constraint surface decoder must preserve observed endpoint shapes')
        surface = reconstructed.reshape(len(states), 2, grid[0], cells)
        surface_target = states.reshape_as(surface)

    def mse(predicted, target):
        return ((predicted - target.detach()).square() * area).sum(-1).mean()

    zero = states.new_zeros(())
    values = {name: zero for name in (
        'physics', 'information', 'static', 'information_spatial_quantile',
        'reconstruction_surface', 'statistical_surface', 'statistical_information', 'pinn_total')}
    values['reconstruction_information'] = mse(decoded[:, :, ~static], info_target[:, :, ~static])
    values['reconstruction'] = values['reconstruction_information']
    if surface is not None:
        values['reconstruction_surface'] = mse(surface, surface_target)
        values['reconstruction'] = .5 * (values['reconstruction_surface']
                                         + values['reconstruction_information'])

    if weights.distribution:
        values['statistical_information'] = spatial_quantile_loss(
            decoded[:, :, ~static], info_target[:, :, ~static], area)
        values['information_spatial_quantile'] = values['statistical_information']
        if surface is not None:
            values['statistical_surface'] = spatial_quantile_loss(surface, surface_target, area)
            values['information_spatial_quantile'] = .5 * (values['statistical_surface']
                                                          + values['statistical_information'])
    if weights.static:
        # Endpoint 1 is the observed origin, never a future label.
        values['static'] = mse(decoded[:, :, static], info_target[:, 1:2, static])
    if weights.pinn:
        values.update(manifold.pinn(
            decoded_information[:, 0], decoded_information[:, 1],
            information[:, 0], information[:, 1], raw_pair[:, 0], dt,
            include_tendency=False,
        ))
    values['regularization'] = (weights.reconstruction * values['reconstruction']
                                + weights.distribution * values['information_spatial_quantile']
                                + weights.static * values['static']
                                + weights.pinn * values['pinn_total'])
    if any(not bool(torch.isfinite(value).all()) for value in values.values()):
        raise FloatingPointError('Nonfinite observed reconstruction constraint objective')
    return values
