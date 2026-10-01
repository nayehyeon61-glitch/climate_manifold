"""Pairwise manifold constraints on observed information reconstructions.

The forecasting branch trains E--F--D separately. This objective never invokes
F, consumes its predictions, or reads future targets. Its two observed frames
are origin minus one archive step and origin, with co-located information.
PINN temporal derivatives therefore describe reconstructed observations, not a
forecast trajectory. All pairs retain a common pointwise reconstruction anchor;
the optional constraint families are selected in pairs or statistics alone.
By default a dedicated surface reconstruction decoder and the information
decoder are used. The forecast decoder is never shared by that default route.
The previous information-only and shared-decoder routes remain selectable.
"""
from __future__ import annotations

import math

import torch

from .joint_objective import JointObjectiveWeights
from .constraint_protocol import CONSTRAINT_DECODERS
from .statistical_objective import make_statistical_config, validate_statistical_config, spatial_statistical_loss


PAIRS = ('pinn_statistical', 'pinn_static', 'statistical_static', 'statistical')
DECODER_MODES = CONSTRAINT_DECODERS


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
            or not torch.isfinite(dt).all() or not bool((dt == manifold.config.step_hours).all())):
        raise ValueError(f'constraint_dt_hours must be [batch, 1] consecutive {manifold.config.step_hours}h observed intervals')
    return states.detach(), information.detach(), dt.detach().to(states)[:, 0]


def reconstruction_constraint_losses(pipeline, batch, weights, constraint_pair, *,
                                     decoder_mode='separate_surface_and_information',
                                     statistical_config=None):
    """Return manifold-only losses and their weighted ``regularization`` sum.

    By default ``separate_surface_and_information`` reconstructs observed fields
    through ``pipeline.reconstruction_decoder`` and information through D_I,
    using their equal average for reconstruction and selected statistical losses.
    The forecast surface decoder D is not invoked. In ``information_only`` mode,
    reconstruction and statistical losses use dynamic information alone. The surface
    decoder is not invoked and its reported losses are zero; its forecasting
    path remains trainable. ``surface_and_information`` restores the legacy
    equal average of surface and dynamic-information terms. Guided raw forecasts
    only allow the two independent auxiliary-decoder modes; their forecast
    output head is never used by this objective. W2/KL losses are spatial
    marginals; signed_measure retains spatial support and signed magnitudes.
    None of these are ensemble CRPS or temporal distribution losses.
    All modes exclude static information from these terms. ``static`` compares
    both reconstructed endpoints to origin terrain. PINN contributes physical
    residuals and closure regularization without duplicated tendency loss.
    """
    if decoder_mode not in DECODER_MODES:
        raise ValueError(f'decoder_mode must be one of {DECODER_MODES}')
    _validate_weights(weights, constraint_pair)
    if statistical_config is not None and not weights.distribution:
        raise ValueError('Statistical config requires an active statistical constraint')
    statistical_config = (make_statistical_config() if statistical_config is None
                          else validate_statistical_config(statistical_config))
    bridge = getattr(pipeline, 'bridge', None)
    manifold = getattr(bridge, 'manifold', None)
    if manifold is None or not hasattr(manifold, 'temporal'):
        raise ValueError('Split constraints require a trainable ClimateManifold representation')
    if bridge.mode not in ('latent', 'guided') or bridge.anchor != 'none':
        raise ValueError('Split constraints require an unanchored latent bridge or guided bridge')
    if bridge.mode == 'guided' and decoder_mode == 'surface_and_information':
        raise ValueError('Guided constraints require information_only or a separate reconstruction decoder')
    if manifold.info_head is None or manifold.info_metadata is None:
        raise ValueError('Split constraints require enriched inputs and an information decoder')
    if weights.pinn and manifold.pinn is None:
        raise ValueError('Positive PINN weight requires an enabled Hybrid PINN')
    reconstruction_decoder = getattr(pipeline, 'reconstruction_decoder', None)
    if decoder_mode == 'separate_surface_and_information' and reconstruction_decoder is None:
        raise ValueError('Separate surface constraints require a dedicated reconstruction_decoder')

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
    # The default auxiliary path uses D_rec, bypassing forecast D entirely.
    # Forecast D remains trainable on E--F--D, while E is shared across routes.
    raw_pair = manifold.raw_encode(states, information)
    decoded_information = manifold.info_head(raw_pair)
    if decoded_information.shape != information.shape:
        raise ValueError('Constraint information decoder must preserve observed endpoint shapes')
    decoded = decoded_information.reshape(len(states), 2, info_shape[0], cells)
    info_target = information.reshape_as(decoded)
    surface = surface_target = None
    if decoder_mode != 'information_only':
        reconstructed = (reconstruction_decoder(raw_pair)
                         if decoder_mode == 'separate_surface_and_information'
                         else manifold.core.manifold.decode(raw_pair))
        if reconstructed.shape != states.shape:
            raise ValueError('Constraint surface decoder must preserve observed endpoint shapes')
        surface = reconstructed.reshape(len(states), 2, grid[0], cells)
        surface_target = states.reshape_as(surface)

    def mse(predicted, target):
        return ((predicted - target.detach()).square() * area).sum(-1).mean()

    zero = states.new_zeros(())
    values = {name: zero for name in (
        'physics', 'information', 'static', 'information_spatial_quantile',
        'reconstruction_surface', 'statistical_surface', 'statistical_information', 'pinn_total',
        'statistical_total', 'statistical_kl_entropy', 'statistical_target_entropy',
        'statistical_reconstructed_entropy', 'statistical_cross_entropy',
        'statistical_signed_measure', 'statistical_signed_spatial_js',
        'statistical_signed_mass_mse')}
    values['reconstruction_information'] = mse(decoded[:, :, ~static], info_target[:, :, ~static])
    values['reconstruction'] = values['reconstruction_information']
    if surface is not None:
        values['reconstruction_surface'] = mse(surface, surface_target)
        values['reconstruction'] = .5 * (values['reconstruction_surface']
                                         + values['reconstruction_information'])

    if weights.distribution:
        information_scores = spatial_statistical_loss(
            decoded[:, :, ~static], info_target[:, :, ~static], area, statistical_config)
        values['statistical_information'] = information_scores['loss']
        combined_scores = information_scores
        if surface is not None:
            surface_scores = spatial_statistical_loss(surface, surface_target, area, statistical_config)
            values['statistical_surface'] = surface_scores['loss']
            combined_scores = {key: .5*(surface_scores[key]+information_scores[key])
                               for key in information_scores}
        values['statistical_total'] = combined_scores['loss']
        if statistical_config['kind'] == 'w2':
            # Preserve the old W2 metric; never put KL values in a quantile column.
            values['information_spatial_quantile'] = values['statistical_total']
        elif statistical_config['kind'] == 'kl_entropy':
            values['statistical_kl_entropy'] = values['statistical_total']
            for key in ('target_entropy', 'reconstructed_entropy', 'cross_entropy'):
                values['statistical_'+key] = combined_scores[key]
        else:
            values['statistical_signed_measure'] = values['statistical_total']
            for key in ('signed_spatial_js', 'signed_mass_mse'):
                values['statistical_'+key] = combined_scores[key]
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
                                + weights.distribution * values['statistical_total']
                                + weights.static * values['static']
                                + weights.pinn * values['pinn_total'])
    if any(not bool(torch.isfinite(value).all()) for value in values.values()):
        raise FloatingPointError('Nonfinite observed reconstruction constraint objective')
    return values
