"""Optional temporal supervision of decoded spatial-marginal transport.

This is flow-inspired *quantile velocity matching* in normalized field units
per day. It does not train a generative Flow Matching vector field, a latent
probability density, or calibrated ensemble uncertainty. W2/KL reconstruction
objectives remain separate and unchanged. Future labels enter this loss only.
"""
from __future__ import annotations

import math

import torch

from .constraint_protocol import CONSTRAINT_DECODERS


def make_statistical_flow_config(weight=0., quantiles=32):
    if (isinstance(weight, bool) or not isinstance(weight, (int, float))
            or not math.isfinite(weight) or weight < 0):
        raise ValueError('statistical_flow_weight must be finite and nonnegative')
    if isinstance(quantiles, bool) or not isinstance(quantiles, int) or not 1 <= quantiles <= 512:
        raise ValueError('statistical_flow_quantiles must be an integer in [1, 512]')
    if not weight:
        return None
    return {'kind': 'quantile_transport_velocity', 'version': 1,
            'weight': float(weight), 'quantiles': quantiles,
            'time_unit': 'days', 'path': 'forecast_latent_auxiliary_decoders',
            'origin': 'observed_shared_anchor',
            'estimator': 'area_weighted_inverse_cdf_midpoints_v1'}


def validate_statistical_flow_config(config):
    if not isinstance(config, dict):
        raise ValueError('Expected statistical_flow_config')
    expected = make_statistical_flow_config(config.get('weight'), config.get('quantiles'))
    if expected is None or config != expected:
        raise ValueError('Inconsistent statistical_flow_config')
    return dict(config)


def _validate_route(pair, values):
    if 'statistical' not in (pair or '').split('_'):
        raise ValueError('Statistical flow requires a statistical constraint pair')
    for key, expected in (('bridge', 'latent'), ('training_mode', 'joint'),
                          ('anchor', 'none'), ('representation', 'climate_manifold')):
        if key in values and values[key] != expected:
            raise ValueError('Statistical flow requires a joint unanchored latent climate manifold')


def statistical_flow_config_from_args(args, pair):
    weight = getattr(args, 'statistical_flow_weight', 0.)
    quantiles = getattr(args, 'statistical_flow_quantiles', None)
    config = make_statistical_flow_config(weight, 32 if quantiles is None else quantiles)
    if config is None:
        if quantiles is not None:
            raise ValueError('--statistical-flow-quantiles requires positive --statistical-flow-weight')
        return None
    _validate_route(pair, vars(args))
    return config


def statistical_flow_config_from_payload(payload):
    """Historical checkpoints and reports have no temporal Statistical loss."""
    config = payload.get('statistical_flow_config')
    if config is None:
        if payload.get('statistical_flow_weight', 0.) != 0.:
            raise ValueError('Missing statistical_flow_config')
        return None
    config = validate_statistical_flow_config(config)
    _validate_route(payload.get('constraint_pair'), payload.get('config', {}))
    if ('statistical_flow_weight' in payload
            and payload['statistical_flow_weight'] != config['weight']):
        raise ValueError('statistical_flow_weight disagrees with statistical_flow_config')
    return config


def weighted_spatial_quantiles(fields, area, quantiles=32):
    """Inverse-CDF samples per field, preserving geographic mass through sort."""
    if (not isinstance(fields, torch.Tensor) or not fields.is_floating_point()
            or fields.ndim < 2 or fields.shape[-1] < 1
            or not isinstance(area, torch.Tensor) or area.ndim != 1
            or fields.shape[-1] != len(area)):
        raise ValueError('Spatial quantiles require floating [..., cells] fields and cell areas')
    if isinstance(quantiles, bool) or not isinstance(quantiles, int) or not 1 <= quantiles <= 512:
        raise ValueError('Quantile count must be an integer in [1, 512]')
    if (not torch.isfinite(fields).all() or not torch.isfinite(area).all()
            or not (area > 0).all()):
        raise ValueError('Spatial quantiles require finite fields and positive areas')
    dtype = torch.float64 if fields.dtype == torch.float64 else torch.float32
    values = fields.to(dtype)
    weights = area.detach().to(values)
    ordered, indices = values.sort(dim=-1)
    ordered_weights = weights.expand_as(values).gather(-1, indices)
    cdf = (ordered_weights / ordered_weights.sum(-1, keepdim=True)).cumsum(-1)
    probabilities = (torch.arange(quantiles, device=values.device, dtype=dtype) + .5) / quantiles
    levels = probabilities.expand(*values.shape[:-1], quantiles).contiguous()
    ranks = torch.searchsorted(cdf.contiguous(), levels).clamp_max(values.shape[-1] - 1)
    return ordered.gather(-1, ranks)


def quantile_transport_velocity_loss(prediction, target, origin, area, lead_hours, quantiles=32):
    """Match adjacent quantile velocities; both paths start at true origin.

    ``prediction``/``target`` are [batch, leads, variables, cells], and origin
    is [batch, variables, cells]. Labels must correspond to ``lead_hours``.
    Quantile rank is a one-dimensional marginal transport coupling, not a
    correspondence between geographic cells or atmospheric material parcels.
    """
    if (prediction.ndim != 4 or target.shape != prediction.shape
            or origin.shape != (prediction.shape[0], *prediction.shape[2:])
            or min(prediction.shape) < 1):
        raise ValueError('Quantile flow requires matching [batch, leads, variables, cells] fields and origin')
    leads = torch.as_tensor(lead_hours, device=prediction.device).detach()
    if (leads.ndim != 1 or len(leads) != prediction.shape[1]
            or not torch.isfinite(leads).all() or not (leads > 0).all()
            or not (leads[1:] > leads[:-1]).all()):
        raise ValueError('Quantile flow lead hours must be positive, finite, increasing and match targets')
    predicted_q = weighted_spatial_quantiles(prediction, area, quantiles)
    target_q = weighted_spatial_quantiles(target.detach().to(prediction), area, quantiles)
    origin_q = weighted_spatial_quantiles(origin.detach().to(prediction), area, quantiles)
    dt_days = torch.diff(torch.cat((leads.new_zeros(1), leads))).to(predicted_q) / 24.
    # A shared observed anchor prevents matching all subsequent increments
    # while retaining an unconstrained, constant distribution offset.
    predicted_path = torch.cat((origin_q[:, None], predicted_q), dim=1)
    target_path = torch.cat((origin_q[:, None], target_q), dim=1)
    velocity_error = (predicted_path.diff(dim=1) - target_path.diff(dim=1)) / dt_days[None, :, None, None]
    loss = velocity_error.square().mean()
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite statistical flow velocity objective')
    return loss


def _field_target(value, batch_size, features, name, steps=None):
    expected = (batch_size, features) if steps is None else (batch_size, steps, features)
    if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
            or value.ndim != len(expected) or value.shape[0] != batch_size
            or value.shape[-1] != features or steps is not None and value.shape[1] < steps):
        raise ValueError(name+' must contain matching floating field targets')
    return value.detach() if steps is None else value[:, :steps].detach()


def forecast_statistical_flow_losses(pipeline, prediction, batch, lead_hours, config, *,
                                     decoder_mode='separate_surface_and_information'):
    """Add temporal distribution supervision through F's actual latent path.

    This function never calls F or encodes a future target. Dedicated surface
    D_rec and D_I read denormalized predicted latent coordinates. Legacy shared
    D and information-only decoder modes follow the observed objective's scope.
    The temporal term is independent of whether reconstruction uses W2 or KL.
    """
    mean = prediction.get('mean')
    if not isinstance(mean, torch.Tensor):
        raise ValueError('Statistical flow requires forecast predictions')
    zero = mean.new_zeros(())
    result = {name: zero for name in ('statistical_flow', 'statistical_flow_surface',
              'statistical_flow_information', 'statistical_flow_regularization')}
    if config is None:
        return result
    config = validate_statistical_flow_config(config)
    if decoder_mode not in CONSTRAINT_DECODERS:
        raise ValueError('Unknown statistical flow decoder mode')
    bridge = getattr(pipeline, 'bridge', None)
    manifold = getattr(bridge, 'manifold', None)
    if (manifold is None or bridge.mode != 'latent' or bridge.anchor != 'none'
            or not hasattr(manifold, 'temporal') or manifold.info_head is None
            or manifold.info_metadata is None
            or getattr(getattr(pipeline, 'config', None), 'training_mode', 'joint') != 'joint'):
        raise ValueError('Statistical flow requires a joint unanchored latent manifold with information decoder')
    future = prediction.get('predicted_latent')
    if (mean.ndim != 3 or mean.shape[1] != len(lead_hours)
            or not isinstance(future, torch.Tensor)
            or future.shape != (len(mean), mean.shape[1], manifold.config.manifold_dim)
            or not torch.isfinite(future).all()):
        raise ValueError('Statistical flow requires the actual forecast latent trajectory')
    batch_size, steps = mean.shape[:2]
    dt = batch.get('dt_hours')
    if dt is not None:
        if (not isinstance(dt, torch.Tensor) or dt.ndim != 2
                or dt.shape[0] != batch_size or dt.shape[1] < steps
                or not torch.isfinite(dt).all() or not (dt > 0).all()):
            raise ValueError('Statistical flow dt_hours must be finite positive [batch, horizon] intervals')
        target_leads = dt[:, :steps].detach().cumsum(dim=1)
        requested_leads = torch.as_tensor(lead_hours, device=dt.device, dtype=dt.dtype)
        if not torch.equal(target_leads, requested_leads.expand_as(target_leads)):
            raise ValueError('Statistical flow lead hours must align with target dt_hours')
    grid = manifold.config.grid
    info_shape = manifold.info_metadata['shape']
    if tuple(info_shape[1:]) != tuple(grid[1:]):
        raise ValueError('Statistical flow surface and information grids must match')
    kinds = [variable['kind'] for variable in manifold.info_metadata['variables']]
    if len(kinds) != info_shape[0] or any(kind not in ('static', 'dynamic') for kind in kinds):
        raise ValueError('Information variables must declare matching dynamic/static kinds')
    dynamic = torch.tensor([kind == 'dynamic' for kind in kinds], device=future.device)
    if not bool(dynamic.any()):
        raise ValueError('Statistical flow requires dynamic information fields')
    cells = math.prod(grid[1:])
    area = manifold.temporal.area.flatten().to(future)
    raw = future * manifold.core.latent_scale + manifold.core.latent_mean
    decoded_info = manifold.info_head(raw)
    if decoded_info.shape != (batch_size, steps, math.prod(info_shape)):
        raise ValueError('Statistical flow information decoder returned incompatible fields')
    origin_info = _field_target(batch.get('information'), batch_size, math.prod(info_shape), 'information')
    target_info = _field_target(batch.get('statistical_flow_information_targets'),
                               batch_size, math.prod(info_shape), 'statistical_flow_information_targets', steps)
    predicted_info = decoded_info.reshape(batch_size, steps, info_shape[0], cells)[:, :, dynamic]
    target_info = target_info.reshape(batch_size, steps, info_shape[0], cells)[:, :, dynamic]
    origin_info = origin_info.reshape(batch_size, info_shape[0], cells)[:, dynamic]
    result['statistical_flow_information'] = quantile_transport_velocity_loss(
        predicted_info, target_info, origin_info, area, lead_hours, config['quantiles'])
    result['statistical_flow'] = result['statistical_flow_information']
    if decoder_mode != 'information_only':
        decoder = (getattr(pipeline, 'reconstruction_decoder', None)
                   if decoder_mode == 'separate_surface_and_information'
                   else manifold.core.manifold.decode)
        if decoder is None:
            raise ValueError('Separate statistical flow requires a dedicated reconstruction_decoder')
        decoded_surface = decoder(raw)
        if decoded_surface.shape != (batch_size, steps, manifold.config.state_dim):
            raise ValueError('Statistical flow surface decoder returned incompatible fields')
        origin = _field_target(batch.get('origin'), batch_size, manifold.config.state_dim, 'origin')
        target = _field_target(batch.get('targets'), batch_size, manifold.config.state_dim, 'targets', steps)
        result['statistical_flow_surface'] = quantile_transport_velocity_loss(
            decoded_surface.reshape(batch_size, steps, grid[0], cells),
            target.reshape(batch_size, steps, grid[0], cells),
            origin.reshape(batch_size, grid[0], cells), area, lead_hours, config['quantiles'])
        result['statistical_flow'] = .5 * (result['statistical_flow_surface']
                                          + result['statistical_flow_information'])
    result['statistical_flow_regularization'] = config['weight'] * result['statistical_flow']
    return result
