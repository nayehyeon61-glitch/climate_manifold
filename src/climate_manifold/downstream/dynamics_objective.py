"""Future-label supervision for the optional, deterministic dynamics encoder.

The main forecast and this auxiliary objective consume the same E_d features.
Only this loss function receives future targets; neither E_d nor its decoder
accepts them as inputs. This is not the archived distribution-flow objective.
"""
import math

import torch

from .statistical_objective import (make_statistical_config, validate_statistical_config,
                                    spatial_statistical_loss)


def dynamics_contract(config):
    if config.dynamics_encoder == 'off':
        return None
    return {
        'version': 'climate_manifold.dual_dynamics.v1',
        'encoder': config.dynamics_encoder, 'mode': config.dynamics_mode,
        'hidden_dim': config.dynamics_hidden_dim, 'channels': config.dynamics_channels,
        'input': 'observed_raw_history_and_physical_time_differences',
        'spatial_encoder_shared': False,
        'forecast_auxiliary_encoder_shared': True,
        'integration': ('spatial_latent_plus_projected_dynamics_history'
                        if config.bridge == 'latent' else 'raw_spatial_dynamics_token_fusion'),
        'future_targets_input': False, 'stochastic_sampling': False,
        'forecast_gradient_to_dynamics_encoder': config.dynamics_mode == 'learned',
        'auxiliary_head': 'independent_lead_conditioned_future_surface_decoder',
        'auxiliary_inference_used': False,
    }


def validate_dynamics_auxiliary_config(config):
    if not isinstance(config, dict) or set(config) != {
            'version', 'statistical_loss_config', 'reconstruction_weight',
            'statistical_weight', 'tendency_weight'}:
        raise ValueError('Invalid dynamics_auxiliary_config')
    if config['version'] != 'climate_manifold.dynamics_auxiliary.v1':
        raise ValueError('Unknown dynamics auxiliary version')
    for name in ('reconstruction_weight', 'statistical_weight', 'tendency_weight'):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('Dynamics objective weights must be finite and nonnegative: '+name)
    validate_statistical_config(config['statistical_loss_config'])
    return dict(config)


def dynamics_auxiliary_config_from_args(args, predictor_config, spatial_statistical_config=None):
    requested = getattr(args, 'dynamics_statistical_loss', None)
    weights = {key: getattr(args, 'dynamics_'+key, default) for key, default in (
        ('reconstruction_weight', .1), ('statistical_weight', .1), ('tendency_weight', 0.))}
    for key, value in weights.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError('Dynamics objective weights must be finite and nonnegative: '+key)
    if predictor_config.dynamics_encoder == 'off':
        if requested is not None:
            raise ValueError('--dynamics-statistical-loss requires an enabled dynamics encoder')
        return None
    kind = requested or (spatial_statistical_config or {}).get('kind', 'signed_measure')
    statistics = (dict(spatial_statistical_config)
                  if spatial_statistical_config and spatial_statistical_config['kind'] == kind
                  else make_statistical_config(kind))
    return validate_dynamics_auxiliary_config({
        'version': 'climate_manifold.dynamics_auxiliary.v1',
        'statistical_loss_config': statistics, **weights})


def dynamics_auxiliary_config_from_payload(payload, predictor_config):
    expected = dynamics_contract(predictor_config)
    if payload.get('dynamics_contract') != expected:
        raise ValueError('Checkpoint dynamics_contract disagrees with predictor config')
    config = payload.get('dynamics_auxiliary_config')
    if expected is None:
        if config is not None:
            raise ValueError('Disabled dynamics cannot have an auxiliary objective')
        return None
    return validate_dynamics_auxiliary_config(config)


def dynamics_auxiliary_losses(pipeline, prediction, batch, lead_hours, config):
    """Per-field/lead spatial statistics plus optional pointwise/tendency fit.

Targets are selected by physical lead from the dense future label sequence.
Tendency uses physical units per hour normalized by training-only scales, as
the main forecast objective does. Nonuniform supported leads use their own dt.
"""
    config = validate_dynamics_auxiliary_config(config)
    history = prediction.get('dynamics_history_latent')
    if pipeline.dynamics_auxiliary_head is None or history is None:
        raise ValueError('Dynamics auxiliary loss requires shared E_d history features')
    if (lead_hours.ndim != 1 or not len(lead_hours) or not torch.isfinite(lead_hours).all()
            or lead_hours[0] <= 0 or not (lead_hours[1:] > lead_hours[:-1]).all()):
        raise ValueError('Dynamics leads must be positive, finite and strictly increasing')
    zero = history.new_zeros(())
    result = {name: zero for name in ('dynamics_reconstruction', 'dynamics_statistical',
                                      'dynamics_tendency', 'dynamics_regularization')}
    if not any(config[key] for key in ('reconstruction_weight', 'statistical_weight', 'tendency_weight')):
        return result
    steps = lead_hours / pipeline.a_config.step_hours
    if not torch.allclose(steps, steps.round(), rtol=0, atol=1e-5):
        raise ValueError('Dynamics labels require leads aligned to archive time steps')
    indices = steps.round().long()-1
    targets = batch['targets']
    if (targets.ndim != 3 or indices[-1] >= targets.shape[1]
            or targets.shape[0] != history.shape[0]):
        raise ValueError('Dynamics future labels do not cover the requested leads')
    target = targets.index_select(1, indices.to(targets.device)).detach()
    future = pipeline.dynamics_auxiliary_head(history, lead_hours)
    if future.shape != target.shape or not torch.isfinite(future).all() or not torch.isfinite(target).all():
        raise ValueError('Dynamics prediction and future targets must be matching finite fields')
    temporal = pipeline.bridge.manifold.temporal
    result['dynamics_reconstruction'] = ((future-target).square()*temporal.metric).sum(-1).mean()
    channels, height, width = pipeline.a_config.grid
    if config['statistical_weight']:
        scores = spatial_statistical_loss(
            future.reshape(*future.shape[:2], channels, height*width),
            target.reshape(*target.shape[:2], channels, height*width),
            temporal.area.flatten().to(future), config['statistical_loss_config'])
        result['dynamics_statistical'] = scores['loss']
        for name, value in scores.items():
            if name != 'loss':
                result['dynamics_'+name] = value
    if config['tendency_weight']:
        origin = batch['origin'].detach()[:, None]
        dt = torch.diff(torch.cat((lead_hours.new_zeros(1), lead_hours)))[None, :, None]
        error = (torch.cat((origin, future), 1).diff(dim=1)
                 - torch.cat((origin, target), 1).diff(dim=1))
        error = error * temporal.scale / dt / temporal.tendency_scale
        result['dynamics_tendency'] = (error.square()*temporal.metric).sum(-1).mean()
    result['dynamics_regularization'] = (
        config['reconstruction_weight']*result['dynamics_reconstruction']
        + config['statistical_weight']*result['dynamics_statistical']
        + config['tendency_weight']*result['dynamics_tendency'])
    if not torch.isfinite(result['dynamics_regularization']):
        raise FloatingPointError('Nonfinite dynamics auxiliary loss')
    return result
