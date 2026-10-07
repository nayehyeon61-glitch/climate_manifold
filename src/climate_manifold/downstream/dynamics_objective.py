"""Future-label supervision for the optional, deterministic dynamics encoder.

The main forecast and this auxiliary objective consume the same observed E_d
features. Optional latent targets use a *separate, detached* call to that E_d on
truth windows, exclusively inside the loss/diagnostic branch. Those labels do
not enter the main forecast or auxiliary rollout. This is not distribution flow.
"""
import math

import torch

from .statistical_objective import (make_statistical_config, validate_statistical_config,
                                    spatial_statistical_loss)
from .dynamics_rollout import lead_steps


def dynamics_contract(config):
    if config.dynamics_encoder == 'off':
        return None
    result = {
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
    if getattr(config, 'dynamics_auxiliary_architecture', 'direct') == 'latent_rollout':
        result.update({
            'version': 'climate_manifold.dual_dynamics.v2',
            'auxiliary_head': 'independent_autoregressive_latent_transition_and_shared_field_decoder',
            'auxiliary_architecture': 'latent_rollout',
            'auxiliary_transition': 'F_d_auxiliary_shared_residual_step_in_physical_days',
            'auxiliary_initial_state': 'last_observed_dynamics_encoder_state',
            'auxiliary_teacher_forcing': False,
            'target_encoder': 'online_stop_gradient',
            'target_semantics': 'same_length_strided_truth_window',
            'future_target_encoding': 'loss_or_diagnostics_only_never_forecast_input',
        })
    return result


def validate_dynamics_auxiliary_config(config):
    base_keys = {'version', 'statistical_loss_config', 'reconstruction_weight',
                 'statistical_weight', 'tendency_weight'}
    if not isinstance(config, dict):
        raise ValueError('Invalid dynamics_auxiliary_config')
    version = config.get('version')
    extra = ({'supervision', 'latent_weight', 'target_encoder', 'target_semantics'}
             if version == 'climate_manifold.dynamics_auxiliary.v2' else set())
    if set(config) != base_keys | extra:
        raise ValueError('Invalid dynamics_auxiliary_config')
    if version not in ('climate_manifold.dynamics_auxiliary.v1',
                       'climate_manifold.dynamics_auxiliary.v2'):
        raise ValueError('Unknown dynamics auxiliary version')
    weight_names = ('reconstruction_weight', 'statistical_weight', 'tendency_weight')
    for name in (*weight_names, *(('latent_weight',) if extra else ())):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('Dynamics objective weights must be finite and nonnegative: '+name)
    if extra:
        if config['supervision'] not in ('field', 'field_latent'):
            raise ValueError('Dynamics supervision must be field or field_latent')
        if config['target_encoder'] != 'online_stop_gradient' or config['target_semantics'] != 'same_length_strided_truth_window':
            raise ValueError('Invalid dynamics latent target semantics')
        if config['reconstruction_weight'] <= 0 or config['statistical_weight'] <= 0:
            raise ValueError('Latent rollout comparison requires positive field reconstruction and statistical weights')
        if ((config['supervision'] == 'field' and config['latent_weight'] != 0)
                or (config['supervision'] == 'field_latent' and config['latent_weight'] <= 0)):
            raise ValueError('Dynamics latent weight must be zero for field and positive for field_latent')
    validate_statistical_config(config['statistical_loss_config'])
    return dict(config)


def dynamics_auxiliary_config_from_args(args, predictor_config, spatial_statistical_config=None):
    requested = getattr(args, 'dynamics_statistical_loss', None)
    architecture = getattr(predictor_config, 'dynamics_auxiliary_architecture', 'direct')
    supervision = getattr(args, 'dynamics_supervision', 'field')
    latent_weight = getattr(args, 'dynamics_latent_weight', None)
    if supervision not in ('field', 'field_latent'):
        raise ValueError('Dynamics supervision must be field or field_latent')
    if architecture == 'direct' and (supervision != 'field' or latent_weight is not None):
        raise ValueError('Dynamics latent supervision requires --dynamics-auxiliary-architecture latent_rollout')
    if supervision == 'field' and latent_weight is not None:
        raise ValueError('--dynamics-latent-weight is only valid with field_latent supervision')
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
    config = {
        'version': 'climate_manifold.dynamics_auxiliary.v1',
        'statistical_loss_config': statistics, **weights}
    if architecture == 'latent_rollout':
        config.update({
            'version': 'climate_manifold.dynamics_auxiliary.v2',
            'supervision': supervision,
            'latent_weight': (0. if supervision == 'field' else .1 if latent_weight is None else latent_weight),
            'target_encoder': 'online_stop_gradient',
            'target_semantics': 'same_length_strided_truth_window',
        })
    return validate_dynamics_auxiliary_config(config)


def dynamics_auxiliary_config_from_payload(payload, predictor_config):
    expected = dynamics_contract(predictor_config)
    if payload.get('dynamics_contract') != expected:
        raise ValueError('Checkpoint dynamics_contract disagrees with predictor config')
    config = payload.get('dynamics_auxiliary_config')
    if expected is None:
        if config is not None:
            raise ValueError('Disabled dynamics cannot have an auxiliary objective')
        return None
    config = validate_dynamics_auxiliary_config(config)
    rollout = getattr(predictor_config, 'dynamics_auxiliary_architecture', 'direct') == 'latent_rollout'
    if (config['version'] == 'climate_manifold.dynamics_auxiliary.v2') != rollout:
        raise ValueError('Dynamics auxiliary config version disagrees with predictor architecture')
    return config


def future_dynamics_targets(pipeline, batch, lead_hours):
    """Detached E_d truth-window labels with exactly the observed window layout.

    ``dynamics_target_sequence`` is a dense normalized archive span beginning at
    the first observed history time. Its prefix at ``history_stride`` positions
    must equal ``history``; its suffix after the observed origin must equal the
    corresponding dense ``targets``. For a future archive step k, encode indices
    k + [0, stride, ..., (T-1)*stride]. Reset E_d's context for every window, just
    as for observed histories. Future information is used only to construct loss
    labels, never as an input to the actual prediction path.
    """
    if pipeline.dynamics_encoder is None:
        raise ValueError('Dynamics latent labels require an enabled dynamics encoder')
    config = pipeline.a_config
    steps = lead_steps(lead_hours, config.step_hours)
    sequence = batch.get('dynamics_target_sequence')
    history, targets = batch['history'], batch['targets']
    last_history = (config.history_steps - 1) * config.history_stride
    required = last_history + int(steps[-1].item()) + 1
    if (not isinstance(history, torch.Tensor) or history.ndim != 3
            or history.shape[1] != config.history_steps or not history.shape[0]
            or not isinstance(targets, torch.Tensor) or targets.ndim != 3
            or targets.shape[0] != history.shape[0] or targets.shape[2] != history.shape[2]
            or targets.shape[1] < int(steps[-1].item())):
        raise ValueError('Dynamics target sequence requires matching observed history and future labels')
    if (not isinstance(sequence, torch.Tensor) or sequence.ndim != 3
            or sequence.shape[0] != history.shape[0] or sequence.shape[2] != history.shape[2]
            or sequence.shape[1] < required or not torch.isfinite(sequence).all()):
        raise ValueError('Dynamics target sequence must be a finite dense span covering history and requested future leads')
    prefix_indices = torch.arange(config.history_steps, device=sequence.device) * config.history_stride
    # A dataset/collation mismatch would silently define a different coordinate
    # target, so detect it before constructing loss labels.
    if not torch.equal(sequence.index_select(1, prefix_indices), history.detach().to(sequence)):
        raise ValueError('Dynamics target sequence history prefix disagrees with observed history')
    max_step = int(steps[-1].item())
    if not torch.equal(sequence[:, last_history+1:last_history+1+max_step],
                       targets[:, :max_step].detach().to(sequence)):
        raise ValueError('Dynamics target sequence future suffix disagrees with future labels')
    outputs = []
    with torch.no_grad():
        # Iterate over leads rather than stacking B*L whole histories; bounded
        # teacher peak memory and no gradients into labels or the teacher call.
        for step in steps.detach().cpu().tolist():
            window = sequence.detach().index_select(1, prefix_indices + step)
            outputs.append(pipeline.dynamics_encoder(window)[:, -1])
        latent = torch.stack(outputs, dim=1).detach()
    if not torch.isfinite(latent).all():
        raise FloatingPointError('Nonfinite dynamics latent target')
    return latent


def dynamics_latent_diagnostics(predicted, target):
    """Scale/collapse audits, not evidence that latent distances are physical."""
    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError('Dynamics latent predictions and labels must have matching (B,L,D) shapes')
    return {
        'dynamics_latent': (predicted-target.detach()).square().mean(),
        'dynamics_target_latent_variance': target.detach().flatten(0, 1).var(dim=0, unbiased=False).mean(),
        'dynamics_target_latent_rms': target.detach().square().mean().sqrt(),
        'dynamics_predicted_latent_variance': predicted.detach().flatten(0, 1).var(dim=0, unbiased=False).mean(),
        'dynamics_predicted_latent_rms': predicted.detach().square().mean().sqrt(),
    }


def dynamics_auxiliary_losses(pipeline, prediction, batch, lead_hours, config):
    """Per-field/lead spatial statistics plus optional pointwise/tendency fit.

Targets are selected by physical lead from the dense future label sequence.
Tendency uses physical units per hour normalized by training-only scales, as
the main forecast objective does. Nonuniform supported leads use their own dt.
"""
    config = validate_dynamics_auxiliary_config(config)
    rollout = config['version'] == 'climate_manifold.dynamics_auxiliary.v2'
    actual_rollout = getattr(pipeline.config, 'dynamics_auxiliary_architecture', 'direct') == 'latent_rollout'
    if rollout != actual_rollout:
        raise ValueError('Dynamics auxiliary loss disagrees with predictor architecture')
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
    indices = lead_steps(lead_hours, pipeline.a_config.step_hours)-1
    targets = batch['targets']
    if (targets.ndim != 3 or indices[-1] >= targets.shape[1]
            or targets.shape[0] != history.shape[0]):
        raise ValueError('Dynamics future labels do not cover the requested leads')
    target = targets.index_select(1, indices.to(targets.device)).detach()
    if rollout:
        trajectory = pipeline.dynamics_auxiliary_head.rollout(history, lead_hours)
        future = trajectory['fields']
        result['dynamics_latent'] = zero
        if config['supervision'] == 'field_latent':
            latent_target = future_dynamics_targets(pipeline, batch, lead_hours)
            result.update(dynamics_latent_diagnostics(trajectory['latent'], latent_target))
    else:
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
    if rollout:
        result['dynamics_regularization'] = (result['dynamics_regularization']
                                             + config['latent_weight']*result['dynamics_latent'])
    if not torch.isfinite(result['dynamics_regularization']):
        raise FloatingPointError('Nonfinite dynamics auxiliary loss')
    return result
