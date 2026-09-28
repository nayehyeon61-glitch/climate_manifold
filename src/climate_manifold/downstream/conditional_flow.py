"""Observed-history conditional Flow Matching of future spatial marginals.

The auxiliary vector field is separate from forecast F and forecast D. It sees
two observed encodings and the reconstructed *origin* distribution as context.
True future quantiles supervise a detached straight interpolation path only.
Sampling solves a deterministic ODE from a noisy observed distribution; it is
not an SDE, a geographic transport map, or a calibrated weather-field ensemble.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .constraint_protocol import constraint_decoder_from_payload
from .reconstruction_objective import _observed_pair
from .statistical_flow import _field_target, weighted_spatial_quantiles


def make_conditional_flow_config(weight=0., quantiles=32, hidden_dim=128, noise_scale=.2):
    if (isinstance(weight, bool) or not isinstance(weight, (int, float))
            or not math.isfinite(weight) or weight < 0):
        raise ValueError('conditional_flow_weight must be finite and nonnegative')
    for name, value, upper in (('quantiles', quantiles, 512), ('hidden_dim', hidden_dim, 4096)):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
            raise ValueError(f'conditional_flow_{name} must be an integer in [1, {upper}]')
    if (isinstance(noise_scale, bool) or not isinstance(noise_scale, (int, float))
            or not math.isfinite(noise_scale) or noise_scale <= 0):
        raise ValueError('conditional_flow_noise_scale must be finite and positive')
    if not weight:
        return None
    return {'kind': 'observed_conditional_flow', 'version': 1,
            'weight': float(weight), 'quantiles': quantiles, 'hidden_dim': hidden_dim,
            'noise_scale': float(noise_scale), 'time': 'flow_tau',
            'physical_lead_unit': 'days', 'representation': 'normalized_spatial_quantiles',
            'source': 'observed_origin_plus_gaussian_noise',
            'condition': 'observed_pair_latent_and_reconstructed_origin_quantiles',
            'path': 'detached_linear_interpolation', 'velocity': 'joint_lead_temporal_conv_v1',
            'sampling_rearrangement': 'sort_per_variable'}


def validate_conditional_flow_config(config):
    if not isinstance(config, dict):
        raise ValueError('Expected conditional_flow_config')
    expected = make_conditional_flow_config(config.get('weight'), config.get('quantiles'),
                                            config.get('hidden_dim'), config.get('noise_scale'))
    if expected is None or config != expected:
        raise ValueError('Inconsistent conditional_flow_config')
    return dict(config)


def _validate_route(pair, values, decoder_mode=None):
    if 'statistical' not in (pair or '').split('_'):
        raise ValueError('Conditional flow requires a statistical constraint pair')
    for name, required in (('bridge', 'latent'), ('training_mode', 'joint'),
                           ('anchor', 'none'), ('representation', 'climate_manifold')):
        if name in values and values[name] != required:
            raise ValueError('Conditional flow requires a joint unanchored latent climate manifold')
    decoder_mode = decoder_mode or values.get('constraint_decoder') or 'separate_surface_and_information'
    if decoder_mode not in ('separate_surface_and_information', 'information_only'):
        raise ValueError('Conditional flow requires auxiliary decoders independent of forecast D')
    if values.get('statistical_flow_weight', 0.):
        raise ValueError('Conditional flow and forecast statistical flow are mutually exclusive')


def conditional_flow_config_from_args(args, pair):
    options = {name: getattr(args, 'conditional_flow_'+name, None)
               for name in ('quantiles', 'hidden_dim', 'noise_scale')}
    config = make_conditional_flow_config(getattr(args, 'conditional_flow_weight', 0.),
        **{name: value for name, value in options.items() if value is not None})
    if config is None:
        if any(value is not None for value in options.values()):
            raise ValueError('Conditional flow settings require positive --conditional-flow-weight')
        return None
    _validate_route(pair, vars(args))
    return config


def conditional_flow_config_from_payload(payload):
    config = payload.get('conditional_flow_config')
    if config is None:
        if payload.get('conditional_flow_weight', 0.) != 0.:
            raise ValueError('Missing conditional_flow_config')
        return None
    config = validate_conditional_flow_config(config)
    values = {**payload.get('config', {}),
              'statistical_flow_weight': payload.get('statistical_flow_weight',
                  (payload.get('statistical_flow_config') or {}).get('weight', 0.))}
    _validate_route(payload.get('constraint_pair'), values, constraint_decoder_from_payload(payload))
    if ('conditional_flow_weight' in payload
            and payload['conditional_flow_weight'] != config['weight']):
        raise ValueError('conditional_flow_weight disagrees with conditional_flow_config')
    return config


class ObservedConditionalFlow(nn.Module):
    """Joint-lead velocity head, without registering shared E or D modules.

    Spatial latent channels are summarized by mean and standard deviation at
    each observed time. Global representations retain all latent coordinates.
    The current decoded quantiles preserve a differentiable D_rec/D_I route.
    A kernel-three temporal convolution couples adjacent future leads; this
    auxiliary predicts the complete requested lead sequence simultaneously.
    """
    def __init__(self, manifold, config, decoder_mode='separate_surface_and_information'):
        super().__init__()
        self.config = validate_conditional_flow_config(config)
        if decoder_mode not in ('separate_surface_and_information', 'information_only'):
            raise ValueError('Conditional flow requires auxiliary decoders independent of forecast D')
        if (getattr(manifold, 'info_head', None) is None
                or getattr(manifold, 'info_metadata', None) is None):
            raise ValueError('Conditional flow requires enriched information and its decoder')
        self.decoder_mode = decoder_mode
        self.surface_grid = tuple(manifold.config.grid)
        self.info_shape = tuple(manifold.info_metadata['shape'])
        if self.info_shape[1:] != self.surface_grid[1:]:
            raise ValueError('Conditional flow surface and information grids must match')
        variables = manifold.info_metadata['variables']
        if (len(variables) != self.info_shape[0]
                or any(v['kind'] not in ('static', 'dynamic') for v in variables)):
            raise ValueError('Conditional flow requires valid information variable kinds')
        self.dynamic_indices = tuple(i for i, v in enumerate(variables) if v['kind'] == 'dynamic')
        if not self.dynamic_indices:
            raise ValueError('Conditional flow requires dynamic information fields')
        self.surface_names = ([] if decoder_mode == 'information_only' else
                              list(manifold.core.physics.variable_names))
        if self.surface_names and len(self.surface_names) != self.surface_grid[0]:
            raise ValueError('Conditional flow surface variable count must match the grid')
        self.variable_names = (['surface:'+name for name in self.surface_names]
            + ['information:'+variables[i]['name'] for i in self.dynamic_indices])
        self.variable_count = len(self.variable_names)
        self.latent_dim = manifold.config.manifold_dim
        self.latent_grid = (tuple(manifold.config.latent_grid)
                           if manifold.config.representation_kind == 'spatial' else None)
        pooled_dim = 2*self.latent_grid[0] if self.latent_grid else self.latent_dim
        self.state_size = self.variable_count*self.config['quantiles']
        hidden = self.config['hidden_dim']
        self.context_net = nn.Sequential(nn.Linear(2*pooled_dim+self.state_size, hidden),
                                         nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.velocity_net = nn.Sequential(nn.Linear(self.state_size+hidden+4, hidden),
                                          nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.temporal = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        # Normal initialization is intentional: a zero output matrix would block
        # E and auxiliary-D gradients on the first optimization step.
        self.output = nn.Linear(hidden, self.state_size)

    def context(self, raw_pair, decoded_quantiles):
        if raw_pair.ndim != 3 or raw_pair.shape[1:] != (2, self.latent_dim):
            raise ValueError('Conditional flow requires two observed latent encodings')
        if decoded_quantiles.shape != (len(raw_pair), self.variable_count, self.config['quantiles']):
            raise ValueError('Conditional flow origin quantiles have incompatible dimensions')
        latent = raw_pair
        if self.latent_grid:
            latent = latent.reshape(len(raw_pair), 2, self.latent_grid[0], -1)
            average = latent.mean(-1)
            deviation = (latent.var(-1, unbiased=False)+1e-6).sqrt()
            latent = torch.cat((average, deviation), dim=-1)
        return self.context_net(torch.cat((latent.flatten(1), decoded_quantiles.flatten(1)), -1))

    def forward(self, state, tau, lead_hours, context):
        if (state.ndim != 4 or state.shape[2:] != (self.variable_count, self.config['quantiles'])
                or context.shape != (len(state), self.config['hidden_dim'])):
            raise ValueError('Conditional flow velocity requires [batch, leads, variables, quantiles]')
        batch, steps = state.shape[:2]
        leads = _lead_hours(lead_hours, state, steps)
        tau = torch.as_tensor(tau, device=state.device, dtype=state.dtype)
        if tau.numel() not in (1, batch) or not torch.isfinite(tau).all() or not ((tau >= 0) & (tau <= 1)).all():
            raise ValueError('Conditional flow tau must have one [0,1] value per joint trajectory')
        tau = tau.reshape(-1, 1).expand(batch, steps)
        physical = (leads/24.).expand(batch, steps)
        times = torch.stack((tau, physical, (math.pi*tau).sin(), (math.pi*tau).cos()), -1)
        features = torch.cat((state.flatten(2), context[:, None].expand(-1, steps, -1), times), -1)
        hidden = self.velocity_net(features)
        hidden = hidden + torch.nn.functional.silu(self.temporal(hidden.transpose(1, 2)).transpose(1, 2))
        return self.output(hidden).reshape_as(state)


def _lead_hours(lead_hours, reference, steps=None):
    leads = torch.as_tensor(lead_hours, device=reference.device, dtype=reference.dtype).detach()
    if (leads.ndim != 1 or len(leads) < 1 or (steps is not None and len(leads) != steps)
            or not torch.isfinite(leads).all() or not (leads > 0).all()
            or not (leads[1:] > leads[:-1]).all()):
        raise ValueError('Conditional flow lead hours must be positive, finite, increasing and match targets')
    return leads


def _prepare_observed(pipeline, batch):
    head = getattr(pipeline, 'conditional_flow', None)
    bridge = getattr(pipeline, 'bridge', None)
    manifold = getattr(bridge, 'manifold', None)
    if (not isinstance(head, ObservedConditionalFlow) or manifold is None
            or bridge.mode != 'latent' or bridge.anchor != 'none'
            or getattr(getattr(pipeline, 'config', None), 'training_mode', 'joint') != 'joint'):
        raise ValueError('Conditional flow requires its auxiliary head and joint unanchored manifold')
    states, information, _ = _observed_pair(batch, manifold)
    raw_pair = manifold.raw_encode(states, information)
    cells = math.prod(head.surface_grid[1:])
    area = manifold.temporal.area.flatten().to(states)
    origin_info = information[:, 1].reshape(len(states), head.info_shape[0], cells)[:, head.dynamic_indices]
    decoded_info = manifold.info_head(raw_pair[:, 1])
    if decoded_info.shape != information[:, 1].shape:
        raise ValueError('Conditional flow information decoder has incompatible dimensions')
    decoded_info = decoded_info.reshape(len(states), head.info_shape[0], cells)[:, head.dynamic_indices]
    source_fields, decoded_fields = [origin_info], [decoded_info]
    if head.decoder_mode != 'information_only':
        decoder = getattr(pipeline, 'reconstruction_decoder', None)
        if decoder is None:
            raise ValueError('Conditional flow requires a dedicated reconstruction_decoder')
        decoded_surface = decoder(raw_pair[:, 1])
        if decoded_surface.shape != states[:, 1].shape:
            raise ValueError('Conditional flow surface decoder has incompatible dimensions')
        source_fields.insert(0, states[:, 1].reshape(len(states), head.surface_grid[0], cells))
        decoded_fields.insert(0, decoded_surface.reshape(len(states), head.surface_grid[0], cells))
    source_q = weighted_spatial_quantiles(torch.cat(source_fields, dim=1).detach(), area, head.config['quantiles'])
    decoded_q = weighted_spatial_quantiles(torch.cat(decoded_fields, dim=1), area, head.config['quantiles'])
    context = head.context(raw_pair, decoded_q)
    return head, source_q.detach(), context, area


def observed_conditional_flow_losses(pipeline, batch, lead_hours, config, *, generator=None):
    """CFM loss from observed pairs; future labels never enter context/encoders.

    One tau is drawn for the complete future trajectory. All future leads form
    one vector-valued state; the velocity head couples their hidden features.
    Target/source interpolation and target velocities are detached labels.
    """
    if config is None:
        reference = batch.get('constraint_states', batch.get('history'))
        zero = reference.new_zeros(()) if isinstance(reference, torch.Tensor) else torch.tensor(0.)
        return {'conditional_flow': zero, 'conditional_flow_regularization': zero}
    config = validate_conditional_flow_config(config)
    head = getattr(pipeline, 'conditional_flow', None)
    if not isinstance(head, ObservedConditionalFlow) or head.config != config:
        raise ValueError('Conditional flow objective and head configurations must match')
    reference = batch.get('constraint_states')
    if not isinstance(reference, torch.Tensor) or reference.ndim != 3:
        raise ValueError('Conditional flow requires observed constraint_states')
    leads = _lead_hours(lead_hours, reference)
    steps, batch_size = len(leads), len(reference)
    dt = batch.get('dt_hours')
    if (not isinstance(dt, torch.Tensor) or dt.ndim != 2 or dt.shape[0] != batch_size
            or dt.shape[1] < steps or not torch.isfinite(dt[:, :steps]).all()
            or not (dt[:, :steps] > 0).all()
            or not torch.equal(dt[:, :steps].cumsum(1).to(leads), leads.expand(batch_size, -1))):
        raise ValueError('Conditional flow lead hours must align with target dt_hours')
    cells = math.prod(head.surface_grid[1:])
    target_info = _field_target(batch.get('statistical_flow_information_targets'), batch_size,
                               math.prod(head.info_shape), 'statistical_flow_information_targets', steps)
    target_fields = [target_info.reshape(batch_size, steps, head.info_shape[0], cells)[:, :, head.dynamic_indices]]
    if head.decoder_mode != 'information_only':
        targets = _field_target(batch.get('targets'), batch_size, math.prod(head.surface_grid), 'targets', steps)
        target_fields.insert(0, targets.reshape(batch_size, steps, head.surface_grid[0], cells))
    head, source_q, context, area = _prepare_observed(pipeline, batch)
    target_q = weighted_spatial_quantiles(torch.cat(target_fields, dim=2).detach().to(source_q),
                                          area, config['quantiles']).detach()
    noise = torch.randn(target_q.shape, device=source_q.device, dtype=source_q.dtype, generator=generator)
    source = (source_q[:, None]+config['noise_scale']*noise).detach()
    tau = torch.rand((batch_size, 1, 1, 1), device=source.device, dtype=source.dtype, generator=generator)
    interpolated = ((1-tau)*source+tau*target_q).detach()
    velocity_target = (target_q-source).detach()
    velocity = head(interpolated, tau, leads, context)
    loss = (velocity-velocity_target).square().mean()
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite observed conditional Flow Matching objective')
    return {'conditional_flow': loss, 'conditional_flow_regularization': config['weight']*loss}


@torch.no_grad()
def sample_observed_conditional_flow(pipeline, batch, lead_hours, *, members=8, steps=32, seed=7):
    """Generate future marginal scenarios without any future-label access.

    The midpoint ODE integrates flow time tau in [0,1], with physical leads as
    conditions. Final per-variable sorting rearranges generated vectors into
    valid monotone quantile functions; it is postprocessing, not a learned
    monotonicity guarantee. Returned fields are NORMALIZED marginal quantiles.
    """
    for name, value in (('members', members), ('steps', steps)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'Conditional flow sampling {name} must be a positive integer')
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('Conditional flow sampling seed must be a nonnegative integer')
    # Preserve every submodule's prior state, including deliberately frozen E.
    modes = {module: module.training for module in pipeline.modules()}
    pipeline.eval()
    try:
        head, source, context, _ = _prepare_observed(pipeline, batch)
        leads = _lead_hours(lead_hours, source)
        batch_size, variables, quantiles = source.shape
        generator = torch.Generator(device=source.device).manual_seed(seed)
        shape = (batch_size, members, len(leads), variables, quantiles)
        noise = torch.randn(shape, device=source.device, dtype=source.dtype, generator=generator)
        state = source[:, None, None]+head.config['noise_scale']*noise
        state = state.reshape(batch_size*members, len(leads), variables, quantiles)
        condition = context[:, None].expand(-1, members, -1).reshape(batch_size*members, -1)
        delta = 1./steps
        for index in range(steps):
            tau = index*delta
            first = head(state, tau, leads, condition)
            midpoint = state+0.5*delta*first
            state = state+delta*head(midpoint, tau+0.5*delta, leads, condition)
            if not torch.isfinite(state).all():
                raise FloatingPointError('Nonfinite conditional flow ODE trajectory')
        quantile_levels = (torch.arange(quantiles, device=source.device, dtype=source.dtype)+.5)/quantiles
        return {'quantiles': state.reshape(shape).sort(-1).values,
                'quantile_levels': quantile_levels, 'variable_names': list(head.variable_names),
                'lead_hours': leads}
    finally:
        for module, training in modes.items():
            module.training = training
