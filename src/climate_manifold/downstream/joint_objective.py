"""Auxiliary objectives on the actual jointly trained forecasting trajectory.

The forecasting trainer supplies state/tendency supervision separately. Nothing
here calls A's former sampler or latent drift. Information targets and future
surface fields enter losses only, never the forecasting network's inputs.

``distribution`` is an optional deterministic *spatial marginal* quantile loss.
It is not ensemble CRPS, a calibrated predictive distribution, or a claim that
the former stochastic A objective has been retained unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math

import torch

from ..dynamics import trajectory_pinn_losses


@dataclass(frozen=True)
class JointObjectiveWeights:
    reconstruction: float = .1
    physics: float = .01
    information: float = .1
    static: float = .05
    distribution: float = 0.
    pinn: float = 0.

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f'Joint {field.name} weight must be finite and nonnegative')


def spatial_quantile_loss(prediction, target, area, quantiles=32):
    """Approximate weighted 1-D W2 squared for each field's spatial marginal.

Inputs end in a flattened spatial-cell dimension. Each field is treated
separately; sorting never pools variables, batch elements, or forecast leads.
Equal probability midpoint quadrature approximates the inverse-CDF integral.
Geographic area follows its cell through sorting, preserving unequal grid
weights. This loses location information and therefore supplements, rather than
replaces, pointwise forecasting and information losses.
"""
    if (prediction.shape != target.shape or prediction.ndim < 2
            or area.ndim != 1 or prediction.shape[-1] != len(area)):
        raise ValueError('Spatial quantiles require matching [..., cells] fields and cell areas')
    if isinstance(quantiles, bool) or not isinstance(quantiles, int) or quantiles < 1:
        raise ValueError('Quantile count must be a positive integer')
    if (not torch.isfinite(prediction).all() or not torch.isfinite(target).all()
            or not torch.isfinite(area).all() or (area <= 0).any()):
        raise ValueError('Spatial quantiles require finite fields and positive areas')
    area = area.to(prediction)
    probabilities = (torch.arange(quantiles, device=prediction.device, dtype=prediction.dtype) + .5) / quantiles

    def inverse_cdf(value):
        ordered, indices = value.sort(dim=-1)
        weights = area.expand_as(value).gather(-1, indices)
        cdf = (weights / weights.sum(-1, keepdim=True)).cumsum(-1)
        levels = probabilities.expand(*value.shape[:-1], quantiles).contiguous()
        ranks = torch.searchsorted(cdf.contiguous(), levels).clamp_max(value.shape[-1] - 1)
        return ordered.gather(-1, ranks)

    return (inverse_cdf(prediction) - inverse_cdf(target.detach())).square().mean()


def _information_target(batch, origin, steps):
    future = batch.get('information_targets')
    if (future is None or future.ndim != 3 or future.shape[0] != len(origin)
            or future.shape[1] < steps or future.shape[-1] != origin.shape[-1]
            or not torch.isfinite(future).all()):
        raise ValueError('Joint information/PINN loss requires matching finite information_targets')
    return future[:, :steps].detach()


def joint_losses(pipeline, prediction, batch, weights, lead_hours):
    """Return named auxiliary losses and their weighted ``regularization`` sum.

Reconstruction covers observed history. Surface diagnostic matching uses the
actual future prediction. Latent-path information/PINN terms require an
unanchored latent bridge; decoded-grid ClimODE supports reconstruction and
surface diagnostics but cannot silently pretend to evolve latent coordinates.
All terms can be disabled individually, including for fair objective ablations.
"""
    if not isinstance(weights, JointObjectiveWeights):
        raise TypeError('weights must be JointObjectiveWeights')
    mean = prediction['mean']
    if mean.ndim != 3 or mean.shape[1] != len(lead_hours):
        raise ValueError('Joint prediction must be [batch, forecast leads, state features]')
    zero = mean.new_zeros(())
    values = {name: zero for name in ('reconstruction', 'physics', 'information', 'static',
                                     'information_spatial_quantile', 'pinn_total')}
    values['regularization'] = zero
    if not any(getattr(weights, field.name) for field in fields(weights)):
        return values
    manifold = pipeline.bridge.manifold
    if manifold is None:
        raise ValueError('Joint auxiliary objectives require a trainable representation bridge')
    if not hasattr(manifold, 'temporal'):
        raise ValueError('Joint objectives require ClimateManifold; legacy plain AE is a frozen control')

    if weights.reconstruction:
        history = batch['history']
        information = batch.get('information')
        info_history = (None if information is None else
                        information[:, None].expand(-1, history.shape[1], -1))
        raw_history = manifold.raw_encode(history, info_history)
        reconstructed = manifold.core.manifold.decode(raw_history)
        values['reconstruction'] = ((reconstructed - history.detach()).square()
                                    * manifold.temporal.metric).sum(-1).mean()
    if weights.physics:
        future = batch['targets'][:, :mean.shape[1]].detach()
        values['physics'] = manifold.core.physics.reconstruction_losses(mean, future)['physics']

    needs_information = any((weights.information, weights.static, weights.distribution, weights.pinn))
    if needs_information:
        if pipeline.bridge.mode != 'latent' or pipeline.bridge.anchor != 'none':
            raise ValueError('Information/PINN joint losses require an unanchored latent bridge')
        if manifold.info_head is None or batch.get('information') is None:
            raise ValueError('Information/PINN joint losses require enriched inputs and an information decoder')
        origin_info = batch['information'].detach()
        future_q, origin_q = prediction.get('predicted_latent'), prediction.get('origin_latent')
        expected = (len(mean), mean.shape[1], manifold.config.manifold_dim)
        if (future_q is None or origin_q is None or future_q.shape != expected
                or origin_q.shape != (len(mean), manifold.config.manifold_dim)):
            raise ValueError('Joint information/PINN loss requires the actual forecast latent trajectory')
        raw_path = torch.cat((origin_q[:, None], future_q), dim=1)
        raw_path = raw_path * manifold.core.latent_scale + manifold.core.latent_mean
        decoded = manifold.info_head(raw_path)
        shape = manifold.info_metadata['shape']
        cells = shape[1] * shape[2]
        decoded_fields = decoded.reshape(len(mean), mean.shape[1] + 1, shape[0], cells)
        origin_fields = origin_info.reshape(len(mean), shape[0], cells)
        static = torch.tensor([v['kind'] == 'static' for v in manifold.info_metadata['variables']],
                              device=mean.device)
        area = manifold.temporal.area.flatten().to(mean)
        future_info = (_information_target(batch, origin_info, mean.shape[1])
                       if weights.information or weights.distribution or weights.pinn else None)
        if weights.information:
            if not bool((~static).any()):
                raise ValueError('Information loss requires at least one dynamic information field')
            target_fields = future_info.reshape(len(mean), mean.shape[1], shape[0], cells)
            present_error = decoded_fields[:, 0, ~static] - origin_fields[:, ~static]
            future_error = decoded_fields[:, 1:, ~static] - target_fields[:, :, ~static]
            values['information_origin'] = (present_error.square() * area).sum(-1).mean()
            values['information_future'] = (future_error.square() * area).sum(-1).mean()
            values['information'] = .5 * (values['information_origin'] + values['information_future'])
        if weights.static:
            if not bool(static.any()):
                raise ValueError('Static loss requires static terrain information fields')
            error = decoded_fields[:, :, static] - origin_fields[:, None, static]
            values['static'] = (error.square() * area).sum(-1).mean()
        if weights.distribution:
            if not bool((~static).any()):
                raise ValueError('Spatial distribution loss requires dynamic information fields')
            target_fields = future_info.reshape(len(mean), mean.shape[1], shape[0], cells)
            values['information_spatial_quantile'] = spatial_quantile_loss(
                decoded_fields[:, 1:, ~static], target_fields[:, :, ~static], area)
        if weights.pinn:
            if manifold.pinn is None:
                raise ValueError('Positive PINN weight requires an enabled Hybrid PINN')
            canonical = torch.arange(1, mean.shape[1] + 1, device=lead_hours.device,
                                     dtype=lead_hours.dtype) * 6
            if not torch.equal(lead_hours, canonical):
                raise ValueError('Joint trajectory PINN requires consecutive 6h forecast leads')
            path = {'raw_latents': raw_path, 'information': decoded,
                    'states': torch.cat((prediction['reconstructed_origin'][:, None], mean), dim=1)}
            values.update(trajectory_pinn_losses(manifold, batch, path))

    values['regularization'] = (weights.reconstruction * values['reconstruction']
                                + weights.physics * values['physics']
                                + weights.information * values['information']
                                + weights.static * values['static']
                                + weights.distribution * values['information_spatial_quantile']
                                + weights.pinn * values['pinn_total'])
    if any(not bool(torch.isfinite(value).all()) for value in values.values()):
        raise FloatingPointError('Nonfinite joint representation objective')
    return values
