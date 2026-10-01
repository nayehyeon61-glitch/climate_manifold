"""Selectable field-statistics objectives, never predictive uncertainty scores.

W2/KL compare value marginals. ``signed_measure`` instead compares positive and
negative normalized-field mass at its original geographic locations.
"""
import math
import torch

from .joint_objective import spatial_quantile_loss

STATISTICAL_LOSSES = ('w2', 'kl_entropy', 'signed_measure')


def make_statistical_config(kind='w2', *, bins=64, value_range=6., bandwidth=.2):
    if kind == 'w2':
        return {'kind': 'w2', 'quantiles': 32,
                'estimator': 'area_weighted_inverse_cdf_midpoints_v1'}
    if kind == 'signed_measure':
        return {'kind': kind, 'epsilon': 1e-6,
                'zero_reference': 'normalized_field_zero',
                'spatial_weight': 1., 'mass_weight': 1.,
                'estimator': 'area_weighted_signed_spatial_js_and_mass_v1'}
    if kind != 'kl_entropy':
        raise ValueError('Unknown statistical_loss: '+str(kind))
    if isinstance(bins, bool) or not isinstance(bins, int) or not 3 <= bins <= 512:
        raise ValueError('kl_bins must be an integer in [3, 512]')
    for name, value in (('kl_range', value_range), ('kl_bandwidth', bandwidth)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(name+' must be positive and finite')
    return {'kind': kind, 'bins': bins, 'range': float(value_range),
            'bandwidth': float(bandwidth), 'epsilon': 1e-6,
            'direction': 'observed||reconstructed',
            'estimator': 'area_weighted_sigmoid_bins_with_tail_bins_v1'}


def validate_statistical_config(config):
    if not isinstance(config, dict):
        raise ValueError('Expected statistical_loss_config')
    expected = make_statistical_config(config.get('kind'), bins=config.get('bins', 64),
        value_range=config.get('range', 6.), bandwidth=config.get('bandwidth', .2))
    if config != expected:
        raise ValueError('Inconsistent statistical_loss_config')
    return dict(config)


def statistical_config_from_payload(payload):
    """Old paired records implicitly used W2; raw/non-statistical arms used none."""
    active = 'statistical' in (payload.get('constraint_pair') or '').split('_')
    config, kind = payload.get('statistical_loss_config'), payload.get('statistical_loss')
    if not active:
        if config is not None or kind is not None:
            raise ValueError('statistical_loss requires a statistical constraint pair')
        return None
    if config is None:
        if kind not in (None, 'w2'):
            raise ValueError('Missing statistical_loss_config for '+str(kind))
        return make_statistical_config()
    config = validate_statistical_config(config)
    if kind is not None and kind != config['kind']:
        raise ValueError('statistical_loss disagrees with statistical_loss_config')
    return config


def statistical_config_from_args(args, pair):
    kind = getattr(args, 'statistical_loss', None)
    options = [getattr(args, name, None) for name in ('kl_bins', 'kl_range', 'kl_bandwidth')]
    active = 'statistical' in (pair or '').split('_')
    if not active:
        if kind is not None or any(value is not None for value in options):
            raise ValueError('--statistical-loss/--kl-* require a statistical --constraint-pair')
        return None
    kind = kind or 'w2'
    if kind != 'kl_entropy' and any(value is not None for value in options):
        raise ValueError('--kl-* options require --statistical-loss kl_entropy')
    return make_statistical_config(kind,
        bins=64 if options[0] is None else options[0],
        value_range=6. if options[1] is None else options[1],
        bandwidth=.2 if options[2] is None else options[2])


def spatial_kl_entropy(prediction, target, area, config):
    """KL(P_observed || Q_reconstructed) via differentiable shared-bin masses.

    B-1 fixed sigmoid boundaries span [-range, +range]; B bins include two
    open tails. Each grid cell contributes soft membership summing to one,
    weighted by geographic area. Bins are shared by target/prediction and all
    cases; their positions never depend on predictions or held-out data.
    Values are in the existing normalized field coordinates. The temperature
    is bandwidth in those same units. Epsilon is a per-bin pseudocount.
    """
    config = validate_statistical_config(config)
    if config['kind'] != 'kl_entropy':
        raise ValueError('KL requires a kl_entropy config')
    if (prediction.shape != target.shape or prediction.ndim < 2 or area.ndim != 1
            or prediction.shape[-1] != len(area)):
        raise ValueError('Spatial KL requires matching [..., cells] fields and cell areas')
    if (not torch.isfinite(prediction).all() or not torch.isfinite(target).all()
            or not torch.isfinite(area).all() or not (area > 0).all()):
        raise ValueError('Spatial KL requires finite fields and positive areas')
    # At least float32 keeps finite log probabilities under mixed precision.
    dtype = torch.float64 if prediction.dtype == torch.float64 else torch.float32
    predicted = prediction.to(dtype)
    truth = target.detach().to(device=prediction.device, dtype=dtype)
    weights = area.detach().to(predicted)
    weights = weights / weights.sum()
    edges = torch.linspace(-config['range'], config['range'], config['bins']-1,
                           device=prediction.device, dtype=dtype)

    def histogram(values):
        cdf = torch.sigmoid((edges - values[..., None]) / config['bandwidth'])
        membership = torch.cat((cdf[..., :1], cdf.diff(dim=-1), 1-cdf[..., -1:]), -1)
        mass = (membership * weights[:, None]).sum(-2)
        mass = mass + config['epsilon']
        return mass / mass.sum(-1, keepdim=True)

    p, q = histogram(truth), histogram(predicted)
    log_p, log_q = p.log(), q.log()
    target_entropy = -(p * log_p).sum(-1).mean()
    reconstructed_entropy = -(q * log_q).sum(-1).mean()
    cross_entropy = -(p * log_q).sum(-1).mean()
    # Round-off may produce a tiny negative divergence near exact agreement.
    loss = (p * (log_p-log_q)).sum(-1).clamp_min(0).mean()
    return {'loss': loss, 'target_entropy': target_entropy,
            'reconstructed_entropy': reconstructed_entropy, 'cross_entropy': cross_entropy}


def spatial_statistical_loss(prediction, target, area, config=None):
    config = make_statistical_config() if config is None else validate_statistical_config(config)
    if config['kind'] == 'w2':
        return {'loss': spatial_quantile_loss(prediction, target, area, quantiles=config['quantiles'])}
    if config['kind'] == 'signed_measure':
        return spatial_signed_measure(prediction, target, area, config)
    return spatial_kl_entropy(prediction, target, area, config)


def spatial_signed_measure(prediction, target, area, config):
    """Compare signed spatial patterns AND their area-weighted magnitude.

    For each field, m_i^+ = a_i relu(x_i) and m_i^- = a_i relu(-x_i),
    where geographic areas a sum to one. Signs refer to zero in the existing
    normalized input coordinates (the training normalization reference), never
    zero physical pressure. No per-example centering or sorting is performed.
    Thus the spatial support and its positive/negative locations are preserved.

    Each sign contributes a JS divergence of location probabilities and the
    squared difference of total mass. Probabilities use an epsilon pseudomass
    distributed according to area: (m_i + eps a_i)/(sum_i m_i + eps).
    Empty signs therefore have a finite area prior; the separate mass term
    distinguishes empty, weak and strong fields even after normalization.
    """
    config = validate_statistical_config(config)
    if config['kind'] != 'signed_measure':
        raise ValueError('Signed measure requires a signed_measure config')
    if (prediction.shape != target.shape or prediction.ndim < 2 or area.ndim != 1
            or prediction.shape[-1] != len(area)):
        raise ValueError('Signed measure requires matching [..., cells] fields and cell areas')
    if (not prediction.is_floating_point() or not target.is_floating_point()
            or not torch.isfinite(prediction).all() or not torch.isfinite(target).all()
            or not torch.isfinite(area).all() or not (area > 0).all()):
        raise ValueError('Signed measure requires finite floating fields and positive areas')
    dtype = torch.float64 if prediction.dtype == torch.float64 else torch.float32
    predicted = prediction.to(dtype)
    truth = target.detach().to(device=prediction.device, dtype=dtype)
    weights = area.detach().to(predicted)
    weights = weights / weights.sum()

    def signed_parts(values):
        mass = torch.stack((values.relu(), (-values).relu()), dim=-2) * weights
        total = mass.sum(-1, keepdim=True)
        probability = (mass + config['epsilon'] * weights) / (total + config['epsilon'])
        return probability, total.squeeze(-1)

    p, target_mass = signed_parts(truth)
    q, predicted_mass = signed_parts(predicted)
    midpoint = .5 * (p + q)
    spatial_js = (.5 * (p * (p.log() - midpoint.log())
                            + q * (q.log() - midpoint.log())).sum(-1)).clamp_min(0).mean()
    mass_mse = (predicted_mass - target_mass).square().mean()
    return {'loss': config['spatial_weight'] * spatial_js + config['mass_weight'] * mass_mse,
            'signed_spatial_js': spatial_js, 'signed_mass_mse': mass_mse}
