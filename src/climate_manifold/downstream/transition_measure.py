"""Observed-transition relational supervision for dynamics encoder v2.

The bank is a TRAIN-only empirical measure, not a reverse physics solver or a
stochastic forecast generator. Similar current fields retrieve distinct past
histories. Their observed histories and subsequent changes define relational
labels; only the query E_d receives gradients. No bank is needed at inference.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral, Real

import numpy as np
import torch

from ..temporal_supervision import area_weights
from .dynamics_information import (DYNAMICS_INFORMATION_FIELDS, DYNAMICS_INFORMATION_MODES,
                                   resolve_dynamics_information, select_dynamics_information)

_VERSION = 'climate_manifold.transition_measure.v1'
_BANK_VERSION = 'climate_manifold.transition_bank.v1'
_DEFAULTS = {
    'weight': .1, 'neighbors': 8, 'bank_size': 4096,
    'endpoint_temperature': 1., 'relation_temperature': 1., 'latent_temperature': 1.,
    'history_weight': 1., 'future_weight': 1., 'geometry_weight': .1,
    'pool_lat': 4, 'pool_lon': 8, 'exclusion_steps': 0,
}
_SEMANTICS = {
    'version': _VERSION,
    'retrieval': 'current_endpoint_only_area_weighted_normalized_coarse_field',
    'target': 'observed_history_and_future_changes_per_physical_day',
    'support_encoder': 'online_stop_gradient',
    'train_only': True, 'inference_bank_required': False,
    'overlap_policy': 'exclude_entire_observed_and_future_intervals',
}


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'Transition {name} must be an integer >= {minimum}')


def validate_transition_config(config):
    """Strict persisted contract; unknown/missing knobs cannot silently default."""
    if not isinstance(config, dict) or set(config) != set(_DEFAULTS) | set(_SEMANTICS) | {'forecast_steps', 'information_mode', 'information_fields'}:
        raise ValueError('Invalid dynamics_transition_config keys')
    if any(config[key] != value or (isinstance(value, bool) and type(config[key]) is not bool)
           for key, value in _SEMANTICS.items()):
        raise ValueError('Invalid transition measure semantics')
    for key in ('neighbors', 'bank_size', 'pool_lat', 'pool_lon', 'exclusion_steps'):
        _integer(config[key], key, 2 if key == 'neighbors' else 0 if key == 'exclusion_steps' else 1)
    _integer(config['forecast_steps'], 'forecast_steps')
    mode = config['information_mode']
    if mode not in DYNAMICS_INFORMATION_MODES or config['information_fields'] != (
            list(DYNAMICS_INFORMATION_FIELDS) if mode == 'geopotential_terrain' else []):
        raise ValueError('Invalid transition dynamics information contract')
    if config['bank_size'] < config['neighbors']:
        raise ValueError('Transition bank_size must be >= neighbors')
    for key in ('weight', 'endpoint_temperature', 'relation_temperature', 'latent_temperature',
                'history_weight', 'future_weight', 'geometry_weight'):
        value = config[key]
        if (isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
                or value < 0 or key in ('weight', 'endpoint_temperature', 'relation_temperature',
                                      'latent_temperature') and value == 0):
            raise ValueError(f'Transition {key} must be finite and '
                             + ('positive' if key in ('weight', 'endpoint_temperature',
                                                       'relation_temperature', 'latent_temperature')
                                else 'nonnegative'))
    if config['history_weight'] + config['future_weight'] <= 0:
        raise ValueError('Transition history_weight and future_weight cannot both be zero')
    return dict(config)


def make_transition_config(args, predictor_config):
    version = getattr(predictor_config, 'dynamics_version', 'v1')
    explicit = {key: getattr(args, 'transition_'+key, None) for key in _DEFAULTS}
    if version == 'v1':
        if any(value is not None for value in explicit.values()):
            raise ValueError('Transition options require --dynamics-version v2')
        return None
    if (version != 'v2' or getattr(predictor_config, 'dynamics_encoder', 'off') == 'off'
            or getattr(predictor_config, 'dynamics_mode', 'learned') not in ('learned', 'zero')):
        raise ValueError('Transition v2 requires an enabled dynamics encoder')
    mode = getattr(predictor_config, 'dynamics_information', 'off')
    return validate_transition_config({**_SEMANTICS, 'forecast_steps': getattr(args, 'horizon_steps', None),
        'information_mode': mode, 'information_fields': list(DYNAMICS_INFORMATION_FIELDS) if mode == 'geopotential_terrain' else [], **{
        key: _DEFAULTS[key] if explicit[key] is None else explicit[key] for key in _DEFAULTS}})


def _indices_hash(indices):
    return hashlib.sha256(json.dumps([int(i) for i in indices]).encode()).hexdigest()


def validate_transition_bank_manifest(manifest, config):
    keys = {'version', 'train_only', 'count', 'starts', 'starts_sha256',
            'training_starts_count', 'training_starts_sha256', 'train_prefix_end',
            'history_span_steps', 'horizon_steps', 'future_steps', 'step_hours', 'exclusion_steps',
            'train_first_time_ns', 'train_last_time_ns', 'effective_pool_lat', 'effective_pool_lon',
            'train_states_sha256', 'normalization_sha256', 'transition_config_sha256',
            'information_mode', 'information_fields', 'train_information_sha256'}
    if not isinstance(manifest, dict) or set(manifest) != keys:
        raise ValueError('Invalid dynamics_transition_bank manifest')
    if manifest['version'] != _BANK_VERSION or manifest['train_only'] is not True:
        raise ValueError('Transition bank must be train-only')
    for key in ('count', 'training_starts_count', 'train_prefix_end', 'history_span_steps',
                'horizon_steps', 'future_steps', 'step_hours', 'effective_pool_lat', 'effective_pool_lon'):
        _integer(manifest[key], key)
    _integer(manifest['exclusion_steps'], 'exclusion_steps', 0)
    if (manifest['information_mode'] != config['information_mode']
            or manifest['information_fields'] != config['information_fields']):
        raise ValueError('Transition bank information contract disagrees')
    info_hash = manifest['train_information_sha256']
    if config['information_mode'] == 'off':
        if info_hash is not None:
            raise ValueError('Disabled dynamics information cannot have an information hash')
    elif (not isinstance(info_hash, str) or len(info_hash) != 64
          or any(c not in '0123456789abcdef' for c in info_hash)):
        raise ValueError('Transition bank needs a train-only information fingerprint')
    starts = manifest['starts']
    if not isinstance(starts, list) or len(starts) != manifest['count']:
        raise ValueError('Transition bank starts/count disagree')
    for start in starts:
        _integer(start, 'bank start', 0)
    if (len(starts) < 2 or starts != sorted(set(starts))
            or manifest['count'] > min(config['bank_size'], manifest['training_starts_count'])
            or manifest['starts_sha256'] != _indices_hash(starts)
            or manifest['exclusion_steps'] != config['exclusion_steps']
            or manifest['future_steps'] != config['forecast_steps']
            or manifest['future_steps'] > manifest['horizon_steps']
            or starts[-1] + manifest['history_span_steps'] + manifest['horizon_steps'] > manifest['train_prefix_end']
            or manifest['effective_pool_lat'] > config['pool_lat']
            or manifest['effective_pool_lon'] > config['pool_lon']):
        raise ValueError('Transition bank indices or configuration disagree')
    for key in ('train_first_time_ns', 'train_last_time_ns'):
        if isinstance(manifest[key], bool) or not isinstance(manifest[key], Integral):
            raise ValueError('Transition bank requires integer timestamp bounds')
    expected_last = manifest['train_first_time_ns'] + (manifest['train_prefix_end']-1)*manifest['step_hours']*3600*10**9
    if manifest['train_last_time_ns'] != expected_last:
        raise ValueError('Transition bank timestamp bounds disagree')
    for key in ('training_starts_sha256', 'train_states_sha256', 'normalization_sha256', 'transition_config_sha256'):
        if (not isinstance(manifest[key], str) or len(manifest[key]) != 64
                or any(c not in '0123456789abcdef' for c in manifest[key])):
            raise ValueError('Invalid hash in transition bank: '+key)
    if manifest['transition_config_sha256'] != hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest():
        raise ValueError('Transition bank config fingerprint disagrees')
    return manifest


def transition_config_from_payload(payload, predictor_config):
    """Validate metadata without loading any training observations at inference."""
    version = getattr(predictor_config, 'dynamics_version', 'v1')
    config = payload.get('dynamics_transition_config')
    manifest = payload.get('dynamics_transition_bank')
    if version == 'v1':
        if config is not None or manifest is not None:
            raise ValueError('Dynamics v1 checkpoint cannot contain transition v2 metadata')
        return None
    if (version != 'v2' or getattr(predictor_config, 'dynamics_encoder', 'off') == 'off'
            or getattr(predictor_config, 'dynamics_mode', 'learned') not in ('learned', 'zero')):
        raise ValueError('Transition v2 checkpoint requires an enabled dynamics encoder')
    config = validate_transition_config(config)
    if config['information_mode'] != getattr(predictor_config, 'dynamics_information', 'off'):
        raise ValueError('Transition configuration disagrees with dynamics information inputs')
    manifest = validate_transition_bank_manifest(manifest, config)
    metadata = payload.get('a_metadata', {})
    a_config, split = metadata.get('config', {}), metadata.get('split', {})
    if not {'history_steps', 'history_stride', 'horizon_steps', 'step_hours'} <= set(a_config) or 'train' not in split:
        raise ValueError('Transition checkpoint requires the A training split and time contract')
    span = (a_config['history_steps']-1)*a_config['history_stride']+1
    prefix = int(split['train'][-1])+span+a_config['horizon_steps']
    if (manifest['history_span_steps'] != span or manifest['horizon_steps'] != a_config['horizon_steps']
            or manifest['step_hours'] != a_config['step_hours'] or manifest['train_prefix_end'] != prefix
            or not set(manifest['starts']) <= set(split['train'])):
        raise ValueError('Transition bank disagrees with the checkpoint training split or time contract')
    expected_leads = [i*a_config['step_hours'] for i in range(1, config['forecast_steps']+1)]
    if payload.get('lead_hours') != expected_leads:
        raise ValueError('Transition future labels must match the main forecast lead window')
    training_hash = payload.get('training_contract', {}).get('train_starts_sha256')
    if training_hash is None or manifest['training_starts_sha256'] != training_hash:
        raise ValueError('Transition bank does not match actual training loader starts')
    return config


class TransitionBank:
    """Bounded exact neighbor search over a deterministic TRAIN-window subset.

    Full archive future/held-out arrays are never retained. Descriptors are
    pooled with physical area weights; normalized per-cell training scales are
    inherited from the existing data contract. Search is chunked, never NxN.
    """
    def __init__(self, data, a_config, starts, transition_config):
        self.config = validate_transition_config(transition_config)
        self.a_config = a_config
        raw_starts = list(starts)
        if not raw_starts:
            raise ValueError('Transition bank needs actual nonempty training loader starts')
        for start in raw_starts:
            _integer(start, 'training start', 0)
        training_starts = [int(i) for i in raw_starts]
        if len(set(training_starts)) != len(training_starts):
            raise ValueError('Transition training starts must be unique')
        allowed = {int(i) for i in data['split']['train']}
        if not set(training_starts) <= allowed:
            raise ValueError('Transition bank accepts only training split starts')
        prefix = int(data['train_end'])
        span, horizon = a_config.history_span_steps, a_config.horizon_steps
        expected_prefix = max(allowed)+span+horizon
        if prefix != expected_prefix or max(training_starts)+span+horizon > prefix:
            raise ValueError('Transition train prefix disagrees with training windows')
        if self.config['forecast_steps'] > horizon:
            raise ValueError('Transition forecast_steps exceeds the archive label contract')
        self.window_length = span+horizon
        self.training_starts = frozenset(training_starts)
        sorted_starts = sorted(training_starts)
        count = min(len(sorted_starts), self.config['bank_size'])
        if count < 2:
            raise ValueError('Transition bank needs at least two training windows')
        positions = np.linspace(0, len(sorted_starts)-1, count, dtype=np.int64)
        self.starts = torch.tensor([sorted_starts[i] for i in positions], dtype=torch.long)
        candidate_starts = self.starts.numpy()
        query_starts = np.asarray(training_starts)
        gap = self.window_length-1+self.config['exclusion_steps']
        # Count nonoverlapping candidates for every actual training query before
        # any optimizer step, in O(N log K) without an all-pairs allocation.
        blocked = (np.searchsorted(candidate_starts, query_starts+gap, side='right')
                   - np.searchsorted(candidate_starts, query_starts-gap, side='left'))
        if np.any(count-blocked < 2):
            raise ValueError('Transition v2 needs at least two nonoverlapping training candidates '
                             'per query; increase training windows/bank size or reduce the history/horizon')
        self.origins = self.starts+span-1
        self.times = np.asarray(data['times'][:prefix]).astype('datetime64[ns]').astype(np.int64).copy()
        expected_dt = a_config.step_hours*3600*10**9
        if (len(self.times) != prefix or not np.all(np.diff(self.times) == expected_dt)):
            raise ValueError('Transition bank requires fixed physical time steps')
        raw = np.asarray(data['states'][:prefix], dtype=np.float32)
        mean, scale = np.asarray(data['mean']), np.asarray(data['scale'])
        if (raw.shape != (prefix, a_config.state_dim) or mean.shape != (a_config.state_dim,)
                or scale.shape != mean.shape or not np.isfinite(raw).all()
                or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0)):
            raise ValueError('Invalid training fields or training normalization in transition bank')
        self.states = torch.as_tensor(((raw-mean)/scale).astype(np.float32).copy())
        self.information_contract = resolve_dynamics_information(
            data.get('information_metadata'), self.config['information_mode'], a_config.grid)
        self.information = None
        if self.information_contract is not None:
            source = data.get('information')
            if source is None:
                raise ValueError('Transition geopotential_terrain requires normalized information archive')
            chunks = [select_dynamics_information(
                source[first:min(first+256, prefix)], data['information_metadata'],
                self.config['information_mode'], raw_grid=a_config.grid)
                for first in range(0, prefix, 256)]
            selected = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
            self.information = torch.from_numpy(selected.copy())
            cells = math.prod(a_config.grid[1:])
            terrain = self.information[:, 2*cells:]
            if not torch.equal(terrain, terrain[:1].expand_as(terrain)):
                raise ValueError('Dynamics terrain must be constant across training timestamps')
        channels, height, width = a_config.grid
        self.pool_lat, self.pool_lon = min(self.config['pool_lat'], height), min(self.config['pool_lon'], width)
        area = torch.as_tensor(area_weights(data['schema']), dtype=torch.float32)
        if area.shape != (height, width):
            raise ValueError('Transition area/grid mismatch')
        self._cells, coarse_area = [], []
        lat_edges = np.linspace(0, height, self.pool_lat+1, dtype=int)
        lon_edges = np.linspace(0, width, self.pool_lon+1, dtype=int)
        for y in range(self.pool_lat):
            for x in range(self.pool_lon):
                sy, sx = slice(lat_edges[y], lat_edges[y+1]), slice(lon_edges[x], lon_edges[x+1])
                weights = area[sy, sx]
                self._cells.append((sy, sx, weights/weights.sum()))
                coarse_area.append(weights.sum())
        # Weighted Euclidean norm equals equal-channel, coarse-area MSE.
        self._metric_sqrt = torch.stack(coarse_area).div(channels).sqrt()
        pooled = torch.cat([self.describe(self.states[i:i+512]) for i in range(0, prefix, 512)])
        if self.information is not None:
            z_pooled = torch.cat([self.describe_geopotential(self.information[i:i+512])
                                  for i in range(0, prefix, 512)])
            # Equal field weights over the raw channels and two geopotential
            # heights. Static terrain conditions E_d, never a temporal target.
            pooled = torch.cat((pooled*math.sqrt(channels/(channels+2)),
                                z_pooled*math.sqrt(2/(channels+2))), dim=-1)
        self.current_descriptors = pooled[self.origins]
        history_indices = self.starts[:, None]+torch.arange(a_config.history_steps)[None]*a_config.history_stride
        future_steps = self.config['forecast_steps']
        future_indices = self.origins[:, None]+torch.arange(1, future_steps+1)[None]
        self.history_descriptors = pooled[history_indices]-self.current_descriptors[:, None]
        days = torch.arange(1, future_steps+1, dtype=torch.float32)*a_config.step_hours/24.
        self.future_descriptors = (pooled[future_indices]-self.current_descriptors[:, None])/days[None, :, None]
        self.manifest = {
            'version': _BANK_VERSION, 'train_only': True, 'count': count,
            'starts': self.starts.tolist(), 'starts_sha256': _indices_hash(self.starts.tolist()),
            'training_starts_count': len(training_starts), 'training_starts_sha256': _indices_hash(training_starts),
            'train_prefix_end': prefix, 'history_span_steps': span, 'horizon_steps': horizon, 'future_steps': future_steps,
            'step_hours': a_config.step_hours, 'exclusion_steps': self.config['exclusion_steps'],
            'train_first_time_ns': int(self.times[0]), 'train_last_time_ns': int(self.times[-1]),
            'effective_pool_lat': self.pool_lat, 'effective_pool_lon': self.pool_lon,
            'train_states_sha256': hashlib.sha256(np.ascontiguousarray(raw).view(np.uint8)).hexdigest(),
            'normalization_sha256': hashlib.sha256(mean.tobytes()+scale.tobytes()).hexdigest(),
            'transition_config_sha256': hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest(),
            'information_mode': self.config['information_mode'],
            'information_fields': self.config['information_fields'],
            'train_information_sha256': (hashlib.sha256(self.information.numpy().view(np.uint8)).hexdigest()
                                         if self.information is not None else None),
        }
        validate_transition_bank_manifest(self.manifest, self.config)

    def describe(self, fields):
        """Detached CPU pooled fields, preserving coarse position and channels."""
        fields = torch.as_tensor(fields).detach().to(device='cpu', dtype=torch.float32)
        if fields.shape[-1] != self.a_config.state_dim or not torch.isfinite(fields).all():
            raise ValueError('Transition descriptors require finite normalized fields')
        prefix = fields.shape[:-1]
        grid = fields.reshape(-1, *self.a_config.grid)
        coarse = torch.stack([(grid[..., y, x]*weights).sum((-2, -1))
                              for y, x, weights in self._cells], dim=-1)
        return (coarse*self._metric_sqrt).reshape(*prefix, -1)

    def describe_geopotential(self, information):
        information = torch.as_tensor(information).detach().to(device='cpu', dtype=torch.float32)
        height, width = self.a_config.grid[1:]
        if information.shape[-1] != 3*height*width or not torch.isfinite(information[..., :2*height*width]).all():
            raise ValueError('Transition dynamics information must contain finite z500,z850,terrain')
        prefix = information.shape[:-1]
        z = information.reshape(-1, 3, height, width)[:, :2]
        coarse = torch.stack([(z[..., y, x]*weights).sum((-2, -1))
                              for y, x, weights in self._cells], dim=-1)
        metric = self._metric_sqrt*math.sqrt(self.a_config.grid[0]/2)
        return (coarse*metric).reshape(*prefix, -1)

    def _query_descriptors(self, batch, *, future=False):
        values = batch['targets'][:, :self.config['forecast_steps']] if future else batch['history']
        raw = self.describe(values)
        if self.information is None:
            return raw
        key = 'dynamics_information_targets' if future else 'dynamics_information_history'
        info = batch.get(key)
        expected_time = self.config['forecast_steps'] if future else self.a_config.history_steps
        if (not isinstance(info, torch.Tensor) or info.ndim != 3 or info.shape[0] != raw.shape[0]
                or info.shape[1] < expected_time or not future and info.shape[1] != expected_time):
            raise ValueError('Transition requires matching observed/future dynamics information: '+key)
        z = self.describe_geopotential(info[:, :expected_time])
        channels = self.a_config.grid[0]
        return torch.cat((raw*math.sqrt(channels/(channels+2)), z*math.sqrt(2/(channels+2))), dim=-1)

    def _query_starts(self, batch):
        times = batch.get('origin_time_ns')
        if not isinstance(times, torch.Tensor) or times.ndim != 1 or not len(times):
            raise ValueError('Transition queries require origin_time_ns for every history')
        times = times.detach().cpu().numpy()
        if not np.issubdtype(times.dtype, np.integer):
            raise ValueError('Transition origin timestamps must be integer nanoseconds')
        origins = np.searchsorted(self.times, times)
        if np.any(origins >= len(self.times)) or not np.array_equal(self.times[origins], times):
            raise ValueError('Transition queries must be inside training timestamps')
        starts = origins-self.a_config.history_span_steps+1
        if not set(starts.tolist()) <= self.training_starts:
            raise ValueError('Transition queries must belong to actual training loader starts')
        return torch.as_tensor(starts, dtype=torch.long)

    def retrieve(self, batch):
        """Current-only exact top-K; no future targets are inspected here."""
        starts = self._query_starts(batch)
        history = batch.get('history')
        expected = (len(starts), self.a_config.history_steps, self.a_config.state_dim)
        if not isinstance(history, torch.Tensor) or tuple(history.shape) != expected:
            raise ValueError('Transition history does not match configured observed windows')
        current = self._query_descriptors(batch)[:, -1]
        # Inclusive whole-window intervals; a positive exclusion expands the gap.
        forbidden_gap = self.window_length-1+self.config['exclusion_steps']
        eligible = (starts[:, None]-self.starts[None]).abs() > forbidden_gap
        eligible_counts = eligible.sum(-1)
        count = min(self.config['neighbors'], int(eligible_counts.max()))
        if int(eligible_counts.min()) < 2:
            raise ValueError('Transition v2 needs at least two nonoverlapping training candidates '
                             'per query; increase training windows/bank size or reduce the history/horizon')
        best_d = torch.empty(len(starts), 0)
        best_i = torch.empty(len(starts), 0, dtype=torch.long)
        for first in range(0, len(self.starts), 512):
            stop = min(first+512, len(self.starts))
            distance = (current[:, None]-self.current_descriptors[None, first:stop]).square().sum(-1)
            distance = distance.masked_fill(~eligible[:, first:stop], float('inf'))
            indices = torch.arange(first, stop)[None].expand(len(starts), -1)
            all_d, all_i = torch.cat((best_d, distance), 1), torch.cat((best_i, indices), 1)
            order = torch.argsort(all_d, dim=1, stable=True)[:, :count]
            best_d, best_i = all_d.gather(1, order), all_i.gather(1, order)
        log_prior = torch.log_softmax(-best_d/self.config['endpoint_temperature'], dim=-1)
        valid = torch.isfinite(best_d)
        # Duplicate the first valid index only in padding; it has zero mass and
        # is excluded from diagnostics. This keeps support queries train-safe.
        best_i = torch.where(valid, best_i, best_i[:, :1])
        selected = self.starts[best_i]
        return {'valid_mask': valid, 'indices': selected, 'bank_positions': best_i, 'distances': best_d,
                'prior_weights': log_prior.exp(), 'log_prior_weights': log_prior,
                'query_starts': starts}

    def empirical_measure(self, pipeline, batch):
        """Re-encode every selected history with the current E_d, never a cache."""
        result = self.retrieve(batch)
        positions, starts = result['bank_positions'], result['indices']
        indices = starts[..., None]+torch.arange(self.a_config.history_steps)*self.a_config.history_stride
        histories = self.states[indices]
        parameter = next(pipeline.dynamics_encoder.parameters())
        flat = histories.flatten(0, 1)
        info_flat = None if self.information is None else self.information[indices].flatten(0, 1)
        with torch.no_grad():
            pieces = []
            for first in range(0, len(flat), 64):
                inputs = flat[first:first+64].to(parameter)
                encoded = (pipeline.dynamics_encoder(inputs) if info_flat is None else
                           pipeline.dynamics_encoder(inputs, info_flat[first:first+64].to(parameter)))
                pieces.append(encoded[:, -1])
            support = torch.cat(pieces).detach()
        support = support.reshape(*starts.shape, -1)
        if not torch.isfinite(support).all():
            raise FloatingPointError('Nonfinite transition support latent')
        descriptors = self._query_descriptors(batch)
        current = descriptors[:, -1]
        query_history = descriptors-current[:, None]
        targets = batch.get('targets')
        horizon = self.config['forecast_steps']
        if (not isinstance(targets, torch.Tensor) or targets.ndim != 3
                or targets.shape[0] != len(starts) or targets.shape[1] < horizon
                or targets.shape[2] != self.a_config.state_dim):
            raise ValueError('Transition supervision needs the complete configured future label window')
        days = torch.arange(1, horizon+1, dtype=torch.float32)*self.a_config.step_hours/24.
        query_future = (self._query_descriptors(batch, future=True)-current[:, None])/days[None, :, None]
        history_distance = (query_history[:, None]-self.history_descriptors[positions]).square().sum(-1).mean(-1)
        future_distance = (query_future[:, None]-self.future_descriptors[positions]).square().sum(-1).mean(-1)
        result.update(support_latent=support, history_distance=history_distance,
                      future_distance=future_distance)
        return result


def transition_measure_losses(pipeline, prediction, batch, bank, config):
    """Match distributions over distinct histories, never their latent mean."""
    config = validate_transition_config(config)
    if bank.config != config:
        raise ValueError('Transition loss configuration disagrees with its bank')
    latent = prediction.get('dynamics_history_latent')
    if (pipeline.dynamics_encoder is None or not isinstance(latent, torch.Tensor)
            or latent.ndim != 3 or latent.shape[1] != pipeline.a_config.history_steps
            or not torch.isfinite(latent).all()):
        raise ValueError('Transition loss requires finite shared E_d history features')
    measure = bank.empirical_measure(pipeline, batch)
    support = measure['support_latent'].to(latent).detach()
    query = latent[:, -1]
    if support.shape[0] != len(query) or support.shape[-1] != query.shape[-1]:
        raise ValueError('Transition support and query latent shapes disagree')
    distance = (query[:, None]-support).square().mean(-1)
    physical = (config['history_weight']*measure['history_distance']
                + config['future_weight']*measure['future_distance']).to(latent).detach()
    log_prior = measure['log_prior_weights'].to(latent).detach()
    log_target = torch.log_softmax(log_prior-physical/config['relation_temperature'], dim=-1).detach()
    log_student = torch.log_softmax(log_prior-distance/config['latent_temperature'], dim=-1)
    target, prior = log_target.exp(), log_prior.exp()
    valid = measure['valid_mask'].to(device=latent.device)
    safe_log_target = log_target.masked_fill(~valid, 0.)
    safe_log_student = log_student.masked_fill(~valid, 0.)
    kl = (target*(safe_log_target-safe_log_student)).sum(-1).mean()
    geometry = (prior*(distance/(1+distance)-physical/(1+physical)).square()).sum(-1).mean()
    result = {
        'transition_kl': kl, 'transition_geometry': geometry,
        'transition_regularization': config['weight']*(kl+config['geometry_weight']*geometry),
        'transition_effective_neighbors': (1/prior.square().sum(-1)).mean().detach(),
        'transition_latent_variance': query.detach().var(0, unbiased=False).mean(),
        'transition_support_variance': (prior[..., None]*(support-(prior[..., None]*support).sum(1, keepdim=True)).square()).sum(1).mean(),
        'transition_target_entropy': (-(target*safe_log_target).sum(-1)).mean(),
        'transition_neighbors': valid.sum(-1).to(latent).mean(),
    }
    if any(not torch.isfinite(value).all() for value in result.values()):
        raise FloatingPointError('Nonfinite transition measure objective')
    return result
