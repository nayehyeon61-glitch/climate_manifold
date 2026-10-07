"""Post-forecast auxiliary F_d audits, separate from downstream model M scores."""
from dataclasses import replace
import math

import numpy as np
import torch

from ..archive import field_grid
from ..temporal_supervision import area_weights
from .statistical_objective import make_statistical_config, spatial_signed_measure


def dynamics_evaluation_windows(data, config, split, rollout_steps, stride=1, max_cases=0,
                                *, trained_horizon_steps=None):
    """Preserve checkpoint splits and restrict extended labels to that split's coverage.

    Selection precedes stride/count truncation so independently trained field and
    field+latent arms receive exactly the same origins. Never recompute the data
    contract with an extended horizon: that would change split/normalization.
    """
    minimum = config.horizon_steps if trained_horizon_steps is None else trained_horizon_steps
    if (isinstance(rollout_steps, bool) or not isinstance(rollout_steps, int)
            or rollout_steps < minimum):
        raise ValueError('Dynamics rollout steps must be an integer >= checkpoint horizon')
    starts = np.asarray(data['split'][split], dtype=np.int64)
    if not len(starts):
        raise ValueError('Dynamics evaluation requires nonempty checkpoint split')
    original_last = int(starts.max()) + config.history_span_steps + config.horizon_steps - 1
    last_allowed = min(original_last, len(data['states']) - 1)
    valid = starts[starts + config.history_span_steps + rollout_steps - 1 <= last_allowed]
    selected = valid[::stride]
    if max_cases:
        selected = selected[:max_cases]
    if not len(selected):
        raise ValueError('Dynamics rollout horizon leaves no origins inside the checkpoint split label coverage')
    from .train import windows
    selected_data = {**data, 'split': {**data['split'], split: selected.tolist()}}
    extended = replace(config, horizon_steps=rollout_steps)
    ds = windows(selected_data, extended, split, dynamics_target_sequence=True)
    def label(index):
        return str(np.datetime64(data['times'][int(index)], 'ns')) + 'Z'
    metadata = {
        'original_split_origin_count': int(len(starts)),
        'horizon_eligible_origin_count': int(len(valid)),
        'excluded_for_rollout_coverage_count': int(len(starts) - len(valid)),
        'selected_origin_count': int(len(selected)), 'origin_stride': stride, 'max_cases': max_cases,
        'original_split_last_label_index': original_last,
        'original_data_contract_horizon_steps': config.horizon_steps,
        'last_allowed_label_index': last_allowed,
        'last_allowed_label_time': label(last_allowed),
        'selected_first_origin_time': label(selected[0] + config.history_span_steps - 1),
        'selected_last_origin_time': label(selected[-1] + config.history_span_steps - 1),
        'selection_order': 'checkpoint_split_then_rollout_coverage_then_stride_then_max_cases',
        'normalization_and_split': 'unchanged_checkpoint_data_contract',
    }
    return ds, metadata


class SignedFieldDiagnostics:
    """Signed spatial JS and signed mass MSE per variable and lead.

    Fields are normalized using the checkpoint's train-only per-cell reference;
    positive/negative do not mean a threshold at zero physical pressure.
    """
    def __init__(self, schema, lead_hours):
        self.grid = field_grid(schema)
        self.names = [v['name'] for v in schema['variables']]
        self.leads = list(map(float, lead_hours))
        self.area = torch.as_tensor(area_weights(schema), dtype=torch.float64).flatten()
        self.config = make_statistical_config('signed_measure')
        self.count = 0
        self.sums = {key: torch.zeros(len(self.leads), len(self.names), dtype=torch.float64)
                     for key in ('signed_spatial_js', 'signed_mass_mse')}

    def update(self, predicted, target):
        if (predicted.shape != target.shape or predicted.ndim != 3
                or predicted.shape[1] != len(self.leads)
                or not torch.isfinite(predicted).all() or not torch.isfinite(target).all()):
            raise FloatingPointError('Invalid or nonfinite signed-field diagnostic')
        shape = (*predicted.shape[:2], self.grid[0], self.grid[1] * self.grid[2])
        predicted = predicted.detach().cpu().double().reshape(shape)
        target = target.detach().cpu().double().reshape(shape)
        for lead in range(len(self.leads)):
            for variable in range(len(self.names)):
                scores = spatial_signed_measure(predicted[:, lead, variable], target[:, lead, variable],
                                               self.area, self.config)
                for key in self.sums:
                    self.sums[key][lead, variable] += scores[key] * len(predicted)
        self.count += len(predicted)

    def result(self):
        if not self.count:
            return {'case_count': 0, 'aggregate': None, 'per_variable': {}}
        scores = {key: value / self.count for key, value in self.sums.items()}
        return {
            'case_count': self.count,
            'aggregate': {key: float(value.mean()) for key, value in scores.items()},
            'per_variable': {name: {
                'aggregate': {key: float(value[:, c].mean()) for key, value in scores.items()},
                'by_lead': [dict(lead_hours=lead, **{key: float(value[j, c])
                    for key, value in scores.items()}) for j, lead in enumerate(self.leads)]}
                for c, name in enumerate(self.names)},
            'reference': 'checkpoint_training_normalized_zero_per_grid_cell_not_zero_physical_pressure',
            'definition': 'equal_average_of_positive_and_negative_area_weighted_spatial_JS_and_total_mass_MSE',
        }


class DynamicsLatentDiagnostics:
    """Coordinate errors and collapse checks, accumulating variance across origins.

    Variance is first calculated across origins separately for every fixed lead
    and coordinate, then averaged over coordinates. Spatial/temporal variance
    within a single example is never substituted for variation across examples.
    """
    def __init__(self, lead_hours, variance_floor=1e-8):
        self.leads = list(map(float, lead_hours))
        self.variance_floor = float(variance_floor)
        self.count = 0
        self.sums = {}
        self.moments = {}

    def update(self, predicted, target, origin):
        if (predicted.shape != target.shape or predicted.ndim != 3
                or predicted.shape[1] != len(self.leads)
                or origin.shape != (predicted.shape[0], predicted.shape[-1])
                or not all(torch.isfinite(x).all() for x in (predicted, target, origin))):
            raise FloatingPointError('Invalid or nonfinite dynamics latent diagnostic')
        predicted, target, origin = [x.detach().cpu().double() for x in (predicted, target, origin)]
        values = {'latent_mse': (predicted-target).square().mean(-1),
                  'latent_persistence_mse': (origin[:, None]-target).square().mean(-1),
                  'predicted_rms2': predicted.square().mean(-1),
                  'target_rms2': target.square().mean(-1)}
        for key, value in values.items():
            self.sums[key] = self.sums.get(key, torch.zeros_like(value[0])) + value.sum(0)
        n = len(predicted)
        for name, value in (('target', target), ('predicted', predicted)):
            mean = value.mean(0)
            m2 = (value-mean).square().sum(0)
            if name not in self.moments:
                self.moments[name] = (mean, m2)
            else:
                old_mean, old_m2 = self.moments[name]
                delta = mean-old_mean
                total = self.count+n
                self.moments[name] = (old_mean+delta*n/total,
                                     old_m2+m2+delta.square()*self.count*n/total)
        self.count += n

    def result(self):
        if not self.count:
            return {'case_count': 0, 'aggregate': None, 'by_lead': []}
        scores = {key: value/self.count for key, value in self.sums.items()}
        variance = {name: m2.mean(-1)/self.count for name, (_, m2) in self.moments.items()}
        def summarize(index=None):
            get = lambda x: float(x.mean() if index is None else x[index])
            target_var, pred_var = get(variance['target']), get(variance['predicted'])
            mse, persistence = get(scores['latent_mse']), get(scores['latent_persistence_mse'])
            return {
                'latent_rmse': math.sqrt(mse), 'latent_persistence_rmse': math.sqrt(persistence),
                'normalized_latent_rmse': math.sqrt(mse/max(target_var, self.variance_floor)),
                'normalized_latent_persistence_rmse': math.sqrt(persistence/max(target_var, self.variance_floor)),
                'target_across_origin_coordinate_variance': target_var,
                'predicted_across_origin_coordinate_variance': pred_var,
                'target_latent_rms': math.sqrt(get(scores['target_rms2'])),
                'predicted_latent_rms': math.sqrt(get(scores['predicted_rms2'])),
                'variance_floor_active': target_var < self.variance_floor,
            }
        return {'case_count': self.count, 'aggregate': summarize(),
                'by_lead': [dict(lead_hours=lead, **summarize(j)) for j, lead in enumerate(self.leads)],
                'target_variance_floor': self.variance_floor,
                'variance_definition': 'population_variance_across_origins_per_fixed_lead_and_coordinate_then_mean_coordinates',
                'normalization': 'RMSE_divided_by_sqrt_max_target_across_origin_coordinate_variance_and_floor',
                'limits': 'Encoder-specific coordinate audits cannot rank different encoders or establish physical similarity. '
                          'A small latent error with collapsed target variance is not evidence of useful dynamics.'}
