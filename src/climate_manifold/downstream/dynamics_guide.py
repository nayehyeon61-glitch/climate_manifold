"""Independent causal dynamics representation for latent and guided forecasts.

Only observed normalized raw fields enter ``DynamicsGuide``.  Its finite
differences are normalized-field changes per physical day, not derivatives in
SI units and not an imposed PDE.  The learned dynamics state at each observed
time depends only on the history up to that time.  Future labels supervise the
separate auxiliary head; they never enter this encoder or a forecast's input.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..spatial import GeographicConv2d
from .guided_transformer import GuidedTransformerPredictor, _positive_grid


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def _history(value, dimension, name, steps=None):
    if (value.ndim != 3 or not value.shape[0] or not value.shape[1]
            or value.shape[-1] != dimension
            or (steps is not None and value.shape[1] != steps)
            or not torch.isfinite(value).all()):
        raise ValueError(f'{name} must be a finite configured (batch, time, dimension) sequence')


class _ConvGRUCell(nn.Module):
    def __init__(self, hidden, periodic_lon):
        super().__init__()
        self.gates = GeographicConv2d(2 * hidden, 2 * hidden, periodic_lon=periodic_lon)
        self.candidate = GeographicConv2d(2 * hidden, hidden, periodic_lon=periodic_lon)

    def forward(self, inputs, previous):
        reset, update = self.gates(torch.cat((inputs, previous), dim=1)).sigmoid().chunk(2, dim=1)
        candidate = self.candidate(torch.cat((inputs, reset * previous), dim=1)).tanh()
        return (1. - update) * previous + update * candidate


class DynamicsGuide(nn.Module):
    """E_d(raw history) -> a causal spatial sequence, independent of E_s.

    ``conv_gru`` accumulates a recurrent spatial memory. ``temporal_conv`` uses
    left-padded dilated temporal convolutions at each pooled location after a
    spatial stem. Its receptive field covers the configured observed history.
    Neither variant shares parameters with the statistical encoder.
    """
    implementation_variant = 'causal_observed_dynamics_guide_v1'

    def __init__(self, raw_grid, dynamics_grid, history_steps, hidden=32,
                 kind='conv_gru', history_dt_hours=24., periodic_lon=False):
        super().__init__()
        self.raw_grid = _positive_grid(raw_grid, 'Raw grid')
        self.dynamics_grid = _positive_grid(dynamics_grid, 'Dynamics grid')
        _positive_integer(history_steps, 'History length')
        _positive_integer(hidden, 'Hidden width')
        if kind not in ('conv_gru', 'temporal_conv'):
            raise ValueError('Dynamics kind must be conv_gru or temporal_conv')
        if (isinstance(history_dt_hours, bool) or not math.isfinite(history_dt_hours)
                or history_dt_hours <= 0):
            raise ValueError('History spacing must be finite and positive')
        self.history_steps, self.hidden, self.kind = history_steps, hidden, kind
        self.history_dt_hours, self.periodic_lon = float(history_dt_hours), bool(periodic_lon)
        self.dimension = math.prod(self.raw_grid)
        self.dynamics_dimension = math.prod(self.dynamics_grid)
        self.stem = nn.Sequential(
            GeographicConv2d(2 * self.raw_grid[0] + 1, hidden, periodic_lon=periodic_lon),
            nn.SiLU(),
        )
        if kind == 'conv_gru':
            self.cell = _ConvGRUCell(hidden, periodic_lon)
        else:
            # Receptive field = 1 + 2 * (1 + 2 + ... + 2**(n-1)).
            layers = max(1, math.ceil(math.log2((history_steps + 1) / 2)))
            self.temporal_layers = nn.ModuleList([
                nn.Conv1d(hidden, hidden, kernel_size=3, dilation=2 ** i)
                for i in range(layers)
            ])
        self.readout = nn.Conv2d(hidden, self.dynamics_grid[0], kernel_size=1)

    def forward(self, history):
        _history(history, self.dimension, 'Raw history', self.history_steps)
        batch, steps = history.shape[:2]
        height, width = self.dynamics_grid[1:]
        fields = history.reshape(batch * steps, *self.raw_grid)
        pooled = F.adaptive_avg_pool2d(fields, (height, width)).reshape(
            batch, steps, self.raw_grid[0], height, width)
        # Pooling and differences commute. The first observed frame has no
        # predecessor in this window, so its derivative is explicitly zero.
        days = self.history_dt_hours / 24.
        changes = torch.cat((torch.zeros_like(pooled[:, :1]),
                             (pooled[:, 1:] - pooled[:, :-1]) / days), dim=1)
        interval = pooled.new_full((batch, steps, 1, height, width), days)
        embedded = self.stem(torch.cat((pooled, changes, interval), dim=2).reshape(
            batch * steps, 2 * self.raw_grid[0] + 1, height, width))
        embedded = embedded.reshape(batch, steps, self.hidden, height, width)
        if self.kind == 'conv_gru':
            state = embedded.new_zeros(batch, self.hidden, height, width)
            states = []
            for frame in embedded.unbind(1):
                state = self.cell(frame, state)
                states.append(state)
            temporal = torch.stack(states, dim=1)
        else:
            temporal = embedded.permute(0, 3, 4, 2, 1).reshape(
                batch * height * width, self.hidden, steps)
            for layer in self.temporal_layers:
                # No right padding, bidirectional layers or time normalization.
                temporal = temporal + F.silu(layer(F.pad(temporal, (2 * layer.dilation[0], 0))))
            temporal = temporal.reshape(batch, height, width, self.hidden, steps).permute(0, 4, 3, 1, 2)
        latent = self.readout(temporal.reshape(batch * steps, self.hidden, height, width))
        return latent.reshape(batch, steps, self.dynamics_dimension)


class LatentDynamicsAdapter(nn.Module):
    """Add a dynamics residual to the statistical latent, preserving its grid.

    The zero control has exactly the same projection parameters but no
    input-dependent dynamics content. The projection starts at zero, keeping
    the original latent predictor's input unchanged at initialization.
    """
    def __init__(self, spatial_grid, dynamics_grid):
        super().__init__()
        self.spatial_grid = _positive_grid(spatial_grid, 'Spatial grid')
        self.dynamics_grid = _positive_grid(dynamics_grid, 'Dynamics grid')
        self.projection = nn.Conv2d(self.dynamics_grid[0], self.spatial_grid[0], kernel_size=1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, spatial_history, dynamics_history, mode='learned'):
        _history(spatial_history, math.prod(self.spatial_grid), 'Spatial history')
        if mode not in ('learned', 'zero', 'only'):
            raise ValueError('Dynamics mode must be learned, zero or only')
        batch, steps = spatial_history.shape[:2]
        shape = (batch, steps, math.prod(self.dynamics_grid))
        if mode == 'zero':
            dynamics_history = spatial_history.new_zeros(shape)
        elif (dynamics_history is None or dynamics_history.shape != shape
              or not torch.isfinite(dynamics_history).all()):
            raise ValueError('Learned dynamics requires the configured finite dynamics history')
        correction = self.projection(dynamics_history.reshape(batch * steps, *self.dynamics_grid))
        if correction.shape[-2:] != self.spatial_grid[1:]:
            correction = F.interpolate(correction, size=self.spatial_grid[1:],
                                       mode='bilinear', align_corners=False)
        correction = correction.reshape(batch, steps, -1)
        # only: P(E_d) replaces the E_s latent as the forecast model input.
        return correction if mode == 'only' else spatial_history + correction


class DynamicsAuxiliaryHead(nn.Module):
    """Decode future normalized raw fields solely from the last E_d context.

    This training head is separate from the downstream forecast model. Positive
    physical lead times condition its output; future states are labels only.
    A nonzero random readout lets auxiliary supervision reach E_d immediately,
    including while the main route's residual projection is still zero.
    """
    def __init__(self, dynamics_grid, raw_grid, hidden=32, periodic_lon=False):
        super().__init__()
        self.dynamics_grid = _positive_grid(dynamics_grid, 'Dynamics grid')
        self.raw_grid = _positive_grid(raw_grid, 'Raw grid')
        _positive_integer(hidden, 'Hidden width')
        self.hidden = hidden
        self.stem = GeographicConv2d(self.dynamics_grid[0], hidden, periodic_lon=periodic_lon)
        self.lead_embed = nn.Sequential(nn.Linear(4, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.head = nn.Sequential(
            nn.SiLU(), GeographicConv2d(hidden, hidden, periodic_lon=periodic_lon),
            nn.SiLU(), nn.Conv2d(hidden, self.raw_grid[0], kernel_size=1),
        )

    def forward(self, dynamics_history, lead_hours):
        _history(dynamics_history, math.prod(self.dynamics_grid), 'Dynamics history')
        if (lead_hours.ndim != 1 or not len(lead_hours) or not torch.isfinite(lead_hours).all()
                or lead_hours[0] <= 0 or not (lead_hours[1:] > lead_hours[:-1]).all()):
            raise ValueError('Lead hours must be finite, positive and strictly increasing')
        batch, leads = len(dynamics_history), len(lead_hours)
        context = self.stem(dynamics_history[:, -1].reshape(batch, *self.dynamics_grid))
        days = lead_hours.to(dynamics_history) / 24.
        lead = self.lead_embed(GuidedTransformerPredictor._time_features(days))
        query = context[:, None] + lead[None, :, :, None, None]
        fields = self.head(query.reshape(batch * leads, self.hidden, *self.dynamics_grid[1:]))
        if fields.shape[-2:] != self.raw_grid[1:]:
            fields = F.interpolate(fields, size=self.raw_grid[1:], mode='bilinear', align_corners=False)
        return fields.reshape(batch, leads, math.prod(self.raw_grid))
