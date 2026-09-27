"""Time-conditioned ConvLSTM forecaster on the raw or learned spatial grid.

The sigmoid/tanh convolutional gates implement the standard ConvLSTM equations
from Shi et al., NeurIPS 2015, https://arxiv.org/abs/1506.04214. OpenSTL provides
another implementation at https://github.com/chengtan9907/OpenSTL (see
``openstl/modules/convlstm_modules.py``). This is an independent implementation,
not an imported pretrained model or reproduction of OpenSTL's training recipe.

Our adaptation keeps the geographic padding and flattened bridge contract,
conditions on actual elapsed time, and predicts residuals in field units/day.
Every observed frame is consumed once. Forecast states are generated from the
memory and then fed back to that same cell; no future target is an input.
"""
from __future__ import annotations

import math
from numbers import Integral

import torch
from torch import nn

from ..spatial import GeographicConv2d
from .baselines import calendar_features
from .spatial_baselines import origin_information


class ConvLSTMCell(nn.Module):
    """Single convolutional memory cell, without peephole connections."""

    def __init__(self, in_channels, hidden_channels, *, periodic_lon=False):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = GeographicConv2d(in_channels + hidden_channels,
                                      4 * hidden_channels,
                                      periodic_lon=periodic_lon)

    def forward(self, inputs, hidden, memory):
        input_gate, forget_gate, candidate, output_gate = self.gates(
            torch.cat((inputs, hidden), dim=1)
        ).chunk(4, dim=1)
        memory = forget_gate.sigmoid() * memory + input_gate.sigmoid() * candidate.tanh()
        hidden = output_gate.sigmoid() * memory.tanh()
        return hidden, memory


class ConvLSTMPredictor(nn.Module):
    """Shared ConvLSTM for history assimilation and autoregressive forecasts.

    ``history_dt_hours`` describes the *observed* sequence spacing; future
    intervals are taken from successive ``lead_hours``. Both relative time and
    interval length enter learned layers, so 24-hour history and 6-hour forecasts
    are never treated as equal recurrent steps. This is a discrete recurrent
    model: adding intermediate query times changes its numerical trajectory.
    Asking for the same lead prefix, however, gives exactly the same prefix.
    """

    upstream_source = 'https://github.com/chengtan9907/OpenSTL'
    upstream_commit = 'eecf8a3078f0a178dbc7b28723da20f94ce36985'
    implementation_variant = 'convlstm_time_conditioned_adaptation_v1'
    # These provenance identifiers record the audited comparison implementation;
    # the standard gates below were written independently, not copied/vendored.

    def __init__(self, latent_grid, history_steps, hidden=128,
                 history_dt_hours=24., periodic_lon=False, information_channels=0):
        super().__init__()
        self.latent_grid = tuple(latent_grid)
        if (len(self.latent_grid) != 3
                or any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1
                       for v in self.latent_grid)):
            raise ValueError('ConvLSTM requires a positive integer (channels, latitude, longitude) grid')
        for name, value in (('history_steps', history_steps), ('hidden', hidden)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if (isinstance(history_dt_hours, bool)
                or not math.isfinite(history_dt_hours) or history_dt_hours <= 0):
            raise ValueError('History spacing must be finite and positive')
        if (isinstance(information_channels, bool)
                or not isinstance(information_channels, Integral) or information_channels < 0):
            raise ValueError('Information channels must be a nonnegative integer')
        self.history_steps = history_steps
        self.history_dt_hours = float(history_dt_hours)
        self.dimension = math.prod(self.latent_grid)
        self.hidden_channels = hidden
        self.information_channels = information_channels
        self.information_dim = information_channels * math.prod(self.latent_grid[1:])
        self.periodic_lon = bool(periodic_lon)
        channels = self.latent_grid[0]
        # Calendar (four channels), origin-relative days and elapsed days.
        self.cell = ConvLSTMCell(channels + information_channels + 6, hidden,
                                periodic_lon=self.periodic_lon)
        self.readout = nn.Sequential(
            GeographicConv2d(hidden + 6, hidden, periodic_lon=self.periodic_lon),
            nn.SiLU(), nn.Conv2d(hidden, channels, kernel_size=1),
        )

    def _time_fields(self, origin_ns, history, relative_hours, dt_hours):
        height, width = self.latent_grid[1:]
        # Calendar covariates at the queried time are deterministic, not future
        # weather observations. float64 preserves sub-day precision of ns dates.
        time_ns = origin_ns.to(torch.float64) + relative_hours * 3.6e12
        calendar = calendar_features(time_ns, history)
        relative = history.new_full((len(history), 1), relative_hours / 24.)
        interval = history.new_full((len(history), 1), dt_hours / 24.)
        return torch.cat((calendar, relative, interval), dim=1)[:, :, None, None].expand(
            -1, -1, height, width
        )

    def forward(self, history, lead_hours, origin_ns, information=None):
        if (history.ndim != 3 or history.shape[0] < 1
                or history.shape[1:] != (self.history_steps, self.dimension)
                or not history.is_floating_point() or not torch.isfinite(history).all()):
            raise ValueError('Finite floating history must match the configured ConvLSTM grid and history length')
        if origin_ns.shape != (len(history),) or not torch.isfinite(origin_ns).all():
            raise ValueError('One finite origin timestamp is required per history')
        if (lead_hours.ndim != 1 or not len(lead_hours)
                or not torch.isfinite(lead_hours).all() or lead_hours[0] <= 0
                or not (lead_hours[1:] > lead_hours[:-1]).all()):
            raise ValueError('Lead hours must be finite, positive and strictly increasing')
        batch = len(history)
        channels, height, width = self.latent_grid
        observed = history.reshape(batch, self.history_steps, channels, height, width)
        information = origin_information(information, history, self.information_channels,
                                         (height, width))
        hidden = history.new_zeros(batch, self.hidden_channels, height, width)
        memory = torch.zeros_like(hidden)

        def consume(state, time):
            inputs = [state, time]
            if information is not None:
                inputs.append(information)
            return self.cell(torch.cat(inputs, dim=1), hidden, memory)

        for index in range(self.history_steps):
            hour = (index - self.history_steps + 1) * self.history_dt_hours
            dt = 0. if index == 0 else self.history_dt_hours
            time = self._time_fields(origin_ns, history, hour, dt)
            hidden, memory = consume(observed[:, index], time)

        state, previous_hour, result = observed[:, -1], 0., []
        hours = lead_hours.tolist()
        for index, hour in enumerate(hours):
            dt = hour - previous_hour
            time = self._time_fields(origin_ns, history, hour, dt)
            rate = self.readout(torch.cat((hidden, time), dim=1))
            state = state + (dt / 24.) * rate
            result.append(state.flatten(1))
            # The observed origin is already represented in hidden/memory. Feed
            # back only the new prediction, and only if another lead is needed.
            if index + 1 < len(hours):
                hidden, memory = consume(state, time)
            previous_hour = hour
        return torch.stack(result, dim=1), None
