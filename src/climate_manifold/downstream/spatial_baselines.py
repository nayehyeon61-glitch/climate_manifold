"""Grid-preserving recurrent predictors for the joint spatial representation.

The flattened outer contract is only a serialization/bridge convention. Every
learned operation retains the latent grid; hidden_dim denotes channels per cell,
never a small global vector bottleneck.
"""
import math

import torch
from torch import nn

from .baselines import calendar_features
from ..spatial import GeographicConv2d


def origin_information(information, history, channels, spatial_shape):
    """Validate observed, origin-only information for a spatial predictor.

    An extra time axis is rejected so callers cannot accidentally provide a
    future information sequence. Latent callers use channels=0: information
    must enter through their encoder instead of an unrecorded bypass.
    """
    if channels == 0:
        if information is not None:
            raise ValueError('Spatial latent predictors receive auxiliary information through the encoder only')
        return None
    expected = (len(history), channels * math.prod(spatial_shape))
    if (information is None or information.shape != expected
            or not torch.isfinite(information).all()):
        raise ValueError('Finite origin information with the configured spatial grid is required')
    return information.reshape(len(history), channels, *spatial_shape)


class SpatialHistoryPredictor(nn.Module):
    def __init__(self, latent_grid, history_steps, hidden=128, kind='mlp', substeps=2,
                 periodic_lon=False, information_channels=0):
        super().__init__()
        if kind not in ('mlp', 'neural_ode') or min(history_steps, hidden, substeps) < 1:
            raise ValueError('Invalid spatial predictor kind, dimensions or ODE substeps')
        self.latent_grid = tuple(latent_grid)
        if len(self.latent_grid) != 3 or min(self.latent_grid) < 1:
            raise ValueError('Spatial predictors require a (channels, latitude, longitude) latent grid')
        if (not isinstance(information_channels, int) or isinstance(information_channels, bool)
                or information_channels < 0):
            raise ValueError('Information channels must be a nonnegative integer')
        self.dimension = math.prod(self.latent_grid)
        self.history_steps = history_steps
        self.kind, self.substeps = kind, substeps
        self.information_channels = information_channels
        self.information_dim = information_channels * math.prod(self.latent_grid[1:])
        self.periodic_lon = bool(periodic_lon)
        channels = self.latent_grid[0]
        # Full observed history becomes a spatial context field. Raw controls
        # may also condition on the same observed information available to E.
        self.context = nn.Sequential(
            GeographicConv2d(history_steps * channels + 4 + information_channels, hidden, periodic_lon=periodic_lon), nn.SiLU(),
            GeographicConv2d(hidden, hidden, periodic_lon=periodic_lon), nn.SiLU(),
        )
        # "mlp" retains the legacy Euler baseline label. Here its learned rate
        # is a spatial convolutional network, not a globally flattened MLP.
        self.field = nn.Sequential(
            GeographicConv2d(channels + hidden + 1, hidden, periodic_lon=periodic_lon), nn.SiLU(),
            GeographicConv2d(hidden, hidden, periodic_lon=periodic_lon), nn.SiLU(),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, history, lead_hours, origin_ns, information=None):
        if (history.ndim != 3 or history.shape[1:] != (self.history_steps, self.dimension)
                or not torch.isfinite(history).all()):
            raise ValueError('History does not match the configured spatial latent grid and history length')
        if origin_ns.shape != (len(history),) or not torch.isfinite(origin_ns).all():
            raise ValueError('One finite origin timestamp is required per history')
        if (lead_hours.ndim != 1 or not len(lead_hours) or not torch.isfinite(lead_hours).all()
                or lead_hours[0] <= 0 or not (lead_hours[1:] > lead_hours[:-1]).all()):
            raise ValueError('Lead hours must be finite, positive and strictly increasing')
        batch = len(history)
        channels, height, width = self.latent_grid
        observed = history.reshape(batch, self.history_steps, channels, height, width)
        calendar = calendar_features(origin_ns, history)[:, :, None, None].expand(-1, -1, height, width)
        information = origin_information(information, history, self.information_channels, (height, width))
        context_inputs = [observed.flatten(1, 2), calendar]
        if information is not None:
            context_inputs.append(information)
        context = self.context(torch.cat(context_inputs, dim=1))

        def rhs(state, time_days):
            time = state.new_full((batch, 1, height, width), float(time_days))
            return self.field(torch.cat((state, context, time), dim=1))

        state, previous, result = observed[:, -1], 0., []
        for hour in lead_hours.tolist():
            final = hour / 24
            if self.kind == 'mlp':
                state = state + (final - previous) * rhs(state, previous)
            else:
                dt = (final - previous) / self.substeps
                for i in range(self.substeps):
                    t = previous + i * dt
                    k1 = rhs(state, t)
                    k2 = rhs(state + dt * k1 / 2, t + dt / 2)
                    k3 = rhs(state + dt * k2 / 2, t + dt / 2)
                    k4 = rhs(state + dt * k3, t + dt)
                    state = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            result.append(state.flatten(1))
            previous = final
        return torch.stack(result, dim=1), None
