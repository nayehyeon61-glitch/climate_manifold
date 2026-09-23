"""ClimODE-style coupled transport dynamics on an explicit spatial grid.

The equation follows ``Climate_encoder_free_uncertain.pde`` in the MIT-licensed
Aalto-QuML/ClimODE (commit e729d23e8799ce0e075699e76d60227d848d8d0c):
``dz/dt = vx*Dx(z) + vy*Dy(z) + z*(Dx(vx) + Dy(vy))``.  We retain its positive
divergence convention and variable-specific learned velocity dynamics.  This is
an adaptation, not the original physical-grid model or its uncertainty head.
The matched raw control uses this same core on normalized observed fields,
with factor=1 and optional observed origin-information context. It has no
manifold encoder or decoder. Conservation in its normalized cell coordinates
does not establish conservation of physical atmospheric quantities either.

Time is in days, derivatives use latent grid-cell coordinates, and velocities
are latent transport coefficients in cells/day, not observed atmospheric winds.
CNN padding and differences wrap longitude only for a verified complete global
grid. Latitude and regional longitude use one-sided conditioning derivatives
and zero normal transport flux at the outer boundary. The transport equation
uses conservative first-order upwind fluxes instead of upstream's centered
product-rule differences; its physical advection direction is -v. This adds
numerical diffusion and boundary assumptions, deliberately favoring stability.
Coordinates are block averages matching the encoder's ceil-mode average pool.
There is no spherical metric or claim of physical mass conservation in z.

Initial velocities are learned differentiably from *observed* latent history.
To control numerical transport speeds we evolve unconstrained velocity r and
use v = vmax*tanh(r), with a bounded learned dr/dt. This is a deliberate change
from upstream's detached per-example velocity fit and unconstrained dv/dt.
No Gaussian sigma is propagated through the nonlinear decoder: output is
deterministic and probabilistic scores must remain absent.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .baselines import calendar_features
from .spatial_baselines import origin_information


def spatial_derivative(value, axis, periodic=False):
    """Derivative per grid index; centered interior, first-order open edges."""
    if value.shape[axis] < 2:
        raise ValueError('Latent transport requires at least two cells per spatial axis')
    if periodic:
        return (value.roll(-1, axis) - value.roll(1, axis)) / 2
    return torch.gradient(value, dim=axis, edge_order=1)[0]


def latent_transport(state, velocity, periodic_lon=False):
    """Upwind discretization of ClimODE's positive div(v*z), in cell units.

    Face velocity is the mean of neighboring velocities. Positive v transports
    toward decreasing indices, so the upwind value is the right-hand cell.
    Nonperiodic boundaries have zero normal flux; each channel's unweighted
    latent sum is conserved numerically (not a physical mass-conservation law).
    """
    if state.ndim != 4 or velocity.shape != (len(state), 2*state.shape[1], *state.shape[-2:]):
        raise ValueError('Latent transport expects [B,C,H,W] and [B,2C,H,W]')
    c = state.shape[1]
    vx, vy = velocity[:, :c], velocity[:, c:]
    def divergence(component, axis, periodic):
        face_velocity = (component+component.roll(-1, axis))/2
        upwind = torch.where(face_velocity >= 0, state.roll(-1, axis), state)
        outgoing = face_velocity*upwind
        if not periodic:
            outgoing = torch.cat((outgoing.narrow(axis, 0, state.shape[axis]-1),
                                  torch.zeros_like(outgoing.narrow(axis, 0, 1))), axis)
        incoming = outgoing.roll(1, axis)
        # For nonperiodic axes the last outgoing face is already zero, hence
        # the roll sets the first incoming face to zero without another branch.
        return outgoing-incoming
    return divergence(vx, -1, periodic_lon)+divergence(vy, -2, False)


class _GeographicConv(nn.Module):
    def __init__(self, input_channels, output_channels, periodic_lon):
        super().__init__()
        self.periodic_lon = periodic_lon
        self.conv = nn.Conv2d(input_channels, output_channels, 3)

    def forward(self, value):
        value = F.pad(value, (1, 1, 0, 0), mode='circular' if self.periodic_lon else 'replicate')
        return self.conv(F.pad(value, (0, 0, 1, 1), mode='replicate'))


def _spatial_net(inputs, hidden, outputs, periodic):
    # No dropout/BatchNorm: each solver evaluation sees the same vector field.
    return nn.Sequential(_GeographicConv(inputs, hidden, periodic), nn.SiLU(),
                         nn.Conv2d(hidden, outputs, 1))


def _pooled_coordinates(latent_grid, schema, spatial_factor):
    if not isinstance(spatial_factor, int) or isinstance(spatial_factor, bool) or spatial_factor < 1:
        raise ValueError('Latent ClimODE needs the encoder spatial_factor as a positive integer')
    try:
        variable = schema['variables'][0]
        lat = np.asarray(variable['coords']['lat'], dtype=np.float64)
        lon = np.asarray(variable['coords']['lon'], dtype=np.float64)
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError('Latent ClimODE requires archive latitude/longitude schema') from exc
    if (lat.ndim != 1 or lon.ndim != 1 or min(len(lat), len(lon)) < 2
            or not np.isfinite(lat).all() or not np.isfinite(lon).all()
            or np.any(np.abs(lat) > 90)
            or tuple(variable['shape']) != (len(lat), len(lon))):
        raise ValueError('Invalid archive geographic coordinates/shape')
    unwrapped_lon = np.rad2deg(np.unwrap(np.deg2rad(lon)))
    for values in (lat, unwrapped_lon):
        if not (np.all(np.diff(values) > 0) or np.all(np.diff(values) < 0)):
            raise ValueError('Geographic axes must be strictly monotonic without repeated endpoints')
    # Reject a duplicate cyclic endpoint even if its unwrapped axis is monotonic.
    if abs(unwrapped_lon[-1]-unwrapped_lon[0]) >= 360-1e-5:
        raise ValueError('Longitude must not duplicate its cyclic endpoint')
    expected = (math.ceil(len(lat)/spatial_factor), math.ceil(len(lon)/spatial_factor))
    if tuple(latent_grid[1:]) != expected:
        raise ValueError('Latent grid must match the encoder spatial_factor and source grid')
    def pool(values):
        return np.asarray([values[i:i+spatial_factor].mean()
                           for i in range(0, len(values), spatial_factor)])
    difference = np.diff(unwrapped_lon)
    periodic = bool(len(lon) >= 3 and np.allclose(difference, difference[0])
                    and np.isclose(abs(difference[0])*len(lon), 360))
    return pool(lat), pool(unwrapped_lon), periodic


class LatentClimODEPredictor(nn.Module):
    """Trainable spatial transport ODE; flattened API, explicit grid.

    ``spatial_factor`` must be the encoder pooling factor. Merely reshaping an
    arbitrary global vector cannot satisfy this representation contract.
    Matched raw controls use the original grid and spatial_factor=1. Only raw
    controls should enable information_channels; latent forecasts condition
    through the external manifold encoder.
    """
    def __init__(self, latent_grid, schema, *, hidden=128, step_hours=1.,
                 history_dt_hours=24., spatial_factor=None, max_speed=2.,
                 max_acceleration=1., information_channels=0):
        super().__init__()
        if (len(latent_grid) != 3 or any(not isinstance(n, int) or isinstance(n, bool) or n < 1
                                       for n in latent_grid) or min(latent_grid[1:]) < 2):
            raise ValueError('Latent ClimODE requires an explicit [C,H,W] grid with H,W>=2')
        if not isinstance(hidden, int) or hidden < 1:
            raise ValueError('Latent ClimODE hidden width must be positive')
        if (not isinstance(information_channels, int) or isinstance(information_channels, bool)
                or information_channels < 0):
            raise ValueError('Information channels must be a nonnegative integer')
        if any(not math.isfinite(value) or value <= 0
               for value in (step_hours, history_dt_hours, max_speed, max_acceleration)):
            raise ValueError('Integration intervals and latent velocity bounds must be finite and positive')
        if step_hours > 6:
            raise ValueError('Latent ClimODE step_hours must not exceed 6')
        self.grid = tuple(latent_grid)
        self.dimension = math.prod(self.grid)
        self.information_channels = information_channels
        self.information_dim = information_channels * math.prod(self.grid[1:])
        self.step_hours = float(step_hours)
        self.history_dt_hours = float(history_dt_hours)
        self.spatial_factor = spatial_factor
        self.max_speed, self.max_acceleration = float(max_speed), float(max_acceleration)
        lat, lon, self.periodic_lon = _pooled_coordinates(self.grid, schema, spatial_factor)
        self.register_buffer('pooled_lat', torch.tensor(lat, dtype=torch.float32))
        self.register_buffer('pooled_lon', torch.tensor(lon, dtype=torch.float32))
        la, lo = torch.meshgrid(torch.deg2rad(self.pooled_lat), torch.deg2rad(self.pooled_lon), indexing='ij')
        self.register_buffer('position', torch.stack((la.cos(), lo.cos(), la.sin(), lo.sin(),
                                                      la.sin()*lo.cos(), la.sin()*lo.sin()))[None])
        c = self.grid[0]
        # Mean and recency-weighted mean include every observed state. Backward
        # tendency and final state retain chronological/current-state context.
        self.history_encoder = _spatial_net(4*c+10+information_channels, hidden, hidden, self.periodic_lon)
        self.initial_velocity = nn.Conv2d(hidden, 2*c, 1)
        # z, grad_x z, grad_y z, vx, vy, fixed history context, position, clock.
        self.velocity_dynamics = _spatial_net(5*c+hidden+11, hidden, 2*c, self.periodic_lon)

    def _calendar(self, origin_ns, like, elapsed_days=0.):
        # Float64 nanosecond arithmetic avoids losing the sub-day clock.
        elapsed = origin_ns.to(torch.float64) + float(elapsed_days)*24*3.6e12
        return calendar_features(elapsed, like)[..., None, None].expand(-1, -1, *self.grid[1:])

    def _initial(self, history, origin_ns, information=None):
        weights = torch.arange(1, history.shape[1]+1, device=history.device, dtype=history.dtype)
        weighted = (history*weights[None, :, None, None, None]).sum(1)/weights.sum()
        tendency = (history[:, -1]-history[:, -2])/(self.history_dt_hours/24)
        pos = self.position.to(history).expand(len(history), -1, -1, -1)
        features = [history[:, -1], history.mean(1), weighted, tendency,
                    pos, self._calendar(origin_ns, history)]
        if information is not None:
            features.append(information)
        features = torch.cat(features, 1)
        context = self.history_encoder(features)
        raw_velocity = self.initial_velocity(context)
        return torch.cat((history[:, -1], raw_velocity), 1), context

    def _rhs(self, state_velocity, time_days, context, origin_ns):
        c = self.grid[0]
        state, raw_velocity = state_velocity[:, :c], state_velocity[:, c:]
        velocity = self.max_speed*raw_velocity.tanh()
        gradients = (spatial_derivative(state, -1, self.periodic_lon), spatial_derivative(state, -2))
        pos = self.position.to(state).expand(len(state), -1, -1, -1)
        clock = state.new_full((len(state), 1, *self.grid[1:]), float(time_days))
        features = torch.cat((state, *gradients, velocity, context, pos,
                              self._calendar(origin_ns, state, time_days), clock), 1)
        acceleration = self.max_acceleration*self.velocity_dynamics(features).tanh()
        return torch.cat((latent_transport(state, velocity, self.periodic_lon), acceleration), 1)

    def forward(self, history, lead_hours, origin_ns, information=None):
        if (history.ndim != 3 or history.shape[-1] != self.dimension or history.shape[1] < 2
                or not torch.isfinite(history).all()):
            raise ValueError('Latent ClimODE requires finite [B,T>=2,C*H*W] observed spatial latents')
        if origin_ns.shape != (len(history),) or not torch.isfinite(origin_ns).all():
            raise ValueError('One finite origin timestamp is required per history')
        if (lead_hours.ndim != 1 or not len(lead_hours) or not torch.isfinite(lead_hours).all()
                or lead_hours[0] <= 0 or not (lead_hours[1:] > lead_hours[:-1]).all()):
            raise ValueError('Lead hours must be finite, positive and strictly increasing')
        fields = history.reshape(len(history), history.shape[1], *self.grid)
        information = origin_information(information, history, self.information_channels, self.grid[1:])
        state, context = self._initial(fields, origin_ns, information)
        # Bound the sum of directional cell Courant numbers by 0.5. This is a
        # numerical safeguard, not a theorem of long-horizon learned stability.
        max_dt_hours = min(self.step_hours, 24*.5/(2*self.max_speed))
        previous, result = 0., []
        for hour in lead_hours.detach().cpu().tolist():
            count = max(1, math.ceil((hour-previous)/max_dt_hours))
            dt = (hour-previous)/(24*count)
            for index in range(count):
                time = previous/24 + index*dt
                k1 = self._rhs(state, time, context, origin_ns)
                k2 = self._rhs(state+dt*k1/2, time+dt/2, context, origin_ns)
                k3 = self._rhs(state+dt*k2/2, time+dt/2, context, origin_ns)
                k4 = self._rhs(state+dt*k3, time+dt, context, origin_ns)
                state = state + dt*(k1+2*k2+2*k3+k4)/6
            if not torch.isfinite(state).all():
                raise FloatingPointError('Nonfinite latent ClimODE rollout')
            result.append(state[:, :self.grid[0]].flatten(1))
            previous = hour
        return torch.stack(result, 1), None
