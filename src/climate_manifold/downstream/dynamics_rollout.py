"""Autoregressive dynamics-latent auxiliary forecast, with no future inputs.

The same architecture supports field-only, field-plus-latent, and vector-only
supervision. Vector-only retains but freezes/skips the auxiliary field decoder.
Unlike the legacy direct auxiliary decoder, every decoded forecast is produced
from the explicitly rolled-out latent. There is no lead-time shortcut to D_d.
This auxiliary F_d is independent of the downstream forecast model M.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..spatial import GeographicConv2d
from .dynamics_guide import _history, _positive_integer
from .guided_transformer import _positive_grid


def lead_steps(lead_hours, step_hours):
    """Validate physical lead labels and return positive archive-step indices."""
    if (isinstance(step_hours, bool) or not isinstance(step_hours, (int, float))
            or not math.isfinite(step_hours) or step_hours <= 0):
        raise ValueError('Archive step hours must be finite and positive')
    if (not isinstance(lead_hours, torch.Tensor) or lead_hours.ndim != 1
            or not len(lead_hours) or not torch.isfinite(lead_hours).all()
            or lead_hours[0] <= 0 or not (lead_hours[1:] > lead_hours[:-1]).all()):
        raise ValueError('Dynamics leads must be positive, finite and strictly increasing')
    steps = lead_hours / step_hours
    if not torch.allclose(steps, steps.round(), rtol=0, atol=1e-5):
        raise ValueError('Dynamics labels require leads aligned to archive time steps')
    return steps.round().long()


class DynamicsLatentRolloutHead(nn.Module):
    """E_d(history)[-1] -> shared residual transition F_d -> shared decoder D_d.

    One transition advances one archive step, including steps between requested
    nonuniform leads. Time is measured in physical days. Decoding occurs only at
    requested leads and cannot change the latent recurrence. The transition is
    not an imposed physical PDE, and long-horizon stability is not guaranteed.
    """
    implementation_variant = 'autoregressive_dynamics_latent_auxiliary_v1'

    def __init__(self, dynamics_grid, raw_grid, hidden=32, periodic_lon=False,
                 step_hours=24.):
        super().__init__()
        self.dynamics_grid = _positive_grid(dynamics_grid, 'Dynamics grid')
        self.raw_grid = _positive_grid(raw_grid, 'Raw grid')
        _positive_integer(hidden, 'Hidden width')
        if (isinstance(step_hours, bool) or not isinstance(step_hours, (int, float))
                or not math.isfinite(step_hours) or step_hours <= 0):
            raise ValueError('Archive step hours must be finite and positive')
        self.step_hours, self.hidden = float(step_hours), hidden
        channels = self.dynamics_grid[0]
        # The final transition/readout stays randomly initialized: both F_d and
        # D_d can propagate auxiliary gradient into E_d on the first update.
        self.transition = nn.Sequential(
            GeographicConv2d(channels + 1, hidden, periodic_lon=periodic_lon),
            nn.SiLU(),
            GeographicConv2d(hidden, hidden, periodic_lon=periodic_lon),
            nn.SiLU(), nn.Conv2d(hidden, channels, kernel_size=1),
        )
        self.decoder = nn.Sequential(
            GeographicConv2d(channels, hidden, periodic_lon=periodic_lon),
            nn.SiLU(),
            GeographicConv2d(hidden, hidden, periodic_lon=periodic_lon),
            nn.SiLU(), nn.Conv2d(hidden, self.raw_grid[0], kernel_size=1),
        )

    def decode(self, latent):
        """Decode a batch of states (B,D) or trajectories (B,L,D)."""
        if (latent.ndim not in (2, 3) or not all(latent.shape)
                or latent.shape[-1] != math.prod(self.dynamics_grid)
                or not torch.isfinite(latent).all()):
            raise ValueError('Dynamics latent must have finite configured shape (B,D) or (B,L,D)')
        prefix = latent.shape[:-1]
        fields = self.decoder(latent.reshape(-1, *self.dynamics_grid))
        if fields.shape[-2:] != self.raw_grid[1:]:
            fields = F.interpolate(fields, size=self.raw_grid[1:], mode='bilinear', align_corners=False)
        return fields.reshape(*prefix, math.prod(self.raw_grid))

    def rollout(self, dynamics_history, lead_hours, *, decode_fields=True):
        """Advance latent states; optionally omit the independent field readout."""
        if not isinstance(decode_fields, bool):
            raise ValueError('decode_fields must be boolean')
        _history(dynamics_history, math.prod(self.dynamics_grid), 'Dynamics history')
        steps = lead_steps(lead_hours, self.step_hours)
        requested = set(steps.detach().cpu().tolist())
        state = dynamics_history[:, -1].reshape(-1, *self.dynamics_grid)
        dt_days = self.step_hours / 24.
        interval = state.new_full((state.shape[0], 1, *self.dynamics_grid[1:]), dt_days)
        selected = []
        for step in range(1, int(steps[-1].item()) + 1):
            state = state + dt_days * self.transition(torch.cat((state, interval), dim=1))
            if step in requested:
                selected.append(state.flatten(1))
        latent = torch.stack(selected, dim=1)
        if not torch.isfinite(latent).all():
            raise FloatingPointError('Nonfinite auxiliary dynamics latent rollout')
        return {'latent': latent, 'fields': self.decode(latent) if decode_fields else None}

    def forward(self, dynamics_history, lead_hours):
        return self.rollout(dynamics_history, lead_hours)['fields']
