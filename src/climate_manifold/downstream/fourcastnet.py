"""Official FourCastNet AFNO core with causal, fixed-step small-grid adaptation.

The upstream AFNO architecture is vendored and pinned, not re-created using a
generic Fourier layer. The model is freshly trained for this archive. It does
not load the published FourCastNet checkpoint or reproduce its resolution,
variables, training protocol or headline scores.

At each fixed physical-time step, AFNO sees the evolving field plus the fixed
observed history, the actual history offsets, target calendar, lead time and
origin-only auxiliary fields. History sampled every 24 hours is consequently
never shifted as if its samples were six hours apart. Requested leads must be
integer multiples of forecast_step_hours; skipped steps are still integrated.
"""
import math
from types import SimpleNamespace

import torch
from torch import nn

from .._vendor.fourcastnet.afnonet import AFNONet
from .baselines import calendar_features
from .spatial_baselines import origin_information


class FourCastNetPredictor(nn.Module):
    implementation_variant = 'official_afno_fixed_context_autoregressive_v1'
    upstream_source = 'https://github.com/NVlabs/FourCastNet'
    upstream_commit = '93360c1720a9f97aabf970689f21c9fad8737788'
    history_protocol = 'fixed_observed_context_with_actual_offsets'
    pretrained = False

    def __init__(self, latent_grid, history_steps, hidden=128, history_dt_hours=24.,
                 periodic_lon=False, information_channels=0, *, patch_size=2,
                 depth=4, forecast_step_hours=6.):
        super().__init__()
        self.latent_grid = tuple(latent_grid)
        if (len(self.latent_grid) != 3 or any(
                isinstance(v, bool) or not isinstance(v, int) or v < 1
                for v in self.latent_grid)):
            raise ValueError('FourCastNet requires a positive (channels, latitude, longitude) grid')
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
               for v in (history_steps, hidden, patch_size, depth)):
            raise ValueError('History length, hidden width, patch size and depth must be positive integers')
        for name, value in (('history_dt_hours', history_dt_hours),
                            ('forecast_step_hours', forecast_step_hours)):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'FourCastNet {name} must be finite and positive')
        if (isinstance(information_channels, bool) or not isinstance(information_channels, int)
                or information_channels < 0):
            raise ValueError('Information channels must be a nonnegative integer')
        channels, height, width = self.latent_grid
        if height % patch_size or width % patch_size:
            raise ValueError('FourCastNet spatial grid dimensions must be divisible by patch_size')
        self.dimension = math.prod(self.latent_grid)
        self.history_steps = history_steps
        self.history_dt_hours = float(history_dt_hours)
        self.forecast_step_hours = float(forecast_step_hours)
        self.information_channels = information_channels
        self.information_dim = information_channels * height * width
        # Retained for contract metadata. The upstream 2-D FFT is periodic on
        # both patch axes regardless of the archive's geographic longitude flag.
        self.periodic_lon = bool(periodic_lon)
        self.spectral_boundary = 'upstream_fft_periodic_both_patch_axes'
        self.hidden = hidden
        self.patch_size = patch_size
        self.depth = depth
        self.num_blocks = math.gcd(hidden, 8)
        # Current state; fixed observed fields; observation offsets; four
        # target calendar features; target lead and step duration; origin info.
        in_channels = channels * (history_steps + 1) + history_steps + 6 + information_channels
        params = SimpleNamespace(patch_size=patch_size, N_in_channels=in_channels,
                                 N_out_channels=channels, num_blocks=self.num_blocks)
        self.core = AFNONet(params=params, img_size=(height, width),
                            embed_dim=hidden, depth=depth, drop_rate=0., drop_path_rate=0.)

    def forward(self, history, lead_hours, origin_ns, information=None):
        if (history.ndim != 3 or history.shape[1:] != (self.history_steps, self.dimension)
                or not len(history) or not torch.isfinite(history).all()):
            raise ValueError('History does not match the configured spatial grid and history length')
        if origin_ns.shape != (len(history),) or not torch.isfinite(origin_ns).all():
            raise ValueError('One finite origin timestamp is required per history')
        if (lead_hours.ndim != 1 or not len(lead_hours) or not torch.isfinite(lead_hours).all()
                or lead_hours[0] <= 0 or not (lead_hours[1:] > lead_hours[:-1]).all()):
            raise ValueError('Lead hours must be finite, positive and strictly increasing')
        requested = lead_hours.to(dtype=torch.float64) / self.forecast_step_hours
        rounded = requested.round()
        if not torch.allclose(requested, rounded, rtol=0., atol=1e-6) or rounded[0] < 1:
            raise ValueError('Lead hours must be positive integer multiples of forecast_step_hours')
        indices = [int(value) for value in rounded.tolist()]

        batch = len(history)
        channels, height, width = self.latent_grid
        observed = history.reshape(batch, self.history_steps, channels, height, width)
        info = origin_information(information, history, self.information_channels, (height, width))
        # Offsets remain relative to the forecast origin, not relative to the
        # evolving field. Both are present, so the network knows their age.
        offsets = ((torch.arange(self.history_steps, device=history.device, dtype=history.dtype)
                    - (self.history_steps - 1)) * (self.history_dt_hours / 24))
        fixed_inputs = [observed.flatten(1, 2),
                        offsets[None, :, None, None].expand(batch, -1, height, width)]
        if info is not None:
            fixed_inputs.append(info)
        fixed_context = torch.cat(fixed_inputs, dim=1)
        origin = origin_ns.to(device=history.device, dtype=torch.float64)
        state = observed[:, -1]
        outputs = []
        wanted = set(indices)
        for step in range(1, indices[-1] + 1):
            hour = step * self.forecast_step_hours
            target_time = origin + hour * 3.6e12
            calendar = calendar_features(target_time, history)
            durations = history.new_tensor([hour / 24, self.forecast_step_hours / 24])
            timing = torch.cat((calendar, durations[None].expand(batch, -1)), dim=-1)
            inputs = torch.cat((state, fixed_context,
                                timing[:, :, None, None].expand(-1, -1, height, width)), dim=1)
            state = self.core(inputs)
            if step in wanted:
                outputs.append(state.flatten(1))
        return torch.stack(outputs, dim=1), None
