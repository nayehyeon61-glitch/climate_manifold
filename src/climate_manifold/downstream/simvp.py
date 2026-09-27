"""SimVP-gSTA adapted to direct, physical-time-conditioned forecasting.

The vendored OpenSTL spatial encoder, gSTA temporal translator and decoder form
the predictor F. They are internal to F: when used with Climate Manifold the
full path is E_manifold -> [E_SimVP -> translator -> D_SimVP] -> D_manifold.

Unlike the upstream equal-length/recursive block forecast, learned lead queries
mix the translated observed history and its skip features for each requested
lead independently. Explicit history spacing and calendar conditioning support
24-hourly history with 6-hourly predictions without relabeling time intervals.
This is an adapted SimVP-gSTA, not the unchanged official benchmark model.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .._vendor.openstl.gsta import Decoder, Encoder, MidMetaNet
from .baselines import calendar_features
from .spatial_baselines import origin_information


class SimVPPredictor(nn.Module):
    implementation_variant = 'openstl_gsta_direct_lead_v1'
    upstream_source = 'https://github.com/chengtan9907/OpenSTL'
    upstream_commit = 'eecf8a3078f0a178dbc7b28723da20f94ce36985'

    def __init__(self, latent_grid, history_steps, hidden=128, history_dt_hours=24.,
                 periodic_lon=False, information_channels=0):
        super().__init__()
        self.latent_grid = tuple(latent_grid)
        if (len(self.latent_grid) != 3 or any(
                isinstance(v, bool) or not isinstance(v, int) or v < 1
                for v in self.latent_grid)):
            raise ValueError('SimVP requires a positive (channels, latitude, longitude) grid')
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
               for v in (history_steps, hidden)):
            raise ValueError('SimVP history length and hidden width must be positive integers')
        if (isinstance(history_dt_hours, bool) or not math.isfinite(history_dt_hours)
                or history_dt_hours <= 0):
            raise ValueError('SimVP history_dt_hours must be finite and positive')
        if (isinstance(information_channels, bool) or not isinstance(information_channels, int)
                or information_channels < 0):
            raise ValueError('Information channels must be a nonnegative integer')
        self.dimension = math.prod(self.latent_grid)
        self.history_steps = history_steps
        self.history_dt_hours = float(history_dt_hours)
        self.information_channels = information_channels
        self.information_dim = information_channels * math.prod(self.latent_grid[1:])
        self.periodic_lon = bool(periodic_lon)
        self.hidden = hidden
        # GroupNorm(2) requires even channels; four permits singleton maps.
        self.spatial_hidden = max(4, 2 * math.ceil(hidden / 8))
        channels = self.latent_grid[0]
        self.encoder = Encoder(channels + information_channels + 5,
                               self.spatial_hidden, self.periodic_lon)
        self.translator = MidMetaNet(history_steps * self.spatial_hidden,
                                    max(4, hidden), blocks=4, periodic_lon=self.periodic_lon)
        self.decoder = Decoder(self.spatial_hidden, channels, self.periodic_lon)
        # Four calendar features at the target time, lead-days, log lead-days,
        # and observed history spacing. Query produces temporal mixing and FiLM.
        self.lead_query = nn.Sequential(
            nn.Linear(7, max(4, hidden)), nn.SiLU(),
            nn.Linear(max(4, hidden), history_steps + 2 * self.spatial_hidden),
        )

    def _pad_to_stride(self, fields):
        height, width = fields.shape[-2:]
        if width % 2:
            edge = fields[..., :1] if self.periodic_lon else fields[..., -1:]
            fields = torch.cat((fields, edge), dim=-1)
        if height % 2:
            fields = F.pad(fields, (0, 0, 0, 1), mode='replicate')
        return fields

    def forward(self, history, lead_hours, origin_ns, information=None):
        if (history.ndim != 3 or history.shape[1:] != (self.history_steps, self.dimension)
                or not torch.isfinite(history).all() or not len(history)):
            raise ValueError('History does not match the configured spatial grid and history length')
        if origin_ns.shape != (len(history),) or not torch.isfinite(origin_ns).all():
            raise ValueError('One finite origin timestamp is required per history')
        if (lead_hours.ndim != 1 or not len(lead_hours) or not torch.isfinite(lead_hours).all()
                or lead_hours[0] <= 0 or not (lead_hours[1:] > lead_hours[:-1]).all()):
            raise ValueError('Lead hours must be finite, positive and strictly increasing')

        batch = len(history)
        channels, height, width = self.latent_grid
        observed = history.reshape(batch, self.history_steps, channels, height, width)
        raw_info = origin_information(information, history, self.information_channels, (height, width))
        # Calendar features and relative offsets describe the actual observations.
        offsets = (torch.arange(self.history_steps, device=history.device, dtype=torch.float64)
                   - (self.history_steps - 1)) * self.history_dt_hours
        history_times = origin_ns.to(device=history.device, dtype=torch.float64)[:, None] + offsets[None] * 3.6e12
        calendar = calendar_features(history_times, history)
        relative = (offsets / 24).to(history)[None, :, None].expand(batch, -1, -1)
        timing = torch.cat((calendar, relative), dim=-1)
        inputs = [observed, timing[:, :, :, None, None].expand(-1, -1, -1, height, width)]
        if raw_info is not None:
            inputs.append(raw_info[:, None].expand(-1, self.history_steps, -1, -1, -1))
        fields = torch.cat(inputs, dim=2).flatten(0, 1)
        fields = self._pad_to_stride(fields)
        embedded, skip = self.encoder(fields)
        small_shape, full_shape = embedded.shape[-2:], skip.shape[-2:]
        embedded = embedded.reshape(batch, self.history_steps, self.spatial_hidden, *small_shape)
        translated = self.translator(embedded)
        skip = skip.reshape(batch, self.history_steps, self.spatial_hidden, *full_shape)

        leads = lead_hours.to(history)
        target_times = origin_ns.to(device=history.device, dtype=torch.float64)[:, None] + leads.to(torch.float64)[None] * 3.6e12
        target_calendar = calendar_features(target_times, history)
        days = leads / 24
        lead_features = torch.stack((days, torch.log1p(days),
                                     torch.full_like(days, self.history_dt_hours / 24)), dim=-1)
        query = self.lead_query(torch.cat((target_calendar,
                              lead_features[None].expand(batch, -1, -1)), dim=-1))
        weights = query[..., :self.history_steps].softmax(dim=-1)
        scale, bias = query[..., self.history_steps:].chunk(2, dim=-1)
        mixed = torch.einsum('blt,btchw->blchw', weights, translated)
        mixed = mixed * (1 + .1 * scale.tanh()[..., None, None]) + bias[..., None, None]
        mixed_skip = torch.einsum('blt,btchw->blchw', weights, skip)
        result = self.decoder(mixed.flatten(0, 1), mixed_skip.flatten(0, 1))
        result = result[..., :height, :width]
        return result.reshape(batch, len(leads), self.dimension), None
