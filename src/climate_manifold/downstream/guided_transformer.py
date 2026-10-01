"""Raw-field forecasting conditioned on a statistics-supervised spatial guide.

The encoder's latent is additional observed context, not a bottleneck through
which all forecast information must pass. Raw and guide history patches attend
jointly. The output head reconstructs physical patches directly, bypassing the
manifold forecast decoder. No random latent sampling or future guide is used.

``guide_mode='zero'`` preserves guide token count and trainable predictor shape
while removing input-dependent guide information. ``guide_grid=None`` is the
smaller raw-only architecture; it is not a parameter-matched zero-guide control.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .spatial_baselines import origin_information


def _positive_grid(grid, name):
    grid = tuple(grid)
    if len(grid) != 3 or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in grid):
        raise ValueError(f'{name} must be a positive (channels, height, width) grid')
    return grid


def _coordinates(height, width, patch_size):
    """Normalized patch-center features, with periodic longitude features."""
    rows = (torch.arange(math.ceil(height / patch_size), dtype=torch.float32)
            * patch_size + (patch_size - 1) / 2).clamp(max=height - 1)
    cols = (torch.arange(math.ceil(width / patch_size), dtype=torch.float32)
            * patch_size + (patch_size - 1) / 2).clamp(max=width - 1)
    rows = rows * (2. / (height - 1)) - 1. if height > 1 else rows * 0
    # Longitude cell centers avoid duplicating -pi and +pi at the end columns.
    cols = (cols + .5) * (2. / width) - 1.
    y, x = torch.meshgrid(rows, cols, indexing='ij')
    return torch.stack((y, y.square(), torch.sin(math.pi * y), torch.cos(math.pi * y),
                        torch.sin(math.pi * x), torch.cos(math.pi * x)), dim=-1).reshape(-1, 6)


class GuidedTransformerPredictor(nn.Module):
    implementation_variant = 'raw_statistical_guide_joint_attention_v1'
    history_policy = 'all_observed_states'
    pretrained = False

    def __init__(self, raw_grid, guide_grid=None, history_steps=2, hidden=128,
                 depth=2, patch_size=2, heads=4, history_dt_hours=24.,
                 guide_mode='learned', periodic_lon=False, information_channels=0):
        super().__init__()
        self.raw_grid = _positive_grid(raw_grid, 'Raw grid')
        self.guide_grid = None if guide_grid is None else _positive_grid(guide_grid, 'Guide grid')
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
               for v in (history_steps, hidden, depth, patch_size, heads)):
            raise ValueError('History length, width, depth, patch size and heads must be positive integers')
        if hidden % heads:
            raise ValueError('Hidden width must be divisible by attention heads')
        if (isinstance(history_dt_hours, bool) or not math.isfinite(history_dt_hours)
                or history_dt_hours <= 0):
            raise ValueError('history_dt_hours must be finite and positive')
        if guide_mode not in ('learned', 'zero'):
            raise ValueError('Guide mode must be learned or zero')
        if (isinstance(information_channels, bool) or not isinstance(information_channels, int)
                or information_channels < 0):
            raise ValueError('Information channels must be a nonnegative integer')

        self.dimension = math.prod(self.raw_grid)
        self.guide_dimension = 0 if self.guide_grid is None else math.prod(self.guide_grid)
        self.history_steps, self.history_dt_hours = history_steps, float(history_dt_hours)
        self.hidden, self.depth, self.patch_size, self.num_heads = hidden, depth, patch_size, heads
        self.guide_mode, self.periodic_lon = guide_mode, bool(periodic_lon)
        self.information_channels = information_channels
        self.information_dim = information_channels * math.prod(self.raw_grid[1:])
        channels, height, width = self.raw_grid
        self.patch_grid = (math.ceil(height / patch_size), math.ceil(width / patch_size))
        self.raw_patch_count = math.prod(self.patch_grid)

        self.raw_embed = nn.Conv2d(channels, hidden, patch_size, stride=patch_size)
        self.guide_embed = (nn.Conv2d(self.guide_grid[0], hidden, patch_size, stride=patch_size)
                            if self.guide_grid is not None else None)
        self.information_embed = (nn.Conv2d(information_channels, hidden, patch_size, stride=patch_size)
                                  if information_channels else None)
        self.register_buffer('raw_coordinates', _coordinates(height, width, patch_size))
        self.register_buffer('guide_coordinates',
                             _coordinates(*self.guide_grid[1:], patch_size)
                             if self.guide_grid is not None else torch.empty(0, 6))
        self.register_buffer('history_age_days',
                             torch.arange(1 - history_steps, 1, dtype=torch.float32)
                             * self.history_dt_hours / 24.)
        self.position_embed = nn.Linear(6, hidden)
        self.time_embed = nn.Sequential(nn.Linear(4, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.lead_embed = nn.Sequential(nn.Linear(4, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.type_embed = nn.Parameter(torch.empty(3, hidden))
        nn.init.normal_(self.type_embed, std=.02)
        layer = nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=4 * hidden,
                                           dropout=0., activation='gelu', batch_first=True,
                                           norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, depth, norm=nn.LayerNorm(hidden),
                                                 enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
                                  nn.Linear(hidden, channels * patch_size * patch_size))

    @staticmethod
    def _time_features(days):
        return torch.stack((days, torch.sign(days) * torch.log1p(days.abs()),
                            torch.sin(days), torch.cos(days)), dim=-1)

    def _patch_tokens(self, fields, embedding, coordinates, token_type, ages):
        batch, times, channels, height, width = fields.shape
        fields = fields.reshape(batch * times, channels, height, width)
        pad_h, pad_w = (-height) % self.patch_size, (-width) % self.patch_size
        if pad_w:
            if self.periodic_lon:
                # index_select also handles patch sizes larger than the width.
                indices = torch.arange(width + pad_w, device=fields.device) % width
                fields = fields.index_select(-1, indices)
            else:
                fields = F.pad(fields, (0, pad_w, 0, 0), mode='replicate')
        if pad_h:
            fields = F.pad(fields, (0, 0, 0, pad_h), mode='replicate')
        tokens = embedding(fields).flatten(2).transpose(1, 2).reshape(batch, times, -1, self.hidden)
        tokens = tokens + self.position_embed(coordinates)[None, None]
        tokens = tokens + self.time_embed(self._time_features(ages))[None, :, None]
        return (tokens + self.type_embed[token_type]).flatten(1, 2)

    def forward(self, history, lead_hours, origin_ns=None, information=None, *, guide_history=None):
        if (history.ndim != 3 or history.shape[1:] != (self.history_steps, self.dimension)
                or not len(history) or not torch.isfinite(history).all()):
            raise ValueError('History must contain the configured finite observed raw sequence')
        batch = len(history)
        if origin_ns is not None and (origin_ns.shape != (batch,) or not torch.isfinite(origin_ns).all()):
            raise ValueError('One finite origin timestamp is required per history')
        if (lead_hours.ndim != 1 or not len(lead_hours) or not torch.isfinite(lead_hours).all()
                or lead_hours[0] <= 0 or not (lead_hours[1:] > lead_hours[:-1]).all()):
            raise ValueError('Lead hours must be finite, positive and strictly increasing')
        if self.guide_grid is None and guide_history is not None:
            raise ValueError('Raw-only Transformer cannot receive guide history')
        if self.guide_grid is not None:
            if guide_history is not None and (
                    guide_history.shape != (batch, self.history_steps, self.guide_dimension)
                    or not torch.isfinite(guide_history).all()):
                raise ValueError('Guide must contain only the configured finite observed history')
            if self.guide_mode == 'learned' and guide_history is None:
                raise ValueError('Learned guide requires observed guide_history')
            if self.guide_mode == 'zero':
                guide_history = history.new_zeros(batch, self.history_steps, self.guide_dimension)
        auxiliary = origin_information(information, history, self.information_channels, self.raw_grid[1:])
        observed = history.reshape(batch, self.history_steps, *self.raw_grid)
        chunks = [self._patch_tokens(observed, self.raw_embed, self.raw_coordinates,
                                     0, self.history_age_days)]
        if self.guide_grid is not None:
            guide = guide_history.reshape(batch, self.history_steps, *self.guide_grid)
            chunks.append(self._patch_tokens(guide, self.guide_embed, self.guide_coordinates,
                                             1, self.history_age_days))
        if auxiliary is not None:
            chunks.append(self._patch_tokens(auxiliary[:, None], self.information_embed,
                                             self.raw_coordinates, 2, self.history_age_days[-1:]))
        memory = self.transformer(torch.cat(chunks, dim=1))
        start = (self.history_steps - 1) * self.raw_patch_count
        current_tokens = memory[:, start:start + self.raw_patch_count]
        # Forecast leads are queries only: no target or future encoder state can
        # enter observed attention. Direct leads can be evaluated independently.
        days = lead_hours.to(history) / 24.
        query = current_tokens[:, None] + self.lead_embed(self._time_features(days))[None, :, None]
        patches = self.head(query)
        ph, pw = self.patch_grid
        channels, height, width = self.raw_grid
        patches = patches.reshape(batch, len(days), ph, pw, channels, self.patch_size, self.patch_size)
        rate = patches.permute(0, 1, 4, 2, 5, 3, 6).reshape(
            batch, len(days), channels, ph * self.patch_size, pw * self.patch_size)
        rate = rate[..., :height, :width].flatten(2)
        return history[:, -1, None] + days[None, :, None] * rate, None
