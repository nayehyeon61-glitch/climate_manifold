"""Fuse observed raw history with a statistics-supervised latent guide.

E(history) -> guide latent; FusionTransformer(raw, guide) -> raw + delta; then
any matched-raw forecast family consumes the fused history unchanged and emits
physical fields directly. The forecast decoder D is never used.

The delta head is zero-initialized, so at step 0 the fused history equals the
raw history and a guided arm starts from exactly its raw counterpart's input.
``guide_mode='zero'`` keeps the fusion module and token count but removes all
input-dependent guide information, isolating guide content from added capacity.
"""
import math

import torch
from torch import nn

from .guided_transformer import GuidedTransformerPredictor, _coordinates, _positive_grid


class GuidedFusion(nn.Module):
    implementation_variant = 'raw_guide_residual_fusion_transformer_v1'

    # Shared patch tokenization with the joint guided Transformer.
    _patch_tokens = GuidedTransformerPredictor._patch_tokens
    _time_features = staticmethod(GuidedTransformerPredictor._time_features)

    def __init__(self, raw_grid, guide_grid, history_steps, hidden=128, depth=2,
                 patch_size=2, heads=4, history_dt_hours=24., guide_mode='learned',
                 periodic_lon=False):
        super().__init__()
        self.raw_grid = _positive_grid(raw_grid, 'Raw grid')
        self.guide_grid = _positive_grid(guide_grid, 'Guide grid')
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
               for v in (history_steps, hidden, depth, patch_size, heads)):
            raise ValueError('History length, width, depth, patch size and heads must be positive integers')
        if hidden % heads:
            raise ValueError('Hidden width must be divisible by attention heads')
        if guide_mode not in ('learned', 'zero'):
            raise ValueError('Guide mode must be learned or zero')
        self.dimension = math.prod(self.raw_grid)
        self.guide_dimension = math.prod(self.guide_grid)
        self.history_steps, self.history_dt_hours = history_steps, float(history_dt_hours)
        self.hidden, self.depth, self.patch_size, self.num_heads = hidden, depth, patch_size, heads
        self.guide_mode, self.periodic_lon = guide_mode, bool(periodic_lon)
        channels, height, width = self.raw_grid
        self.patch_grid = (math.ceil(height / patch_size), math.ceil(width / patch_size))
        self.raw_patch_count = math.prod(self.patch_grid)

        self.raw_embed = nn.Conv2d(channels, hidden, patch_size, stride=patch_size)
        self.guide_embed = nn.Conv2d(self.guide_grid[0], hidden, patch_size, stride=patch_size)
        self.register_buffer('raw_coordinates', _coordinates(height, width, patch_size))
        self.register_buffer('guide_coordinates', _coordinates(*self.guide_grid[1:], patch_size))
        self.register_buffer('history_age_days',
                             torch.arange(1 - history_steps, 1, dtype=torch.float32)
                             * self.history_dt_hours / 24.)
        self.position_embed = nn.Linear(6, hidden)
        self.time_embed = nn.Sequential(nn.Linear(4, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.type_embed = nn.Parameter(torch.empty(2, hidden))
        nn.init.normal_(self.type_embed, std=.02)
        layer = nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=4 * hidden,
                                           dropout=0., activation='gelu', batch_first=True,
                                           norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, depth, norm=nn.LayerNorm(hidden),
                                                 enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
                                  nn.Linear(hidden, channels * patch_size * patch_size))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, history, guide_history=None):
        batch = len(history)
        if (history.ndim != 3 or history.shape[1:] != (self.history_steps, self.dimension)
                or not batch or not torch.isfinite(history).all()):
            raise ValueError('History must contain the configured finite observed raw sequence')
        if self.guide_mode == 'zero':
            guide_history = history.new_zeros(batch, self.history_steps, self.guide_dimension)
        elif (guide_history is None
              or guide_history.shape != (batch, self.history_steps, self.guide_dimension)
              or not torch.isfinite(guide_history).all()):
            raise ValueError('Learned guide requires the configured finite observed guide history')
        observed = history.reshape(batch, self.history_steps, *self.raw_grid)
        guide = guide_history.reshape(batch, self.history_steps, *self.guide_grid)
        tokens = torch.cat((
            self._patch_tokens(observed, self.raw_embed, self.raw_coordinates, 0, self.history_age_days),
            self._patch_tokens(guide, self.guide_embed, self.guide_coordinates, 1, self.history_age_days)),
            dim=1)
        memory = self.transformer(tokens)[:, :self.history_steps * self.raw_patch_count]
        patches = self.head(memory)
        ph, pw = self.patch_grid
        channels, height, width = self.raw_grid
        patches = patches.reshape(batch, self.history_steps, ph, pw, channels,
                                  self.patch_size, self.patch_size)
        delta = patches.permute(0, 1, 4, 2, 5, 3, 6).reshape(
            batch, self.history_steps, channels, ph * self.patch_size, pw * self.patch_size)
        return history + delta[..., :height, :width].flatten(2)
