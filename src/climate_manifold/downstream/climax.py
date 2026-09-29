"""Official ClimaX backbone adapted to this project's forecast interface.

The actual Microsoft variable-tokenization/cross-attention/ViT/head code is
vendored, not replaced by a generic Transformer. Model size, variables and grid
are configurable and trained from scratch. These runs do not reproduce the
published pretrained foundation model or its benchmark training protocol.

Upstream ClimaX conditions a single observed field on lead_hours / 100. Thus
only the last observed state enters this predictor (including each lead of a
multi-lead batch). Earlier fields and calendar timestamps are validated for the
shared interface but not used. Origin-only information is tokenized as extra
input variables; the output selects only the prognostic channels. The same
backbone works in raw space and in the jointly learned spatial latent space.
"""
import math

import torch
from torch import nn

from .._vendor.climax.arch import ClimaX
from .spatial_baselines import origin_information


class ClimaXPredictor(nn.Module):
    implementation_variant = 'official_climax_configured_direct_lead_v1'
    upstream_source = 'https://github.com/microsoft/ClimaX'
    upstream_commit = '6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c'
    history_policy = 'last_observed_state'
    pretrained = False

    def __init__(self, latent_grid, history_steps, hidden=128, history_dt_hours=24.,
                 periodic_lon=False, information_channels=0, depth=4, patch_size=2,
                 variable_names=None):
        super().__init__()
        self.latent_grid = tuple(latent_grid)
        if (len(self.latent_grid) != 3 or any(
                isinstance(v, bool) or not isinstance(v, int) or v < 1
                for v in self.latent_grid)):
            raise ValueError('ClimaX requires a positive (channels, latitude, longitude) grid')
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
               for v in (history_steps, hidden, depth, patch_size)):
            raise ValueError('ClimaX history length, hidden width, depth and patch size must be positive integers')
        if hidden % 4:
            raise ValueError('ClimaX hidden width must be divisible by four for 2D sine/cosine embedding')
        channels, height, width = self.latent_grid
        if height % patch_size or width % patch_size:
            raise ValueError('ClimaX grid height and width must be divisible by patch_size')
        if (isinstance(history_dt_hours, bool) or not math.isfinite(history_dt_hours)
                or history_dt_hours <= 0):
            raise ValueError('ClimaX history_dt_hours must be finite and positive')
        if (isinstance(information_channels, bool) or not isinstance(information_channels, int)
                or information_channels < 0):
            raise ValueError('Information channels must be a nonnegative integer')
        if variable_names is None:
            variable_names = tuple(f'channel_{i:03d}' for i in range(channels))
        else:
            if isinstance(variable_names, str):
                raise ValueError('variable_names must contain one unique name per prognostic channel')
            variable_names = tuple(variable_names)
        if (len(variable_names) != channels or len(set(variable_names)) != channels
                or any(not isinstance(v, str) or not v for v in variable_names)
                or any(v.startswith('__origin_information_') for v in variable_names)):
            raise ValueError('variable_names must contain one unique nonempty name per prognostic channel')

        self.dimension = math.prod(self.latent_grid)
        self.history_steps = history_steps
        self.history_dt_hours = float(history_dt_hours)
        self.information_channels = information_channels
        self.information_dim = information_channels * height * width
        # Retained as domain metadata; the official patch/attention operations
        # use absolute positions, without adding a new periodic padding rule.
        self.periodic_lon = bool(periodic_lon)
        self.hidden, self.depth, self.patch_size = hidden, depth, patch_size
        self.variable_names = variable_names
        self.input_variables = variable_names + tuple(
            f'__origin_information_{i:03d}' for i in range(information_channels))
        self.num_heads = next(n for n in (8, 4, 2, 1) if hidden % n == 0)
        self.model = ClimaX(
            default_vars=list(self.input_variables), img_size=[height, width],
            patch_size=patch_size, embed_dim=hidden, depth=depth,
            decoder_depth=2, num_heads=self.num_heads, mlp_ratio=4.,
            drop_path=0.1, drop_rate=0.1, parallel_patch_embed=False,
        )

    def forward(self, history, lead_hours, origin_ns, information=None):
        if (history.ndim != 3 or history.shape[1:] != (self.history_steps, self.dimension)
                or not len(history) or not torch.isfinite(history).all()):
            raise ValueError('History does not match the configured spatial grid and history length')
        if origin_ns.shape != (len(history),) or not torch.isfinite(origin_ns).all():
            raise ValueError('One finite origin timestamp is required per history')
        if (lead_hours.ndim != 1 or not len(lead_hours) or not torch.isfinite(lead_hours).all()
                or lead_hours[0] <= 0 or not (lead_hours[1:] > lead_hours[:-1]).all()):
            raise ValueError('Lead hours must be finite, positive and strictly increasing')
        batch = len(history)
        channels, height, width = self.latent_grid
        current = history[:, -1].reshape(batch, channels, height, width)
        auxiliary = origin_information(information, history, self.information_channels, (height, width))
        if auxiliary is not None:
            current = torch.cat((current, auxiliary), dim=1)
        result = []
        for hour in lead_hours.to(history).unbind():
            # Original dataset.py defines lead_times = hours / 100.
            leads = (hour / 100.).expand(batch)
            _, prediction = self.model(
                current, None, leads, self.input_variables, self.variable_names,
                metric=None, lat=None,
            )
            result.append(prediction.flatten(1))
        return torch.stack(result, dim=1), None
