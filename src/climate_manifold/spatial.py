"""Spatial climate representation for joint encoder--predictor--decoder training.

Flattening is an interface convention only: every learned E/D operation is a
local convolution on a geographic grid. There is no global latent bottleneck,
intrinsic A drift, auxiliary sampler, or future-information input here.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .manifold_physics import SurfacePhysics
from .temporal_supervision import TemporalObjective


class GeographicConv2d(nn.Module):
    """3x3 convolution with latitude edges and optional cyclic longitude.

    Replicate padding is used at both regional longitude boundaries and latitude
    boundaries. Only a complete cyclic longitude grid wraps across its seam.
    """
    def __init__(self, in_channels, out_channels, *, periodic_lon=False):
        super().__init__()
        self.periodic_lon = bool(periodic_lon)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3)

    def forward(self, fields):
        fields = F.pad(fields, (1, 1, 0, 0), mode='circular' if self.periodic_lon else 'replicate')
        fields = F.pad(fields, (0, 0, 1, 1), mode='replicate')
        return self.conv(fields)


def downsample_latlon(lat, lon, factor):
    """Mean coordinates of the same nonoverlapping bins as the spatial encoder.

    Ceil pooling retains a smaller last bin when the source size is odd. These
    coordinates consequently need not be uniformly spaced even on a uniform
    source grid. Operators claiming physical geographic derivatives must use
    that spacing; transport defined in latent cell-index coordinates is a
    separate model choice and has cell/day rather than physical velocity units.
    """
    if isinstance(factor, bool) or not isinstance(factor, (int, np.integer)) or factor < 1:
        raise ValueError('Coordinate downsampling factor must be a positive integer')
    result = []
    for values in (lat, lon):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
            raise ValueError('Spatial coordinates must be finite one-dimensional arrays')
        if not (np.all(np.diff(values) > 0) or np.all(np.diff(values) < 0)):
            raise ValueError('Spatial coordinates must be strictly monotone')
        result.append(np.asarray([values[i:i+factor].mean() for i in range(0, len(values), factor)]))
    return tuple(result)


class SpatialEncoder(nn.Module):
    def __init__(self, input_grid, latent_grid, hidden_dim, factor, periodic_lon):
        super().__init__()
        self.input_grid, self.latent_grid = tuple(input_grid), tuple(latent_grid)
        self.factor = factor
        self.features = nn.Sequential(
            GeographicConv2d(input_grid[0], hidden_dim, periodic_lon=periodic_lon), nn.SiLU(),
            GeographicConv2d(hidden_dim, hidden_dim, periodic_lon=periodic_lon), nn.SiLU(),
        )
        self.channels = nn.Conv2d(hidden_dim, latent_grid[0], kernel_size=1)

    def forward(self, states):
        if states.ndim < 1 or states.shape[-1] != math.prod(self.input_grid):
            raise ValueError('Spatial encoder requires a flattened field with the configured grid size')
        leading = states.shape[:-1]
        features = self.features(states.reshape(-1, *self.input_grid))
        # No artificial zeros enter an incomplete edge bin.
        if self.factor != 1:
            features = F.avg_pool2d(features, self.factor, stride=self.factor,
                                   ceil_mode=True, count_include_pad=False)
        latent = self.channels(features)
        return latent.reshape(*leading, math.prod(self.latent_grid))


class SpatialDecoder(nn.Module):
    def __init__(self, latent_grid, output_grid, hidden_dim, factor, periodic_lon):
        super().__init__()
        self.latent_grid, self.output_grid = tuple(latent_grid), tuple(output_grid)
        self.factor = factor
        self.channels = nn.Conv2d(latent_grid[0], hidden_dim, kernel_size=1)
        self.features = nn.Sequential(
            GeographicConv2d(hidden_dim, hidden_dim, periodic_lon=periodic_lon), nn.SiLU(),
            GeographicConv2d(hidden_dim, output_grid[0], periodic_lon=periodic_lon),
        )

    def forward(self, latent):
        if latent.ndim < 1 or latent.shape[-1] != math.prod(self.latent_grid):
            raise ValueError('Spatial decoder requires a flattened configured latent grid')
        leading = latent.shape[:-1]
        fields = F.silu(self.channels(latent.reshape(-1, *self.latent_grid)))
        # Repeat exact pooling bins, then trim only the incomplete final bin.
        # This preserves cell alignment for odd sizes and supports cyclic convs
        # at the actual source-grid seam instead of an interpolated extra column.
        fields = fields.repeat_interleave(self.factor, -2).repeat_interleave(self.factor, -1)
        fields = fields[..., :self.output_grid[1], :self.output_grid[2]]
        return self.features(fields).reshape(*leading, math.prod(self.output_grid))


class SpatialManifoldAE(nn.Module):
    def __init__(self, config, periodic_lon):
        super().__init__()
        kwargs = dict(hidden_dim=config.spatial_hidden_dim, factor=config.spatial_downsample,
                      periodic_lon=periodic_lon)
        self.encoder = SpatialEncoder(config.grid, config.latent_grid, **kwargs)
        self.decoder = SpatialDecoder(config.latent_grid, config.grid, **kwargs)

    def encode(self, states):
        return self.encoder(states)

    def decode(self, latent):
        return self.decoder(latent)


class SpatialManifoldCore(nn.Module):
    def __init__(self, config, schema, mean, scale):
        super().__init__()
        self.config = config
        self.physics = SurfacePhysics(schema, mean, scale)
        if self.physics.grid != config.grid:
            raise ValueError('Spatial configuration and archive field grid must match')
        self.manifold = SpatialManifoldAE(config, self.physics.periodic_lon)
        # Joint training keeps raw latent coordinates. Buffers retain the common
        # bridge/checkpoint interface, without a separate post-training seal.
        self.register_buffer('latent_mean', torch.zeros(config.manifold_dim))
        self.register_buffer('latent_scale', torch.ones(config.manifold_dim))
        self.register_buffer('manifold_ready', torch.tensor(False))

    def decode(self, latent):
        return self.manifold.decode(latent * self.latent_scale + self.latent_mean)


class SpatialClimateManifold(nn.Module):
    def __init__(self, config, schema, mean, scale, statistics, info_metadata=None,
                 pinn_config=None, information_mean=None, information_scale=None):
        super().__init__()
        if config.representation_kind != 'spatial':
            raise ValueError('SpatialClimateManifold requires representation_kind=spatial')
        self.core = SpatialManifoldCore(config, schema, mean, scale)
        self.temporal = TemporalObjective(schema, mean, scale, statistics)
        self.info_metadata = info_metadata
        self.information = self.info_head = None
        if info_metadata is not None:
            shape = tuple(info_metadata['shape'])
            if (len(shape) != 3 or any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1 for n in shape)
                    or shape[1:] != config.grid[1:] or shape[0] != len(info_metadata['variables'])):
                raise ValueError('Information fields must share the source spatial grid and declared variables')
            info_coords = info_metadata.get('grid')
            if info_coords is not None:
                coords = schema['variables'][0]['coords']
                for name in ('lat', 'lon'):
                    if not np.array_equal(np.asarray(info_coords[name]), np.asarray(coords[name])):
                        raise ValueError('Information coordinates must match the source spatial grid')
            kwargs = dict(hidden_dim=config.spatial_hidden_dim, factor=config.spatial_downsample,
                          periodic_lon=self.core.physics.periodic_lon)
            self.information = SpatialEncoder(shape, config.latent_grid, **kwargs)
            self.info_head = SpatialDecoder(config.latent_grid, shape, **kwargs)
        self.pinn = None
        if pinn_config is not None:
            from .hybrid_pinn import HybridPINN, HybridPINNConfig
            if isinstance(pinn_config, dict):
                pinn_config = HybridPINNConfig(**pinn_config)
            if info_metadata is None or information_mean is None or information_scale is None:
                raise ValueError('Hybrid PINN requires enriched information and its training-only normalization')
            # The existing closure consumes flattened latent features. Physical
            # residuals are evaluated on decoded physical variables, never on
            # the learned channel coordinates themselves.
            self.pinn = HybridPINN(pinn_config, info_metadata, information_mean, information_scale,
                                   config.manifold_dim, config.spatial_hidden_dim)
        self.phase = 'A'

    @property
    def config(self):
        return self.core.config

    def raw_encode(self, states, information=None):
        latent = self.core.manifold.encode(states)
        if self.information is not None:
            if information is None or information.shape[:-1] != states.shape[:-1]:
                raise ValueError('Enriched representation requires matching observed origin information')
            latent = latent + self.information(information)
        elif information is not None:
            raise ValueError('Surface-only representation does not accept enriched information')
        return latent

    def encode(self, states, information=None):
        return (self.raw_encode(states, information) - self.core.latent_mean) / self.core.latent_scale
