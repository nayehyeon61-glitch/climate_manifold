"""Local elementary sine/cosine initialization compatible with ClimaX.

This module is independently implemented from the positional-encoding formula;
it does not include the upstream MAE-derived helper or interpolation routines.
For d features, frequency k is 10000**(-2*k/d). Features contain all sine
values followed by all cosine values. On a grid, x features precede y features
and positions follow row-major patch ordering.
"""
import numpy as np


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Return [number of positions, embed_dim] for any position array."""
    if embed_dim <= 0 or embed_dim % 2:
        raise ValueError('Sine/cosine embedding width must be positive and even')
    frequencies = np.power(10000., -2. * np.arange(embed_dim // 2) / embed_dim)
    phase = np.asarray(pos, dtype=np.float64).reshape(-1, 1) * frequencies
    return np.concatenate((np.sin(phase), np.cos(phase)), axis=1)


def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False):
    """Encode patch columns then rows, optionally preceded by a zero token."""
    if embed_dim <= 0 or embed_dim % 4:
        raise ValueError('2D sine/cosine embedding width must be divisible by four')
    if grid_size_h <= 0 or grid_size_w <= 0:
        raise ValueError('Embedding grid dimensions must be positive')
    columns = np.tile(np.arange(grid_size_w), grid_size_h)
    rows = np.repeat(np.arange(grid_size_h), grid_size_w)
    encoded = np.concatenate((
        get_1d_sincos_pos_embed_from_grid(embed_dim // 2, columns),
        get_1d_sincos_pos_embed_from_grid(embed_dim // 2, rows),
    ), axis=1)
    if cls_token:
        encoded = np.vstack((np.zeros(embed_dim), encoded))
    return encoded
