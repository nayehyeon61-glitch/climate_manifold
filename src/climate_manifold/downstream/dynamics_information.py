"""Strict observed geopotential/terrain channels for the optional E_d input.

Inputs have already been normalized with TRAIN-only per-channel area statistics
by the information archive. Geopotential heights and terrain are metres before
normalization; terrain retains its spatial pattern and is a static condition.
"""
from __future__ import annotations

import math
from numbers import Integral

import numpy as np
import torch

DYNAMICS_INFORMATION_FIELDS = ('z500', 'z850', 'terrain_height')
DYNAMICS_INFORMATION_MODES = ('off', 'geopotential_terrain')


def resolve_dynamics_information(metadata, mode, raw_grid=None):
    if mode not in DYNAMICS_INFORMATION_MODES:
        raise ValueError('Dynamics information must be off or geopotential_terrain')
    if mode == 'off':
        return None
    if not isinstance(metadata, dict):
        raise ValueError('Dynamics geopotential_terrain requires information metadata')
    variables, shape, grid = metadata.get('variables'), metadata.get('shape'), metadata.get('grid')
    if (not isinstance(variables, list) or not isinstance(shape, (list, tuple)) or len(shape) != 3
            or any(isinstance(i, bool) or not isinstance(i, Integral) or i < 1 for i in shape)
            or shape[0] != len(variables) or not isinstance(grid, dict)):
        raise ValueError('Invalid dynamics information grid/shape metadata')
    if raw_grid is not None and tuple(shape[1:]) != tuple(raw_grid[-2:]):
        raise ValueError('Dynamics information must match the observed raw spatial grid')
    for axis, size in zip(('lat', 'lon'), shape[1:]):
        coords = np.asarray(grid.get(axis, []))
        if coords.shape != (size,) or not np.issubdtype(coords.dtype, np.number) or not np.isfinite(coords).all():
            raise ValueError('Invalid dynamics information grid coordinates: '+axis)
    names = [variable.get('name') for variable in variables]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate dynamics information channel names')
    missing = set(DYNAMICS_INFORMATION_FIELDS)-set(names)
    if missing:
        raise ValueError('Missing required dynamics information fields: '+', '.join(sorted(missing)))
    indices = [names.index(name) for name in DYNAMICS_INFORMATION_FIELDS]
    for name, index in zip(DYNAMICS_INFORMATION_FIELDS, indices):
        variable = variables[index]
        kind = 'static' if name == 'terrain_height' else 'dynamic'
        if variable.get('unit') != 'm' or variable.get('kind') != kind:
            raise ValueError(f'Dynamics information {name} requires canonical metres and kind={kind}')
    return {'mode': mode, 'fields': list(DYNAMICS_INFORMATION_FIELDS), 'indices': indices,
            'grid': [3, *shape[1:]], 'dynamic_indices': [0, 1], 'static_indices': [2],
            'units': ['m', 'm', 'm']}


def select_dynamics_information(values, metadata, mode, *, raw_grid=None):
    """Select materialized archive values, preserving leading axes and type.

    For a sharded information archive first read one integer or contiguous slice
    and then call this helper; it never assumes advanced shard indexing works.
    """
    contract = resolve_dynamics_information(metadata, mode, raw_grid)
    if contract is None:
        return None
    if not isinstance(values, (np.ndarray, torch.Tensor)):
        raise ValueError('Select dynamics information from a materialized numpy/tensor slice')
    shape = metadata['shape']
    if values.ndim < 1 or values.shape[-1] != math.prod(shape):
        raise ValueError('Dynamics information values disagree with flattened archive shape')
    if not (bool(torch.isfinite(values).all()) if isinstance(values, torch.Tensor)
            else bool(np.isfinite(values).all())):
        raise ValueError('Dynamics information values must be finite')
    leading = values.shape[:-1]
    grouped = values.reshape(*leading, shape[0], shape[1]*shape[2])
    selected = grouped[..., contract['indices'], :]
    return selected.reshape(*leading, math.prod(contract['grid']))
