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
# (name, unit, kind); dynamic fields first, then static spatial conditions.
_MODE_FIELDS = {
    'geopotential_terrain': (('z500', 'm', 'dynamic'), ('z850', 'm', 'dynamic'),
                             ('terrain_height', 'm', 'static')),
    # Every information-archive channel: all pressure-level fields and both terrain conditions.
    'pressure_terrain': (('z850', 'm', 'dynamic'), ('z500', 'm', 'dynamic'), ('z250', 'm', 'dynamic'),
                         ('u850', 'm/s', 'dynamic'), ('v850', 'm/s', 'dynamic'),
                         ('t850', 'K', 'dynamic'), ('t500', 'K', 'dynamic'),
                         ('terrain_height', 'm', 'static'), ('terrain_slope', '1', 'static')),
}
DYNAMICS_INFORMATION_MODES = ('off', *_MODE_FIELDS)


def dynamics_information_fields(mode):
    """Ordered field names for a mode (empty for off)."""
    if mode not in DYNAMICS_INFORMATION_MODES:
        raise ValueError('Dynamics information must be one of '+', '.join(DYNAMICS_INFORMATION_MODES))
    return [] if mode == 'off' else [name for name, _, _ in _MODE_FIELDS[mode]]


def dynamics_information_counts(mode):
    """(dynamic, static) channel counts for a mode."""
    if mode == 'off':
        return 0, 0
    kinds = [kind for _, _, kind in _MODE_FIELDS[mode]]
    return kinds.count('dynamic'), kinds.count('static')


def resolve_dynamics_information(metadata, mode, raw_grid=None):
    if mode not in DYNAMICS_INFORMATION_MODES:
        raise ValueError('Dynamics information must be one of '+', '.join(DYNAMICS_INFORMATION_MODES))
    if mode == 'off':
        return None
    if not isinstance(metadata, dict):
        raise ValueError('Dynamics '+mode+' requires information metadata')
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
    specs = _MODE_FIELDS[mode]
    fields = [name for name, _, _ in specs]
    missing = set(fields)-set(names)
    if missing:
        raise ValueError('Missing required dynamics information fields: '+', '.join(sorted(missing)))
    indices = [names.index(name) for name in fields]
    for (name, unit, kind), index in zip(specs, indices):
        variable = variables[index]
        if variable.get('unit') != unit or variable.get('kind') != kind:
            units = 'canonical metres' if unit == 'm' else f'unit={unit}'
            raise ValueError(f'Dynamics information {name} requires {units} and kind={kind}')
    dynamic, static = dynamics_information_counts(mode)
    return {'mode': mode, 'fields': fields, 'indices': indices,
            'grid': [len(fields), *shape[1:]], 'dynamic_indices': list(range(dynamic)),
            'static_indices': list(range(dynamic, dynamic+static)),
            'units': [unit for _, unit, _ in specs]}


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
