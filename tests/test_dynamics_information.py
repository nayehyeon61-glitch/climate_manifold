"""Canonical dynamics information ordering, units and observed terrain content."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from climate_manifold.downstream.dynamics_information import (
    resolve_dynamics_information, select_dynamics_information,
)


def metadata():
    return {'shape': [4, 2, 3], 'grid': {'lat': [-30., 30.], 'lon': [0., 120., 240.]},
            'variables': [{'name': n, 'unit': u, 'kind': k} for n, u, k in [
                ('z850', 'm', 'dynamic'), ('t2m', 'K', 'dynamic'),
                ('terrain_height', 'm', 'static'), ('z500', 'm', 'dynamic')]]}


@pytest.mark.parametrize('as_tensor', [False, True])
def test_named_order_preserves_leading_axes_and_spatial_terrain(as_tensor):
    meta = metadata()
    values = np.arange(2*3*24, dtype=np.float32).reshape(2, 3, 24)
    if as_tensor:
        values = torch.from_numpy(values).requires_grad_()
    contract = resolve_dynamics_information(meta, 'geopotential_terrain', (4, 2, 3))
    assert contract['indices'] == [3, 0, 2]
    selected = select_dynamics_information(values, meta, 'geopotential_terrain', raw_grid=(4, 2, 3))
    assert selected.shape == (2, 3, 18)
    actual = selected.detach().numpy() if as_tensor else selected
    source = values.detach().numpy() if as_tensor else values
    np.testing.assert_array_equal(actual, source.reshape(2, 3, 4, 6)[..., [3, 0, 2], :].reshape(2, 3, 18))
    assert np.ptp(actual[..., 12:]) > 0
    if as_tensor:
        selected.sum().backward()
        assert torch.count_nonzero(values.grad.reshape(2, 3, 4, 6)[:, :, 1]) == 0


def test_off_needs_no_information_and_strict_metadata_errors():
    assert resolve_dynamics_information(None, 'off') is None
    assert select_dynamics_information(None, None, 'off') is None
    with pytest.raises(ValueError, match='metadata'):
        resolve_dynamics_information(None, 'geopotential_terrain')
    changed = metadata()
    changed['variables'][3]['unit'] = 'm2/s2'
    with pytest.raises(ValueError, match='canonical metres'):
        resolve_dynamics_information(changed, 'geopotential_terrain')
    changed = metadata()
    changed['variables'][2]['kind'] = 'dynamic'
    with pytest.raises(ValueError, match='kind=static'):
        resolve_dynamics_information(changed, 'geopotential_terrain')
    changed = metadata()
    changed['variables'][3]['name'] = 'z250'
    with pytest.raises(ValueError, match='z500'):
        resolve_dynamics_information(changed, 'geopotential_terrain')
    with pytest.raises(ValueError, match='spatial grid'):
        resolve_dynamics_information(metadata(), 'geopotential_terrain', (4, 3, 2))
    with pytest.raises(ValueError, match='materialized'):
        select_dynamics_information(object(), metadata(), 'geopotential_terrain')
