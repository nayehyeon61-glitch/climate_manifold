import math

import numpy as np
import pytest
import torch

from climate_manifold._vendor.climax.pos_embed import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
)
from climate_manifold.downstream.climax import ClimaXPredictor


def test_position_formula_frequencies_axis_order_and_odd_variable_count():
    # Three variables do not need padding: each gets an eight-feature vector.
    variables = get_1d_sincos_pos_embed_from_grid(8, [0., 1., 2.])
    frequencies = [1., .1, .01, .001]
    expected = np.array([
        [math.sin(position * f) for f in frequencies]
        + [math.cos(position * f) for f in frequencies]
        for position in (0., 1., 2.)
    ])
    np.testing.assert_allclose(variables, expected, rtol=1e-14, atol=1e-15)
    # A rectangular grid exposes any row/column or flattening-order swap.
    patches = get_2d_sincos_pos_embed(8, 2, 3)
    expected_patches = np.array([
        [math.sin(col), math.sin(col / 100.), math.cos(col), math.cos(col / 100.),
         math.sin(row), math.sin(row / 100.), math.cos(row), math.cos(row / 100.)]
        for row in range(2) for col in range(3)
    ])
    np.testing.assert_allclose(patches, expected_patches, rtol=1e-14, atol=1e-15)
    with_prefix = get_2d_sincos_pos_embed(8, 2, 3, cls_token=True)
    np.testing.assert_array_equal(with_prefix[0], np.zeros(8))
    np.testing.assert_array_equal(with_prefix[1:], patches)


@pytest.mark.parametrize('grid,information_channels', [((4, 16, 32), 2), ((32, 8, 16), 0)])
def test_climax_raw_and_spatial_latent_train(grid, information_channels):
    """Both comparison arms execute the official backbone and backpropagate."""
    torch.manual_seed(23)
    model = ClimaXPredictor(grid, 3, hidden=32, depth=2,
                           information_channels=information_channels)
    history = torch.randn(2, 3, math.prod(grid), requires_grad=True)
    information = (torch.randn(2, information_channels * grid[1] * grid[2], requires_grad=True)
                   if information_channels else None)
    leads = torch.tensor([6., 24., 120.])
    origins = torch.tensor([0, 6 * 3600 * 10**9], dtype=torch.int64)
    prediction, uncertainty = model(history, leads, origins, information)
    assert prediction.shape == (2, 3, math.prod(grid))
    assert uncertainty is None and torch.isfinite(prediction).all()
    before = model.model.var_agg.in_proj_weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    prediction.square().mean().backward()
    assert history.grad[:, -1].abs().sum() > 0
    assert history.grad[:, :-1].abs().sum() == 0
    if information is not None:
        assert information.grad.abs().sum() > 0
    assert model.model.token_embeds[0].proj.weight.grad.abs().sum() > 0
    assert model.model.blocks[0].attn.qkv.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    assert not torch.equal(before, model.model.var_agg.in_proj_weight)


def test_climax_last_state_direct_leads_and_checkpoint_roundtrip():
    torch.manual_seed(5)
    model = ClimaXPredictor((2, 4, 8), 3, hidden=16, depth=1,
                           variable_names=('t2m', 'msl')).eval()
    history = torch.randn(1, 3, 64)
    origins = torch.tensor([0])
    leads = torch.tensor([6., 24.])
    first, _ = model(history, leads, origins)
    altered_history = history.clone()
    altered_history[:, :-1] += 100
    second, _ = model(altered_history, leads, origins + 10**15)
    torch.testing.assert_close(first, second)
    single, _ = model(history, leads[-1:], origins)
    torch.testing.assert_close(first[:, -1:], single)
    assert not torch.allclose(first[:, 0], first[:, 1])
    copied = ClimaXPredictor((2, 4, 8), 3, hidden=16, depth=1,
                            variable_names=('t2m', 'msl')).eval()
    copied.load_state_dict(model.state_dict())
    torch.testing.assert_close(copied(history, leads, origins)[0], first)


def test_climax_singleton_patch_and_input_information_contract():
    model = ClimaXPredictor((1, 2, 2), 1, hidden=8, depth=1)
    output, _ = model(torch.zeros(1, 1, 4), torch.tensor([6.]), torch.tensor([0]))
    assert output.shape == (1, 1, 4)
    with pytest.raises(ValueError, match='encoder only'):
        model(torch.zeros(1, 1, 4), torch.tensor([6.]), torch.tensor([0]),
              information=torch.zeros(1, 4))
    raw = ClimaXPredictor((1, 2, 2), 1, hidden=8, depth=1, information_channels=1)
    with pytest.raises(ValueError, match='origin information'):
        raw(torch.zeros(1, 1, 4), torch.tensor([6.]), torch.tensor([0]),
            information=torch.zeros(1, 2, 4))


@pytest.mark.parametrize('kwargs,match', [
    ({'hidden': 18}, 'divisible by four'),
    ({'latent_grid': (2, 5, 8)}, 'divisible by patch_size'),
    ({'patch_size': 0}, 'positive integers'),
    ({'depth': 0}, 'positive integers'),
    ({'variable_names': ('same', 'same')}, 'unique'),
])
def test_climax_invalid_config(kwargs, match):
    config = dict(latent_grid=(2, 4, 8), history_steps=2, hidden=16, depth=1)
    config.update(kwargs)
    with pytest.raises(ValueError, match=match):
        ClimaXPredictor(**config)
