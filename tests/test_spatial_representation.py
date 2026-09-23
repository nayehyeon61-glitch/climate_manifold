"""Spatial capacity, locality, boundaries, and E/D supervision contracts."""
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_pinn_training import pinn_prepared
from climate_manifold.architecture import ManifoldConfig
from climate_manifold.downstream.joint_objective import JointObjectiveWeights, joint_losses
from climate_manifold.spatial import (
    GeographicConv2d, SpatialClimateManifold, SpatialDecoder, SpatialEncoder,
    downsample_latlon,
)


def spatial_fixture(previous, data, **changes):
    config = replace(previous.config, representation_kind='spatial', latent_channels=8,
                     spatial_hidden_dim=12, **changes)
    return SpatialClimateManifold(config, data['schema'], data['mean'], data['scale'],
        data['statistics'], data['information_metadata'], pinn_config=previous.pinn.config,
        information_mean=data['information_mean'], information_scale=data['information_scale'])


def test_old_config_remains_global_and_spatial_capacity_is_not_global_dimension():
    old = {'state_dim': 128, 'grid': [4, 4, 8], 'manifold_dim': 16}
    config = ManifoldConfig(**old)
    assert config.representation_kind == 'global'
    assert config.latent_grid is None
    assert config.manifold_dim == 16
    spatial = replace(config, representation_kind='spatial', latent_channels=32)
    assert spatial.latent_grid == (32, 2, 4)
    assert spatial.manifold_dim == 256 > spatial.state_dim
    assert ManifoldConfig(**asdict(spatial)) == spatial
    with pytest.raises(ValueError, match='manifold_dim < state_dim'):
        replace(config, manifold_dim=256)


@pytest.mark.parametrize('field,value', [('latent_channels', 0), ('latent_channels', 1.5),
    ('spatial_downsample', True), ('spatial_downsample', float('inf')),
    ('spatial_hidden_dim', float('nan')), ('history_steps', 2.5)])
def test_spatial_dimensions_are_finite_positive_integers(field, value):
    with pytest.raises(ValueError, match='integral'):
        ManifoldConfig(state_dim=128, grid=(4, 4, 8), representation_kind='spatial', **{field: value})


def test_spatial_grid_must_support_both_coordinate_derivatives():
    with pytest.raises(ValueError, match='at least two'):
        ManifoldConfig(state_dim=128, grid=(4, 4, 8), representation_kind='spatial', spatial_downsample=4)


@pytest.mark.parametrize('grid,factor', [((4, 5, 7), 2), ((4, 7, 11), 3), ((4, 4, 8), 1)])
def test_odd_spatial_sizes_restore_exact_grid_and_arbitrary_leading_axes(grid, factor):
    torch.set_num_threads(1)
    config = ManifoldConfig(state_dim=int(np.prod(grid)), grid=grid, representation_kind='spatial',
                            latent_channels=6, spatial_downsample=factor)
    encoder = SpatialEncoder(grid, config.latent_grid, 8, factor, False)
    decoder = SpatialDecoder(config.latent_grid, grid, 8, factor, False)
    states = torch.randn(2, 3, config.state_dim, requires_grad=True)
    z = encoder(states)
    assert z.shape == (2, 3, config.manifold_dim)
    reconstruction = decoder(z)
    assert reconstruction.shape == states.shape
    reconstruction.square().mean().backward()
    assert torch.isfinite(states.grad).all() and states.grad.abs().sum() > 0
    assert not any(isinstance(module, torch.nn.Linear) for module in encoder.modules())
    assert not any(isinstance(module, torch.nn.Linear) for module in decoder.modules())
    assert encoder(states[0, 0]).shape == (config.manifold_dim,)
    assert decoder(z[0, 0]).shape == (config.state_dim,)


def test_periodic_longitude_only_wraps_global_grid_and_latitude_never_wraps():
    global_conv = GeographicConv2d(1, 1, periodic_lon=True)
    regional_conv = GeographicConv2d(1, 1, periodic_lon=False)
    with torch.no_grad():
        for module in (global_conv, regional_conv):
            module.conv.weight.fill_(1.)
            module.conv.bias.zero_()
    impulse = torch.zeros(1, 1, 7, 8)
    impulse[0, 0, 3, 0] = 1
    assert global_conv(impulse)[0, 0, 3, -1] == 1
    assert regional_conv(impulse)[0, 0, 3, -1] == 0
    impulse.zero_()
    impulse[0, 0, 0, 3] = 1
    assert global_conv(impulse)[0, 0, -1, 3] == 0
    assert regional_conv(impulse)[0, 0, -1, 3] == 0


def test_spatial_encoder_does_not_mix_remote_cells_into_global_vector():
    encoder = SpatialEncoder((1, 20, 24), (3, 10, 12), 4, 2, False)
    original = torch.zeros(1, 20*24)
    changed = original.clone().reshape(1, 1, 20, 24)
    changed[0, 0, 8, 8] = 1.
    difference = (encoder(changed.flatten(1)) - encoder(original)).reshape(1, 3, 10, 12)
    assert difference[..., 2:6, 2:6].abs().sum() > 0
    assert torch.count_nonzero(difference[..., 8:, :]) == 0
    assert torch.count_nonzero(difference[..., :, 8:]) == 0


def test_coarse_coordinates_follow_partial_pooling_bins():
    lat, lon = downsample_latlon([40., 30., 20., 10., 0.], [0., 45., 90., 135., 180., 225., 270.], 2)
    np.testing.assert_array_equal(lat, [35., 15., 0.])
    np.testing.assert_array_equal(lon, [22.5, 112.5, 202.5, 270.])


def test_enriched_spatial_forecast_information_and_pinn_have_joint_gradients(pinn_prepared):
    old, batch, data, _ = pinn_prepared
    model = spatial_fixture(old, data)
    model.core.physics.fit(batch['origin'])
    assert model.phase == 'A'
    torch.testing.assert_close(model.core.latent_mean, torch.zeros(model.config.manifold_dim))
    torch.testing.assert_close(model.core.latent_scale, torch.ones(model.config.manifold_dim))
    info = batch['information'][:, None].expand(-1, batch['history'].shape[1], -1)
    history = model.encode(batch['history'], info)
    assert history.shape == (*batch['history'].shape[:2], model.config.manifold_dim)
    predictor = torch.nn.Conv2d(model.config.latent_channels, model.config.latent_channels, 1)
    q = history[:, -1]
    origin = q
    future = []
    for _ in range(2):
        q = q + .25 * predictor(q.reshape(-1, *model.config.latent_grid)).flatten(1)
        future.append(q)
    future = torch.stack(future, 1)
    output = dict(mean=model.core.decode(future), predicted_latent=future, origin_latent=origin,
                  reconstructed_origin=model.core.decode(origin))
    pipe = SimpleNamespace(bridge=SimpleNamespace(manifold=model, mode='latent', anchor='none'))
    values = joint_losses(pipe, output, batch,
                         JointObjectiveWeights(physics=.01, information=.1, static=.05, pinn=.01),
                         torch.tensor([6., 12.]))
    loss = values['regularization'] + (output['mean'] - batch['targets'][:, :2]).square().mean()
    loss.backward()
    for module in (model.core.manifold.encoder, model.core.manifold.decoder, model.information,
                   model.info_head, model.pinn.closure_head, predictor):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients) > 0
    assert not hasattr(model.core.manifold, 'latent_drift')
    assert not hasattr(model, 'a_sampler')
    assert model.info_head(future).shape == (2, 2, batch['information'].shape[-1])


def test_spatial_requires_matching_observed_information(pinn_prepared):
    old, batch, data, _ = pinn_prepared
    model = spatial_fixture(old, data)
    with pytest.raises(ValueError, match='observed origin'):
        model.encode(batch['origin'])
    with pytest.raises(ValueError, match='observed origin'):
        model.encode(batch['history'], batch['information'])
    bad = {**data['information_metadata'], 'shape': [3, 5, 8]}
    with pytest.raises(ValueError, match='source spatial grid'):
        SpatialClimateManifold(model.config, data['schema'], data['mean'], data['scale'], data['statistics'], bad)
    surface = SpatialClimateManifold(model.config, data['schema'], data['mean'], data['scale'], data['statistics'])
    with pytest.raises(ValueError, match='Surface-only'):
        surface.encode(batch['origin'], batch['information'])


def test_same_spatial_representation_initialization_is_predictor_independent(pinn_prepared):
    old, _, data, _ = pinn_prepared
    torch.manual_seed(191)
    first = spatial_fixture(old, data)
    torch.manual_seed(191)
    second = spatial_fixture(old, data)
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)
