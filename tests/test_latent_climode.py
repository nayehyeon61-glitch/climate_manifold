"""Numerical/causality gates for the spatial-latent ClimODE adaptation."""
import numpy as np
import pytest
import torch

from climate_manifold.downstream.latent_climode import (
    LatentClimODEPredictor, latent_transport, spatial_derivative,
)


def schema(h=4, w=8, *, global_lon=True):
    return {'variables': [{'name': 'msl', 'shape': [h, w], 'coords': {
        'lat': np.linspace(-75, 75, h).tolist(),
        'lon': (np.arange(w)*(360/w if global_lon else 2.)).tolist(),
    }}]}


def predictor(**kwargs):
    torch.set_num_threads(1)
    return LatentClimODEPredictor((3, 2, 4), schema(), hidden=8,
                                  spatial_factor=2, **kwargs)


def test_climode_transport_sign_and_constant_periodic_state():
    # Positive upstream div(v*z) moves material toward decreasing x: a rising
    # ramp has a positive interior tendency, not its negative.
    state = torch.arange(6.).view(1, 1, 1, 6).expand(1, 2, 4, 6)
    velocity = torch.cat((torch.ones_like(state), torch.zeros_like(state)), 1)
    derivative = latent_transport(state, velocity)
    torch.testing.assert_close(derivative[..., 1:-1], torch.ones_like(state[..., 1:-1]))
    torch.testing.assert_close(latent_transport(torch.ones_like(state), velocity, True),
                               torch.zeros_like(state))


@pytest.mark.parametrize('periodic', [True, False])
def test_variable_transport_conserves_latent_sum_and_cfl_step_is_positive(periodic):
    torch.manual_seed(14)
    state = torch.rand(2, 3, 4, 6)
    velocity = 2*torch.randn(2, 6, 4, 6).tanh()
    tendency = latent_transport(state, velocity, periodic)
    torch.testing.assert_close(tendency.sum((-2, -1)), torch.zeros(2, 3), atol=2e-6, rtol=0)
    # |vx|+|vy|<=4 cells/day and dt=.125 day gives the implemented CFL=.5.
    assert (state+.125*tendency).min() >= 0


def test_geographic_coordinate_pooling_matches_encoder_partial_blocks():
    model = LatentClimODEPredictor((2, 3, 4), schema(5, 7), hidden=8, spatial_factor=2)
    np.testing.assert_allclose(model.pooled_lat, [-56.25, 18.75, 75.])
    lon = np.asarray(schema(5, 7)['variables'][0]['coords']['lon'])
    np.testing.assert_allclose(model.pooled_lon, [lon[:2].mean(), lon[2:4].mean(),
                                                lon[4:6].mean(), lon[6]], rtol=1e-6)
    assert model.periodic_lon
    regional = LatentClimODEPredictor((2, 2, 4), schema(global_lon=False), hidden=8, spatial_factor=2)
    assert not regional.periodic_lon


def test_invalid_grid_and_global_vector_contract_are_rejected():
    with pytest.raises(ValueError, match='spatial_factor'):
        LatentClimODEPredictor((2, 2, 4), schema())
    with pytest.raises(ValueError, match='match the encoder'):
        LatentClimODEPredictor((2, 3, 4), schema(), spatial_factor=2)
    with pytest.raises(ValueError, match='H,W>=2'):
        LatentClimODEPredictor((64, 1, 1), schema(), spatial_factor=4)
    duplicate = schema()
    duplicate['variables'][0]['coords']['lon'] = np.linspace(0, 360, 8).tolist()
    with pytest.raises(ValueError, match='duplicate'):
        LatentClimODEPredictor((2, 2, 4), duplicate, spatial_factor=2)
    model = predictor()
    with pytest.raises(ValueError, match='observed spatial latents'):
        model(torch.randn(1, 6, 64), torch.tensor([6.]), torch.zeros(1, dtype=torch.long))


def test_forecast_gradients_reach_every_history_and_velocity_network():
    torch.manual_seed(7)
    model = predictor(step_hours=3.)
    history = torch.randn(2, 6, 24, requires_grad=True)
    origins = torch.tensor([978307200000000000, 978393600000000000])
    future, std = model(history, torch.tensor([6., 12.]), origins)
    assert future.shape == (2, 2, 24) and std is None
    (future-torch.randn_like(future)).square().mean().backward()
    assert torch.isfinite(history.grad).all()
    assert (history.grad.abs().sum((0, 2)) > 0).all()
    for module in (model.history_encoder, model.initial_velocity, model.velocity_dynamics):
        gradients = [p.grad for p in module.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients) > 0
    assert not torch.equal(future[:, 0], future[:, 1])


def test_solver_sampling_does_not_change_trajectory_and_eval_is_deterministic():
    torch.manual_seed(8)
    model = predictor(step_hours=1.).eval()
    history = torch.randn(2, 4, 24)
    origins = torch.tensor([978307200000000000, 978393600000000000])
    with torch.no_grad():
        dense, _ = model(history, torch.tensor([6., 12.]), origins)
        single, _ = model(history, torch.tensor([12.]), origins)
        repeat, _ = model(history, torch.tensor([6., 12.]), origins)
    torch.testing.assert_close(dense[:, -1], single[:, -1], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(dense, repeat, atol=0, rtol=0)


def test_full_120h_rollout_is_finite_under_saturated_velocity():
    torch.manual_seed(12)
    model = predictor(step_hours=6.).eval()
    # An intentionally strong initialization saturates the velocity bound and
    # makes the internal CFL cap (3h) stricter than requested 6h solver steps.
    with torch.no_grad():
        model.initial_velocity.bias.fill_(10.)
        history = torch.rand(2, 6, 24)
        future, std = model(history, torch.arange(6., 121., 6), torch.zeros(2, dtype=torch.long))
    assert future.shape == (2, 20, 24) and std is None and torch.isfinite(future).all()
    initial_sum = history[:, -1].reshape(2, 3, 2, 4).sum((-2, -1))
    future_sum = future.reshape(2, 20, 3, 2, 4).sum((-2, -1))
    torch.testing.assert_close(future_sum, initial_sum[:, None].expand_as(future_sum), atol=5e-6, rtol=1e-6)
    assert future.min() >= 0


def test_spatial_conditioning_derivative_wraps_global_seam_only():
    ramp = torch.arange(4.).reshape(1, 1, 1, 4).expand(1, 1, 2, 4)
    torch.testing.assert_close(spatial_derivative(ramp, -1), torch.ones_like(ramp))
    periodic = spatial_derivative(ramp, -1, True)
    torch.testing.assert_close(periodic[..., 0], -torch.ones_like(periodic[..., 0]))
