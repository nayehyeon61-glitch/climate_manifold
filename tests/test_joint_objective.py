"""Joint auxiliary supervision follows F's trajectory, without target leakage."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from test_pinn_training import pinn_prepared
from climate_manifold.downstream.joint_objective import (
    JointObjectiveWeights, joint_losses, spatial_quantile_loss,
)


def setup_path(manifold, batch, steps=2):
    """Small actual differentiable F with no dependence on future targets."""
    predictor = torch.nn.Linear(manifold.config.manifold_dim, manifold.config.manifold_dim)
    q = manifold.encode(batch['origin'], batch['information'])
    origin = q
    future = []
    for _ in range(steps):
        q = q + .25 * predictor(q)
        future.append(q)
    future = torch.stack(future, dim=1)
    prediction = {'mean': manifold.core.decode(future), 'predicted_latent': future,
                  'origin_latent': origin, 'reconstructed_origin': manifold.core.decode(origin)}
    pipe = SimpleNamespace(bridge=SimpleNamespace(manifold=manifold, mode='latent', anchor='none'))
    return pipe, predictor, prediction


def no_aux(**changes):
    return replace(JointObjectiveWeights(reconstruction=0., physics=0., information=0., static=0.), **changes)


def test_joint_information_and_pinn_follow_actual_forecaster(pinn_prepared, monkeypatch):
    manifold, batch, _, _ = pinn_prepared
    pipe, predictor, prediction = setup_path(manifold, batch)
    batch = {**batch, 'information_targets': batch['information_targets'].clone().requires_grad_()}
    def forbidden(*args, **kwargs):
        raise AssertionError('Separate A dynamics/sampler entered the joint objective')
    monkeypatch.setattr(manifold, 'rollout', forbidden)
    monkeypatch.setattr(manifold.core.manifold.latent_drift, 'forward', forbidden)
    weights = no_aux(information=.1, static=.05, pinn=.01)
    values = joint_losses(pipe, prediction, batch, weights, torch.tensor([6., 12.]))
    values['regularization'].backward()
    groups = (manifold.core.manifold.encoder, manifold.core.manifold.decoder,
              manifold.information, manifold.info_head, predictor, manifold.pinn.closure_head)
    for group in groups:
        gradients = [parameter.grad for parameter in group.parameters() if parameter.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(gradient.abs().sum() for gradient in gradients) > 0
    assert batch['information_targets'].grad is None
    assert all(parameter.grad is None for parameter in manifold.a_sampler.parameters())
    assert all(parameter.grad is None for parameter in manifold.a_context.parameters())


def test_future_information_changes_supervision_not_generated_trajectory(pinn_prepared):
    manifold, batch, _, _ = pinn_prepared
    pipe, predictor, prediction = setup_path(manifold, batch)
    before = prediction['mean'].detach().clone()
    weights = no_aux(information=1.)
    baseline = joint_losses(pipe, prediction, batch, weights, torch.tensor([6., 12.]))
    altered = {**batch, 'information_targets': batch['information_targets'] + 10.}
    changed = joint_losses(pipe, prediction, altered, weights, torch.tensor([6., 12.]))
    assert not torch.equal(baseline['information_future'], changed['information_future'])
    torch.testing.assert_close(before, prediction['mean'], rtol=0, atol=0)
    torch.testing.assert_close(baseline['information_origin'], changed['information_origin'])


def test_static_supervision_uses_origin_terrain_not_future_static_targets(pinn_prepared):
    manifold, batch, _, _ = pinn_prepared
    pipe, _, prediction = setup_path(manifold, batch)
    weights = no_aux(static=1.)
    # Static supervision can operate with no future upper-air target at all.
    causal_batch = {key: value for key, value in batch.items() if key != 'information_targets'}
    values = joint_losses(pipe, prediction, causal_batch, weights, torch.tensor([6., 12.]))
    assert values['static'] > 0
    altered = {**causal_batch, 'information_targets': torch.full_like(batch['information_targets'], float('nan'))}
    unchanged = joint_losses(pipe, prediction, altered, weights, torch.tensor([6., 12.]))
    torch.testing.assert_close(values['static'], unchanged['static'])


def test_all_auxiliary_weights_off_require_no_future_information_or_bridge():
    output = {'mean': torch.randn(2, 2, 4, requires_grad=True)}
    values = joint_losses(None, output, {}, no_aux(), torch.tensor([6., 12.]))
    assert values['regularization'] == 0
    assert not values['regularization'].requires_grad


def test_reconstruction_and_surface_physics_use_history_and_forecast(pinn_prepared):
    manifold, batch, _, _ = pinn_prepared
    pipe, predictor, prediction = setup_path(manifold, batch)
    weights = no_aux(reconstruction=.1, physics=.01)
    values = joint_losses(pipe, prediction, batch, weights, torch.tensor([6., 12.]))
    expected = manifold.core.physics.reconstruction_losses(prediction['mean'], batch['targets'][:, :2])['physics']
    torch.testing.assert_close(values['physics'], expected)
    values['regularization'].backward()
    assert predictor.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in manifold.info_head.parameters())


def test_quantile_marginal_preserves_area_and_does_not_claim_spatial_location():
    area = torch.tensor([.25, .75])
    prediction = torch.tensor([[0., 2.]], requires_grad=True)
    truth = torch.tensor([[0., 0.]])
    score = spatial_quantile_loss(prediction, truth, area, quantiles=4)
    # Three quarters of geographic mass has value two.
    assert score.item() == pytest.approx(3.)
    score.backward()
    assert prediction.grad[0, 1] > 0
    # Equal-area permutation preserves the spatial marginal while changing locations.
    assert spatial_quantile_loss(torch.tensor([[1., 2.]]), torch.tensor([[2., 1.]]), torch.ones(2)) == 0


def test_quantile_loss_separates_variables_and_leads():
    prediction = torch.tensor([[[[1., 1.], [4., 4.]], [[2., 2.], [8., 8.]]]])
    truth = torch.zeros_like(prediction)
    assert spatial_quantile_loss(prediction, truth, torch.ones(2)) == (1 + 16 + 4 + 64) / 4


def test_invalid_auxiliary_contracts_fail_explicitly(pinn_prepared):
    manifold, batch, _, _ = pinn_prepared
    pipe, _, prediction = setup_path(manifold, batch)
    with pytest.raises(ValueError, match='consecutive 6h'):
        joint_losses(pipe, prediction, batch, no_aux(pinn=1.), torch.tensor([6., 18.]))
    pipe.bridge.mode = 'decoded'
    with pytest.raises(ValueError, match='latent bridge'):
        joint_losses(pipe, prediction, batch, no_aux(information=1.), torch.tensor([6., 12.]))
    for value in (-1, float('nan'), True):
        with pytest.raises(ValueError, match='finite and nonnegative'):
            no_aux(pinn=value)
