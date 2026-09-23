"""Regression gates for causal, unanchored, multi-step A dynamics."""
import copy

import pytest
import torch

from test_pinn_training import pinn_prepared
from climate_manifold.architecture import drift_per_day
from climate_manifold.dynamics import (
    dynamics_losses, pure_drift_rollout, rollout_steps, trajectory_pinn_losses,
)
from climate_manifold.model import ClimateManifold


def _nonzero_gradients(module):
    gradients = [p.grad for p in module.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum() for g in gradients) > 0


def test_dynamics_curriculum_starts_with_short_free_forecasts():
    assert [rollout_steps(epoch, 2, 4) for epoch in range(1, 11)] == [1, 1, 1, 1, 2, 2, 4, 4, 4, 4]
    assert rollout_steps(10**100, 1, 20) == 20
    assert rollout_steps(5, 1, 3) == 3
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            rollout_steps(bad, 1, 4)
        with pytest.raises(ValueError, match="positive integer"):
            rollout_steps(1, bad, 4)
        with pytest.raises(ValueError, match="positive integer"):
            rollout_steps(1, 1, bad)


def test_free_rollout_has_no_target_conditioning_anchor_or_sampler(pinn_prepared):
    model, batch, _, _ = pinn_prepared
    original_parameter_names = set(dict(model.named_parameters()))
    metrics, path = dynamics_losses(model, batch, 4)
    changed = dict(batch)
    changed["targets"] = batch["targets"] + 25
    changed["information_targets"] = batch["information_targets"] - 17
    changed_metrics, changed_path = dynamics_losses(model, changed, 4)
    for name in path:
        torch.testing.assert_close(path[name], changed_path[name], rtol=0, atol=0)
    assert not torch.isclose(metrics["latent_dynamics"], changed_metrics["latent_dynamics"])
    assert not torch.isclose(metrics["direct_state"], changed_metrics["direct_state"])
    z = model.raw_encode(batch["origin"], batch["information"])
    torch.testing.assert_close(path["states"][:, 0], model.core.manifold.decode(z))
    assert not torch.allclose(path["states"][:, 0], batch["origin"])
    for step in range(4):
        z = z + .25 * model.core.manifold.latent_drift(z)
        torch.testing.assert_close(z, path["raw_latents"][:, step + 1])
    sum(metrics[name] for name in ("latent_dynamics", "decoded_drift", "direct_state", "direct_information")).backward()
    assert set(dict(model.named_parameters())) == original_parameter_names
    assert all(p.grad is None for p in model.a_sampler.parameters())
    assert all(p.grad is None for p in model.a_context.parameters())
    for module in (model.core.manifold.encoder, model.core.manifold.decoder,
                   model.core.manifold.latent_drift, model.information, model.info_head):
        _nonzero_gradients(module)


def test_future_teacher_uses_origin_information_and_is_detached(pinn_prepared, monkeypatch):
    model, batch, _, _ = pinn_prepared
    future_surface = batch["targets"].detach().requires_grad_(True)
    future_info = batch["information_targets"].detach().requires_grad_(True)
    batch = {**batch, "targets": future_surface, "information_targets": future_info}
    raw_encode = model.raw_encode
    calls = []

    def record(state, information=None):
        calls.append((state.detach().clone(), information.detach().clone(), torch.is_grad_enabled()))
        return raw_encode(state, information)

    monkeypatch.setattr(model, "raw_encode", record)
    metrics, _ = dynamics_losses(model, batch, 4)
    assert len(calls) == 2
    assert calls[0][2]
    assert not calls[1][2]
    torch.testing.assert_close(calls[1][0], future_surface[:, :4])
    torch.testing.assert_close(calls[1][1], batch["information"][:, None].expand(-1, 4, -1))
    assert not torch.allclose(calls[1][1], future_info[:, :4])
    sum(metrics[k] for k in ("latent_dynamics", "direct_state", "direct_information", "direct_static", "decoded_drift")).backward()
    assert future_surface.grad is None
    assert future_info.grad is None
    _nonzero_gradients(model.core.manifold.encoder)


def test_one_step_matches_previous_latent_tendency_and_pinn_formulas(pinn_prepared):
    model, batch, _, _ = pinn_prepared
    values, path = dynamics_losses(model, batch, 1)
    x, y = batch["origin"], batch["targets"][:, 0]
    information = batch["information"]
    z = model.raw_encode(x, information)
    next_z = z + batch["dt_hours"][:, 0, None] / 24 * model.core.manifold.latent_drift(z)
    zy = model.raw_encode(y, information)
    t = model.temporal
    surface0, surface1 = model.core.manifold.decode(z), model.core.manifold.decode(next_z)
    error = ((surface1 - surface0) - (y - x)) * t.scale / batch["dt_hours"][:, 0, None] / t.tendency_scale
    expected_tendency = (error.square() * t.metric).sum(-1).mean()
    torch.testing.assert_close(values["latent_dynamics"], (next_z - zy.detach()).square().mean())
    torch.testing.assert_close(values["decoded_drift"], expected_tendency)
    torch.testing.assert_close(values["direct_state"], ((surface1-y).square()*t.metric).sum(-1).mean())
    expected_pinn = model.pinn(model.info_head(z), model.info_head(next_z), information,
                               batch["information_targets"][:, 0], z, batch["dt_hours"][:, 0])
    expected_pinn["pinn_surface_tendency"] = expected_tendency
    expected_pinn["pinn_total"] = expected_pinn["pinn_total"] + model.pinn.config.tendency_weight * expected_tendency
    actual = trajectory_pinn_losses(model, batch, path)
    assert actual.keys() == expected_pinn.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected_pinn[name], rtol=2e-5, atol=1e-7)


def test_late_pair_pinn_gradients_reach_initial_encoder_and_existing_dynamics(pinn_prepared):
    model, batch, _, _ = pinn_prepared
    _, path = dynamics_losses(model, batch, 4)
    path["raw_latents"].retain_grad()
    # Supervise only the final generated pair while retaining its graph all the
    # way back to the initial encoding; no intermediate truth is substituted.
    final_pair = {key: value[:, -2:] for key, value in path.items()}
    final_batch = {**batch, "origin": batch["targets"][:, 2],
                   "targets": batch["targets"][:, 3:4],
                   "information": batch["information_targets"][:, 2],
                   "information_targets": batch["information_targets"][:, 3:4],
                   "dt_hours": batch["dt_hours"][:, 3:4]}
    values = trajectory_pinn_losses(model, final_batch, final_pair)
    values["pinn_total"].backward()
    assert path["raw_latents"].grad[:, -1].abs().sum() > 0
    for module in (model.core.manifold.encoder, model.core.manifold.decoder,
                   model.core.manifold.latent_drift, model.information, model.info_head, model.pinn):
        _nonzero_gradients(module)
    assert all(p.grad is None for p in model.a_sampler.parameters())


def test_static_future_targets_cannot_redefine_fixed_terrain(pinn_prepared):
    model, batch, _, _ = pinn_prepared
    metrics, _ = dynamics_losses(model, batch, 4)
    cells = model.info_metadata["shape"][1] * model.info_metadata["shape"][2]
    static = torch.tensor([v["kind"] == "static" for v in model.info_metadata["variables"]]).repeat_interleave(cells)
    changed = copy.copy(batch)
    changed["information_targets"] = batch["information_targets"].clone()
    changed["information_targets"][:, :, static] += 1234
    other, _ = dynamics_losses(model, changed, 4)
    torch.testing.assert_close(metrics["direct_static"], other["direct_static"], rtol=0, atol=0)
    torch.testing.assert_close(metrics["direct_information"], other["direct_information"], rtol=0, atol=0)
    torch.testing.assert_close(metrics["latent_dynamics"], other["latent_dynamics"], rtol=0, atol=0)


def test_rollout_reductions_average_over_leads(pinn_prepared):
    model, batch, _, _ = pinn_prepared
    metrics, path = dynamics_losses(model, batch, 4)
    weights = model.temporal.metric
    individual = [((path["states"][:, k+1]-batch["targets"][:, k]).square()*weights).sum(-1).mean()
                  for k in range(4)]
    torch.testing.assert_close(metrics["direct_state"], torch.stack(individual).mean())
    assert not torch.isclose(metrics["direct_state"], torch.stack(individual).sum())
    assert float(metrics["dynamics_rollout_steps"]) == 4


def test_raw_rollout_is_unchanged_by_sealing_and_matches_q_drift(pinn_prepared):
    model, batch, data, _ = pinn_prepared
    before = pure_drift_rollout(model, batch["origin"], batch["information"], batch["dt_hours"][:, :4])
    model.seal(torch.tensor((data["states"][:data["train_end"]]-data["mean"])/data["scale"]),
               torch.tensor(data["information"][:data["train_end"]]))
    after = pure_drift_rollout(model, batch["origin"], batch["information"], batch["dt_hours"][:, :4])
    for name in before:
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
    q = model.encode(batch["origin"], batch["information"])
    for step in range(4):
        q = q + .25 * drift_per_day(model.core, q)
        torch.testing.assert_close(model.core.decode(q), after["states"][:, step+1], rtol=2e-5, atol=2e-6)


def test_surface_only_dynamics_and_input_contract(pinn_prepared):
    original, batch, data, _ = pinn_prepared
    model = ClimateManifold(original.config, data["schema"], data["mean"], data["scale"], data["statistics"])
    batch = {k: v for k, v in batch.items() if not k.startswith("information")}
    metrics, path = dynamics_losses(model, batch, 3)
    assert path["information"] is None
    assert "direct_information" not in metrics and "direct_static" not in metrics
    for dt in (torch.empty(2, 0), torch.full((2, 21), 6.), torch.full((2, 1), 12.), torch.tensor([[6.], [float("nan")]])):
        with pytest.raises(ValueError):
            pure_drift_rollout(model, batch["origin"], None, dt)
    with pytest.raises(ValueError, match="steps"):
        dynamics_losses(model, batch, 21)
    with pytest.raises(ValueError, match="finite"):
        pure_drift_rollout(model, batch["origin"] * float("nan"), None, batch["dt_hours"][:, :1])
    with pytest.raises(ValueError, match="information"):
        pure_drift_rollout(original, batch["origin"], None, batch["dt_hours"][:, :1])
    bad_path = dict(path)
    bad_path["states"] = path["states"][:, :-1]
    with pytest.raises(ValueError, match="shape"):
        dynamics_losses(model, batch, 3, rollout=bad_path)
