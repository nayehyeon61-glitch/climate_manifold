"""Causal, unanchored A-stage dynamics in raw latent coordinates.

The rollout uses existing A parameters only. Future observations enter loss
targets and PINN masks, never the generated trajectory. Sealing changes the
public q coordinates, not this raw-z training objective.
"""
from __future__ import annotations

import numbers

import torch


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def rollout_steps(epoch, interval, max_steps):
    """Grow 1, 2, 4, ... from the first dynamics-active curriculum phase.

Epochs are joint-A epochs, excluding a preceding PINN closure-only warm-up.
The reconstruction-only first phase still reports a one-step diagnostic.
    """
    epoch = _positive_integer(epoch, "epoch")
    interval = _positive_integer(interval, "interval")
    max_steps = _positive_integer(max_steps, "max_steps")
    exponent = max(0, (epoch - 1) // interval - 1)
    # Cap before exponentiation, including for excessively large epoch values.
    return min(max_steps, 1 << min(exponent, max_steps.bit_length()))


def _validate_dt(model, origin, dt_hours):
    if (origin.ndim != 2 or origin.shape[0] == 0
            or origin.shape[1] != model.config.state_dim or not origin.is_floating_point()):
        raise ValueError("Dynamics origin must be nonempty floating [batch, state_dim]")
    if not bool(torch.isfinite(origin).all()):
        raise ValueError("Dynamics origin must be finite")
    dt = torch.as_tensor(dt_hours, device=origin.device, dtype=origin.dtype)
    if dt.ndim != 2 or dt.shape[0] != len(origin) or not 1 <= dt.shape[1] <= model.config.horizon_steps:
        raise ValueError("Dynamics dt_hours must be [batch, 1..horizon_steps]")
    if (model.config.step_hours != 6 or not bool(torch.isfinite(dt).all())
            or not bool((dt == 6).all())):
        raise ValueError("Dynamics requires canonical consecutive 6h intervals")
    return dt


def _information_contract(model, origin, information):
    if model.info_head is None:
        if information is not None:
            raise ValueError("Surface-only dynamics does not accept information")
        return
    features = model.info_metadata["shape"]
    expected_dim = features[0] * features[1] * features[2]
    if (information is None or information.shape != (len(origin), expected_dim)
            or not information.is_floating_point() or not bool(torch.isfinite(information).all())):
        raise ValueError("Enriched dynamics requires finite matching origin information")


def pure_drift_rollout(model, origin, information, dt_hours):
    """Generate [B,K+1,...] raw latents and decoder outputs without anchoring.

There is no teacher forcing, auxiliary flow, sampler noise, or origin-residual
addition. The origin output is D(E(x_t,I_t)), not the observed x_t.
    """
    dt = _validate_dt(model, origin, dt_hours)
    _information_contract(model, origin, information)
    z = model.raw_encode(origin, information)
    latent_path = [z]
    for step in range(dt.shape[1]):
        z = z + (dt[:, step, None] / 24) * model.core.manifold.latent_drift(z)
        if not bool(torch.isfinite(z).all()):
            raise FloatingPointError("Nonfinite free latent dynamics rollout")
        latent_path.append(z)
    raw_latents = torch.stack(latent_path, dim=1)
    states = model.core.manifold.decode(raw_latents)
    decoded_info = None if model.info_head is None else model.info_head(raw_latents)
    if (not bool(torch.isfinite(states).all())
            or (decoded_info is not None and not bool(torch.isfinite(decoded_info).all()))):
        raise FloatingPointError("Nonfinite pure dynamics decoder output")
    return {"raw_latents": raw_latents, "states": states, "information": decoded_info}


def _validate_path(model, batch, steps, path):
    b, dimension = batch["origin"].shape
    shapes = {"raw_latents": (b, steps + 1, model.config.manifold_dim),
              "states": (b, steps + 1, dimension)}
    if model.info_head is not None:
        shapes["information"] = (b, steps + 1, batch["information"].shape[-1])
    elif path.get("information") is not None:
        raise ValueError("Surface-only dynamics cannot have decoded information")
    for name, expected in shapes.items():
        value = path.get(name)
        if value is None or value.shape != expected:
            raise ValueError(f"Dynamics {name} must have shape {expected}")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"Nonfinite dynamics {name}")


def _future_targets(batch, key, steps, origin):
    target = batch[key]
    if (target.ndim != 3 or target.shape[0] != len(origin) or target.shape[1] < steps
            or target.shape[-1] != origin.shape[-1]):
        raise ValueError(f"Dynamics {key} must contain matching [batch, >=steps, features] targets")
    target = target[:, :steps].detach()
    if not bool(torch.isfinite(target).all()):
        raise ValueError(f"Dynamics {key} must be finite")
    return target


def _tendency_error(model, batch, path, steps):
    truth = _future_targets(batch, "targets", steps, batch["origin"])
    truth = torch.cat((batch["origin"].detach()[:, None], truth), dim=1)
    t = model.temporal
    dt = batch["dt_hours"][:, :steps].to(path["states"])
    return ((path["states"].diff(dim=1) - truth.diff(dim=1))
            * t.scale / dt[:, :, None] / t.tendency_scale)


def dynamics_losses(model, batch, steps, rollout=None):
    """Return (metrics, path) for deterministic multi-step A representation loss.

Latent targets use the same origin information at every future surface state.
Teacher encoding is detached. Future dynamic information is supervised through
the information decoder; static fields always target the origin values.
All reductions average over forecast leads rather than scaling with horizon.
    """
    steps = _positive_integer(steps, "steps")
    if steps > model.config.horizon_steps:
        raise ValueError("Dynamics steps exceeds configured horizon")
    origin = batch["origin"]
    dt_all = batch["dt_hours"]
    if dt_all.ndim != 2 or dt_all.shape[1] < steps:
        raise ValueError("Dynamics dt_hours has fewer intervals than requested")
    dt = _validate_dt(model, origin, dt_all[:, :steps])
    information = batch.get("information")
    _information_contract(model, origin, information)
    truth = _future_targets(batch, "targets", steps, origin)
    path = pure_drift_rollout(model, origin, information, dt) if rollout is None else rollout
    _validate_path(model, batch, steps, path)
    with torch.no_grad():
        fixed_information = (None if information is None
                             else information.detach()[:, None].expand(-1, steps, -1))
        target_latents = model.raw_encode(truth, fixed_information)
    state_error = path["states"][:, 1:] - truth
    tendency_error = _tendency_error(model, batch, path, steps)
    state_per_lead = (state_error.square() * model.temporal.metric).sum(-1).mean(0)
    tendency_per_lead = (tendency_error.square() * model.temporal.metric).sum(-1).mean(0)
    metrics = {
        "latent_dynamics": (path["raw_latents"][:, 1:] - target_latents).square().mean(),
        "decoded_drift": tendency_per_lead.mean(),
        "direct_state": state_per_lead.mean(),
        "dynamics_rollout_steps": origin.new_tensor(float(steps)),
        "dynamics_latent_step_rms": path["raw_latents"].diff(dim=1).detach().square().mean().sqrt(),
    }
    for lead in range(steps):
        hours = (lead + 1) * model.config.step_hours
        metrics[f"dynamics_state_mse_{hours}h"] = state_per_lead[lead].detach()
        metrics[f"dynamics_tendency_mse_{hours}h"] = tendency_per_lead[lead].detach()
    if information is not None:
        future_information = _future_targets(batch, "information_targets", steps, information)
        sh = model.info_metadata["shape"]
        cells = sh[1] * sh[2]
        static = torch.tensor([v["kind"] == "static" for v in model.info_metadata["variables"]],
                              device=origin.device).repeat_interleave(cells)
        area = model.temporal.area.flatten().repeat(sh[0])
        predicted_information = path["information"][:, 1:]
        dynamic_error = (predicted_information - future_information).square() * area
        static_error = (predicted_information - information.detach()[:, None]).square() * area
        metrics["direct_information"] = dynamic_error[:, :, ~static].sum(-1).mean() / max(1, int((~static).sum()) // cells)
        metrics["direct_static"] = static_error[:, :, static].sum(-1).mean() / max(1, int(static.sum()) // cells)
    if any(not bool(torch.isfinite(value).all()) for value in metrics.values()):
        raise FloatingPointError("Nonfinite dynamics objective")
    return metrics, path


def trajectory_pinn_losses(model, batch, rollout):
    """Apply existing Hybrid PINN once to all generated adjacent field pairs.

The latent closure is conditioned on generated z_k, never an encoded future
observation. Observed endpoints enter only detached supervision and masks.
    """
    if model.pinn is None or model.phase != "A":
        raise ValueError("Trajectory PINN requires an enabled A-stage Hybrid PINN")
    states = rollout["states"]
    if states.ndim != 3:
        raise ValueError("Trajectory PINN states must be [batch, steps+1, features]")
    steps = states.shape[1] - 1
    _positive_integer(steps, "steps")
    _validate_dt(model, batch["origin"], batch["dt_hours"][:, :steps])
    _information_contract(model, batch["origin"], batch.get("information"))
    _validate_path(model, batch, steps, rollout)
    information = batch["information"]
    truth = _future_targets(batch, "information_targets", steps, information)
    truth = torch.cat((information.detach()[:, None], truth), dim=1)
    decoded = rollout["information"]
    flatten = lambda x: x.reshape(-1, x.shape[-1])
    values = model.pinn(
        flatten(decoded[:, :-1]), flatten(decoded[:, 1:]),
        flatten(truth[:, :-1]), flatten(truth[:, 1:]),
        flatten(rollout["raw_latents"][:, :-1]), batch["dt_hours"][:, :steps].reshape(-1),
    )
    error = _tendency_error(model, batch, rollout, steps)
    values["pinn_surface_tendency"] = (error.square() * model.temporal.metric).sum(-1).mean()
    values["pinn_total"] = (values["pinn_total"]
                            + model.pinn.config.tendency_weight * values["pinn_surface_tendency"])
    return values
