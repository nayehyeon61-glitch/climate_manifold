"""Evaluate A's deterministic latent drift with pure decoder outputs.

This is a representation/dynamics audit, separate from the origin-anchored A
auxiliary ensemble. Future fields are used for scoring and latent diagnostics
only. In particular, future upper-air information is never read by the loader.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .dynamics import pure_drift_rollout
from .downstream.metrics import ForecastMetrics, LatentDiagnostics
from .downstream.train import windows
from .physical_information import digest
from .train import data_contract, load_checkpoint, write_json


FORMAT = "climate_manifold.dynamics_evaluation.v1"


def forecast_drift(model, origin, information, dt_hours):
    """Pure decoder forecast; neither residual anchoring nor auxiliary noise."""
    rollout = pure_drift_rollout(model, origin, information, dt_hours)
    if not all(torch.isfinite(value).all() for value in rollout.values()
               if isinstance(value, torch.Tensor)):
        raise FloatingPointError("Nonfinite pure decoder drift over requested horizon")
    raw = rollout["raw_latents"]
    q = (raw - model.core.latent_mean) / model.core.latent_scale
    if not torch.isfinite(q).all():
        raise FloatingPointError("Nonfinite standardized drift coordinates")
    return {
        "mean": rollout["states"][:, 1:],
        "std": None,
        "reconstructed_origin": rollout["states"][:, 0],
        "predicted_latent": q[:, 1:],
        "origin_latent": q[:, 0],
    }


def _encode_targets(model, fields, information):
    condition = None if information is None else information[:, None].expand(-1, fields.shape[1], -1)
    return model.encode(fields, condition)


def evaluate(checkpoint, archive, output, *, information=None, split="validation",
             max_cases=0, origin_stride=1, steps=20, device="cpu", forecast_output=None):
    """Score sealed legacy or dynamics-trained A on the same held-out origins."""
    output = Path(output)
    if split not in ("expert_validation", "validation", "test"):
        raise ValueError("Use expert_validation, validation, or test")
    if max_cases < 0 or origin_stride < 1 or steps < 1:
        raise ValueError("Invalid evaluation counts or horizon")
    if output.exists() or (forecast_output and Path(forecast_output).exists()):
        raise FileExistsError("Choose new report/forecast paths")
    if forecast_output and (Path(forecast_output).suffix != ".npz"
                            or Path(forecast_output).resolve() == output.resolve()):
        raise ValueError("Forecast path must be a distinct .npz file")

    model, payload = load_checkpoint(checkpoint, device)
    model.eval().requires_grad_(False)
    if not bool(model.core.manifold_ready):
        raise ValueError("Dynamics evaluation requires sealed A coordinates")
    if steps > model.config.horizon_steps:
        raise ValueError("Evaluation horizon exceeds checkpoint data contract")
    data = data_contract(archive, information, payload["mode"], model.config, payload)
    dataset = windows(data, model.config, split, origin_stride, max_cases)
    lead_hours = (np.arange(1, steps + 1) * model.config.step_hours).tolist()
    metrics = ForecastMetrics(payload["schema"], payload["mean"], payload["scale"], lead_hours)
    latent_metrics = LatentDiagnostics(payload["schema"], lead_hours)
    origins, successful, failures, diagnostic_failures = [], [], [], []
    inference_seconds = 0.
    saved = False
    checkpoint_sha = digest(checkpoint)
    cuda = str(device).startswith("cuda")
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    with torch.no_grad():
        for index in range(len(dataset)):
            batch = {key: value[None].to(device) for key, value in dataset[index].items()}
            origin_ns = int(batch["origin_time_ns"][0].cpu())
            label = str(np.datetime64(origin_ns, "ns")) + "Z"
            origins.append(label)
            dt = batch["dt_hours"][:, :steps]
            if not torch.equal(dt, torch.full_like(dt, model.config.step_hours)):
                raise ValueError("Observed time increments differ from checkpoint physical clock")
            if cuda:
                torch.cuda.synchronize(device)
            forecast_started = time.perf_counter()
            try:
                prediction = forecast_drift(model, batch["origin"], batch.get("information"), dt)
            except FloatingPointError as exc:
                failures.append({"origin": label, "error": str(exc)})
                continue
            finally:
                if cuda:
                    torch.cuda.synchronize(device)
                inference_seconds += time.perf_counter() - forecast_started

            truth = batch["targets"][:, :steps]
            metrics.update(prediction["mean"], truth, batch["origin"],
                           reconstructed_origin=prediction["reconstructed_origin"])
            successful.append(label)
            target_q = None
            # This block runs after the entire forecast. Future information is
            # never requested: both target and cycle encodings use origin I_t.
            try:
                target_q = _encode_targets(model, truth, batch.get("information"))
                reconstruction = model.core.decode(target_q)
                cycle_q = _encode_targets(model, prediction["mean"], batch.get("information"))
                latent_metrics.update(prediction, target_q, reconstruction, cycle_q, truth)
            except FloatingPointError as exc:
                diagnostic_failures.append({"origin": label, "error": str(exc)})
                target_q = None

            if forecast_output and not saved:
                path = Path(forecast_output)
                path.parent.mkdir(parents=True, exist_ok=True)
                origin_time = np.datetime64(origin_ns, "ns")
                mean, scale = np.asarray(payload["mean"]), np.asarray(payload["scale"])
                values = {
                    "mean": prediction["mean"][0].cpu().numpy() * scale + mean,
                    "truth": truth[0].cpu().numpy() * scale + mean,
                    "origin": batch["origin"][0].cpu().numpy() * scale + mean,
                    "reconstructed_origin": prediction["reconstructed_origin"][0].cpu().numpy() * scale + mean,
                    "predicted_latent": prediction["predicted_latent"][0].cpu().numpy(),
                    "origin_latent": prediction["origin_latent"][0].cpu().numpy(),
                    "lead_hours": np.asarray(lead_hours),
                    "valid_times": origin_time + np.asarray(lead_hours).astype("timedelta64[h]"),
                    "origin_time": origin_time,
                    "schema_json": json.dumps(payload["schema"]),
                    "checkpoint_sha256": checkpoint_sha,
                    "mode": "pure_decoder_drift",
                    "latent_coordinates": "sealed q=(raw_z-latent_mean)/latent_scale",
                }
                if target_q is not None:
                    values["diagnostic_target_latent"] = target_q[0].cpu().numpy()
                np.savez_compressed(path, **values)
                saved = True

    if cuda:
        torch.cuda.synchronize(device)
    all_finite = len(successful) == len(origins)
    report = {
        "format": FORMAT,
        "mode": "pure_decoder_drift",
        "checkpoint_sha256": checkpoint_sha,
        "archive_sha256": payload["archive_sha256"],
        "information_sha256": payload["information_sha256"],
        "config": payload["config"],
        "training_dynamics_metadata": payload.get("dynamics_training"),
        "source_commit": payload.get("source_commit"),
        "split": split,
        "selection_split": split == "expert_validation",
        "origin_times": origins,
        "successful_origin_times": successful,
        "lead_hours": lead_hours,
        "finite_forecast_fraction": len(successful) / len(origins),
        "ranking_allowed": all_finite,
        "failed_origins": failures,
        "scores": metrics.result(),
        "latent_diagnostics": latent_metrics.result(),
        "latent_diagnostic_failures": diagnostic_failures,
        "inference_seconds": inference_seconds,
        "evaluation_seconds": time.perf_counter() - started,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "cuda_peak_memory_bytes": torch.cuda.max_memory_allocated(device) if cuda else None,
        "forecast_output_written": saved,
        "conditioning": {
            "origin_information_only": True,
            "future_information_input": False,
            "origin_residual_anchor": False,
            "auxiliary_sampler": False,
        },
        "limits": (
            "Deterministic pure-decoder A drift audit, not the legacy anchored auxiliary ensemble. "
            "Scores are conditional on finite whole-horizon forecasts; any failure disables ranking. "
            "Compare only identical data, splits, origins, and leads. Latent scores use each A's sealed "
            "coordinates and cannot rank different encoders. expert_validation selected A and is not "
            "an untouched test split. Forecast NPZ contains the first successful origin only."
        ),
    }
    write_json(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "archive", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--information")
    parser.add_argument("--forecast-output")
    parser.add_argument("--split", choices=["expert_validation", "validation", "test"], default="validation")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--origin-stride", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    report = evaluate(**vars(parser.parse_args(argv)))
    print(json.dumps(report["scores"]["aggregate"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
