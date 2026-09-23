"""Pure A-drift evaluation gates: no residual shortcut or future information."""
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from test_pinn_training import pinn_prepared
from climate_manifold import dynamics_evaluate as evaluation
from climate_manifold.model import FORMAT
from climate_manifold.physical_information import digest, information_digest
from climate_manifold.train import load_checkpoint, write_json


def checkpoint_fixture(prepared, directory, dynamics_metadata=None):
    model, _, data, archive = prepared
    model.seal(torch.tensor((data["states"][:data["train_end"]] - data["mean"]) / data["scale"]),
               torch.tensor(data["information"][:data["train_end"]]))
    information = Path(archive).parent / "pinn-information.npz"
    persisted = {key: value.tolist() if isinstance(value, np.ndarray) else value
                 for key, value in data.items() if key not in ("states", "times", "information")}
    payload = {
        **persisted, "format": FORMAT, "config": asdict(model.config),
        "model": model.state_dict(), "stage": "A", "mode": "enriched",
        "pinn_config": asdict(model.pinn.config),
        "archive_sha256": digest(archive), "information_sha256": information_digest(information),
    }
    if dynamics_metadata is not None:
        payload["dynamics_training"] = dynamics_metadata
    path = directory / "a.pt"
    torch.save(payload, path)
    write_json(path.with_suffix(".manifest.json"), {"checkpoint_sha256": digest(path)})
    return path, archive, information


def test_drift_is_pure_decoding_not_legacy_anchored_rollout(pinn_prepared):
    model, batch, _, _ = pinn_prepared
    model.eval()
    with torch.no_grad():
        forecast = evaluation.forecast_drift(model, batch["origin"], batch["information"], batch["dt_hours"][:, :3])
        latent = model.raw_encode(batch["origin"], batch["information"])
        expected = []
        for _ in range(3):
            latent = latent + .25 * model.core.manifold.latent_drift(latent)
            expected.append(model.core.manifold.decode(latent))
        torch.testing.assert_close(forecast["mean"], torch.stack(expected, 1))
        legacy = model.rollout(batch["history"], batch["information"], members=2, steps=3, drift_only=True)
        offset = batch["origin"] - forecast["reconstructed_origin"]
        assert not torch.allclose(offset, torch.zeros_like(offset))
        torch.testing.assert_close(legacy[:, 0, 1:], forecast["mean"] + offset[:, None])
        assert forecast["std"] is None


@pytest.mark.parametrize("metadata", [None, {"format": "climate_manifold.dynamics_training.v1", "enabled": True, "max_steps": 4}])
def test_legacy_and_new_a_evaluate_and_reload_identically(pinn_prepared, tmp_path, metadata):
    checkpoint, archive, information = checkpoint_fixture(pinn_prepared, tmp_path, metadata)
    output, forecast = tmp_path / "evaluation.json", tmp_path / "forecast.npz"
    report = evaluation.evaluate(checkpoint, archive, output, information=information,
                                 forecast_output=forecast, max_cases=2, steps=3)
    assert report["format"] == "climate_manifold.dynamics_evaluation.v1"
    assert report["mode"] == "pure_decoder_drift"
    assert report["training_dynamics_metadata"] == metadata
    assert report["finite_forecast_fraction"] == 1 and report["ranking_allowed"]
    assert report["scores"]["case_count"] == report["latent_diagnostics"]["case_count"] == 2
    assert report["lead_hours"] == [6, 12, 18]
    assert report["evaluation_seconds"] >= report["inference_seconds"] >= 0
    assert not report["selection_split"]
    assert "gaussian_crps" not in report["scores"]["per_variable"]["msl"]["aggregate"]
    restored, payload = load_checkpoint(checkpoint)
    data = evaluation.data_contract(archive, information, payload["mode"], restored.config, payload)
    row = evaluation.windows(data, restored.config, "validation", max_windows=2)[0]
    with torch.no_grad():
        prediction = evaluation.forecast_drift(restored, row["origin"][None], row["information"][None], row["dt_hours"][None, :3])
    with np.load(forecast, allow_pickle=False) as saved:
        expected = prediction["mean"][0].numpy() * np.asarray(payload["scale"]) + np.asarray(payload["mean"])
        np.testing.assert_array_equal(saved["mean"], expected)
        np.testing.assert_array_equal(saved["predicted_latent"], prediction["predicted_latent"][0].numpy())
        assert saved["diagnostic_target_latent"].shape == (3, restored.config.manifold_dim)
        assert "std" not in saved.files
        assert saved["valid_times"][0] - saved["origin_time"] == np.timedelta64(6, "h")


def test_future_information_never_loaded_even_for_target_encoding(pinn_prepared, tmp_path, monkeypatch):
    checkpoint, archive, information = checkpoint_fixture(pinn_prepared, tmp_path)
    original = evaluation.data_contract
    reads = []

    def origin_only_data(*args, **kwargs):
        data = original(*args, **kwargs)
        origin = data["split"]["validation"][0] + args[3].history_span_steps - 1
        values = data["information"]

        class OriginOnly:
            def __getitem__(self, index):
                assert index == origin, "Future information was accessed"
                reads.append(index)
                return values[index]

        return {**data, "information": OriginOnly()}

    monkeypatch.setattr(evaluation, "data_contract", origin_only_data)
    report = evaluation.evaluate(checkpoint, archive, tmp_path / "causal.json", information=information,
                                 steps=4, max_cases=1)
    assert len(reads) == 1
    assert report["latent_diagnostics"]["case_count"] == 1
    assert report["conditioning"]["future_information_input"] is False


def test_failed_entire_horizon_disables_ranking_and_reports_survivors(pinn_prepared, tmp_path, monkeypatch):
    checkpoint, archive, information = checkpoint_fixture(pinn_prepared, tmp_path)
    original = evaluation.forecast_drift
    calls = []

    def fail_first(*args):
        calls.append(1)
        if len(calls) == 1:
            raise FloatingPointError("nonfinite lead 120h")
        return original(*args)

    monkeypatch.setattr(evaluation, "forecast_drift", fail_first)
    report = evaluation.evaluate(checkpoint, archive, tmp_path / "failed.json", information=information,
                                 max_cases=2, steps=20)
    assert not report["ranking_allowed"]
    assert report["finite_forecast_fraction"] == .5
    assert report["scores"]["case_count"] == 1
    assert report["failed_origins"][0]["origin"] == report["origin_times"][0]
    assert report["successful_origin_times"] == report["origin_times"][1:]


def test_nonfinite_late_output_rejects_whole_forecast(pinn_prepared, monkeypatch):
    model, batch, _, _ = pinn_prepared
    original = evaluation.pure_drift_rollout

    def corrupt_last(*args):
        result = original(*args)
        result["states"] = result["states"].clone()
        result["states"][:, -1, 0] = float("nan")
        return result

    monkeypatch.setattr(evaluation, "pure_drift_rollout", corrupt_last)
    with pytest.raises(FloatingPointError, match="requested horizon"):
        evaluation.forecast_drift(model, batch["origin"], batch["information"], batch["dt_hours"])


def test_horizon_contract_selection_label_and_output_protection(pinn_prepared, tmp_path):
    checkpoint, archive, information = checkpoint_fixture(pinn_prepared, tmp_path)
    with pytest.raises(ValueError, match="horizon exceeds"):
        evaluation.evaluate(checkpoint, archive, tmp_path / "too-long.json", information=information, steps=21)
    output = tmp_path / "selection.json"
    result = evaluation.evaluate(checkpoint, archive, output, information=information,
                                 split="expert_validation", steps=1, max_cases=1)
    assert result["selection_split"]
    with pytest.raises(FileExistsError):
        evaluation.evaluate(checkpoint, archive, output, information=information, steps=1, max_cases=1)
