"""Shared field/archive I/O; no forecast model dependencies."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import xarray as xr

COORDINATE_ALIASES = {
    "valid_time": "time",
    "datetime": "time",
    "latitude": "lat",
    "longitude": "lon",
}


def _open_dataset(path: str | Path) -> xr.Dataset:
    value = str(path)
    return xr.open_zarr(value) if value.endswith(".zarr") else xr.open_dataset(value)


def _normalise_coordinates(dataset: xr.Dataset) -> xr.Dataset:
    rename = {
        old: new
        for old, new in COORDINATE_ALIASES.items()
        if (old in dataset.coords or old in dataset.dims)
        and new not in dataset.coords
        and new not in dataset.dims
    }
    return dataset.rename(rename)


def _json_values(values: np.ndarray) -> list[Any]:
    result = []
    for value in values.tolist():
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        result.append(value)
    return result


def _coarsen_global_fields(
    dataset: xr.Dataset,
    target_lat_points: int,
    target_lon_points: int,
) -> xr.Dataset:
    factors = {}
    if "lat" in dataset.dims:
        factors["lat"] = max(1, int(np.ceil(dataset.sizes["lat"] / target_lat_points)))
    if "lon" in dataset.dims:
        factors["lon"] = max(1, int(np.ceil(dataset.sizes["lon"] / target_lon_points)))
    return dataset.coarsen(factors, boundary="pad").mean(skipna=True) if factors else dataset


def load_state_archive(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    archive_path = Path(path)
    with np.load(archive_path, allow_pickle=False) as archive:
        states = archive["states"].astype(np.float32)
        times = archive["times"].astype("datetime64[ns]")
    schema = json.loads(
        archive_path.with_suffix(".schema.json").read_text(encoding="utf-8")
    )
    if states.ndim != 2 or states.shape[1] != schema["state_dim"]:
        raise ValueError("Monthly archive does not match its schema")
    return states, times, schema
