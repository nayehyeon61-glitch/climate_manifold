"""Synthetic fields for software tests, not evidence of forecast skill."""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from climate_manifold.fixed_step_data import prepare_fixed_step_archive
from climate_manifold.archive import load_archive
from climate_manifold.physical_information import prepare

def synthetic_archive(directory, seed=19, count=480):
    """Two toy propagation modes, shared across all four fields; labels for audit only."""
    rng = np.random.default_rng(seed)
    lat = np.linspace(-67.5, 67.5, 4)
    lon = np.arange(8) * 45.0
    yy, xx = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    # Alternating directions affect the FULL state; labels are never training input.
    regime = ((np.arange(count) // 40) % 2).astype(int)
    increments = np.where(regime == 0, 0.13, -0.18)
    phase = np.cumsum(increments)
    latent = []
    for i in range(count):
        wave = np.cos(yy) * np.sin(xx - phase[i])
        secondary = np.sin(2 * xx + 0.7 * phase[i]) * np.cos(2 * yy)
        north = np.sin(yy) * np.cos(xx - phase[i])
        latent.append(np.stack((wave + 0.25 * secondary, -0.6 * wave + 0.5 * north,
                                np.cos(yy) * np.cos(xx - phase[i]), north + 0.3 * secondary)))
    fields = np.asarray(latent) + rng.normal(0, 0.06, (count, 4, 4, 8))
    # Plausible-looking units are cosmetic; these are NOT simulated physical weather.
    fields = fields * np.array([900, 6, 8, 8])[None, :, None, None]
    fields += np.array([101000, 285, 0, 0])[None, :, None, None]
    dataset = xr.Dataset({name: (("time", "lat", "lon"), fields[:, j].astype(np.float32))
                          for j, name in enumerate(("msl", "t2m", "u10", "v10"))},
                         coords={"time": pd.date_range("2001-01-01", periods=count, freq="6h"),
                                 "lat": lat, "lon": lon})
    raw = directory / "synthetic-fields.nc"
    dataset.to_netcdf(raw, engine="scipy")
    archive, _ = prepare_fixed_step_archive(raw, directory / "synthetic-states.npz", step_hours=6,
                                           target_lat_points=4, target_lon_points=8)
    return archive, regime


def synthetic_information(archive,directory):
    states,times,schema=load_archive(archive)
    coords=schema['variables'][0]['coords'];lat=np.array(coords['lat']);lon=np.array(coords['lon'])
    fields=states.reshape(-1,4,len(lat),len(lon));wave=(fields[:,0]-101000)/900
    data={name:(('time','lat','lon'),value.astype('float32')) for name,value in
          dict(z850=1500+80*wave,z500=5500+100*wave,z250=10500+150*wave,
               u850=1.5*fields[:,2],v850=1.4*fields[:,3]).items()}
    data['terrain_height']=(('lat','lon'),(500+400*np.cos(np.deg2rad(lat))[:,None]*np.cos(np.deg2rad(lon))[None]).astype('float32'))
    ds=xr.Dataset(data,coords={'time':times,'lat':lat,'lon':lon})
    for name in ds:ds[name].attrs['units']='m/s' if name.startswith(('u','v')) else 'm'
    raw=directory/'synthetic-information.nc';ds.to_netcdf(raw,engine='scipy')
    return prepare(archive,raw,directory/'information.npz')


def synthetic_pinn_information(archive, directory):
    """Add matched pressure-level T/u/v/omega and true surface pressure.

    Preserve the existing synthetic-information helper and its Z/terrain inputs.
    The generated legacy sidecar also permits testing that missing PINN inputs
    are rejected instead of filled with zeros.
    """
    directory = Path(directory)
    synthetic_information(archive, directory)
    with xr.open_dataset(directory / "synthetic-information.nc") as source:
        fields = source.load()
    wave = (fields["z850"].values - 1500.0) / 80.0
    t500 = 255.0 + 3.0 * wave
    t850 = 280.0 + 4.0 * wave
    extras = {
        "t500": (t500, "K"),
        "t850": (t850, "K"),
        "u500": (1.25 * fields["u850"].values + 3.0, "m/s"),
        "v500": (1.15 * fields["v850"].values, "m/s"),
        "w500": (0.02 * wave, "Pa/s"),
        "w850": (0.03 * wave, "Pa/s"),
        "sp": (
            101000.0 - 11.7 * fields["terrain_height"].values[None]
            + 500.0 * wave,
            "Pa",
        ),
    }
    for name, (values, units) in extras.items():
        fields[name] = (("time", "lat", "lon"), values.astype(np.float32))
        fields[name].attrs["units"] = units
    # A reasonable dry layer thickness, not an exact discretized PDE solution.
    fields["z500"] = (
        ("time", "lat", "lon"),
        (fields["z850"].values
         + 287.05 / 9.80665 * 0.5 * (t500 + t850) * np.log(850.0 / 500.0))
        .astype(np.float32),
    )
    fields["z500"].attrs["units"] = "m"
    raw = directory / "synthetic-pinn-information.nc"
    fields.to_netcdf(raw, engine="scipy")
    return prepare(archive, raw, directory / "pinn-information.npz", pinn=True)
