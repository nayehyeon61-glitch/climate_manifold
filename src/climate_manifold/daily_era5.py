"""Read daily ERA5 files in place; write resumable, coarse 24h model archives.

Only derived outputs below the supplied root are written. Each raw file is
opened separately, and only selected fields/pressure levels are pooled. A date
label at 00 UTC does not establish whether the source is an instantaneous or
daily-aggregated field: source cell_methods are retained without reinterpretation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .data import _normalise_coordinates, _coarsen_global_fields
from .physical_information import FORMAT, PRIMARY, canonical_unit, digest, terrain_slope

SURFACE = ('msl', 't2m', 'u10', 'v10')
INFORMATION = (*PRIMARY, 't850', 't500')
DYNAMIC = tuple(name for name in INFORMATION if not name.startswith('terrain_'))
PREPARATION_FORMAT = 'climate_manifold.daily_era5_preparation.v1'


def _json(path, value):
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def _npz(path, **values):
    temporary = path.with_suffix(path.suffix+'.partial')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def _inside(root, path):
    path = Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Path must remain inside {root}: {path}')
    return path


def _coordinates(ds):
    ds = _normalise_coordinates(ds)
    if any(axis not in ds.coords or ds[axis].dims != (axis,) for axis in ('lat','lon')):
        raise ValueError('Daily ERA5 requires one-dimensional latitude/longitude coordinates')
    ds = ds.assign_coords(lon=ds.lon % 360).sortby('lat').sortby('lon')
    for axis in ('lat','lon'):
        values = ds[axis].values
        if not np.isfinite(values).all() or len(values)<3 or not np.all(np.diff(values)>0):
            raise ValueError(f'Invalid or duplicate {axis} coordinates')
    return ds


def _convert(array, name, source_name, overrides):
    unit = overrides.get(source_name, overrides.get(name, array.attrs.get('units')))
    if unit is None:
        raise ValueError(f'{source_name}: units missing; inspect the source and provide --units-json explicitly')
    compact = str(unit).replace(' ', '').replace('²','2').replace('¹','1').replace('⁻','-').lower()
    attrs = dict(array.attrs)
    if name=='msl':
        if compact in ('hpa','mbar','millibars'): array = array*100
        elif compact not in ('pa','pascal','pascals'): raise ValueError(f'{source_name}: unsupported pressure unit {unit}')
    elif name.startswith('z') or name=='terrain_height':
        if compact in ('m**2s**-2','m2s-2','m^2/s^2','m**2s**(-2)','m^2s^-2','m2/s2'):
            array = array/9.80665
        elif compact not in ('m','meter','metre','meters','metres'):
            raise ValueError(f'{source_name}: unsupported geopotential/height unit {unit}')
    elif name.startswith(('u','v')):
        if compact not in ('m/s','ms**-1','ms-1','ms^-1'):
            raise ValueError(f'{source_name}: unsupported wind unit {unit}')
    elif name.startswith('t'):
        if compact not in ('k','kelvin'):
            raise ValueError(f'{source_name}: temperature must be declared in kelvin')
    target = 'Pa' if name=='msl' else canonical_unit(name)
    array.attrs = {**attrs, 'units': target, 'source_units': str(unit)}
    return array


def _field(ds, name, overrides):
    source = 'mslp' if name=='msl' and 'mslp' in ds else name
    if source in ds:
        value = ds[source]
    elif name[0] in 'zutv' and name[1:].isdigit() and name[0] in ds:
        source = name[0]
        value = ds[source]
        dimension = next((d for d in ('pressure_level','level','isobaricInhPa','plev') if d in value.dims), None)
        if dimension is None: raise ValueError(f'{name}: no pressure-level dimension')
        levels = value[dimension]
        unit = str(levels.attrs.get('units','')).lower()
        target = int(name[1:])
        if unit in ('pa','pascals'): target *= 100
        elif unit not in ('hpa','millibars','mbar',''):
            raise ValueError(f'Unsupported pressure coordinate units: {unit}')
        if not unit:
            # The required ERA5 levels distinguish hPa and Pa numerically.
            if target not in levels.values and target*100 in levels.values: target *= 100
        if target not in levels.values: raise ValueError(f'Missing pressure level for {name}')
        value = value.sel({dimension: target}, drop=True)
    else:
        raise ValueError(f'Missing required source variable: {name}')
    if set(value.dims) != {'time','lat','lon'}:
        raise ValueError(f'{name}: expected time/lat/lon after pressure selection, got {value.dims}')
    return _convert(value, name, source, overrides).reset_coords(drop=True)


def prepare(root, output, *, start='1979-01-01', end='2025-12-31',
            target_lat_points=16, target_lon_points=32, orography=None, units=None):
    root = Path(root).resolve(strict=True)
    daily = _inside(root, root/'daily')
    output = _inside(root, output)
    if output == root or output.is_relative_to(daily):
        raise ValueError('Derived output must be a separate subdirectory, never the raw daily directory')
    if min(target_lat_points,target_lon_points)<4:
        raise ValueError('Use at least four cells per spatial dimension')
    overrides = {} if units is None else dict(units)
    oro = _inside(root, orography or root/'era5_orography_0p25.nc')
    if not oro.is_file():
        raise FileNotFoundError(f'Provide --orography pointing to the existing file under root: {oro}')
    dates = pd.date_range(start,end,freq='D')
    if len(dates)<2: raise ValueError('Need at least two days')
    files = [_inside(root,daily/(date.strftime('%Y%m%d')+'.nc')) for date in dates]
    missing = [str(path) for path in files if not path.is_file()]
    if missing: raise FileNotFoundError(f'Missing daily files ({len(missing)}): {missing[:5]}')
    inventory = [{'path':str(path.relative_to(root)), 'size':path.stat().st_size,
                  'mtime_ns':path.stat().st_mtime_ns} for path in files]
    plan = {'format':PREPARATION_FORMAT, 'root':str(root), 'start':str(dates[0]), 'end':str(dates[-1]),
            'files':inventory, 'source_identity':'relative path, size and mtime; raw contents are not cryptographically hashed',
            'orography':str(oro), 'orography_sha256':digest(oro), 'unit_overrides':overrides,
            'target_grid':[target_lat_points,target_lon_points], 'step_hours':24,
            'surface_variables':list(SURFACE), 'information_variables':list(INFORMATION)}
    identity = hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True,exist_ok=True)
    plan_path = output/'preparation_plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError('Existing preparation plan differs; use a new output directory')
    elif any(output.iterdir()):
        raise FileExistsError('Refuse to overwrite an unowned nonempty preparation directory')
    else: _json(plan_path,plan)
    manifest_path = output/'preparation_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, expected in manifest['output_sha256'].items():
            if digest(output/name) != expected: raise ValueError(f'Prepared file changed: {name}')
        print('Reusing verified daily archives:',output,flush=True)
        return output/'surface.npz', output/'information.npz'

    with xr.open_dataset(oro) as source:
        source = _coordinates(source)
        height = source['terrain_height'] if 'terrain_height' in source else source['z']
        if 'time' in height.dims:
            if height.sizes['time'] != 1: raise ValueError('Orography must be static')
            height = height.isel(time=0,drop=True)
        if set(height.dims) != {'lat','lon'}: raise ValueError('Orography requires lat/lon only')
        height = _convert(height,'terrain_height','terrain_height',overrides)
        raw_lat,raw_lon = source.lat.values.copy(),source.lon.values.copy()
        pooled = _coarsen_global_fields(height.to_dataset(name='terrain_height'),target_lat_points,target_lon_points).load()
    coords = {axis:pooled[axis].values.tolist() for axis in ('lat','lon')}
    terrain = pooled.terrain_height.transpose('lat','lon').values.astype(np.float32)
    if not np.isfinite(terrain).all(): raise ValueError('Nonfinite pooled terrain')
    slope = terrain_slope(terrain,np.array(coords['lat']),np.array(coords['lon'])).astype(np.float32)
    shape = terrain.shape
    cache = output/'monthly_cache'; cache.mkdir(exist_ok=True)
    months = sorted(set(dates.strftime('%Y-%m')))
    state_chunks,info_chunks,source_attrs = [],[],None
    for month in months:
        indices = np.flatnonzero(dates.strftime('%Y-%m')==month)
        target = cache/(month+'.npz')
        if not target.exists():
            states,information,attributes = [],[],None
            for i in indices:
                with xr.open_dataset(files[i]) as source:
                    source = _coordinates(source)
                    if source.sizes.get('time') != 1:
                        raise ValueError(f'{files[i]}: expected one daily time sample')
                    actual = np.asarray(source.time.values).astype('datetime64[ns]')[0]
                    if actual != dates[i].to_datetime64():
                        raise ValueError(f'{files[i]}: time does not equal the filename date at 00 UTC')
                    if not np.array_equal(source.lat.values,raw_lat) or not np.array_equal(source.lon.values,raw_lon):
                        raise ValueError(f'{files[i]}: source grid differs from orography')
                    fields = {name:_field(source,name,overrides) for name in (*SURFACE,*DYNAMIC)}
                    if attributes is None:
                        attributes = {name:{k:str(v) for k,v in field.attrs.items()} for name,field in fields.items()}
                    fields = _coarsen_global_fields(xr.Dataset(fields),target_lat_points,target_lon_points).load()
                    surface = np.stack([fields[n].transpose('time','lat','lon').values[0] for n in SURFACE]).astype(np.float32)
                    dynamic = {n:fields[n].transpose('time','lat','lon').values[0] for n in DYNAMIC}
                    dynamic.update(terrain_height=terrain,terrain_slope=slope)
                    extra = np.stack([dynamic[n] for n in INFORMATION]).astype(np.float32)
                    if not np.isfinite(surface).all() or not np.isfinite(extra).all():
                        raise ValueError(f'{files[i]}: missing/nonfinite pooled cells; no imputation')
                    states.append(surface.reshape(-1)); information.append(extra.reshape(-1))
            _npz(target,states=np.stack(states),information=np.stack(information),
                 times=dates[indices].values,plan_sha256=identity,attrs_json=json.dumps(attributes))
        with np.load(target,allow_pickle=False) as saved:
            if str(saved['plan_sha256']) != identity or not np.array_equal(saved['times'],dates[indices].values):
                raise ValueError(f'Invalid cached month: {month}')
            state_chunks.append(saved['states']); info_chunks.append(saved['information'])
            if source_attrs is None: source_attrs=json.loads(str(saved['attrs_json']))
        print(f'Prepared {month}: {indices[-1]+1}/{len(dates)} days',flush=True)
    states,information = np.concatenate(state_chunks),np.concatenate(info_chunks)
    del state_chunks,info_chunks
    times = dates.values.astype('datetime64[ns]')
    cells = int(np.prod(shape)); variables=[]
    for i,name in enumerate(SURFACE):
        variables.append({'name':name,'dims':['lat','lon'],'shape':list(shape),
            'slice':[i*cells,(i+1)*cells],'coords':coords,'attrs':source_attrs[name]})
    feature_names = [f'field:{name}:{i}' for name in SURFACE for i in range(cells)]
    schema = {'format':'climate_diffusion.fixed_step_state.v1','source_fields':str(daily),
        'source_integrated':None,'state_dim':states.shape[1],'field_dim':states.shape[1],
        'integrated_feature_names':[],'variables':variables,'forecast_step_hours':24,
        'state_time_semantics':'source_daily_value_at_recorded_time',
        'aggregation':'spatial_block_mean_only; no temporal resampling or interpolation',
        'missing_value_policy':'finite pooled cells required; no label imputation',
        'target_lat_points':target_lat_points,'target_lon_points':target_lon_points,
        'preparation_plan_sha256':identity}
    archive = output/'surface.npz'
    _npz(archive,states=states,observed_mask=np.ones(states.shape,dtype=np.uint8),times=times,
         feature_names=np.asarray(feature_names))
    _json(archive.with_suffix('.schema.json'),schema)
    metadata = {'format':FORMAT,'surface_sha256':digest(archive),'source_sha256':identity,
        'variables':[{'name':n,'unit':canonical_unit(n),
            'kind':'static' if n.startswith('terrain_') else 'dynamic',
            'pressure_hpa':int(n[1:]) if n[0] in 'zutv' and n[1:].isdigit() else None} for n in INFORMATION],
        'grid':coords,'shape':[len(INFORMATION),*shape],
        'time_policy':'UTC exact 24h; origin information fixed throughout forecast',
        'msl':'already in surface input; not duplicated'}
    info = output/'information.npz'
    _npz(info,data=information,observed_mask=np.ones(information.shape,dtype=np.uint8),
         times=times,metadata_json=json.dumps(metadata))
    _json(manifest_path,{'format':PREPARATION_FORMAT,'plan_sha256':identity,'days':len(dates),
        'grid':list(shape),'forecast_step_hours':24,
        'output_sha256':{p.name:digest(p) for p in (archive,archive.with_suffix('.schema.json'),info)},
        'unused_source_fields':'lsm, q, other levels and unselected surface variables are not inputs to this experiment'})
    print('Daily preparation complete:',output,flush=True)
    return archive,info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--start',default='1979-01-01')
    parser.add_argument('--end',default='2025-12-31')
    parser.add_argument('--orography')
    parser.add_argument('--target-lat-points',type=int,default=16)
    parser.add_argument('--target-lon-points',type=int,default=32)
    parser.add_argument('--units-json',help='Explicit source-unit overrides only when source metadata is missing/incorrect')
    args=parser.parse_args(argv)
    prepare(args.root,args.output,start=args.start,end=args.end,orography=args.orography,
            target_lat_points=args.target_lat_points,target_lon_points=args.target_lon_points,
            units=json.loads(args.units_json) if args.units_json else None)


if __name__=='__main__':
    main()
