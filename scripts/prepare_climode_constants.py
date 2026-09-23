"""Align measured orography and land-sea mask to the existing A archive grid."""
import argparse
from pathlib import Path
import numpy as np
import xarray as xr
from prepare_era5_extra import regrid,canonical
from climate_manifold.archive import load_archive
from climate_manifold.downstream.climode import validate_constants


def prepare(archive,fields,output,method='linear'):
    output=Path(output)
    if output.suffix!='.npz':raise ValueError('Constants output must end in .npz')
    if output.exists():raise FileExistsError(output)
    _,_,schema=load_archive(archive)
    coords=schema['variables'][0]['coords'];lat=np.asarray(coords['lat']);lon=np.asarray(coords['lon'])
    with xr.open_dataset(fields) as source:
        ds=canonical(source)
        oro_name=next((n for n in ('orography','terrain_height','z') if n in ds),None)
        mask_name=next((n for n in ('lsm','land_sea_mask') if n in ds),None)
        if oro_name is None or mask_name is None:raise ValueError('Real orography and land-sea mask are both required')
        oro,lsm=ds[oro_name].squeeze(drop=True),ds[mask_name].squeeze(drop=True)
        if set(oro.dims)!={'lat','lon'} or set(lsm.dims)!={'lat','lon'}:
            raise ValueError('Constants must be static lat/lon fields')
        unit=oro.attrs.get('units','')
        if unit in ('m**2 s**-2','m2 s-2','m^2/s^2','m2/s2'):oro=oro/9.80665
        elif unit not in ('m','metres','meters'):raise ValueError('Declare orography height m or geopotential m2/s2 units')
        constants={'lat':lat,'lon':lon,'orography':regrid(oro,lat,lon,method).values,
                   'lsm':regrid(lsm,lat,lon,method).values,'orography_units':'m'}
        validate_constants(constants,schema)
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**constants)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('archive','fields','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--method',choices=['linear','coarsen-match'],default='linear')
    print(prepare(**vars(p.parse_args())))

if __name__=='__main__':main()
