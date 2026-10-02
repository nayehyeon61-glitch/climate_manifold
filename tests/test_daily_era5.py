"""Daily source units, causal time contracts and supported forecasting routes."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from climate_manifold.archive import load_archive
from climate_manifold.daily_era5 import prepare
from climate_manifold.physical_information import digest, load_information
from climate_manifold.downstream.train import parser, train, load_predictor, initialize_manifold, windows
from climate_manifold.downstream.evaluate import evaluate
from climate_manifold.downstream.compare import compare


def daily_sources(root, count=90):
    daily=root/'daily'; daily.mkdir(parents=True)
    lat=np.linspace(80,-80,8); lon=np.arange(16)*22.5
    yy,xx=np.meshgrid(np.deg2rad(lat),np.deg2rad(lon),indexing='ij')
    levels=np.array([1000,925,850,700,600,500,400,300,250,200,150,100,50])
    terrain=400+80*np.cos(xx)*np.cos(yy)
    oro=xr.Dataset({'z':(('latitude','longitude'),terrain*9.80665)},
                   coords={'latitude':lat,'longitude':lon})
    oro.z.attrs['units']='m² s⁻²'
    oro.to_netcdf(root/'era5_orography_0p25.nc',engine='scipy')
    for i,date in enumerate(pd.date_range('2001-01-01',periods=count,freq='D')):
        wave=np.cos(yy)*np.sin(xx-i*.15)
        fields={'mslp':1000+10*wave,'t2m':280+4*wave,'u10':6*wave,'v10':4*np.cos(xx-i*.15)}
        values={name:(('time','latitude','longitude'),field[None].astype('float32')) for name,field in fields.items()}
        for name in ('z','u','v','t'):
            stack=np.stack([(12000-p*10+20*wave)*9.80665 if name=='z' else
                            260+p*.02+wave if name=='t' else 5*wave+p*.001 for p in levels])
            values[name]=(('time','level','latitude','longitude'),stack[None].astype('float32'))
        ds=xr.Dataset(values,coords={'time':[date],'level':levels,'latitude':lat,'longitude':lon})
        ds.level.attrs['units']='hPa'
        for name in ds.data_vars:
            ds[name].attrs['units']='hPa' if name=='mslp' else 'K' if name.startswith('t') else 'm**2 s**-2' if name=='z' else 'm s**-1'
        ds.mslp.attrs['cell_methods']='time: mean'
        ds.to_netcdf(daily/date.strftime('%Y%m%d.nc'),engine='scipy')
    return root


@pytest.fixture(scope='module')
def daily_prepared(tmp_path_factory):
    root=daily_sources(tmp_path_factory.mktemp('daily'))
    output=root/'derived'
    original={p.name:digest(p) for p in (root/'daily').glob('*.nc')}
    archive,info=prepare(root,output,start='2001-01-01',end='2001-03-31',target_lat_points=4,target_lon_points=8)
    return root,archive,info,original


def test_daily_conversion_units_source_preservation_and_resume(daily_prepared,monkeypatch):
    root,archive,info,original=daily_prepared
    states,times,schema=load_archive(archive)
    assert states.shape==(90,4*4*8)
    assert schema['forecast_step_hours']==24
    assert np.all(np.diff(times)==np.timedelta64(24,'h'))
    assert states[:,:32].mean()==pytest.approx(100000,rel=1e-5)
    assert schema['variables'][0]['attrs']['units']=='Pa'
    assert schema['variables'][0]['attrs']['cell_methods']=='time: mean'
    assert schema['state_time_semantics']=='source_daily_value_at_recorded_time'
    data,meta=load_information(info,archive,times,schema)
    assert meta['shape']==[9,4,8]
    assert all(v['unit']=='m' for v in meta['variables'] if v['name'].startswith('z'))
    assert original=={p.name:digest(p) for p in (root/'daily').glob('*.nc')}
    monkeypatch.setattr(xr,'open_dataset',lambda *a,**k:pytest.fail('Completed preparation reopened raw NetCDF'))
    assert prepare(root,archive.parent,start='2001-01-01',end='2001-03-31',
                   target_lat_points=4,target_lon_points=8)==(archive,info)


def test_daily_rejects_writes_outside_root_and_missing_days(tmp_path):
    root=daily_sources(tmp_path/'source',count=2)
    with pytest.raises(ValueError,match='inside'):
        prepare(root,tmp_path/'outside',start='2001-01-01',end='2001-01-02')
    with pytest.raises(FileNotFoundError,match='Missing daily'):
        prepare(root,root/'derived',start='2001-01-01',end='2001-01-03')
    with pytest.raises(ValueError,match='separate subdirectory'):
        prepare(root,root/'daily'/'bad',start='2001-01-01',end='2001-01-02')


def test_daily_separate_readonly_source_and_write_root(tmp_path):
    root=daily_sources(tmp_path/'source',count=2)
    work=tmp_path/'work';work.mkdir()
    before={str(p.relative_to(root)):digest(p) for p in root.rglob('*.nc')}
    kwargs=dict(start='2001-01-01',end='2001-01-02',target_lat_points=4,target_lon_points=8,write_root=work)
    archive,info=prepare(root,work/'prepared',**kwargs)
    assert archive.is_relative_to(work) and info.is_relative_to(work)
    assert before=={str(p.relative_to(root)):digest(p) for p in root.rglob('*.nc')}
    assert {p.name for p in root.iterdir()}=={'daily','era5_orography_0p25.nc'}
    with pytest.raises(ValueError,match='inside'):
        prepare(root,tmp_path/'elsewhere',**kwargs)
    with pytest.raises(ValueError,match='read-only'):
        prepare(root,root/'derived',**{**kwargs,'write_root':tmp_path})
    (work/'prepared'/'monthly_cache'/'escaped').symlink_to(root/'daily',target_is_directory=True)
    with pytest.raises(ValueError,match='inside'):
        prepare(root,work/'prepared',**kwargs)


def test_daily_rejects_timestamp_mismatch(tmp_path):
    root=daily_sources(tmp_path/'source',count=2)
    path=root/'daily'/'20010102.nc'
    with xr.open_dataset(path) as opened: ds=opened.load()
    ds=ds.assign_coords(time=[np.datetime64('2001-01-02T06:00','ns')])
    ds.to_netcdf(path,engine='scipy')
    with pytest.raises(ValueError,match='filename date'):
        prepare(root,root/'derived',start='2001-01-01',end='2001-01-02',target_lat_points=4,target_lon_points=8)


def _args(archive,info,output,bridge,mode='learned'):
    argv=['--archive',str(archive),'--information',str(info),'--output',str(output),
          '--model','transformer','--bridge',bridge,'--training-mode','joint',
          '--initialization','fresh','--latent-layout','spatial','--raw-backend','matched',
          '--epochs','1','--batch-size','16','--max-windows','2','--window-stride','1',
          '--horizon-steps','2','--history-steps','3','--history-stride','1',
          '--hidden-dim','8','--latent-channels','3','--spatial-hidden-dim','8',
          '--weather-depth','1','--weather-patch-size','2','--transformer-heads','2',
          '--spatial-variable-conditioning','--device','cpu']
    argv+=['--regularization','none'] if bridge=='raw' else [
        '--constraint-pair','statistical','--statistical-loss','signed_measure']
    if bridge=='guided': argv+=['--guide-mode',mode]
    return parser().parse_args(argv)


def test_daily_all_routes_train_reload_evaluate_compare(daily_prepared,tmp_path):
    _,archive,info,_=daily_prepared
    reports=[];test_reports=[]
    for bridge,mode in [('raw','learned'),('latent','learned'),('guided','learned'),('guided','zero')]:
        name=bridge+'-'+mode
        args=_args(archive,info,tmp_path/(name+'.pt'),bridge,mode)
        checkpoint=train(args)
        model,payload=load_predictor(checkpoint)
        assert payload['lead_hours']==[24.,48.]
        assert model.a_config.step_hours==24 and model.a_config.horizon_steps==2
        assert payload['predictor_provenance']['history_dt_hours']==24
        if bridge!='raw':
            assert payload['constraint_contract']['observed_pair']=='origin-24h,origin'
            assert 'origin-24h,origin' in payload['objective_semantics']['reconstruction']
            manifold,_,data=initialize_manifold(args)
            row=windows(data,manifold.config,'validation',max_windows=1,reconstruction_constraints=True)[0]
            torch.testing.assert_close(row['constraint_dt_hours'],torch.tensor([24.]))
            assert data['statistics']['step_hours']==24
        path=tmp_path/(name+'.json')
        report=evaluate(checkpoint,archive,path,information=info,max_cases=2)
        assert report['finite_forecast_fraction']==1.
        reports.append(path)
        test_path=tmp_path/(name+'.test.json')
        test_report=evaluate(checkpoint,archive,test_path,information=info,split='test',max_cases=2)
        assert test_report['split']=='test'
        assert not set(test_report['origin_times']) & set(report['origin_times'])
        assert test_report['checkpoint_sha256']==report['checkpoint_sha256']
        test_reports.append(test_path)
    result=compare(reports,tmp_path/'comparison.json')
    assert len(result['seed_summary'])==4 and result['ranking_allowed']
    result_test=compare(test_reports,tmp_path/'comparison.test.json')
    assert len(result_test['seed_summary'])==4 and result_test['ranking_allowed']
    bad=json.loads(reports[2].read_text())
    bad['constraint_contract']['observed_pair']='origin-6h,origin'
    altered=tmp_path/'bad.json';altered.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='constraint_contract'):
        compare([reports[0],altered],tmp_path/'bad-comparison.json')


def test_daily_rejects_unvalidated_forecaster(daily_prepared,tmp_path):
    _,archive,info,_=daily_prepared
    args=_args(archive,info,tmp_path/'unsupported.pt','raw')
    args.model='persistence'
    with pytest.raises(ValueError,match='Daily support requires'):
        initialize_manifold(args)
