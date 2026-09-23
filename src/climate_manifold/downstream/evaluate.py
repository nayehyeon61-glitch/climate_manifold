"""Evaluate downstream models on held-out physical fields and identical origins."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from ..train import data_contract,write_json
from ..physical_information import digest
from .train import load_predictor,windows
from .metrics import ForecastMetrics


def evaluate(checkpoint,archive,output,*,information=None,split='validation',max_cases=0,
             origin_stride=1,device='cpu',forecast_output=None):
    output=Path(output)
    if split not in ('calibration','validation','test'):raise ValueError('Use a held-out downstream split')
    if max_cases<0 or origin_stride<1:raise ValueError('Invalid evaluation counts')
    if output.exists() or (forecast_output and Path(forecast_output).exists()):raise FileExistsError('Choose new report/forecast paths')
    if forecast_output and (Path(forecast_output).suffix!='.npz' or Path(forecast_output).resolve()==output.resolve()):
        raise ValueError('Forecast path must be a distinct .npz file')
    model,p=load_predictor(checkpoint,device)
    a=p['a_metadata'];data=data_contract(archive,information,a['mode'],model.a_config,a)
    ds=windows(data,model.a_config,split,origin_stride,max_cases)
    leads=torch.tensor(p['lead_hours'],device=device,dtype=torch.float32)
    metrics=ForecastMetrics(a['schema'],a['mean'],a['scale'],p['lead_hours'])
    origins=[];successful=[];failures=[];saved=False
    if str(device).startswith('cuda'):
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    with torch.no_grad():
        for index in range(len(ds)):
            batch={k:v[None].to(device) for k,v in ds[index].items()}
            origin_ns=int(batch['origin_time_ns'][0].cpu())
            label=str(np.datetime64(origin_ns,'ns'))+'Z';origins.append(label)
            try:
                prediction=model(batch['history'],batch.get('information'),batch['origin_time_ns'],leads)
            except FloatingPointError as exc:
                failures.append({'origin':label,'error':str(exc)});continue
            metrics.update(prediction['mean'],batch['targets'][:,:len(leads)],batch['origin'],
                           prediction['std'],prediction['reconstructed_origin'])
            successful.append(label)
            if forecast_output and not saved:
                path=Path(forecast_output);path.parent.mkdir(parents=True,exist_ok=True)
                origin=np.datetime64(origin_ns,'ns')
                values={'mean':prediction['mean'][0].cpu().numpy()*np.asarray(a['scale'])+np.asarray(a['mean']),
                        'truth':batch['targets'][0,:len(leads)].cpu().numpy()*np.asarray(a['scale'])+np.asarray(a['mean']),
                        'lead_hours':np.asarray(p['lead_hours']),
                        'valid_times':origin+np.asarray(p['lead_hours']).astype('timedelta64[h]'),
                        'origin_time':origin,'schema_json':json.dumps(a['schema']),'checkpoint_sha256':digest(checkpoint)}
                if prediction['std'] is not None:
                    values['std']=prediction['std'][0].cpu().numpy()*np.asarray(a['scale'])
                np.savez_compressed(path,**values);saved=True
    if str(device).startswith('cuda'):torch.cuda.synchronize()
    report={'format':'climate_manifold.downstream_evaluation.v1','checkpoint_sha256':digest(checkpoint),
            'a_sha256':p['a_sha256'],'archive_sha256':a['archive_sha256'],
            'information_sha256':a['information_sha256'],'config':p['config'],'conditioning':p['conditioning'],
            'implementation':p['implementation'],'training_contract':p['training_contract'],
            'constants_sha256':p['constants_sha256'],
            'split':split,'origin_times':origins,'successful_origin_times':successful,'lead_hours':p['lead_hours'],
            'finite_forecast_fraction':len(successful)/len(origins),'failed_origins':failures,
            'scores':metrics.result(),'inference_seconds':time.perf_counter()-start,
            'trainable_parameters':p['trainable_parameters'],'total_parameters':p['total_parameters'],
            'training_seconds':p['training_seconds'],'seed':p['options']['seed'],
            'cuda_peak_memory_bytes':torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None,
            'forecast_output_written':saved,
            'limits':'Scores are conditional on finite forecasts; failed origins are explicit. Gaussian uncertainty is marginal, not a coherent stochastic trajectory ensemble.'}
    write_json(output,report);return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','archive','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--information');p.add_argument('--forecast-output')
    p.add_argument('--split',choices=['calibration','validation','test'],default='validation')
    p.add_argument('--max-cases',type=int,default=0);p.add_argument('--origin-stride',type=int,default=1)
    p.add_argument('--device',default='cpu')
    r=evaluate(**vars(p.parse_args(argv)));print(json.dumps(r['scores']['aggregate']));return 0

if __name__=='__main__':raise SystemExit(main())
