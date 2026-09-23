"""Train a downstream predictor with a frozen Climate Manifold or plain AE."""
import argparse
import hashlib
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from ..architecture import ManifoldConfig
from ..model import ClimateManifold
from ..train import load_checkpoint as load_a, data_contract, write_json, source_commit
from ..physical_information import digest
from ..temporal_supervision import TemporalWindowDataset, TemporalObjective
from .pipeline import ForecastPipeline, PredictorConfig
from .protocol import validate_experiment

FORMAT = 'climate_manifold.downstream.v1'


class CausalWindows(TemporalWindowDataset):
    def __init__(self,*args,information=None,**kwargs):
        super().__init__(*args,**kwargs);self.information=information

    def __getitem__(self,index):
        row=super().__getitem__(index)
        if self.information is not None:
            origin=self.starts[index]+self.config.history_span_steps-1
            row['information']=torch.from_numpy(self.information[origin].copy())
        return row


def windows(data,config,split,stride=1,max_windows=0):
    starts=data['split'][split][::stride]
    if max_windows:starts=starts[:max_windows]
    return CausalWindows(data['states'],data['times'],config,starts,data['mean'],data['scale'],data['schema'],information=data['information'])


def new_a(metadata):
    model=ClimateManifold(ManifoldConfig(**metadata['config']),metadata['schema'],metadata['mean'],metadata['scale'],
        metadata['statistics'],metadata['information_metadata'],pinn_config=metadata.get('pinn_config'),
        information_mean=metadata.get('information_mean'),information_scale=metadata.get('information_scale'))
    # Only used to construct a saved pipeline, whose strict state loading follows.
    model.core.manifold_ready.fill_(True)
    return model


def load_predictor(path,device='cpu'):
    manifest=Path(path).with_suffix('.manifest.json')
    if not manifest.exists() or json.loads(manifest.read_text())['checkpoint_sha256']!=digest(path):
        raise ValueError('Downstream checkpoint manifest/hash mismatch')
    p=torch.load(path,map_location='cpu',weights_only=False)
    if p.get('format')!=FORMAT or not p.get('a_was_sealed'):
        raise ValueError('Not a standalone downstream checkpoint with a sealed A')
    config=PredictorConfig(**p['config'])
    representation=None
    if config.representation == 'plain_ae':
        from .plain_ae import new_plain_ae
        representation=new_plain_ae(p['a_metadata'])
        representation.core.manifold_ready.fill_(True)
    model=ForecastPipeline(new_a(p['a_metadata']),config,p['constants'],p['a_metadata']['schema'],representation)
    model.load_state_dict(p['model'],strict=True)
    return model.to(device).eval(),p


def forecast_loss(output,batch,temporal,lead_hours,tendency_weight):
    target=batch['targets'][:,:len(lead_hours)]
    error=output['mean']-target
    state_mse=(error.square()*temporal.metric).sum(-1).mean()
    if output['std'] is None:fit=state_mse
    else:
        sigma=output['std']
        fit=((sigma.log()+.5*(error/sigma).square()+.5*math.log(2*math.pi))*temporal.metric).sum(-1).mean()
    dt=torch.diff(torch.cat((lead_hours.new_zeros(1),lead_hours)))[None,:,None]
    pred_path=torch.cat((batch['origin'][:,None],output['mean']),1)
    true_path=torch.cat((batch['origin'][:,None],target),1)
    delta_error=(pred_path.diff(dim=1)-true_path.diff(dim=1))*temporal.scale/dt/temporal.tendency_scale
    tendency=(delta_error.square()*temporal.metric).sum(-1).mean()
    loss=fit+tendency_weight*tendency
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite downstream training loss')
    return {'loss':loss,'state_mse':state_mse,'tendency_mse':tendency,'fit':fit}


def train(args):
    output=Path(args.output)
    if output.suffix!='.pt':raise ValueError('Output must end in .pt')
    if any(output.with_suffix(s).exists() for s in ('.pt','.manifest.json','.metrics.json','.metadata.json')):
        raise FileExistsError('Choose a new downstream checkpoint path')
    if min(args.epochs,args.batch_size,args.horizon_steps,args.window_stride)<1 or args.max_windows<0:
        raise ValueError('Invalid training counts')
    if not math.isfinite(args.learning_rate) or args.learning_rate<=0 or not math.isfinite(args.tendency_weight) or args.tendency_weight<0:
        raise ValueError('Invalid learning rate/tendency weight')
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    a,p=load_a(args.a_checkpoint)
    if not bool(a.core.manifold_ready):raise ValueError('A must be sealed')
    if args.horizon_steps>a.config.horizon_steps:raise ValueError('Horizon exceeds the A data contract')
    constants=None
    if args.model=='climode':
        from .climode import load_constants
        if not args.constants:raise ValueError('ClimODE requires --constants with real orography and lsm')
        constants=load_constants(args.constants,p['schema'])
    config=PredictorConfig(model=args.model,bridge=args.bridge,anchor=args.anchor,hidden_dim=args.hidden_dim,
        ode_substeps=args.ode_substeps,condition_information=not args.no_information_conditioning,
        climode_attention=not args.no_climode_attention,climode_step_hours=args.climode_step_hours,
        velocity_iterations=args.velocity_iterations,representation=args.representation)
    experiment=validate_experiment(config,args.experiment)
    if args.experiment == 'primary' and p['mode'] == 'enriched' and not config.condition_information:
        raise ValueError('Primary enriched comparisons require equal origin information access; use auxiliary for this ablation')
    representation=representation_payload=None
    if args.representation == 'plain_ae':
        from .plain_ae import load_plain_ae
        if not args.ae_checkpoint:raise ValueError('Plain AE control requires --ae-checkpoint')
        representation,representation_payload=load_plain_ae(args.ae_checkpoint)
        if representation_payload['a_sha256'] != digest(args.a_checkpoint):
            raise ValueError('Plain AE was trained against a different reference A/data contract')
    elif args.ae_checkpoint:
        raise ValueError('--ae-checkpoint is only used with --representation plain_ae')
    data=data_contract(args.archive,args.information,p['mode'],a.config,p)
    # All variants share train / downstream-selection calibration / untouched validation,test.
    loaders=[DataLoader(windows(data,a.config,name,args.window_stride,args.max_windows),batch_size=args.batch_size,
        shuffle=(name=='train'),generator=torch.Generator().manual_seed(args.seed)) for name in ('train','calibration')]
    # A and AE construction consume different RNG amounts. Reset so equal-size
    # latent predictors start with identical weights for the same forecast seed.
    torch.manual_seed(args.seed)
    model=ForecastPipeline(a,config,constants,p['schema'],representation).to(args.device)
    temporal=TemporalObjective(p['schema'],p['mean'],p['scale'],p['statistics']).to(args.device)
    parameters=[x for x in model.parameters() if x.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=args.learning_rate) if parameters else None
    leads=torch.arange(1,args.horizon_steps+1,device=args.device,dtype=torch.float32)*a.config.step_hours
    best=float('inf');best_state=None;rows=[];started=time.perf_counter();best_epoch=0
    for epoch in range(1,args.epochs+1):
        row={'epoch':epoch}
        for training,loader in zip((True,False),loaders):
            model.train(training);total={};count=0
            with torch.set_grad_enabled(training and optimizer is not None):
                for batch in loader:
                    batch={k:v.to(args.device) for k,v in batch.items()}
                    prediction=model(batch['history'],batch.get('information'),batch['origin_time_ns'],leads)
                    losses=forecast_loss(prediction,batch,temporal,leads,args.tendency_weight)
                    if training and optimizer is not None:
                        optimizer.zero_grad(set_to_none=True);losses['loss'].backward()
                        torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step()
                    size=len(batch['origin']);count+=size
                    for k,v in losses.items():total[k]=total.get(k,0.)+float(v.detach())*size
            row['train' if training else 'selection']={k:v/count for k,v in total.items()}
        score=row['selection']['state_mse']
        if score<best:
            best,best_epoch=score,epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        rows.append(row)
        print(json.dumps({'model':args.model,'bridge':args.bridge,'epoch':epoch,'selection_state_mse':score}),flush=True)
    model.load_state_dict(best_state)
    if model.bridge.manifold is not None:
        frozen_state=(representation_payload or p)['model']
        for key,value in frozen_state.items():
            if not torch.equal(value.cpu(),model.bridge.manifold.state_dict()[key].cpu()):
                raise AssertionError('Frozen representation changed: '+key)
    metadata={k:v for k,v in p.items() if k!='model'}
    payload={'format':FORMAT,'a_was_sealed':True,'a_metadata':metadata,'a_sha256':digest(args.a_checkpoint),
        'config':asdict(config),'constants':constants,'model':best_state,'horizon_steps':args.horizon_steps,
        'experiment':experiment,
        'representation_sha256':(digest(args.ae_checkpoint) if representation is not None else
                                 digest(args.a_checkpoint) if config.bridge != 'raw' else None),
        'representation_metadata':({k:v for k,v in representation_payload.items() if k not in ('model','a_metadata')}
                                   if representation_payload else None),
        'lead_hours':leads.cpu().tolist(),'options':vars(args),'best_epoch':best_epoch,'best_selection_state_mse':best,
        'selection_split':'calibration','training_seconds':time.perf_counter()-started,'source_commit':source_commit(),
        'implementation':'official_climode_custom_data_adaptation' if args.model=='climode' else 'local_'+args.model,
        'constants_sha256':digest(args.constants) if args.model=='climode' else None,
        'training_contract':{**{key:getattr(args,key) for key in ('epochs','batch_size','learning_rate','tendency_weight','horizon_steps')},
            'train_starts_sha256':hashlib.sha256(json.dumps(loaders[0].dataset.starts).encode()).hexdigest(),
            'selection_starts_sha256':hashlib.sha256(json.dumps(loaders[1].dataset.starts).encode()).hexdigest()},
        'trainable_parameters':sum(x.numel() for x in parameters),'total_parameters':sum(x.numel() for x in model.parameters()),
        'conditioning':{'direct_origin_information':bool(config.condition_information and config.bridge=='raw' and args.model in ('mlp','neural_ode') and p['mode']=='enriched'),
                        'manifold_origin_information':bool(config.bridge!='raw' and p['mode']=='enriched'),
                        'climode_static_constants':args.model=='climode'},
        'a_frozen':True,'resume':'optimizer/RNG resume not implemented'}
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(payload,output)
    write_json(output.with_suffix('.manifest.json'),{'checkpoint_sha256':digest(output),'format':FORMAT})
    write_json(output.with_suffix('.metrics.json'),rows)
    write_json(output.with_suffix('.metadata.json'),{k:v for k,v in payload.items() if k not in ('model','a_metadata','constants')})
    return output


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('a-checkpoint','archive','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--information');p.add_argument('--constants')
    p.add_argument('--model',choices=['mlp','neural_ode','climode','persistence'],default='neural_ode')
    p.add_argument('--experiment',choices=['primary','auxiliary'],default='primary')
    p.add_argument('--representation',choices=['climate_manifold','plain_ae'],default='climate_manifold')
    p.add_argument('--ae-checkpoint')
    p.add_argument('--bridge',choices=['raw','latent','decoded'],default='latent')
    p.add_argument('--anchor',choices=['none','origin'],default='none')
    for name,value in dict(epochs=20,batch_size=2,hidden_dim=128,ode_substeps=2,horizon_steps=20,
                           window_stride=4,max_windows=0,seed=7,velocity_iterations=20).items():
        p.add_argument('--'+name.replace('_','-'),type=int,default=value)
    p.add_argument('--learning-rate',type=float,default=1e-3)
    p.add_argument('--tendency-weight',type=float,default=.1)
    p.add_argument('--climode-step-hours',type=float,default=1.)
    p.add_argument('--no-information-conditioning',action='store_true',help='Remove direct origin information from raw NN baselines; manifold modes always pass information through A only')
    p.add_argument('--no-climode-attention',action='store_true')
    p.add_argument('--device',default='cpu')
    return p


def main(argv=None):
    print(train(parser().parse_args(argv)));return 0

if __name__=='__main__':raise SystemExit(main())
