"""Train Climate Manifold A, including optional physical-time Hybrid PINN."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import resource
import subprocess
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .model import ClimateManifold, FORMAT, curriculum, gradient_diagnostics
from .physical_information import load_information, fit_information, digest, information_digest
from .architecture import ManifoldConfig
from .archive import load_archive, field_grid, build_split, validate_split
from .temporal_supervision import TemporalWindowDataset, NonSingletonBatchSampler, fit_temporal_statistics, area_weights

def source_commit():
    root=Path(__file__).resolve().parents[2]
    if not (root/'.git').exists():return None
    result=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True)
    return result.stdout.strip() if result.returncode==0 else None

def write_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

class Windows(TemporalWindowDataset):
    def __init__(self,*args,information=None,**kwargs):
        super().__init__(*args,**kwargs);self.information=information
    def __getitem__(self,index):
        row=super().__getitem__(index)
        if self.information is not None:
            origin=self.starts[index]+self.config.history_span_steps-1
            row['information']=torch.from_numpy(self.information[origin].copy())
            row['information_targets']=torch.from_numpy(self.information[origin+1:origin+self.config.horizon_steps+1].copy())
        return row

def load_checkpoint(path,device='cpu'):
    p=torch.load(path,map_location='cpu',weights_only=False)
    if p.get('format')!=FORMAT:raise ValueError('Not a standalone Climate Manifold checkpoint; legacy checkpoints require explicit conversion')
    manifest=Path(path).with_suffix('.manifest.json')
    if not manifest.exists() or json.loads(manifest.read_text())['checkpoint_sha256']!=digest(path):
        raise ValueError('Checkpoint manifest/hash mismatch')
    model=ClimateManifold(ManifoldConfig(**p['config']),p['schema'],p['mean'],p['scale'],p['statistics'],p['information_metadata'],
        pinn_config=p.get('pinn_config'),information_mean=p.get('information_mean'),information_scale=p.get('information_scale'))
    model.load_state_dict(p['model']);model.set_phase(p['stage']);model.to(device)
    return model,p

def data_contract(archive,information_path,mode,config,parent=None):
    states,times,schema=load_archive(archive)
    if [v['name'] for v in schema['variables']]!=['msl','t2m','u10','v10'] or schema['forecast_step_hours']!=6:
        raise ValueError('This profile requires canonical msl/t2m/u10/v10 and exact6h')
    info=meta=None
    if mode=='enriched':
        if not information_path:raise ValueError('Enriched mode requires --information; missing fields cannot be fabricated')
        info,meta=load_information(information_path,archive,times,schema)
    elif information_path:raise ValueError('Surface mode must not receive enriched information')
    if parent:
        if parent['archive_sha256']!=digest(archive) or parent['information_sha256']!=(information_digest(information_path) if information_path else None):
            raise ValueError('Parent archive/information hash changed')
        if hasattr(info,'pin'):info.pin(parent.get('information_shards'))
        split=parent['split'];mean=np.asarray(parent['mean']);scale=np.asarray(parent['scale'])
        stats=parent['statistics'];im=parent['information_mean'];isc=parent['information_scale'];its=parent['information_tendency_scale']
    else:
        count=len(states)-config.history_span_steps-config.horizon_steps+1
        split=build_split(count,config.horizon_steps)
        end=split['train'][-1]+config.history_span_steps+config.horizon_steps
        mean=states[:end].mean(0);scale=states[:end].std(0);scale=np.where(scale>1e-6,scale,1).astype(np.float32)
        stats=fit_temporal_statistics(states,times,schema,scale,end)
        im=isc=its=None
        if hasattr(info,'statistics'):
            im,isc,its=info.statistics(end)
        elif info is not None:
            im,isc=fit_information(info,meta,end,schema)
            sh=meta['shape'];d=np.diff(info[:end],axis=0).reshape(-1,*sh)/6
            w=area_weights(schema)
            avg=(d*w).sum((-2,-1)).mean(0)
            var=((d-avg[None,:,None,None])**2*w).sum((-2,-1)).mean(0)
            channel=np.maximum(np.sqrt(var),np.maximum(isc.reshape(sh).mean((-2,-1))*1e-3/6,1e-8))
            its=np.broadcast_to(channel[:,None,None],sh).copy().reshape(-1).astype(np.float32)
    validate_split(split,config.horizon_steps,len(states)-config.history_span_steps-config.horizon_steps+1)
    end=split['train'][-1]+config.history_span_steps+config.horizon_steps
    normalized=(info.normalized(im,isc) if hasattr(info,'normalized') else
                None if info is None else ((info-np.asarray(im))/np.asarray(isc)).astype(np.float32))
    data=dict(states=states,times=times,schema=schema,split=split,mean=mean,scale=scale,statistics=stats,
        information_metadata=meta,information_mean=im,information_scale=isc,information_tendency_scale=its,
        information=normalized,train_end=end)
    return data

def batch_loss(model,batch,args,epoch,streams):
    info=batch.get('information');truth=torch.cat((batch['origin'][:,None],batch['targets']),1)
    aux=model.phase=='A'
    pc=model.pinn.config if aux and model.pinn is not None else None
    warmup=pc is not None and epoch<=pc.warmup_epochs
    if warmup:
        metrics=model.pinn_losses(batch,warmup=True)
        metrics['weighted_pinn']=pc.weight*metrics['pinn_total']
        metrics['loss']=metrics['weighted_pinn']
        metrics['selection']=metrics['pinn_total']
        metrics['pinn_weight']=metrics['loss'].new_tensor(pc.weight)
        metrics['pinn_warmup']=metrics['loss'].new_tensor(1.)
        for k,v in metrics.items():
            if not bool(torch.isfinite(v)):raise FloatingPointError(f'Nonfinite {k}')
        return metrics
    schedule_epoch=epoch-(pc.warmup_epochs if pc is not None else 0)
    generated,qs=model.rollout(batch['history'],info,members=args.members,tau_steps=args.tau_steps,
        generator=streams['ensemble'],auxiliary=aux,return_q=True)
    metrics=model.scores(generated,truth,batch['dt_hours'],batch['pair_observed_mask'])
    teacher=model.teacher_loss(batch,info,streams['fm']);metrics.update(teacher)
    metrics.update(model.geometry_losses(batch,info))
    phase,weights=curriculum(schedule_epoch,args.curriculum_interval)
    if args.profile!='process':
        phase=2 if args.profile=='dynamics' else 1
        _,weights=curriculum(3 if phase==2 else 1,2)
        if args.profile=='information':weights.update(ae_delta=.05,decoded_drift=.05,latent_dynamics=.1,information_geometry=.02)
        for key in ('fm','state_crps','transition_crps','loss_delta','loss_trajectory','info_distribution'):weights[key]=0.
    if args.loss_weights:
        # Overrides set plateau strength, not the activation epoch.
        for key,value in json.loads(args.loss_weights).items():
            if weights[key]>0:weights[key]=float(value)
    if info is not None:
        metrics.update(model.information_scores(qs,info,batch['information_targets'],batch['dt_hours'],
            torch.as_tensor(args.info_scale,device=truth.device),torch.as_tensor(args.info_tendency_scale,device=truth.device)))
    metrics['curriculum_phase']=generated.new_tensor(phase)
    total=generated.sum()*0
    for key,weight in weights.items():
        if key in metrics:metrics['weighted_'+key]=metrics[key]*weight;total=total+metrics['weighted_'+key]
    if pc is not None:
        metrics.update(model.pinn_losses(batch))
        weight=pc.weight*min(1.,schedule_epoch/pc.ramp_epochs)
        metrics['pinn_weight']=generated.new_tensor(weight)
        metrics['pinn_warmup']=generated.new_tensor(0.)
        metrics['weighted_pinn']=weight*metrics['pinn_total']
        total=total+metrics['weighted_pinn']
    # Fixed validation selection below is independent of curriculum weights.
    metrics['loss']=total
    metrics['selection']=metrics['state_crps']+metrics['transition_crps']+.1*metrics['loss_trajectory']+.1*metrics['mean_state']
    if aux:metrics['selection']=metrics['selection']+metrics['reconstruction']+.05*(metrics['ae_delta']+metrics['decoded_drift'])
    if pc is not None:
        # Fixed plateau weight for selection; independent of the training ramp.
        metrics['selection']=metrics['selection']+pc.weight*metrics['pinn_total']
    for k,v in metrics.items():
        if not bool(torch.isfinite(v)):raise FloatingPointError(f'Nonfinite {k}')
    return metrics

def train(args):
    if args.stage!="A":raise ValueError("Climate Manifold trains A only")
    output=Path(args.output)
    if output.suffix!='.pt':raise ValueError('Checkpoint output must end in .pt')
    if any(output.with_suffix(s).exists() for s in ('.pt','.metrics.json','.metadata.json','.manifest.json')):raise FileExistsError('Choose a new output checkpoint')
    if args.batch_size<2 or args.members<2 or args.epochs<1:raise ValueError('Require batch>=2, members>=2, epochs>=1')
    if min(args.tau_steps,args.window_stride,args.curriculum_interval)<1 or min(args.max_windows,args.patience)<0:
        raise ValueError('Invalid sampling/curriculum counts')
    if args.max_windows==1:
        raise ValueError('max_windows must be 0 (all) or >=2 for the manifold metric')
    if not math.isfinite(args.learning_rate) or args.learning_rate<=0 or not math.isfinite(args.weight_decay) or args.weight_decay<0:
        raise ValueError('Invalid optimizer parameters')
    if args.a_quality_max is not None and (not math.isfinite(args.a_quality_max) or args.a_quality_max<=0):
        raise ValueError('Quality threshold must be finite and positive')
    if args.loss_weights:
        overrides=json.loads(args.loss_weights)
        if args.stage!='A' or not isinstance(overrides,dict) or set(overrides)-set(curriculum(999)[1]):
            raise ValueError('Loss overrides are named A curriculum weights only')
        if any(not math.isfinite(float(v)) or float(v)<0 for v in overrides.values()):raise ValueError('Invalid A loss weight')
    pinn_config=None
    if getattr(args,'pinn',False):
        from .hybrid_pinn import HybridPINNConfig
        if args.stage!='A' or args.mode!='enriched':
            raise ValueError('--pinn requires enriched A and co-located pressure fields')
        pinn_config=HybridPINNConfig(levels_hpa=tuple(args.pinn_levels),weight=args.pinn_weight,
            warmup_epochs=args.pinn_warmup_epochs,ramp_epochs=args.pinn_ramp_epochs)
        pinn_config.validate()
    warmup_epochs=pinn_config.warmup_epochs if pinn_config else 0
    if args.stage=='A':
        min_joint=5*args.curriculum_interval+1 if args.profile=='process' else 1
        if pinn_config:min_joint=max(min_joint,pinn_config.ramp_epochs)
        if args.epochs<warmup_epochs+min_joint:
            raise ValueError(f'A must complete warm-up, curriculum and PINN ramp; require >={warmup_epochs+min_joint} epochs')
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    device=args.device
    states,_,schema=load_archive(args.archive)
    config=ManifoldConfig(state_dim=states.shape[1],grid=field_grid(schema),horizon_steps=20,step_hours=6,
        history_steps=args.history_steps,history_stride=args.history_stride,manifold_dim=args.manifold_dim,
        hidden_dim=args.hidden_dim,context_dim=args.context_dim)
    print(f'A manifold_dim={config.manifold_dim} hidden_dim={config.hidden_dim}',flush=True)
    d=data_contract(args.archive,args.information,args.mode,config)
    model=ClimateManifold(config,d['schema'],d['mean'],d['scale'],d['statistics'],d['information_metadata'],
        pinn_config=pinn_config,information_mean=d['information_mean'],information_scale=d['information_scale']).to(device)
    model.core.physics.fit(torch.as_tensor((d['states'][:d['train_end']]-d['mean'])/d['scale'],device=device))
    model.set_phase(args.stage)
    args.info_scale=d['information_scale'];args.info_tendency_scale=d['information_tendency_scale']
    params_a=[p for name,p in model.named_parameters() if p.requires_grad and (name.startswith('core.manifold.') or name.startswith('information.'))]
    ids={id(p) for p in params_a};params_process=[p for p in model.parameters() if p.requires_grad and id(p) not in ids]
    lr=args.learning_rate
    groups=[{'params':ps,'lr':rate,'name':name} for ps,rate,name in
        ((params_a,lr,'representation'),(params_process,lr,'process')) if ps]
    opt=torch.optim.AdamW(groups,weight_decay=args.weight_decay)
    def loader(name,shuffle):
        starts=d['split'][name][::args.window_stride]
        if args.max_windows:starts=starts[:args.max_windows]
        if len(starts)<2:
            raise ValueError(f'{name}: fewer than two selected windows; reduce window_stride or increase max_windows')
        ds=Windows(d['states'],d['times'],config,starts,d['mean'],d['scale'],d['schema'],information=d['information'])
        batches=NonSingletonBatchSampler(len(ds),args.batch_size,shuffle=shuffle,generator=torch.Generator().manual_seed(args.seed))
        return DataLoader(ds,batch_sampler=batches)
    train_name='train';val_name='expert_validation'
    loaders=[loader(train_name,True),loader(val_name,False)]
    rows=[];best=float('inf');best_state=None;best_epoch=0;start=time.perf_counter()
    if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,args.epochs+1):
        if args.stage=='A' and model.pinn is not None:
            model.set_pinn_warmup(epoch<=model.pinn.config.warmup_epochs)
        record={'epoch':epoch}
        for is_train,dl in zip((True,False),loaders):
            model.train(is_train);sums={};count=0
            streams={k:torch.Generator(device=device).manual_seed(args.seed+(epoch*1000 if is_train else 900000)+offset)
                     for k,offset in (('fm',11),('ensemble',29))}
            with torch.set_grad_enabled(is_train):
                for index,batch in enumerate(dl):
                    batch={k:v.to(device) for k,v in batch.items()}
                    values=batch_loss(model,batch,args,epoch,streams)
                    if is_train:
                        if args.gradient_audit and index==0:
                            modules={'encoder':model.core.manifold.encoder,'decoder':model.core.manifold.decoder,
                                     'drift':model.core.manifold.latent_drift,'a_sampler':model.a_sampler,
                                     'information':model.information,'context':model.a_context,
                                     'info_decoder':model.info_head,'pinn':model.pinn}
                            selected={k:values[k] for k in ('reconstruction','ae_delta','decoded_drift','static_l2','information_geometry',
                                'fm','state_crps','transition_crps','loss_trajectory','pinn_total','pinn_surface_tendency') if k in values}
                            selected.update({k:v for k,v in values.items() if k.startswith('weighted_')})
                            record['gradient_first_batch']=gradient_diagnostics(selected,{k:list(m.parameters()) for k,m in modules.items() if m is not None})
                        opt.zero_grad(set_to_none=True);values['loss'].backward()
                        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True);opt.step()
                    b=len(batch['origin']);count+=b
                    for k,v in values.items():sums[k]=sums.get(k,0.)+float(v.detach())*b
            record['train' if is_train else 'validation']={k:v/count for k,v in sums.items()}
        eligible=args.stage!='A' or epoch>=warmup_epochs+min_joint
        record['eligible_for_best']=eligible
        if eligible and record['validation']['selection']<best:
            best=record['validation']['selection'];best_epoch=epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        record.update(runtime_seconds=time.perf_counter()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      cuda_peak_memory_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None)
        rows.append(record);print(json.dumps({'stage':args.stage,'epoch':epoch,'selection':record['validation']['selection']},allow_nan=False),flush=True)
        if args.patience and best_state is not None and epoch-best_epoch>=args.patience:break
    model.load_state_dict(best_state)
    model.set_phase(args.stage)
    if args.stage=='A':
        chosen=rows[best_epoch-1]['validation']
        if args.a_quality_max is not None and max(chosen['ae_delta'],chosen['decoded_drift'])>args.a_quality_max:
            raise ValueError('Best A failed user-specified dynamics quality gate; not sealed')
        x=torch.as_tensor((d['states'][:d['train_end']]-d['mean'])/d['scale'],device=device)
        c=None if d['information'] is None else torch.as_tensor(d['information'][:d['train_end']],device=device)
        model.seal(x,c)
    output.parent.mkdir(parents=True,exist_ok=True)
    persisted={k:v for k,v in d.items() if k not in ('states','times','information')}
    for k,v in persisted.items():
        if isinstance(v,np.ndarray):persisted[k]=v.tolist()
    options={k:v for k,v in vars(args).items() if not k.startswith('info_')}
    payload={**persisted,'format':FORMAT,'stage':args.stage,'mode':args.mode,'config':asdict(config),'model':model.state_dict(),
        'pinn_config':asdict(model.pinn.config) if model.pinn is not None else None,
        'options':options,'best_epoch':best_epoch,'best_selection':best,'archive_sha256':digest(args.archive),
        'information_sha256':information_digest(args.information) if args.information else None,
        'information_shards':d['information'].provenance() if hasattr(d['information'],'provenance') else None,
        'source_commit':source_commit(),
        'source_file_sha256':{str(f.relative_to(Path(__file__).parent)):digest(f) for f in sorted(Path(__file__).parent.glob('*.py'))},
        'optimizer_groups':[{'name':g['name'],'lr':g['lr']} for g in groups],
        'resume':'best weights only; optimizer/RNG not stored; same-stage resume is not implemented',
        'sampling_contract':'origin-fixed information/history; persistent independent member noise; raw-z auxiliary A sampler; 20 physical steps at 6h'}
    torch.save(payload,output);write_json(output.with_suffix('.metrics.json'),rows)
    write_json(output.with_suffix('.metadata.json'),{k:v for k,v in payload.items() if k!='model'})
    write_json(output.with_suffix('.manifest.json'),{'checkpoint_sha256':digest(output),'stage':args.stage})
    return output

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('archive','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--information');p.add_argument('--stage',choices=['A'],default='A')
    p.add_argument('--mode',choices=['surface','enriched'],default='enriched')
    p.add_argument('--profile',choices=['baseline','dynamics','information','process'],default='process')
    for k,v in dict(epochs=60,batch_size=2,members=4,tau_steps=4,history_steps=6,history_stride=4,
                    manifold_dim=ManifoldConfig.manifold_dim,hidden_dim=ManifoldConfig.hidden_dim,
                    context_dim=64,window_stride=4,max_windows=0,seed=7,curriculum_interval=4,patience=0).items():
        p.add_argument('--'+k.replace('_','-'),type=int,default=v)
    p.add_argument('--learning-rate',type=float,default=.001);p.add_argument('--weight-decay',type=float,default=.0001)
    p.add_argument('--loss-weights',help='A-only JSON plateau coefficients; preserves six-phase activation schedule')
    p.add_argument('--a-quality-max',type=float);p.add_argument('--gradient-audit',action='store_true')
    p.add_argument('--pinn',action='store_true',help='Add physical-time Hybrid PINN to enriched A')
    p.add_argument('--pinn-levels',nargs='+',type=int,default=[500,850])
    p.add_argument('--pinn-weight',type=float,default=.1)
    p.add_argument('--pinn-warmup-epochs',type=int,default=1)
    p.add_argument('--pinn-ramp-epochs',type=int,default=3)
    p.add_argument('--device',default='cpu');a=p.parse_args(argv);print(train(a));return 0
if __name__=='__main__':main()
