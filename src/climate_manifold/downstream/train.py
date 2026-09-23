"""Jointly train a manifold and its forecaster; frozen checkpoints remain optional."""
import argparse
import hashlib
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader
from ..architecture import ManifoldConfig
from ..archive import load_archive, field_grid
from ..model import ClimateManifold
from ..train import load_checkpoint as load_a, data_contract, write_json, source_commit
from ..physical_information import digest, information_digest
from ..temporal_supervision import TemporalWindowDataset, TemporalObjective
from .pipeline import ForecastPipeline, PredictorConfig
from .protocol import validate_experiment

FORMAT = 'climate_manifold.downstream.v3'
LEGACY_FORMAT = 'climate_manifold.downstream.v1'


class RawFieldContract:
    """Input metadata for direct forecasting, with no learned representation.

    The bridge only needs geometry, dimensions and input declarations. Keeping
    this separate avoids constructing or loading unused encoder/decoder weights.
    """
    def __init__(self, config, schema, mean, scale, information_metadata, sealed=False):
        from ..manifold_physics import SurfacePhysics
        self.config=config
        self.info_metadata=information_metadata
        self.info_head=self.pinn=None
        self.core=SimpleNamespace(manifold_ready=torch.tensor(sealed),
            physics=SurfacePhysics(schema,mean,scale))


class CausalWindows(TemporalWindowDataset):
    def __init__(self,*args,information=None,information_targets=False,**kwargs):
        super().__init__(*args,**kwargs);self.information=information
        self.information_targets=information_targets

    def __getitem__(self,index):
        row=super().__getitem__(index)
        if self.information is not None:
            origin=self.starts[index]+self.config.history_span_steps-1
            row['information']=torch.from_numpy(self.information[origin].copy())
            if self.information_targets:
                row['information_targets']=torch.from_numpy(
                    self.information[origin+1:origin+self.config.horizon_steps+1].copy())
        return row


def windows(data,config,split,stride=1,max_windows=0,*,information_targets=False):
    starts=data['split'][split][::stride]
    if max_windows:starts=starts[:max_windows]
    if not len(starts):raise ValueError('Forecast training/evaluation requires nonempty windows')
    return CausalWindows(data['states'],data['times'],config,starts,data['mean'],data['scale'],data['schema'],
        information=data['information'],information_targets=information_targets)


def new_a(metadata, sealed=True, raw=False):
    config=ManifoldConfig(**metadata['config'])
    if raw:
        return RawFieldContract(config,metadata['schema'],metadata['mean'],metadata['scale'],
                                metadata['information_metadata'],sealed)
    if config.representation_kind=='spatial':
        from ..spatial import SpatialClimateManifold
        constructor=SpatialClimateManifold
    else:
        constructor=ClimateManifold
    model=constructor(config,metadata['schema'],metadata['mean'],metadata['scale'],
        metadata['statistics'],metadata['information_metadata'],pinn_config=metadata.get('pinn_config'),
        information_mean=metadata.get('information_mean'),information_scale=metadata.get('information_scale'))
    # Only used to construct a saved pipeline, whose strict state loading follows.
    model.core.manifold_ready.fill_(sealed)
    return model


def load_predictor(path,device='cpu'):
    manifest=Path(path).with_suffix('.manifest.json')
    if not manifest.exists() or json.loads(manifest.read_text())['checkpoint_sha256']!=digest(path):
        raise ValueError('Downstream checkpoint manifest/hash mismatch')
    p=torch.load(path,map_location='cpu',weights_only=False)
    if p.get('format') not in (FORMAT,LEGACY_FORMAT,'climate_manifold.downstream.v2'):
        raise ValueError('Not a standalone forecast checkpoint')
    config=PredictorConfig(**p['config'])
    if config.training_mode == 'frozen' and not p.get('a_was_sealed'):
        raise ValueError('Frozen checkpoints require a sealed A')
    representation=None
    if config.representation == 'plain_ae':
        from .plain_ae import new_plain_ae
        representation=new_plain_ae(p['a_metadata'])
        representation.core.manifold_ready.fill_(True)
    model=ForecastPipeline(new_a(p['a_metadata'],bool(p.get('a_was_sealed')),raw=config.bridge=='raw'),config,
        p['constants'],p['a_metadata']['schema'],representation)
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


def initialize_manifold(args):
    """Build from training data, optionally using an A contract or warm start.

    Fresh joint models use identity latent coordinates throughout optimization;
    resealing after training would change the predictor's coordinate system.
    """
    parent = None
    pretrained = args.training_mode == 'frozen' or args.initialization == 'pretrained'
    if args.a_checkpoint:
        reference, parent = load_a(args.a_checkpoint)
        config = reference.config
        mode = parent['mode']
        if args.mode is not None and args.mode != mode:
            raise ValueError('--mode differs from the reference A data contract')
        layout=args.latent_layout or (config.representation_kind if pretrained else 'spatial')
        if pretrained and layout != config.representation_kind:
            raise ValueError('Pretrained global A weights cannot initialize a spatial encoder; use --initialization fresh')
        if not pretrained:
            config=replace(config,representation_kind=layout,latent_channels=args.latent_channels,
                spatial_downsample=args.spatial_downsample,spatial_hidden_dim=args.spatial_hidden_dim)
    else:
        if args.training_mode == 'frozen' or args.initialization == 'pretrained':
            raise ValueError('Frozen/pretrained mode requires --a-checkpoint')
        states, _, schema = load_archive(args.archive)
        config = ManifoldConfig(state_dim=states.shape[1], grid=field_grid(schema),
            history_steps=args.history_steps, history_stride=args.history_stride,
            manifold_dim=args.manifold_dim, hidden_dim=args.manifold_hidden_dim,
            context_dim=args.context_dim, horizon_steps=20, step_hours=6,
            representation_kind=args.latent_layout or 'spatial',latent_channels=args.latent_channels,
            spatial_downsample=args.spatial_downsample,spatial_hidden_dim=args.spatial_hidden_dim)
        mode = args.mode or ('enriched' if args.information else 'surface')
    if args.horizon_steps > config.horizon_steps:
        raise ValueError('Horizon exceeds the representation data contract')
    data = data_contract(args.archive, args.information, mode, config, parent)
    if pretrained:
        if not bool(reference.core.manifold_ready):
            raise ValueError('Pretrained initialization requires a sealed A checkpoint')
        if args.pinn and reference.pinn is None:
            raise ValueError('This pretrained A has no PINN; use fresh initialization with --pinn')
        model, metadata = reference, {k:v for k,v in parent.items() if k != 'model'}
        if args.bridge=='raw':
            model=RawFieldContract(config,data['schema'],data['mean'],data['scale'],
                                   data['information_metadata'],sealed=True)
    else:
        pc = parent.get('pinn_config') if parent else None
        if args.pinn:
            from ..hybrid_pinn import HybridPINNConfig
            pc = asdict(HybridPINNConfig(levels_hpa=tuple(args.pinn_levels),
                weight=.1 if args.pinn_weight in (None,0.) else args.pinn_weight,
                warmup_epochs=0, ramp_epochs=1))
        torch.manual_seed(args.seed)
        if args.bridge=='raw':
            model=RawFieldContract(config,data['schema'],data['mean'],data['scale'],data['information_metadata'])
        elif config.representation_kind=='spatial':
            from ..spatial import SpatialClimateManifold
            constructor=SpatialClimateManifold
        else:
            constructor=ClimateManifold
        if args.bridge!='raw':
            model = constructor(config, data['schema'], data['mean'], data['scale'],
                data['statistics'], data['information_metadata'], pinn_config=pc,
                information_mean=data['information_mean'], information_scale=data['information_scale'])
            normalized = torch.as_tensor((data['states'][:data['train_end']]-data['mean'])/data['scale'])
            model.core.physics.fit(normalized)
        metadata = {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in data.items()
                    if k not in ('states','times','information')}
        metadata.update(config=asdict(config), mode=mode,
            format='climate_manifold.joint_representation_metadata.v1',
            pinn_config=asdict(model.pinn.config) if model.pinn is not None else None,
            archive_sha256=digest(args.archive),
            information_sha256=information_digest(args.information) if args.information else None,
            information_shards=data['information'].provenance() if hasattr(data['information'],'provenance') else None)
    return model, metadata, data


def objective_weights(args, model):
    from .joint_objective import JointObjectiveWeights
    enabled = args.training_mode == 'joint' and args.bridge != 'raw'
    full = enabled and args.regularization == 'full'
    latent = args.bridge == 'latent'
    info = model.info_head is not None
    if args.pinn_weight and full and (not latent or model.pinn is None):
        raise ValueError('Positive --pinn-weight requires a joint latent model with enabled PINN')
    if full and args.distribution_weight and (not latent or not info):
        raise ValueError('Positive --distribution-weight requires an enriched joint latent model')
    return JointObjectiveWeights(
        reconstruction=args.reconstruction_weight if enabled else 0.,
        physics=args.physics_weight if full else 0.,
        information=args.information_weight if full and latent and info else 0.,
        static=args.static_weight if full and latent and info else 0.,
        distribution=args.distribution_weight if full and latent and info else 0.,
        pinn=((model.pinn.config.weight if args.pinn_weight is None else args.pinn_weight)
              if full and latent and model.pinn is not None else 0.))


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
    for name in ('reconstruction_weight','information_weight','static_weight','distribution_weight','physics_weight','pinn_weight'):
        value=getattr(args,name)
        if value is not None and (not math.isfinite(value) or value<0):
            raise ValueError('Objective weights must be finite and nonnegative: '+name)
    if args.training_mode == 'joint' and args.representation != 'climate_manifold':
        raise ValueError('Joint controls use the same climate_manifold with --regularization none; plain_ae is a frozen legacy control')
    a,p,data=initialize_manifold(args)
    raw_backend=args.raw_backend or ('matched' if args.training_mode=='joint' and a.config.representation_kind=='spatial' else 'legacy')
    constants=None
    grid_climode=args.model=='climode' and (args.bridge=='decoded' or (args.bridge=='raw' and raw_backend=='legacy'))
    if grid_climode:
        from .climode import load_constants
        if not args.constants:raise ValueError('ClimODE requires --constants with real orography and lsm')
        constants=load_constants(args.constants,p['schema'])
    elif args.model=='climode' and args.constants:
        raise ValueError('Matched transport ClimODE uses observed information; --constants is only for legacy raw/decoded grid ClimODE')
    config=PredictorConfig(model=args.model,bridge=args.bridge,anchor=args.anchor,hidden_dim=args.hidden_dim,
        ode_substeps=args.ode_substeps,condition_information=not args.no_information_conditioning,
        climode_attention=not args.no_climode_attention,climode_step_hours=args.climode_step_hours,
        velocity_iterations=args.velocity_iterations,representation=args.representation,training_mode=args.training_mode,
        latent_layout=a.config.representation_kind,latent_max_speed=args.latent_max_speed,
        latent_max_acceleration=args.latent_max_acceleration,raw_backend=raw_backend)
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
    weights=objective_weights(args,a)
    # All variants share train / downstream-selection calibration / untouched validation,test.
    target_info=bool(weights.information or weights.static or weights.distribution or weights.pinn)
    loaders=[DataLoader(windows(data,a.config,name,args.window_stride,args.max_windows,information_targets=target_info),batch_size=args.batch_size,
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
                    if config.training_mode == 'joint' and config.bridge != 'raw':
                        from .joint_objective import joint_losses
                        auxiliary=joint_losses(model,prediction,batch,weights,leads)
                        losses.update(auxiliary)
                        losses['loss']=losses['loss']+auxiliary['regularization']
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
    if model.bridge.manifold is not None and config.training_mode == 'frozen':
        frozen_state=(representation_payload or load_a(args.a_checkpoint)[1])['model']
        for key,value in frozen_state.items():
            if not torch.equal(value.cpu(),model.bridge.manifold.state_dict()[key].cpu()):
                raise AssertionError('Frozen representation changed: '+key)
    metadata={k:v for k,v in p.items() if k!='model'}
    payload={'format':FORMAT,'a_was_sealed':bool(a.core.manifold_ready),'a_metadata':metadata,
        'a_sha256':digest(args.a_checkpoint) if args.a_checkpoint else None,
        'config':asdict(config),'constants':constants,'model':best_state,'horizon_steps':args.horizon_steps,
        'experiment':experiment,
        'representation_sha256':(digest(args.ae_checkpoint) if representation is not None else
                                 digest(args.a_checkpoint) if config.bridge != 'raw' and args.a_checkpoint and config.training_mode=='frozen' else None),
        'representation_metadata':({k:v for k,v in representation_payload.items() if k not in ('model','a_metadata')}
                                   if representation_payload else None),
        'lead_hours':leads.cpu().tolist(),'options':vars(args),'best_epoch':best_epoch,'best_selection_state_mse':best,
        'selection_split':'calibration','training_seconds':time.perf_counter()-started,'source_commit':source_commit(),
        'implementation':('raw_climode_transport_adaptation_v1' if args.model=='climode' and args.bridge=='raw' and raw_backend=='matched'
            else 'raw_spatial_'+args.model if args.bridge=='raw' and raw_backend=='matched'
            else 'latent_climode_transport_adaptation_v1' if args.model=='climode' and args.bridge=='latent'
            else 'official_climode_custom_data_adaptation' if args.model=='climode'
            else 'spatial_'+args.model if args.bridge=='latent' and a.config.representation_kind=='spatial'
            else 'local_'+args.model),
        'constants_sha256':digest(args.constants) if constants is not None else None,
        'latent_shape':list(a.config.latent_grid) if a.config.representation_kind=='spatial' and config.bridge=='latent' else None,
        'forecast_state_grid':list(a.config.grid if config.bridge!='latent' else a.config.latent_grid)
            if config.bridge!='latent' or a.config.representation_kind=='spatial' else None,
        'transport_contract':({'velocity_bound_cells_per_day':model.predictor.max_speed,
            'raw_velocity_rate_bound_per_day':model.predictor.max_acceleration,
            'reference_spatial_downsample':a.config.spatial_downsample,
            'speed_scaling':'source_grid_factor' if config.bridge=='raw' else 'latent_grid',
            'uncertainty':'deterministic'}
            if args.model=='climode' and (config.bridge=='latent' or raw_backend=='matched' and config.bridge=='raw') else None),
        'training_contract':{**{key:getattr(args,key) for key in ('epochs','batch_size','learning_rate','tendency_weight','horizon_steps')},
            # Shared requested setting; objective_weights records the effective
            # zero reconstruction coefficient for direct raw controls.
            'reconstruction_weight':args.reconstruction_weight if config.training_mode=='joint' else 0.,
            'train_starts_sha256':hashlib.sha256(json.dumps(loaders[0].dataset.starts).encode()).hexdigest(),
            'selection_starts_sha256':hashlib.sha256(json.dumps(loaders[1].dataset.starts).encode()).hexdigest()},
        'trainable_parameters':sum(x.numel() for x in parameters),'total_parameters':sum(x.numel() for x in model.parameters()),
        'conditioning':{'direct_origin_information':bool(config.condition_information and config.bridge=='raw'
                            and (args.model in ('mlp','neural_ode') or args.model=='climode' and raw_backend=='matched') and p['mode']=='enriched'),
                        'manifold_origin_information':bool(config.bridge!='raw' and p['mode']=='enriched'),
                        'climode_static_constants':constants is not None,
                        'observed_information_available':p['mode']=='enriched'},
        'a_frozen':config.training_mode=='frozen' and config.bridge!='raw',
        'regularization':('none' if config.bridge=='raw' else args.regularization) if config.training_mode=='joint' else 'legacy',
        'objective_weights':asdict(weights),
        'initialization':'pretrained' if config.training_mode=='frozen' else args.initialization,
        'representation_training':'jointly_trained' if config.training_mode=='joint' and config.bridge!='raw' else 'frozen' if config.bridge!='raw' else 'none',
        'latent_coordinates':('not_applicable' if config.bridge=='raw' else
            'fixed pretrained seal' if bool(a.core.manifold_ready) else 'identity; no post-training reseal'),
        'objective_semantics':{'forecast':'field MSE or Gaussian NLL plus physical-time tendency',
            'distribution':'optional deterministic spatial quantiles of dynamic information fields; not ensemble CRPS',
            'physics':'future-field diagnostic matching, not exact conservation',
            'pinn':'same forecast trajectory; no independent A drift or auxiliary sampler'},
        'resume':'optimizer/RNG resume not implemented'}
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(payload,output)
    write_json(output.with_suffix('.manifest.json'),{'checkpoint_sha256':digest(output),'format':FORMAT})
    write_json(output.with_suffix('.metrics.json'),rows)
    write_json(output.with_suffix('.metadata.json'),{k:v for k,v in payload.items() if k not in ('model','a_metadata','constants')})
    return output


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('archive','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--a-checkpoint',help='Optional A data contract; use --initialization pretrained to reuse its weights')
    p.add_argument('--training-mode',choices=['joint','frozen'],default='joint')
    p.add_argument('--initialization',choices=['fresh','pretrained'],default='fresh')
    p.add_argument('--latent-layout',choices=['spatial','global'],default=None,
        help='Fresh default: spatial; pretrained/frozen default: reference checkpoint layout')
    p.add_argument('--raw-backend',choices=['matched','legacy'],default=None,
        help='Joint spatial default: matched spatial forecast core without E/D; legacy preserves former raw models')
    for name,value in dict(latent_channels=32,spatial_downsample=2,spatial_hidden_dim=64).items():
        p.add_argument('--'+name.replace('_','-'),type=int,default=value)
    p.add_argument('--mode',choices=['surface','enriched'],help='Default: enriched when --information is supplied')
    p.add_argument('--regularization',choices=['full','none'],default='full',
        help='none keeps common forecast/tendency/reconstruction losses for matched joint controls')
    for name,value in dict(manifold_dim=64,manifold_hidden_dim=512,context_dim=64,history_steps=6,history_stride=4).items():
        p.add_argument('--'+name.replace('_','-'),type=int,default=value)
    for name,value in dict(reconstruction_weight=.1,information_weight=.1,static_weight=.05,
                           distribution_weight=0.,physics_weight=.01).items():
        p.add_argument('--'+name.replace('_','-'),type=float,default=value)
    p.add_argument('--pinn',action='store_true',help='Initialize Hybrid PINN for joint training with co-located pressure fields')
    p.add_argument('--pinn-levels',nargs='+',type=int,default=[500,850])
    p.add_argument('--pinn-weight',type=float,default=None,help='Default: enabled PINN config weight, otherwise zero')
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
    p.add_argument('--latent-max-speed',type=float,default=2.,help='Latent ClimODE velocity bound in latent cells/day; resolution dependent')
    p.add_argument('--latent-max-acceleration',type=float,default=1.,help='Bound on latent ClimODE raw velocity-coordinate rate per day')
    p.add_argument('--no-information-conditioning',action='store_true',help='Remove direct origin information from raw predictors; manifold modes pass information through the encoder')
    p.add_argument('--no-climode-attention',action='store_true')
    p.add_argument('--device',default='cpu')
    return p


def main(argv=None):
    print(train(parser().parse_args(argv)));return 0

if __name__=='__main__':raise SystemExit(main())
