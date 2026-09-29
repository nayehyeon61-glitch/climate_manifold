"""Official JAX GraphCast, retrained on this archive; an external raw baseline.

No local GNN substitutes the upstream network. This optional runner uses the
pinned WeatherNext GraphCast class with a custom surface-only TaskConfig.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

UPSTREAM_URL = 'https://github.com/google-deepmind/weathernext'
UPSTREAM_COMMIT = 'f2f2c5117d2f864e2d5e7c2f9f220db5e1049dd3'
UPSTREAM_MODULE = 'weathernext.weathernext1_graph.graphcast'
VARIABLES = {'msl': 'mean_sea_level_pressure', 't2m': '2m_temperature',
             'u10': '10m_u_component_of_wind', 'v10': '10m_v_component_of_wind'}
FORCINGS = ('year_progress_sin', 'year_progress_cos', 'day_progress_sin', 'day_progress_cos')
FORMAT = 'climate_manifold.graphcast_official.v1'


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def official_dependencies():
    try:
        import haiku as hk
        import jax
        import jax.numpy as jnp
        import optax
        import xarray as xr
        import xarray_jax
        from weathernext.weathernext1_graph import graphcast
        from weathernext.utils import normalization
    except ImportError as exc:
        raise RuntimeError('Install the isolated official GraphCast environment with '
                           'scripts/install_graphcast_official.sh; no fallback model is used.') from exc
    # Editable installation preserves a verifiable upstream revision. Accept a
    # PEP-610 direct URL wheel install only when its commit is also exact.
    source = Path(graphcast.__file__).resolve()
    root = source.parents[2]
    try:
        commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True,
                                         stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain',
                                          '--untracked-files=no'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        from importlib.metadata import distribution
        info = distribution('weathernext').read_text('direct_url.json')
        commit = json.loads(info or '{}').get('vcs_info', {}).get('commit_id')
        dirty = ''
    if commit != UPSTREAM_COMMIT or dirty:
        raise RuntimeError('GraphCast requires the unmodified pinned upstream commit ' + UPSTREAM_COMMIT)
    return hk, jax, jnp, optax, xr, xarray_jax, graphcast, normalization


def calendar_features(times, longitude):
    """Deterministic known-future clock features, never future observations."""
    seconds = np.asarray(times).astype('datetime64[s]').astype(np.int64)
    year = np.mod(seconds / (86400. * 365.24219), 1.)
    day = np.mod(np.mod(seconds, 86400.)[..., None] / 86400. + np.asarray(longitude)/360., 1.)
    year = np.broadcast_to(year[..., None], day.shape)
    return np.stack([np.sin(2*np.pi*year), np.cos(2*np.pi*year),
                     np.sin(2*np.pi*day), np.cos(2*np.pi*day)], axis=-2).astype(np.float32)


def validate_grid(schema):
    from ..archive import field_grid
    grid = field_grid(schema)
    if [v['name'] for v in schema['variables']] != list(VARIABLES):
        raise ValueError('Official GraphCast adapter requires msl,t2m,u10,v10 in canonical order')
    if schema.get('forecast_step_hours') != 6:
        raise ValueError('Official GraphCast adapter requires exact 6-hourly observations')
    coords = schema['variables'][0]['coords']
    lat, lon = np.asarray(coords['lat']), np.asarray(coords['lon'])
    if len(lat)<2 or len(lon)<4 or not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError('GraphCast requires a finite global latitude/longitude grid')
    spacing = np.diff(lon)
    if (np.any(spacing<=0) or not np.allclose(spacing, spacing[0])
            or not np.isclose(spacing[0]*len(lon), 360., atol=1e-3)):
        raise ValueError('GraphCast baseline requires a periodic global longitude grid without a duplicate endpoint')
    if np.any(np.diff(lat)==0) or not (np.all(np.diff(lat)>0) or np.all(np.diff(lat)<0)):
        raise ValueError('Latitudes must be strictly monotonic')
    if np.min(lat)<-90 or np.max(lat)>90 or np.min(lat)>-45 or np.max(lat)<45:
        raise ValueError('GraphCast baseline requires a global latitude grid')
    return grid, lat.astype(np.float32), lon.astype(np.float32)


def make_batches(data, config, split, stride=1, maximum=0, batch_size=16, shuffle=None):
    starts = np.asarray(data['split'][split][::stride], dtype=int)
    if maximum:
        starts = starts[:maximum]
    if not len(starts):
        raise ValueError('No ' + split + ' origins')
    if shuffle is not None:
        starts = shuffle.permutation(starts)
    origins = starts + config.history_span_steps - 1
    if np.min(origins)<1:
        raise ValueError('Split history must include at least the two latest 6-hourly states')
    grid, _, lon = validate_grid(data['schema'])
    fields = data['states'].reshape(-1, *grid)
    for begin in range(0, len(origins), batch_size):
        indices = origins[begin:begin+batch_size]
        history_indices = indices[:, None]+np.array([-1, 0])
        future_indices = indices[:, None]+np.arange(1, config.horizon_steps+1)
        yield (indices, fields[history_indices].astype(np.float32),
               fields[future_indices].astype(np.float32),
               calendar_features(data['times'][history_indices], lon),
               calendar_features(data['times'][future_indices], lon))


def fit_upstream_normalization(data):
    """Per-variable global moments, fitted to training observations only."""
    grid, _, _ = validate_grid(data['schema'])
    fields = data['states'][:data['train_end']].reshape(-1, *grid).astype(np.float64)
    return {'mean': fields.mean((0, 2, 3)).astype(np.float32).tolist(),
            'scale': np.maximum(fields.std((0, 2, 3)), 1e-6).astype(np.float32).tolist(),
            'residual_scale': np.maximum(np.diff(fields, axis=0).std((0, 2, 3)), 1e-6).astype(np.float32).tolist()}


def make_network(schema, model_config, normalization_stats):
    hk, jax, jnp, optax, xr, xj, graphcast, normalization = official_dependencies()
    _, lat, lon = validate_grid(schema)
    names = tuple(VARIABLES.values())
    task = graphcast.TaskConfig(input_variables=names+FORCINGS, target_variables=names,
        forcing_variables=FORCINGS, pressure_levels=(), input_duration='12h')
    stats = [xr.Dataset({name: np.float32(normalization_stats[key][i]) for i,name in enumerate(names)})
             for key in ('scale', 'mean', 'residual_scale')]
    # Unit scales for bounded known-future clock features avoid upstream warnings.
    for name in FORCINGS:
        stats[0][name] = np.float32(1.); stats[1][name] = np.float32(0.)
    def forward(history, history_clock, future_clock):
        batch = history.shape[0]
        coords = {'batch': np.arange(batch), 'lat': lat, 'lon': lon,
                  'time': np.array([-6, 0], dtype='timedelta64[h]')}
        inputs = xr.Dataset({name:xj.DataArray(history[:, :, i], dims=('batch','time','lat','lon'))
                             for i,name in enumerate(names)}, coords=coords)
        for i,name in enumerate(FORCINGS):
            inputs[name] = xj.DataArray(history_clock[:,:,i], dims=('batch','time','lon'))
        target_coords = {**coords, 'time': np.array([6], dtype='timedelta64[h]')}
        template = xr.Dataset({name:xj.DataArray(jnp.zeros_like(history[:, :1, i]),
            dims=('batch','time','lat','lon')) for i,name in enumerate(names)}, coords=target_coords)
        forcings = xr.Dataset({name:xj.DataArray(jnp.broadcast_to(future_clock[:,None,i,None,:], (batch,1,len(lat),len(lon))), dims=('batch','time','lat','lon'))
                              for i,name in enumerate(FORCINGS)}, coords=target_coords)
        predictor = normalization.InputsAndResiduals(
            graphcast.GraphCast(graphcast.ModelConfig(**model_config), task), *stats)
        result = predictor(inputs, template, forcings=forcings)
        return jnp.stack([xj.unwrap(result[name].transpose('batch','time','lat','lon').data)[:,0]
                          for name in names], axis=1)
    network = hk.without_apply_rng(hk.transform(forward))
    def rollout(params, history, history_clock, future_clock):
        def step(carry, clock):
            fields, clocks = carry
            prediction = network.apply(params, fields, clocks, clock)
            return (jnp.concatenate([fields[:,1:], prediction[:,None]], axis=1),
                    jnp.concatenate([clocks[:,1:], clock[:,None]], axis=1)), prediction
        _, predictions = jax.lax.scan(step, (history,history_clock), jnp.swapaxes(future_clock,0,1))
        return jnp.swapaxes(predictions,0,1)
    return network, rollout


def save_checkpoint(path, params, metadata):
    entries = [(module, name) for module in sorted(params) for name in sorted(params[module])]
    values = {f'p{i:05d}':np.asarray(params[module][name]) for i,(module,name) in enumerate(entries)}
    values['metadata_json'] = np.asarray(json.dumps({**metadata, 'parameter_paths':entries}))
    np.savez_compressed(path, **values)
    return sha256(path)


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata_json']))
        if metadata.get('format') != FORMAT or metadata.get('upstream_commit') != UPSTREAM_COMMIT:
            raise ValueError('Not a compatible pinned official GraphCast checkpoint')
        params = {}
        for i,(module,name) in enumerate(metadata['parameter_paths']):
            params.setdefault(module,{})[name] = archive[f'p{i:05d}'].copy()
    return params, metadata


def _write_json(path, payload):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')


def train(args):
    from ..archive import load_archive, field_grid
    from ..architecture import ManifoldConfig
    from ..train import data_contract, source_commit
    from ..temporal_supervision import area_weights
    hk, jax, jnp, optax, *_ = official_dependencies()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Choose a new GraphCast run directory')
    states, _, schema = load_archive(args.archive)
    config = ManifoldConfig(state_dim=states.shape[1], grid=field_grid(schema),
        history_steps=args.history_steps, history_stride=args.history_stride,
        horizon_steps=20, manifold_dim=min(64, states.shape[1]-1))
    # The same history span determines the existing five-way split; input is
    # separately restricted to t-6,t so its origins match the other experiments.
    data = data_contract(args.archive, None, 'surface', config)
    window_config = replace(config, horizon_steps=args.horizon_steps)
    grid, lat, lon = validate_grid(schema)
    model_config = dict(resolution=float(np.diff(lon).mean()), mesh_size=args.mesh_size,
        latent_size=args.latent_size, gnn_msg_steps=args.message_steps, hidden_layers=1,
        radius_query_fraction_edge_length=1., mesh2grid_edge_normalization_factor=None)
    normalization_stats = fit_upstream_normalization(data)
    network, rollout = make_network(schema, model_config, normalization_stats)
    batch = next(make_batches(data, window_config, 'train', args.window_stride, args.max_windows, args.batch_size))
    params = network.init(jax.random.PRNGKey(args.seed), jnp.asarray(batch[1]), jnp.asarray(batch[3]), jnp.asarray(batch[4][:,0]))
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(args.learning_rate))
    opt_state = optimizer.init(params)
    scale = jnp.asarray(data['scale'].reshape(grid))
    weights = jnp.asarray(area_weights(schema))[None,None,None]
    def loss_fn(params, history, truth, hclock, fclock):
        predicted = rollout(params,history,hclock,fclock)
        return jnp.mean(jnp.sum(jnp.square((predicted-truth)/scale)*weights,axis=(-2,-1)))
    @jax.jit
    def update(params, opt_state, history, truth, hclock, fclock):
        loss, grads = jax.value_and_grad(loss_fn)(params,history,truth,hclock,fclock)
        updates, opt_state = optimizer.update(grads,opt_state,params)
        return optax.apply_updates(params,updates), opt_state, loss
    score = jax.jit(loss_fn)
    rng = np.random.default_rng(args.seed)
    best, best_params, best_epoch = float('inf'), None, None
    history_log = []
    output.mkdir(parents=True)
    start = time.perf_counter()
    for epoch in range(1,args.epochs+1):
        total = count = 0
        for _, history, truth, hclock, fclock in make_batches(data,window_config,'train',args.window_stride,
                args.max_windows,args.batch_size,rng):
            params,opt_state,loss = update(params,opt_state,history,truth,hclock,fclock)
            value = float(loss)
            if not np.isfinite(value):
                raise FloatingPointError('Non-finite official GraphCast training loss')
            total += value*len(history); count += len(history)
        calibration = calibration_count = 0
        for _, history, truth, hclock, fclock in make_batches(data,window_config,'calibration',args.window_stride,
                args.max_windows,args.batch_size):
            value = float(score(params,history,truth,hclock,fclock))
            if not np.isfinite(value):
                raise FloatingPointError('Non-finite official GraphCast calibration loss')
            calibration += value*len(history); calibration_count += len(history)
        calibration /= calibration_count
        row = dict(epoch=epoch,train_normalized_mse=total/count,calibration_normalized_mse=calibration)
        history_log.append(row); print(json.dumps(row),flush=True)
        if calibration<best:
            best,best_epoch = calibration,epoch
            best_params = jax.tree_util.tree_map(lambda x:np.asarray(x).copy(),params)
    metadata = dict(format=FORMAT, source_commit=source_commit(), upstream_url=UPSTREAM_URL,upstream_commit=UPSTREAM_COMMIT,
        upstream_module=UPSTREAM_MODULE,license='Apache-2.0',pretrained=False,
        implementation='official_graphcast_surface_custom_task_v1',model_config=model_config,
        normalization=normalization_stats,config=asdict(config),forecast_horizon_steps=args.horizon_steps,schema=schema,
        archive_sha256=sha256(args.archive),information_sha256=None,split=data['split'],
        mean=np.asarray(data['mean']).tolist(),scale=np.asarray(data['scale']).tolist(),
        statistics=data['statistics'],
        training_seconds=time.perf_counter()-start,seed=args.seed,options=vars(args),
        best_epoch=best_epoch,selection_split='calibration',history=history_log,
        trainable_parameters=sum(np.asarray(v).size for module in best_params.values() for v in module.values()),
        conditioning={'fields':list(VARIABLES),'observed_offsets_hours':[-6,0],
            'information_sidecar':False,'known_future':list(FORCINGS)},
        limitations=['Custom four-surface-variable task; no pressure-level fields, precipitation, terrain, or land/sea mask.',
            'Reduced mesh/width/depth, random initialization: not published operational GraphCast scores or pretrained weights.',
            'Raw-only JAX external baseline; no joint PyTorch encoder/decoder connection.',
            'Input history and sidecar availability differ from the joint manifold arms; cross-model comparison is descriptive, not an isolated manifold effect.',
            'Training uses shared area-weighted normalized field MSE over an autoregressive rollout; no tendency/physics/auxiliary loss.'])
    checkpoint = output/'checkpoint.npz'
    digest = save_checkpoint(checkpoint,best_params,metadata)
    _write_json(output/'metadata.json',{**metadata,'checkpoint_sha256':digest})
    evaluate(checkpoint,args.archive,output/'evaluation.json',split=args.split,
             max_cases=args.max_cases,origin_stride=args.origin_stride,forecast_output=output/'forecast.npz')
    return metadata


def evaluate(checkpoint, archive, output, *, split='validation', max_cases=0, origin_stride=1, forecast_output=None):
    from ..architecture import ManifoldConfig
    from ..train import data_contract
    from .metrics import ForecastMetrics
    import torch
    _, jax, _, *_ = official_dependencies()
    params, metadata = load_checkpoint(checkpoint)
    if sha256(archive) != metadata['archive_sha256']:
        raise ValueError('GraphCast checkpoint/archive hash mismatch')
    if split not in ('calibration','validation','test') or max_cases<0 or origin_stride<1:
        raise ValueError('Invalid held-out evaluation selection')
    output = Path(output)
    if forecast_output and (Path(forecast_output).suffix!='.npz' or Path(forecast_output).resolve()==output.resolve()):
        raise ValueError('Forecast path must be a distinct .npz file')
    if output.exists() or (forecast_output and Path(forecast_output).exists()):
        raise FileExistsError('Choose new GraphCast report and forecast paths')
    config = ManifoldConfig(**metadata['config'])
    data = data_contract(archive,None,'surface',config)
    if data['split'] != metadata['split']:
        raise ValueError('GraphCast split changed')
    _, rollout = make_network(data['schema'],metadata['model_config'],metadata['normalization'])
    predict = jax.jit(rollout)
    window_config = replace(config, horizon_steps=metadata['forecast_horizon_steps'])
    leads = np.arange(1,window_config.horizon_steps+1)*6
    metrics = ForecastMetrics(data['schema'],data['mean'],data['scale'],leads)
    mean, scale = np.asarray(data['mean']),np.asarray(data['scale'])
    origins, successful, failures = [],[],[]
    started = time.perf_counter(); saved = False
    for indices, history, truth, hclock, fclock in make_batches(data,window_config,split,origin_stride,max_cases,1):
        label = str(data['times'][indices[0]].astype('datetime64[ns]'))+'Z'; origins.append(label)
        prediction = np.asarray(predict(params,history,hclock,fclock))
        if not np.isfinite(prediction).all():
            failures.append({'origin':label,'error':'nonfinite forecast'}); continue
        pack = lambda values:torch.from_numpy(((values.reshape(*values.shape[:2],-1)-mean)/scale).astype(np.float32))
        origin = torch.from_numpy(((history[:,-1].reshape(1,-1)-mean)/scale).astype(np.float32))
        metrics.update(pack(prediction),pack(truth),origin)
        successful.append(label)
        if forecast_output and not saved:
            Path(forecast_output).parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(forecast_output,mean=prediction[0].reshape(len(leads),-1),
                truth=truth[0].reshape(len(leads),-1),lead_hours=leads,
                origin_time=data['times'][indices[0]],valid_times=data['times'][indices[0]]+leads.astype('timedelta64[h]'),
                schema_json=json.dumps(data['schema']),checkpoint_sha256=sha256(checkpoint))
            saved = True
    report = dict(format='climate_manifold.external_baseline_evaluation.v1',
        model='graphcast_official',bridge='raw',implementation=metadata['implementation'],
        upstream_commit=UPSTREAM_COMMIT,archive_sha256=metadata['archive_sha256'],information_sha256=None,
        checkpoint_sha256=sha256(checkpoint),seed=metadata['seed'],split=split,
        origin_times=origins,successful_origin_times=successful,failed_origins=failures,
        lead_hours=leads.tolist(),finite_forecast_fraction=len(successful)/len(origins),scores=metrics.result(),
        conditioning=metadata['conditioning'],ranking_allowed=False,
        comparison_scope='external cross-family raw baseline; unequal observed history and sidecar information',
        trainable_parameters=metadata['trainable_parameters'],training_seconds=metadata['training_seconds'],
        evaluation_seconds=time.perf_counter()-started,forecast_output_written=saved,limits=metadata['limitations'])
    _write_json(output,report)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest='command',required=True)
    fit = commands.add_parser('train')
    fit.add_argument('--archive',required=True); fit.add_argument('--output',required=True)
    for name,default in [('epochs',20),('batch-size',16),('seed',7),('history-steps',6),('history-stride',4),
            ('horizon-steps',20),('window-stride',4),('max-windows',0),('mesh-size',2),('latent-size',128),('message-steps',4)]:
        fit.add_argument('--'+name,type=int,default=default)
    fit.add_argument('--learning-rate',type=float,default=1e-3)
    check = commands.add_parser('evaluate')
    check.add_argument('--checkpoint',required=True); check.add_argument('--archive',required=True)
    check.add_argument('--output',required=True); check.add_argument('--forecast-output')
    for sub in (fit,check):
        sub.add_argument('--split',choices=['calibration','validation','test'],default='validation')
        sub.add_argument('--max-cases',type=int,default=0); sub.add_argument('--origin-stride',type=int,default=1)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command=='train':
        for key in ('epochs','batch_size','history_steps','history_stride','horizon_steps','window_stride','latent_size','message_steps'):
            if getattr(args,key)<1:
                raise ValueError(key+' must be positive')
        if (args.mesh_size<0 or args.max_windows<0 or args.max_cases<0 or args.origin_stride<1
                or args.horizon_steps>20 or not np.isfinite(args.learning_rate) or args.learning_rate<=0):
            raise ValueError('Invalid GraphCast model/training options')
        train(args)
    else:
        values = vars(args); values.pop('command'); evaluate(**values)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
