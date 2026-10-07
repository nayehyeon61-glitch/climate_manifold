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
from .metrics import ForecastMetrics,LatentDiagnostics
from .protocol import experiment_contract
from .constraint_protocol import constraint_decoder_from_payload
from .statistical_objective import statistical_config_from_payload
from .statistical_flow import statistical_flow_config_from_payload
from .conditional_flow import conditional_flow_config_from_payload
from .dynamics_diagnostics import (dynamics_evaluation_windows, SignedFieldDiagnostics,
                                   DynamicsLatentDiagnostics)
from .dynamics_objective import future_dynamics_targets
from .dynamics_rollout import lead_steps


def evaluate(checkpoint,archive,output,*,information=None,split='validation',max_cases=0,
             origin_stride=1,device='cpu',forecast_output=None,
             guide_interaction_override='checkpoint',dynamics_diagnostics=False,
             dynamics_rollout_steps=None):
    output=Path(output)
    if split not in ('calibration','validation','test'):raise ValueError('Use a held-out downstream split')
    if max_cases<0 or origin_stride<1:raise ValueError('Invalid evaluation counts')
    if output.exists() or (forecast_output and Path(forecast_output).exists()):raise FileExistsError('Choose new report/forecast paths')
    if forecast_output and (Path(forecast_output).suffix!='.npz' or Path(forecast_output).resolve()==output.resolve()):
        raise ValueError('Forecast path must be a distinct .npz file')
    model,p=load_predictor(checkpoint,device)
    if dynamics_rollout_steps is not None and not dynamics_diagnostics:
        raise ValueError('--dynamics-rollout-steps requires --dynamics-diagnostics')
    if dynamics_diagnostics and getattr(model.config, 'dynamics_auxiliary_architecture', 'direct') != 'latent_rollout':
        raise ValueError('Dynamics diagnostics require a latent_rollout auxiliary checkpoint')
    if guide_interaction_override not in ('checkpoint', 'off'):
        raise ValueError('Guide interaction override must be checkpoint or off; an untrained module cannot be enabled at evaluation')
    interaction_evaluation = None
    trained_interaction = model.config.guide_interaction
    if guide_interaction_override == 'off' and trained_interaction != 'transformer':
        raise ValueError('Guide interaction bypass requires a checkpoint trained with --guide-interaction transformer')
    if trained_interaction == 'transformer':
        bypassed = guide_interaction_override == 'off'
        model.guide_fusion.guide_interaction_bypass = bypassed
        interaction_evaluation = {
            'requested': guide_interaction_override, 'trained': trained_interaction,
            'effective': 'off' if bypassed else trained_interaction, 'bypassed': bypassed,
        }
    statistical_config=statistical_config_from_payload(p)
    dynamics_latent_audit = ({
        'reference_encoder': 'statistical_encoder_E_s',
        'origin_latent': 'E_s_of_observed_origin_with_origin_information',
        'forecast_history_latent': 'actual_predictor_input_E_s_history_plus_projected_E_d_history',
        'diagnostic_target_latent': 'post_forecast_E_s_of_future_truth_with_origin_information',
        'interpretation': 'E_s coordinate/reconstruction audit, not a direct E_d trajectory score',
        'primary_comparison': 'physical_field_RMSE_and_ACC',
        'limits': 'Latent tendencies start from E_s origin, not the mixed predictor input; do not rank different encoders by these coordinate errors.',
    } if p.get('dynamics_contract') is not None and model.config.bridge == 'latent' else None)
    a=p['a_metadata'];data=data_contract(archive,information,a['mode'],model.a_config,a)
    dynamics_evaluation_contract = None
    dynamics_origin_selection = None
    if dynamics_diagnostics:
        # The A data contract can reserve 120 h while M/F_d were trained for a
        # shorter horizon. Preserve A for split validation, but use actual saved
        # forecast leads for the trained-horizon minimum and audit default.
        trained_steps = int(lead_steps(torch.as_tensor(p['lead_hours']),model.a_config.step_hours)[-1])
        rollout_steps = trained_steps if dynamics_rollout_steps is None else dynamics_rollout_steps
        ds, dynamics_origin_selection = dynamics_evaluation_windows(
            data, model.a_config, split, rollout_steps, origin_stride, max_cases,
            trained_horizon_steps=trained_steps)
        dynamics_evaluation_contract = {
            'version': 'climate_manifold.dynamics_evaluation.v1',
            'rollout_steps': rollout_steps, 'step_hours': float(model.a_config.step_hours),
            'target_semantics': 'same_length_strided_truth_window',
            'main_forecast_horizon_steps': trained_steps,
        }
        auxiliary_leads = torch.arange(1, rollout_steps+1, device=device, dtype=torch.float32)*model.a_config.step_hours
        auxiliary_hours = auxiliary_leads.cpu().tolist()
        auxiliary_metrics = ForecastMetrics(a['schema'], a['mean'], a['scale'], auxiliary_hours)
        auxiliary_signed = SignedFieldDiagnostics(a['schema'], auxiliary_hours)
        main_signed = SignedFieldDiagnostics(a['schema'], p['lead_hours'])
        auxiliary_latent = DynamicsLatentDiagnostics(auxiliary_hours)
    else:
        ds=windows(data,model.a_config,split,origin_stride,max_cases)
    leads=torch.tensor(p['lead_hours'],device=device,dtype=torch.float32)
    metrics=ForecastMetrics(a['schema'],a['mean'],a['scale'],p['lead_hours'])
    latent_metrics=LatentDiagnostics(a['schema'],p['lead_hours']) if model.config.bridge=='latent' else None
    diagnostic_failures=[];inference_seconds=0.
    origins=[];successful=[];failures=[];saved=False
    auxiliary_successful=[];auxiliary_failures=[];auxiliary_seconds=0.
    main_cuda_peak=0;auxiliary_cuda_peak=0
    if str(device).startswith('cuda'):
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    with torch.no_grad():
        for index in range(len(ds)):
            batch={k:v[None].to(device) for k,v in ds[index].items()}
            origin_ns=int(batch['origin_time_ns'][0].cpu())
            label=str(np.datetime64(origin_ns,'ns'))+'Z';origins.append(label)
            try:
                if str(device).startswith('cuda'):
                    torch.cuda.synchronize()
                    if dynamics_diagnostics:torch.cuda.reset_peak_memory_stats()
                forecast_start=time.perf_counter()
                prediction=model(batch['history'],batch.get('information'),batch['origin_time_ns'],leads)
            except FloatingPointError as exc:
                failures.append({'origin':label,'error':str(exc)});continue
            finally:
                if str(device).startswith('cuda'):
                    torch.cuda.synchronize()
                    if dynamics_diagnostics:main_cuda_peak=max(main_cuda_peak,torch.cuda.max_memory_allocated())
                inference_seconds+=time.perf_counter()-forecast_start
            metrics.update(prediction['mean'],batch['targets'][:,:len(leads)],batch['origin'],
                           prediction['std'],prediction['reconstructed_origin'])
            successful.append(label)
            auxiliary_prediction=auxiliary_target_latent=None
            if dynamics_diagnostics:
                # M is evaluated first with its checkpoint leads. This independent
                # F_d audit starts from the same observed E_d features; no teacher
                # values enter its recurrence or the already-completed M forecast.
                main_signed.update(prediction['mean'],batch['targets'][:,:len(leads)])
                if str(device).startswith('cuda'):
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                auxiliary_start=time.perf_counter()
                try:
                    auxiliary_prediction=model.dynamics_auxiliary_head.rollout(
                        prediction['dynamics_history_latent'],auxiliary_leads)
                    auxiliary_target_latent=future_dynamics_targets(model,batch,auxiliary_leads)
                    if not all(torch.isfinite(value).all() for value in (
                            auxiliary_prediction['fields'],auxiliary_prediction['latent'],auxiliary_target_latent)):
                        raise FloatingPointError('Nonfinite auxiliary dynamics rollout or latent targets')
                    auxiliary_latent.update(auxiliary_prediction['latent'],auxiliary_target_latent,
                                            prediction['dynamics_history_latent'][:,-1])
                    auxiliary_signed.update(auxiliary_prediction['fields'],batch['targets'][:,:rollout_steps])
                    auxiliary_metrics.update(auxiliary_prediction['fields'],batch['targets'][:,:rollout_steps],batch['origin'])
                    auxiliary_successful.append(label)
                except FloatingPointError as exc:
                    auxiliary_failures.append({'origin':label,'error':str(exc)})
                    auxiliary_prediction=auxiliary_target_latent=None
                finally:
                    if str(device).startswith('cuda'):
                        torch.cuda.synchronize()
                        auxiliary_cuda_peak=max(auxiliary_cuda_peak,torch.cuda.max_memory_allocated())
                    auxiliary_seconds+=time.perf_counter()-auxiliary_start
            target_q=None
            if latent_metrics is not None:
                # Post-forecast audit only: never pass future fields/information to the predictor.
                try:
                    target_q=model.bridge.encode_history(batch['targets'][:,:len(leads)],batch.get('information'))
                    target_reconstruction=model.bridge.decode(target_q)
                    cycle_q=model.bridge.encode_history(prediction['mean'],batch.get('information'))
                    latent_metrics.update(prediction,target_q,target_reconstruction,cycle_q,batch['targets'][:,:len(leads)])
                except FloatingPointError as exc:
                    diagnostic_failures.append({'origin':label,'error':str(exc)});target_q=None
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
                if prediction['predicted_latent'] is not None:
                    values['predicted_latent']=prediction['predicted_latent'][0].cpu().numpy()
                    values['origin_latent']=prediction['origin_latent'][0].cpu().numpy()
                    if target_q is not None:values['diagnostic_target_latent']=target_q[0].cpu().numpy()
                    if p.get('latent_shape'):
                        shape=tuple(p['latent_shape'])
                        values['latent_shape']=np.asarray(shape,dtype=np.int64)
                        values['predicted_latent_spatial']=prediction['predicted_latent'][0].cpu().numpy().reshape(len(leads),*shape)
                if model.config.bridge == 'guided' and prediction.get('history_latent') is not None:
                    values['guide_history_latent'] = prediction['history_latent'][0].cpu().numpy()
                    values['guide_contract_json'] = json.dumps(p.get('guide_contract'), sort_keys=True)
                if p.get('dynamics_contract') is not None:
                    for name in ('dynamics_history_latent', 'forecast_history_latent', 'fused_history'):
                        if prediction.get(name) is not None:
                            values[name] = prediction[name][0].cpu().numpy()
                    if p.get('dynamics_latent_shape') is not None:
                        values['dynamics_latent_shape'] = np.asarray(p['dynamics_latent_shape'], dtype=np.int64)
                    if dynamics_latent_audit is not None:
                        values['latent_diagnostic_semantics_json'] = json.dumps(dynamics_latent_audit, sort_keys=True)
                    values['dynamics_contract_json'] = json.dumps(p['dynamics_contract'], sort_keys=True)
                    values['dynamics_auxiliary_config_json'] = json.dumps(p.get('dynamics_auxiliary_config'), sort_keys=True)
                if interaction_evaluation is not None:
                    values['guide_interaction_contract_json'] = json.dumps(p['guide_interaction_contract'], sort_keys=True)
                    values['guide_interaction_evaluation_json'] = json.dumps(interaction_evaluation, sort_keys=True)
                if dynamics_diagnostics:
                    values['dynamics_evaluation_contract_json']=json.dumps(dynamics_evaluation_contract,sort_keys=True)
                    if auxiliary_prediction is not None:
                        values['auxiliary_dynamics_predicted_latent']=auxiliary_prediction['latent'][0].cpu().numpy()
                        values['auxiliary_dynamics_teacher_latent']=auxiliary_target_latent[0].cpu().numpy()
                        values['auxiliary_dynamics_surface_fields']=auxiliary_prediction['fields'][0].cpu().numpy()*np.asarray(a['scale'])+np.asarray(a['mean'])
                        values['auxiliary_dynamics_surface_truth']=batch['targets'][0,:rollout_steps].cpu().numpy()*np.asarray(a['scale'])+np.asarray(a['mean'])
                        values['auxiliary_dynamics_lead_hours']=np.asarray(auxiliary_hours)
                np.savez_compressed(path,**values);saved=True
    if str(device).startswith('cuda'):torch.cuda.synchronize()
    report={'format':'climate_manifold.downstream_evaluation.v1','checkpoint_sha256':digest(checkpoint),
            'a_sha256':p['a_sha256'],'archive_sha256':a['archive_sha256'],
            'information_sha256':a['information_sha256'],'config':p['config'],'conditioning':p['conditioning'],
            'implementation':p['implementation'],'training_contract':p['training_contract'],
            'constants_sha256':p['constants_sha256'],
            'experiment':p.get('experiment',experiment_contract(p['config'])),
            'representation_sha256':(digest(checkpoint) if model.config.training_mode=='joint' and model.config.bridge!='raw'
                else p.get('representation_sha256',p['a_sha256'] if model.config.bridge!='raw' else None)),
            'representation_config':a['config'],
            'latent_shape':p.get('latent_shape'),
            'forecast_state_grid':p.get('forecast_state_grid'),
            'transport_contract':p.get('transport_contract'),
            'predictor_provenance':p.get('predictor_provenance'),
            'guide_contract':p.get('guide_contract'),
            'dynamics_contract':p.get('dynamics_contract'),
            'dynamics_auxiliary_config':p.get('dynamics_auxiliary_config'),
            'dynamics_latent_shape':p.get('dynamics_latent_shape'),
            'latent_diagnostic_semantics':dynamics_latent_audit,
            'regularization':p.get('regularization','legacy'),
            'objective_weights':p.get('objective_weights',{}),
            'constraint_pair':p.get('constraint_pair'),
            'constraint_path':p.get('constraint_path','forecast_trajectory'),
            'constraint_contract':p.get('constraint_contract'),
            'constraint_decoder':constraint_decoder_from_payload(p),
            'statistical_loss':statistical_config['kind'] if statistical_config else None,
            'statistical_loss_config':statistical_config,
            'statistical_flow_config':statistical_flow_config_from_payload(p),
            'conditional_flow_config':conditional_flow_config_from_payload(p),
            'split_objective_weights':p.get('split_objective_weights'),
            'forecast_parameters':p.get('forecast_parameters'),
            'constraint_parameters':p.get('constraint_parameters'),
            'pinn_config':a.get('pinn_config'),
            'initialization':p.get('initialization','pretrained'),
            'representation_training':p.get('representation_training','frozen' if model.config.bridge!='raw' else 'none'),
            'split':split,'origin_times':origins,'successful_origin_times':successful,'lead_hours':p['lead_hours'],
            'finite_forecast_fraction':len(successful)/len(origins),'failed_origins':failures,
            'scores':metrics.result(),'inference_seconds':inference_seconds,'evaluation_seconds':time.perf_counter()-start,
            'latent_diagnostics':latent_metrics.result() if latent_metrics is not None else None,
            'latent_diagnostic_failures':diagnostic_failures,
            'trainable_parameters':p['trainable_parameters'],'total_parameters':p['total_parameters'],
            'training_seconds':p['training_seconds'],'seed':p['options']['seed'],
            'cuda_peak_memory_bytes':(main_cuda_peak if dynamics_diagnostics else torch.cuda.max_memory_allocated()) if str(device).startswith('cuda') else None,
            'forecast_output_written':saved,
            'limits':'Scores are conditional on finite forecasts; failed origins are explicit. Gaussian uncertainty is marginal, not a coherent stochastic trajectory ensemble.'}
    if dynamics_diagnostics:
        report['dynamics_evaluation_contract']=dynamics_evaluation_contract
        report['dynamics_origin_selection']=dynamics_origin_selection
        report['main_forecast_signed_diagnostics']=main_signed.result()
        report['dynamics_auxiliary_evaluation']={
            'path':'E_d_observed_history_to_autoregressive_F_d_to_D_d',
            'downstream_forecast_model_used':False,'teacher_forcing':False,
            'lead_hours':auxiliary_hours,'origin_times':successful,
            'successful_origin_times':auxiliary_successful,
            'attempted_case_count':len(successful),'successful_case_count':len(auxiliary_successful),
            'finite_auxiliary_fraction':len(auxiliary_successful)/len(successful) if successful else None,
            'auxiliary_failed_origins':auxiliary_failures,
            'scores':auxiliary_metrics.result(),'signed_diagnostics':auxiliary_signed.result(),
            'latent_diagnostics':auxiliary_latent.result(),
            'diagnostic_seconds':auxiliary_seconds,
            'cuda_peak_memory_bytes':auxiliary_cuda_peak if str(device).startswith('cuda') else None,
            'runtime_scope':'auxiliary_rollout_plus_detached_truth_encoding_and_auxiliary_metric_accumulation_excludes_main_M_forecast',
            'memory_scope':'absolute_allocated_peak_during_auxiliary_audit_including_resident_model_batch_and_main_outputs',
            'limits':'This is the independent auxiliary F_d+D_d, not an autoregressive extension of downstream M. '
                     'Scores are conditional on finite auxiliary forecasts; failure coverage is separate from M. '
                     'Do not compare latent RMSE as a ranking of different trained encoders.',
        }
        report['inference_scope']='checkpoint_downstream_M_forecast_only_excludes_auxiliary_dynamics_audit'
        report['cuda_peak_memory_scope']='absolute_allocated_peak_during_main_forecast_including_resident_model_and_batch'
    if interaction_evaluation is not None:
        report['guide_interaction_contract'] = p['guide_interaction_contract']
        report['guide_interaction_evaluation'] = interaction_evaluation
        if interaction_evaluation['bypassed']:
            report['limits'] += (' Guide interaction T1 is bypassed in a checkpoint trained with it. '
                                 'This is a same-checkpoint inference intervention, not an independently trained single-stage baseline.')
    write_json(output,report);return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','archive','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--information');p.add_argument('--forecast-output')
    p.add_argument('--split',choices=['calibration','validation','test'],default='validation')
    p.add_argument('--max-cases',type=int,default=0);p.add_argument('--origin-stride',type=int,default=1)
    p.add_argument('--device',default='cpu')
    p.add_argument('--dynamics-diagnostics',action='store_true',
                   help='Audit the independent latent-rollout dynamics auxiliary head after the main forecast')
    p.add_argument('--dynamics-rollout-steps',type=int,
                   help='Optional auxiliary rollout horizon >= checkpoint horizon; restrict origins to original split label coverage')
    p.add_argument('--guide-interaction-override',choices=['checkpoint','off'],default='checkpoint',
                   help='checkpoint keeps the trained architecture; off bypasses T1 of a trained two-stage checkpoint as an inference ablation')
    r=evaluate(**vars(p.parse_args(argv)));print(json.dumps(r['scores']['aggregate']));return 0

if __name__=='__main__':raise SystemExit(main())
