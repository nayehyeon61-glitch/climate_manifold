"""Validate comparison contracts and produce a JSON/CSV benchmark table."""
import argparse
import csv
import json
import math
import re
from itertools import combinations
from pathlib import Path
import numpy as np
from ..train import write_json
from .statistical_objective import statistical_config_from_payload
from .statistical_flow import statistical_flow_config_from_payload
from .conditional_flow import conditional_flow_config_from_payload
from .protocol import experiment_contract,validate_experiment
from .climode_benchmark import benchmark,write_table
from .constraint_protocol import constraint_decoder_from_payload, normalize_constraint_contract


CONSTRAINT_GROUPS = {
    'pinn_statistical': {'pinn', 'statistical'},
    'pinn_static': {'pinn', 'static'},
    'statistical_static': {'statistical', 'static'},
}

# Preserve the historical three-pair suite; a statistical-only arm is opt-in.
SUPPORTED_CONSTRAINT_GROUPS = {**CONSTRAINT_GROUPS, 'statistical': {'statistical'}}


def _regime(report):
    """Missing metadata identifies the original frozen-representation protocol."""
    cfg = report['config']
    mode = cfg.get('training_mode', 'frozen')
    return {
        'training_mode': mode,
        'regularization': report.get('regularization', 'legacy'),
        'initialization': report.get('initialization', 'pretrained' if cfg['bridge']!='raw' else 'fresh'),
        'representation_training': report.get('representation_training',
            'not_applicable' if cfg['bridge']=='raw' else 'jointly_trained' if mode=='joint' else 'frozen'),
    }


def _arm(row):
    if row.get('constraint_pair'):
        name=row['constraint_pair'] + (':'+row['statistical_loss'] if row.get('statistical_loss') not in (None, 'w2') else '')
        if row.get('bridge') == 'guided':
            name = 'guided:'+row['guide_mode']+':'+name
        flow=row.get('statistical_flow_config')
        conditional=row.get('conditional_flow_config')
        return (name + (':flow='+format(flow['weight'],'.12g') if flow else '')
                + (':cfm='+format(conditional['weight'],'.12g') if conditional else ''))
    if row.get('bridge') == 'guided':
        return 'guided:'+row['guide_mode']+':'+row['regularization']
    if row['representation']=='raw':
        return 'raw'
    if (row['representation']=='climate_manifold' and row['training_mode']=='joint'
            and row['regularization']=='none'):
        return 'forecast_only'
    return row['representation']


def _validate_constraint_pairs(group):
    """Allow raw and validated guided forecast-only controls with observed constraints."""
    split = [row for row in group if row.get('constraint_pair') is not None]
    if not split:
        return
    for row in group:
        if row.get('constraint_pair') is not None:
            continue
        cfg = row['config']
        if cfg.get('bridge') == 'guided':
            if (cfg.get('model') != 'transformer' or cfg.get('training_mode') != 'joint'
                    or cfg.get('latent_layout') != 'spatial' or cfg.get('anchor') != 'none'
                    or row.get('initialization') != 'fresh' or row.get('regularization') != 'none'):
                raise ValueError('Guided forecast-only controls require fresh joint spatial Transformer and regularization=none')
            if any(row.get(key) is not None for key in (
                    'constraint_path','constraint_contract','constraint_decoder','split_objective_weights')):
                raise ValueError('Guided forecast-only controls must not declare a constraint objective')
            weights = row.get('objective_weights')
            if (not isinstance(weights, dict) or not weights
                    or any(isinstance(value, bool) or not isinstance(value, (int, float))
                           or not math.isfinite(value) or value != 0 for value in weights.values())):
                raise ValueError('Guided forecast-only controls require zero effective objective_weights')
            for candidate in split:
                if candidate['config']['bridge'] != 'guided':
                    continue
                for key in ('forecast_parameters','implementation','conditioning'):
                    if row.get(key) is None or row.get(key) != candidate.get(key):
                        raise ValueError('Unfair guided constraint comparison: mismatched '+key)
            continue
        if (cfg.get('bridge') != 'raw' or cfg.get('raw_backend') != 'matched'
                or cfg.get('training_mode') != 'joint' or cfg.get('anchor') != 'none'
                or row.get('initialization') != 'fresh' or row.get('regularization') != 'none'):
            raise ValueError('Unfair comparison: mixed constraint_path experiments require fresh joint matched raw controls')
        if (row.get('constraint_path') is not None or row.get('constraint_contract') is not None
                or row.get('constraint_decoder') is not None
                or row.get('split_objective_weights') is not None):
            raise ValueError('Raw controls must not declare a constraint_path or constraint objective')
        weights = row.get('objective_weights')
        if (not isinstance(weights, dict) or not weights
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) or value != 0 for value in weights.values())):
            raise ValueError('Raw controls require zero effective objective_weights')
        if row.get('constraint_parameters', 0) != 0:
            raise ValueError('Raw controls must have zero constraint_parameters')
    statistical_configs = {}
    common_flow = None
    common_conditional = None
    for row in group:
        config = statistical_config_from_payload(row)
        if config is not None:
            kind = config['kind']
            if kind in statistical_configs and statistical_configs[kind] != config:
                raise ValueError('Unfair comparison: mismatched statistical_loss_config')
            statistical_configs[kind] = config
        flow = statistical_flow_config_from_payload(row)
        if flow is not None:
            specification = {k:v for k,v in flow.items() if k != 'weight'}
            if common_flow is not None and specification != common_flow:
                raise ValueError('Unfair comparison: mismatched statistical_flow_config')
            common_flow = specification
        conditional = conditional_flow_config_from_payload(row)
        if conditional is not None:
            specification = {k:v for k,v in conditional.items() if k != 'weight'}
            if common_conditional is not None and specification != common_conditional:
                raise ValueError('Unfair comparison: mismatched conditional_flow_config')
            common_conditional = specification
    first = split[0]
    common_contract = None
    active_weights = {}
    pinn_config = None
    pinn_reports = [row for row in split if row['constraint_pair'] in ('pinn_statistical','pinn_static')]
    if any(row.get('pinn_config') is not None for row in pinn_reports):
        pinn_config = pinn_reports[0].get('pinn_config')
        if pinn_config is None or any(row.get('pinn_config') != pinn_config for row in pinn_reports):
            raise ValueError('Unfair comparison: mismatched pinn_config')
    for row in split:
        pair = row['constraint_pair']
        if pair not in SUPPORTED_CONSTRAINT_GROUPS:
            raise ValueError('Unknown constraint_pair: '+str(pair))
        if row.get('constraint_path') != 'observed_reconstruction':
            raise ValueError('Split experiments require constraint_path=observed_reconstruction')
        if row['config'].get('bridge') not in ('latent', 'guided') or row['config'].get('training_mode') != 'joint':
            raise ValueError('Split experiments require jointly trained latent or guided forecasts')
        decoder = constraint_decoder_from_payload(row)
        contract = normalize_constraint_contract(row.get('constraint_contract'))
        if (set(contract.get('groups', [])) != SUPPORTED_CONSTRAINT_GROUPS[pair]
                or contract.get('observed_pair') != 'origin-6h,origin'
                or contract.get('pinn_tendency_supervision') is not False):
            raise ValueError('Invalid reconstruction constraint_contract')
        common = {key:value for key,value in contract.items() if key not in ('groups','pair')}
        if common_contract is not None and decoder != common_contract['decoder']:
            raise ValueError('Unfair comparison: mismatched constraint_decoder scope')
        if common_contract is not None and common != common_contract:
            raise ValueError('Unfair comparison: mismatched constraint_contract')
        common_contract = common
        route_first = next(r for r in split if r['config']['bridge'] == row['config']['bridge'])
        if row.get('forecast_parameters') is None or row['forecast_parameters'] != route_first.get('forecast_parameters'):
            raise ValueError('Unfair comparison: mismatched forecast_parameters')
        weights = row.get('split_objective_weights')
        if not isinstance(weights,dict) or set(weights) != {'reconstruction','pinn','statistical','static'}:
            raise ValueError('Split experiments require split_objective_weights')
        for name, weight in weights.items():
            if isinstance(weight,bool) or not isinstance(weight,(int,float)) or not math.isfinite(weight) or weight < 0:
                raise ValueError('Invalid split_objective_weights.'+name)
            active = name=='reconstruction' or name in SUPPORTED_CONSTRAINT_GROUPS[pair]
            if active and weight <= 0:
                raise ValueError('Active constraint needs positive split_objective_weights.'+name)
            if not active and weight != 0:
                raise ValueError('Inactive constraint has nonzero split_objective_weights.'+name)
            if active:
                if name in active_weights and weight != active_weights[name]:
                    raise ValueError('Unfair comparison: mismatched shared split_objective_weights.'+name)
                active_weights[name] = weight
        # Enabling an observed CFM head adds auxiliary parameters. Only compare
        # repeated seeds with the same head presence as capacity-identical.
        same_pair = [r for r in split if r['constraint_pair']==pair
                     and r['config']['bridge']==row['config']['bridge']
                     and r['config'].get('guide_mode', 'learned')==row['config'].get('guide_mode', 'learned')
                     and bool(r.get('conditional_flow_config'))==bool(row.get('conditional_flow_config'))]
        for name in ('total_parameters','trainable_parameters','constraint_parameters','implementation'):
            if row.get(name) != same_pair[0].get(name):
                raise ValueError('Cannot pool different '+name+' as constraint_pair seeds')


def _comparison_training_contract(row, raw_and_split):
    """Only the validated auxiliary-route label differs for raw/guided controls."""
    contract = row.get('training_contract')
    if not raw_and_split:
        return contract
    if not isinstance(contract, dict):
        raise ValueError('Raw/reconstruction comparison requires training_contract')
    expected = 'observed_reconstruction' if row.get('constraint_pair') else None
    if contract.get('constraint_path') != expected:
        raise ValueError('Unfair comparison: invalid training_contract.constraint_path')
    return {key:value for key,value in contract.items() if key != 'constraint_path'}


def _validate_family_inputs(group):
    """Shared observed inputs are distinct from a model's forecast-state grid."""
    for key in ('grid','state_dim','history_steps','history_stride','step_hours'):
        values=[(row.get('representation_config') or {}).get(key) for row in group]
        if any(value is not None for value in values) and any(value!=values[0] for value in values):
            raise ValueError('Unfair comparison: mismatched input '+key)
    raw=[row for row in group if row['config']['bridge']=='raw']
    for row in raw[1:]:
        if row['config'].get('raw_backend','legacy')!=raw[0]['config'].get('raw_backend','legacy'):
            raise ValueError('Cannot pool different raw_backend implementations as same-family seeds')
        if row.get('implementation')!=raw[0].get('implementation'):
            raise ValueError('Cannot pool different implementation versions as same-family raw seeds')


def _validate_guides(group):
    """A statistical guide is field forecasting, with independently audited inputs."""
    if not group or group[0]['config']['model'] != 'transformer':
        return
    first = group[0]
    for row in group:
        cfg = row['config']
        for key, default in (('hidden_dim', 128), ('transformer_heads', 4),
                             ('weather_depth', 4), ('weather_patch_size', 2)):
            if cfg.get(key, default) != first['config'].get(key, default):
                raise ValueError('Unfair comparison: mismatched '+key)
        contract = row.get('guide_contract')
        if cfg['bridge'] != 'guided':
            if contract is not None:
                raise ValueError('guide_contract requires bridge=guided')
            continue
        mode = cfg.get('guide_mode', 'learned')
        expected = {
            'version': 'climate_manifold.statistical_guide.v1', 'mode': mode,
            'raw_input': True,
            'guide_input': 'observed_history_encoded_with_origin_information',
            'fusion': 'joint_raw_guide_token_self_attention',
            'forecast_output': 'physical_fields', 'forecast_decoder_used': False,
            'forecast_gradient_to_encoder': mode == 'learned', 'stochastic_sampling': False,
        }
        if mode not in ('learned', 'zero') or contract != expected:
            raise ValueError('Invalid guide_contract')
        if row.get('representation_config') is None:
            raise ValueError('Guided comparison requires representation_config')
    representations = [row for row in group if row['config']['bridge'] in ('latent','guided')]
    if representations and any(row.get('representation_config') != representations[0].get('representation_config')
                               for row in representations):
        raise ValueError('Unfair comparison: mismatched representation_config')


def _validate_weather_provenance(group):
    """Keep pinned weather cores and common capacity fixed across raw/E/F/D arms.

    State channels, variable names and spatial resolution deliberately differ
    between these arms. This check does not imply equal parameter counts or
    reproduction of a published pretrained benchmark. Older model families
    retain their existing report contract.
    """
    if not group or group[0]['config']['model'] not in ('fourcastnet', 'climax'):
        return
    family = group[0]['config']['model']
    text_fields = ('family', 'implementation', 'upstream_source', 'upstream_commit',
                   'implementation_variant')
    architecture = {'depth': ('weather_depth', 4),
                    'patch_size': ('weather_patch_size', 2),
                    'hidden_dim': ('hidden_dim', 128)}
    first = None
    for row in group:
        provenance = row.get('predictor_provenance')
        if not isinstance(provenance, dict):
            raise ValueError(f'{family} comparisons require predictor_provenance')
        for key in text_fields:
            if not isinstance(provenance.get(key), str) or not provenance[key].strip():
                raise ValueError('Invalid predictor_provenance.'+key)
        if provenance['family'] != family:
            raise ValueError('predictor_provenance.family differs from model')
        if provenance['implementation'] != row.get('implementation'):
            raise ValueError('predictor_provenance.implementation differs from report')
        if re.fullmatch(r'[0-9a-fA-F]{40}', provenance['upstream_commit']) is None:
            raise ValueError('predictor_provenance.upstream_commit must pin a full commit SHA')
        for key, (config_key, default) in architecture.items():
            value = provenance.get(key)
            configured = row['config'].get(config_key, default)
            if (isinstance(value, bool) or not isinstance(value, int) or value < 1
                    or isinstance(configured, bool) or not isinstance(configured, int)
                    or configured < 1):
                raise ValueError('Invalid predictor_provenance.'+key)
            if value != configured:
                raise ValueError('predictor_provenance.'+key+' differs from '+config_key)
        if not isinstance(provenance.get('pretrained'), bool):
            raise ValueError('Invalid predictor_provenance.pretrained')
        common = {key: provenance[key] for key in (*text_fields, *architecture, 'pretrained')}
        if first is not None:
            for key, value in common.items():
                if value != first[key]:
                    raise ValueError('Unfair comparison: mismatched predictor_provenance.'+key)
        first = common


def _validate_transport(group):
    """Check declared grid-index scaling; it is not exact geographic equivalence."""
    transport=[row for row in group if row['config']['model']=='climode' and
        (row['config']['bridge']=='raw' and row['config'].get('raw_backend','legacy')=='matched'
         or row['config']['bridge']=='latent' and row['config'].get('latent_layout')=='spatial')]
    if not transport or not any(row.get('transport_contract') is not None for row in transport):
        return  # Historical reports did not record effective transport bounds.
    if any(not isinstance(row.get('transport_contract'),dict) for row in transport):
        raise ValueError('Unfair comparison: missing transport_contract')
    first=transport[0]['transport_contract']
    speed=[]
    for row in transport:
        contract=row['transport_contract'];cfg=row['config'];rep=row.get('representation_config') or {}
        for key in ('reference_spatial_downsample','raw_velocity_rate_bound_per_day','uncertainty'):
            if contract.get(key)!=first.get(key):
                raise ValueError('Unfair comparison: mismatched transport_contract.'+key)
        factor=contract.get('reference_spatial_downsample')
        if isinstance(factor,bool) or not isinstance(factor,(int,float)) or not math.isfinite(factor) or factor<1 or int(factor)!=factor:
            raise ValueError('Invalid transport_contract.reference_spatial_downsample')
        if rep.get('spatial_downsample') is not None and rep['spatial_downsample']!=factor:
            raise ValueError('Unfair comparison: transport_contract.reference_spatial_downsample differs from representation_config')
        raw=cfg['bridge']=='raw'
        if contract.get('speed_scaling')!=('source_grid_factor' if raw else 'latent_grid'):
            raise ValueError('Invalid transport_contract.speed_scaling')
        for key in ('velocity_bound_cells_per_day','raw_velocity_rate_bound_per_day'):
            value=contract.get(key)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
                raise ValueError('Invalid transport_contract.'+key)
        normalized_speed=contract['velocity_bound_cells_per_day']/(factor if raw else 1)
        speed.append(normalized_speed)
        for key,value in (('latent_max_speed',normalized_speed),
                          ('latent_max_acceleration',contract['raw_velocity_rate_bound_per_day'])):
            if cfg.get(key) is not None and not math.isclose(cfg[key],value,rel_tol=1e-9,abs_tol=1e-12):
                raise ValueError('Unfair comparison: transport_contract differs from '+key)
    if any(not math.isclose(value,speed[0],rel_tol=1e-9,abs_tol=1e-12) for value in speed):
        raise ValueError('Unfair comparison: raw velocity_bound_cells_per_day must equal latent bound times reference_spatial_downsample')


def _direct_effects(pairs):
    """Same-family field scores, never latent-coordinate or mixed-unit ranks."""
    effects, missing = [], []
    for identity, candidate, baseline in pairs:
        if 'climode' not in candidate['scores'] or 'climode' not in baseline['scores']:
            missing.append({key:identity[key] for key in ('model','seed','candidate_arm','pair_key')})
            continue
        scores, reference = candidate['scores']['climode'], baseline['scores']['climode']
        if scores['protocol']!=reference['protocol']:
            raise ValueError('Unfair raw comparison: mismatched metric protocol/climatology')
        values, base_values = scores['per_variable'], reference['per_variable']
        if set(values)!=set(base_values):
            raise ValueError('Unfair raw comparison: mismatched variables')
        for report, result in ((candidate,scores),(baseline,reference)):
            if result['case_count']!=len(report['successful_origin_times']):
                raise ValueError('Metric count differs from successful origins')
        complete = all(report['finite_forecast_fraction']==1 and
                       report['successful_origin_times']==report['origin_times']
                       for report in (candidate,baseline))
        for variable, value in values.items():
            base = base_values[variable]
            if value['units']!=base['units']:
                raise ValueError('Unfair raw comparison: mismatched units')
            for item, report in ((value,candidate),(base,baseline)):
                if [row['lead_hours'] for row in item['by_lead']]!=report['lead_hours']:
                    raise ValueError('Metric lead hours differ from report')
            for row, ref in zip(value['by_lead'],base['by_lead']):
                def valid(metric):
                    return (complete and row.get(metric) is not None and ref.get(metric) is not None
                            and row.get(metric+'_valid_cases')==row['case_count']
                            and ref.get(metric+'_valid_cases')==ref['case_count'])
                def skill(metric):
                    return 1-row[metric]/ref[metric] if valid(metric) and ref[metric]>1e-15 else None
                effects.append({**identity,'variable':variable,'units':value['units'],
                    'lead_hours':row['lead_hours'],'ranking_allowed':complete,
                    'model_rmse':row['rmse'],'raw_rmse':ref['rmse'],
                    'rmse_reduction':ref['rmse']-row['rmse'] if valid('rmse') else None,
                    'rmse_skill_vs_raw':skill('rmse'),
                    'acc_difference':row['acc']-ref['acc'] if valid('acc') else None,
                    'crps_skill_vs_raw':skill('crps')})
    return {'available':bool(effects),'effects':effects,'missing_metric_pairs':missing,
            'ranking_allowed':bool(effects) and not missing and all(row['ranking_allowed'] for row in effects),
            'notes':['Pairs use the same predictor family, training seed, data, origins and lead times.',
                     'Raw inputs and spatial latent inputs can differ in channel count, resolution and total parameters; this is a whole-model comparison.',
                     'Matched transport bounds use declared grid-index scaling; pooled grids are not asserted to have exactly equal geographic displacement.',
                     'Forecast-only versus full E/F/D controls isolate added regularization; raw comparisons do not.',
                     'Positive physical RMSE/CRPS skill or ACC difference favors the named latent or guided candidate.']}


def _constraint_field_effects(identity, candidate, baseline):
    """Per-field pair replacements retain physical units and lead times."""
    if 'climode' not in candidate['scores'] or 'climode' not in baseline['scores']:
        return []
    # Reuse the strict metric checks but remove raw-specific vocabulary.
    result = _direct_effects([(identity,candidate,baseline)])
    rows = []
    for item in result['effects']:
        row = dict(item)
        row['baseline_rmse'] = row.pop('raw_rmse')
        row['rmse_skill_vs_baseline'] = row.pop('rmse_skill_vs_raw')
        row['crps_skill_vs_baseline'] = row.pop('crps_skill_vs_raw')
        rows.append(row)
    return rows


def compare(reports,output,climode_reference_reports=None):
    output=Path(output)
    if any(output.with_suffix(s).exists() for s in ('.json','.csv','.climode.csv','.climode-effects.csv','.raw-effects.csv','.constraint-effects.csv','.statistical-effects.csv','.flow-effects.csv','.conditional-flow-effects.csv','.guide-effects.csv','.route-effects.csv')) or output.exists():
        raise FileExistsError('Choose a new comparison path')
    data=[json.loads(Path(path).read_text()) for path in reports]
    if not data:raise ValueError('At least one evaluation report is required')
    contracts=[validate_experiment(row['config'],row.get('experiment',experiment_contract(row['config']))['suite']) for row in data]
    if len({contract['suite'] for contract in contracts}) != 1:
        raise ValueError('Do not mix primary latent experiments and auxiliary grid/anchor experiments')
    required=('format','a_sha256','archive_sha256','information_sha256','split','origin_times','lead_hours')
    for row in data:
        if row['format']!='climate_manifold.downstream_evaluation.v1':raise ValueError('Expected downstream evaluation reports')
        for key in required:
            if row[key]!=data[0][key]:raise ValueError('Unfair comparison: mismatched '+key)
    same_cases=all(row['successful_origin_times']==data[0]['successful_origin_times'] for row in data)
    # Same-family ablations must differ only in their declared representation/objective.
    # Objective weights are reported separately; the common supervision and budget
    # remain part of training_contract.
    for family in {row['config']['model'] for row in data}:
        group=[row for row in data if row['config']['model']==family]
        _validate_weather_provenance(group)
        _validate_guides(group)
        _validate_constraint_pairs(group)
        _validate_family_inputs(group)
        _validate_transport(group)
        first=group[0]
        raw_and_split=(any(row.get('constraint_pair') for row in group)
                       and any(row['config']['bridge']=='raw'
                               or row['config']['bridge']=='guided' and not row.get('constraint_pair')
                               for row in group))
        training_contract=_comparison_training_contract(first,raw_and_split)
        if len({_regime(row)['training_mode'] for row in group}) != 1:
            raise ValueError('Unfair comparison: mixed frozen and joint training_mode within '+family)
        for row in group[1:]:
            if _comparison_training_contract(row,raw_and_split)!=training_contract:
                raise ValueError('Unfair comparison: mismatched training_contract')
            for key in ('constants_sha256',):
                if row.get(key)!=first.get(key):raise ValueError('Unfair comparison: mismatched '+key)
            if _regime(row)['training_mode']=='joint' and _regime(row)['initialization']!=_regime(first)['initialization']:
                raise ValueError('Unfair comparison: mismatched initialization')
            if row.get('conditioning',{}).get('observed_information_available')!=first.get('conditioning',{}).get('observed_information_available'):
                raise ValueError('Unfair comparison: mismatched observed_information_available')
            for key in ('hidden_dim','ode_substeps','condition_information','climode_attention','climode_step_hours','velocity_iterations',
                        'latent_max_speed','latent_max_acceleration','latent_layout'):
                if row['config'].get(key)!=first['config'].get(key):raise ValueError('Unfair comparison: mismatched '+key)
        latent = [row for row in group if row['config']['bridge']=='latent']
        if latent and _regime(first)['training_mode']=='joint':
            for row in latent:
                if row.get('representation_config') is None:
                    raise ValueError('Joint latent comparison requires representation_config')
                if row['representation_config']!=latent[0]['representation_config']:
                    raise ValueError('Unfair comparison: mismatched representation_config')
                for key in ('conditioning','total_parameters','trainable_parameters'):
                    if row.get('constraint_pair') and key in ('total_parameters','trainable_parameters'):
                        continue  # PINN closure is auxiliary capacity, reported separately.
                    if row.get(key)!=latent[0].get(key):
                        raise ValueError('Unfair comparison: mismatched '+key)
                if row.get('regularization') not in ('none','full'):
                    raise ValueError('Joint latent comparison requires regularization=none or full')
    rows=[]
    for report,contract in zip(data,contracts):
        statistical_config = statistical_config_from_payload(report)
        cfg=report['config'];aggregate=report['scores']['aggregate'] or {}
        rows.append(dict(model=cfg['model'],bridge=cfg['bridge'],anchor=cfg['anchor'],seed=report['seed'],
            representation=contract['representation'],prediction_space=contract['prediction_space'],
            **_regime(report), representation_sha256=report.get('representation_sha256'),
            latent_layout=cfg.get('latent_layout','global') if cfg['bridge'] in ('latent','guided') else None,
            guide_mode=cfg.get('guide_mode','learned') if cfg['bridge']=='guided' else None,
            guide_contract=report.get('guide_contract'),
            latent_shape=report.get('latent_shape'),implementation=report.get('implementation'),
            predictor_provenance=report.get('predictor_provenance'),
            raw_backend=cfg.get('raw_backend','legacy') if cfg['bridge']=='raw' else None,
            objective_weights=report.get('objective_weights'),
            constraint_pair=report.get('constraint_pair'),constraint_path=report.get('constraint_path'),
            constraint_decoder=constraint_decoder_from_payload(report),
            constraint_contract=report.get('constraint_contract'),
            pinn_config=report.get('pinn_config'),
            statistical_loss=statistical_config['kind'] if statistical_config else None,
            statistical_loss_config=statistical_config,
            statistical_flow_config=statistical_flow_config_from_payload(report),
            conditional_flow_config=conditional_flow_config_from_payload(report),
            split_objective_weights=report.get('split_objective_weights'),
            forecast_parameters=report.get('forecast_parameters'),constraint_parameters=report.get('constraint_parameters'),
            normalized_rmse=aggregate.get('normalized_rmse'),wind_speed_rmse_mps=aggregate.get('wind_speed_rmse_mps'),
            finite_forecast_fraction=report['finite_forecast_fraction'],trainable_parameters=report['trainable_parameters'],
            total_parameters=report['total_parameters'],inference_seconds=report['inference_seconds'],
            training_seconds=report['training_seconds'],conditioning=report['conditioning']))
        rows[-1]['candidate_arm']=_arm(rows[-1])
    groups={}
    for row in rows:
        key='/'.join(row[k] for k in ('model','representation','bridge','anchor','training_mode','regularization','initialization'))
        if row['constraint_pair'] or row['bridge']=='guided':
            key += '/'+row['candidate_arm']
        if groups.get(key) and row['guide_contract'] != groups[key][0]['guide_contract']:
            raise ValueError('Cannot pool different guide_contract as seeds within '+key)
        if groups.get(key) and row['conditional_flow_config']!=groups[key][0]['conditional_flow_config']:
            raise ValueError('Cannot pool different conditional_flow_config as seeds within '+key)
        if groups.get(key) and row['statistical_flow_config']!=groups[key][0]['statistical_flow_config']:
            raise ValueError('Cannot pool different statistical_flow_config as seeds within '+key)
        if groups.get(key) and row['objective_weights']!=groups[key][0]['objective_weights']:
            raise ValueError('Cannot pool different objective_weights as seeds within '+key)
        if any(x['seed']==row['seed'] for x in groups.get(key,[])):
            raise ValueError('Duplicate seed within comparison group '+key)
        groups.setdefault(key,[]).append(row)
    summary={}
    for key,group in groups.items():
        if len({x['seed'] for x in group})!=len(group):raise ValueError('Duplicate seed within comparison group '+key)
        scores=[x['normalized_rmse'] for x in group if x['normalized_rmse'] is not None]
        summary[key]={'seeds':[x['seed'] for x in group],'mean_normalized_rmse':float(np.mean(scores)),
                      'sample_std_normalized_rmse':float(np.std(scores,ddof=1)) if len(scores)>1 else None,
                      'scored_runs':len(scores)} if scores else {'seeds':[x['seed'] for x in group],
                      'mean_normalized_rmse':None,'sample_std_normalized_rmse':None,'scored_runs':0}
    ranking_allowed=same_cases and all(r['finite_forecast_fraction']==1 for r in rows)
    paired=[];paired_summary={};direct_pairs=[];constraint_physical=[];statistical_physical=[];flow_physical=[];conditional_physical=[];guide_physical=[];route_physical=[]
    if ranking_allowed and contracts[0]['suite']=='primary':
        indexed={(r['model'],r['seed'],r['representation'],r['training_mode'],r['regularization']):r
                 for r in rows if not r['constraint_pair'] and r['bridge'] != 'guided'}
        source={id(row):report for row,report in zip(rows,data)}
        for row in rows:
            if row['representation']!='climate_manifold':continue
            mode=row['training_mode']
            if row['constraint_pair']:
                controls=[('raw','raw','none')]
            elif mode=='joint':
                # Full and forecast-only models share E/F/D architecture and both
                # learn their representation from future prediction supervision.
                controls=[('forecast_only','climate_manifold','none')] if row['regularization']=='full' else []
                controls += [(name,name,'none') for name in ('raw','plain_ae')]
            else:
                controls=[(name,name,row['regularization']) for name in ('raw','plain_ae')]
            for control,representation,regularization in controls:
                baseline=indexed.get((row['model'],row['seed'],representation,mode,regularization))
                if baseline is None:continue
                identity={'model':row['model'],'seed':row['seed'],'control':control,
                          'candidate_arm':row['candidate_arm'],'training_mode':mode,
                          'constraint_pair':row['constraint_pair'],'constraint_path':row['constraint_path'],
                          'pair_key':row['model']+'/'+row['candidate_arm']+'/vs_'+control,
                          'interpretation':('combined_physical_information_regularization'
                              if control=='forecast_only' else 'representation_and_capacity'
                              if row['candidate_arm']=='forecast_only' and control=='raw' else 'whole_model_comparison')}
                if control=='raw':
                    direct_pairs.append(({**identity,'raw_backend':baseline['raw_backend'],
                        'candidate_implementation':row['implementation'],
                        'raw_implementation':baseline['implementation'],
                        'candidate_parameters':row['total_parameters'],
                        'raw_parameters':baseline['total_parameters']},source[id(row)],source[id(baseline)]))
                error=baseline['normalized_rmse'];ours=row['normalized_rmse']
                if error is None or ours is None:continue
                paired.append({**identity,
                               'rmse_reduction':error-ours,
                               'relative_rmse_reduction':1-ours/error if error>1e-15 else None})
        # Hold the guided forecasting path fixed when adding observed encoder
        # constraints. Auxiliary decoder parameters can differ and stay explicit.
        guide_controls={(row['model'],row['seed'],row['guide_mode']):row for row in rows
                        if row['bridge']=='guided' and not row['constraint_pair']
                        and row['regularization']=='none'}
        for candidate in rows:
            if candidate['bridge']!='guided' or not candidate['constraint_pair']:
                continue
            baseline=guide_controls.get((candidate['model'],candidate['seed'],candidate['guide_mode']))
            if baseline is None:
                continue
            identity={'model':candidate['model'],'seed':candidate['seed'],
                'control':baseline['candidate_arm'],'candidate_arm':candidate['candidate_arm'],
                'training_mode':'joint', 'constraint_pair':candidate['constraint_pair'],
                'pair_key':candidate['model']+'/'+candidate['candidate_arm']+'/vs_'+baseline['candidate_arm'],
                'interpretation':'add_observed_encoder_constraints',
                'baseline_parameters':baseline['total_parameters'],
                'candidate_parameters':candidate['total_parameters'],
                'forecast_parameters':candidate['forecast_parameters']}
            error,ours=baseline['normalized_rmse'],candidate['normalized_rmse']
            if error is not None and ours is not None:
                paired.append({**identity,'rmse_reduction':error-ours,
                    'relative_rmse_reduction':1-ours/error if error>1e-15 else None})
            guide_physical.extend(_constraint_field_effects(
                identity,source[id(candidate)],source[id(baseline)]))
        for family,seed in sorted({(r['model'],r['seed']) for r in rows if r['constraint_pair']}):
            variants = sorted(
                (r for r in rows if r['model']==family and r['seed']==seed and r['constraint_pair']),
                key=lambda r:(list(SUPPORTED_CONSTRAINT_GROUPS).index(r['constraint_pair']),
                              r['bridge']!='latent', r.get('guide_mode')!='zero',
                              r['statistical_loss']=='kl_entropy',
                              (r['statistical_flow_config'] or {}).get('weight',0.),
                              (r['conditional_flow_config'] or {}).get('weight',0.)))
            for baseline,candidate in combinations(variants,2):
                baseline_arm,candidate_arm=baseline['candidate_arm'],candidate['candidate_arm']
                if baseline['bridge'] != candidate['bridge']:
                    if (baseline['constraint_pair'] != candidate['constraint_pair']
                            or baseline['statistical_loss_config'] != candidate['statistical_loss_config']
                            or baseline['statistical_flow_config'] != candidate['statistical_flow_config']
                            or baseline['conditional_flow_config'] != candidate['conditional_flow_config']):
                        continue
                    identity={'model':family,'seed':seed,'control':baseline_arm,
                        'candidate_arm':candidate_arm,'training_mode':'joint',
                        'pair_key':family+'/'+candidate_arm+'/vs_'+baseline_arm,
                        'interpretation':'forecasting_route_and_capacity',
                        'baseline_bridge':baseline['bridge'],'candidate_bridge':candidate['bridge'],
                        'baseline_parameters':baseline['total_parameters'],
                        'candidate_parameters':candidate['total_parameters']}
                    error,ours=baseline['normalized_rmse'],candidate['normalized_rmse']
                    if error is not None and ours is not None:
                        paired.append({**identity,'rmse_reduction':error-ours,
                            'relative_rmse_reduction':1-ours/error if error>1e-15 else None})
                    route_physical.extend(_constraint_field_effects(
                        identity,source[id(candidate)],source[id(baseline)]))
                    continue
                guide_effect = baseline.get('guide_mode') != candidate.get('guide_mode')
                if guide_effect and (baseline['constraint_pair'] != candidate['constraint_pair']
                        or baseline['statistical_loss_config'] != candidate['statistical_loss_config']
                        or baseline['statistical_flow_config'] != candidate['statistical_flow_config']
                        or baseline['conditional_flow_config'] != candidate['conditional_flow_config']):
                    continue
                same_pair=baseline['constraint_pair']==candidate['constraint_pair']
                base_kind,new_kind=baseline['statistical_loss'],candidate['statistical_loss']
                base_flow,new_flow=baseline['statistical_flow_config'],candidate['statistical_flow_config']
                base_cond,new_cond=baseline['conditional_flow_config'],candidate['conditional_flow_config']
                if base_cond!=new_cond and base_flow!=new_flow:
                    continue  # These are two distinct auxiliary flow mechanisms.
                if base_kind and new_kind and base_kind!=new_kind and base_cond!=new_cond:
                    continue
                if not same_pair and base_kind and new_kind and base_cond!=new_cond:
                    continue
                if base_kind and new_kind and base_kind!=new_kind and base_flow!=new_flow:
                    continue  # Do not change reconstruction metric and temporal loss together.
                if not same_pair and base_kind and new_kind and base_flow!=new_flow:
                    continue  # Pair changes keep the shared statistical objective fixed.
                if not same_pair and base_kind and new_kind and base_kind!=new_kind:
                    continue  # Changing both the constraint pair and metric is confounded.
                identity={'model':family,'seed':seed,'control':baseline_arm,
                    'candidate_arm':candidate_arm,'training_mode':'joint',
                    'pair_key':family+'/'+candidate_arm+'/vs_'+baseline_arm}
                conditional_effect = same_pair and base_kind==new_kind and base_cond!=new_cond
                flow_effect = same_pair and base_kind==new_kind and base_flow!=new_flow and not conditional_effect
                if guide_effect:
                    identity.update(interpretation='change_guide_input',
                        baseline_guide_mode=baseline['guide_mode'], candidate_guide_mode=candidate['guide_mode'],
                        baseline_parameters=baseline['total_parameters'], candidate_parameters=candidate['total_parameters'])
                elif conditional_effect:
                    identity.update(interpretation='change_observed_conditional_flow',
                        constraint_pair=candidate['constraint_pair'],statistical_loss=new_kind,
                        baseline_conditional_flow_weight=(base_cond or {}).get('weight',0.),
                        candidate_conditional_flow_weight=(new_cond or {}).get('weight',0.),
                        baseline_parameters=baseline['total_parameters'],
                        candidate_parameters=candidate['total_parameters'])
                elif flow_effect:
                    identity.update(interpretation='change_statistical_flow_weight',
                        constraint_pair=candidate['constraint_pair'],statistical_loss=new_kind,
                        baseline_flow_weight=(base_flow or {}).get('weight',0.),
                        candidate_flow_weight=(new_flow or {}).get('weight',0.))
                elif same_pair:
                    identity.update(interpretation='replace_statistical_loss',
                        constraint_pair=candidate['constraint_pair'],
                        baseline_statistical_loss=base_kind,candidate_statistical_loss=new_kind,
                        statistical_flow_weight=(new_flow or {}).get('weight',0.),
                        conditional_flow_weight=(new_cond or {}).get('weight',0.))
                else:
                    base_groups=SUPPORTED_CONSTRAINT_GROUPS[baseline['constraint_pair']]
                    candidate_groups=SUPPORTED_CONSTRAINT_GROUPS[candidate['constraint_pair']]
                    if len(base_groups) != 2 or len(candidate_groups) != 2:
                        continue  # Single-group ablations are not pair replacements.
                    identity.update(interpretation='replace_one_constraint_group_with_one_fixed',
                        fixed_group=next(iter(base_groups & candidate_groups)),
                        removed_group=next(iter(base_groups-candidate_groups)),
                        added_group=next(iter(candidate_groups-base_groups)))
                error=baseline['normalized_rmse'];ours=candidate['normalized_rmse']
                if error is not None and ours is not None:
                    paired.append({**identity,'rmse_reduction':error-ours,
                        'relative_rmse_reduction':1-ours/error if error>1e-15 else None})
                destination=guide_physical if guide_effect else conditional_physical if conditional_effect else flow_physical if flow_effect else statistical_physical if same_pair else constraint_physical
                destination.extend(_constraint_field_effects(
                    identity,source[id(candidate)],source[id(baseline)]))
        for row in paired:
            key=row['pair_key'] if row['training_mode']=='joint' else row['model']+'/vs_'+row['control']
            paired_summary.setdefault(key,[]).append(row)
        paired_summary={key:{'seeds':[r['seed'] for r in group],
                             'mean_rmse_reduction':float(np.mean([r['rmse_reduction'] for r in group])),
                             'sample_std_rmse_reduction':float(np.std([r['rmse_reduction'] for r in group],ddof=1)) if len(group)>1 else None}
                        for key,group in paired_summary.items()}
    result={'format':'climate_manifold.comparison.v1','rows':rows,'seed_summary':summary,
        'experiment_suite':contracts[0]['suite'],'paired_effects':paired,'paired_summary':paired_summary,
        'constraint_pair_effects':constraint_physical,
        'statistical_loss_effects':statistical_physical,
        'statistical_flow_effects':flow_physical,
        'conditional_flow_effects':conditional_physical,
        'guide_effects':guide_physical,'route_effects':route_physical,
        'direct_comparison':_direct_effects(direct_pairs),
        'same_successful_origins':same_cases,'ranking_allowed':ranking_allowed,
        'notes':['Inspect per-variable and per-lead physical scores in the original reports.',
                 'Different failure subsets cannot be ranked as an equal-case comparison.',
                 'Joint runs relearn encoder, predictor and decoder per seed; their spread includes all three components.',
                 'Legacy frozen runs share fixed representations when representation hashes match; forecast seeds do not measure representation pretraining variance.',
                 'Positive paired RMSE reduction favors the named candidate_arm; pairs share the training seed.',
                 'Joint forecast_only/full pairs isolate the combined added regularization under matched architecture, inputs, initialization and training budget; they do not isolate PINN alone.',
                 'Reconstruction constraint pairs replace one group while holding one fixed; they do not identify a single-group causal benefit or prove reduced overfitting.',
                 'Flow ablations hold the base W2/KL objective fixed; flow also supplies future dynamic-information supervision through auxiliary decoders, so its effects include that added supervision.',
                 'Observed conditional-flow ablations add a separate auxiliary network and future-information supervision; their effects include capacity and supervision changes, not direct gradient to the forecast predictor.',
                 'PINN pairs may add auxiliary closure parameters; forecast and constraint parameter counts are reported separately.',
                 'Guided forecasts predict physical fields from raw observations and encoded observed-history guidance; they have no latent forecast trajectory.',
                 'Guided versus raw or latent comparisons change routes and parameter counts; matched Transformer width/depth does not imply equal capacity.',
                 'Guided forecast-only versus constrained guide holds the forecasting path fixed; the effect includes observed reconstruction/statistical supervision and any added auxiliary decoder capacity, not the statistical loss alone.',
                 'Zero-guide ablations replace guide tokens by zero but retain the same information-route supervision; they do not constitute stochastic encoder experiments.',
                 'Raw-versus-latent and plain-AE comparisons change representation or supervision and are whole-model comparisons, not causal evidence for physical constraints.',
                 'Latent coordinate errors are within-representation diagnostics, never cross-encoder rankings.',
                 'Enriched A supplies extra dynamic information to decoded ClimODE; use surface A to isolate representation alone.']}
    references = ([json.loads(Path(path).read_text()) for path in climode_reference_reports]
                  if climode_reference_reports is not None else None)
    if references is not None or all('climode' in r['scores'] for r in data):
        result['climode_benchmark'] = benchmark(data, references)
    else:
        result['climode_benchmark'] = {'available':False,'reason':'Re-evaluate checkpoints for ClimODE-style metrics'}
    write_json(output,result)
    write_table(output.with_suffix('.climode.csv'),result['climode_benchmark'].get('rows',[]))
    write_table(output.with_suffix('.climode-effects.csv'),result['climode_benchmark'].get('effects',[]))
    write_table(output.with_suffix('.raw-effects.csv'),result['direct_comparison']['effects'])
    write_table(output.with_suffix('.constraint-effects.csv'),result['constraint_pair_effects'])
    write_table(output.with_suffix('.statistical-effects.csv'),result['statistical_loss_effects'])
    write_table(output.with_suffix('.flow-effects.csv'),result['statistical_flow_effects'])
    write_table(output.with_suffix('.conditional-flow-effects.csv'),result['conditional_flow_effects'])
    write_table(output.with_suffix('.guide-effects.csv'),result['guide_effects'])
    write_table(output.with_suffix('.route-effects.csv'),result['route_effects'])
    with output.with_suffix('.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader()
        for row in rows:
            structured = ('guide_contract','conditioning','objective_weights','constraint_contract','split_objective_weights','pinn_config','statistical_loss_config','statistical_flow_config','conditional_flow_config')
            writer.writerow({**row,**{key:json.dumps(row[key],sort_keys=True) for key in structured}})
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reports',nargs='+',required=True);p.add_argument('--output',required=True)
    p.add_argument('--climode-reference-reports',nargs='+',help='Raw ClimODE reports matched by forecast seed; separate field benchmark')
    r=compare(**vars(p.parse_args(argv)));print(json.dumps(r['seed_summary'],indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
