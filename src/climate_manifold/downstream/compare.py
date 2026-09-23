"""Validate comparison contracts and produce a JSON/CSV benchmark table."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from ..train import write_json
from .protocol import experiment_contract,validate_experiment
from .climode_benchmark import benchmark,write_table


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


def compare(reports,output,climode_reference_reports=None):
    output=Path(output)
    if any(output.with_suffix(s).exists() for s in ('.json','.csv','.climode.csv','.climode-effects.csv')) or output.exists():
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
        first=group[0]
        if len({_regime(row)['training_mode'] for row in group}) != 1:
            raise ValueError('Unfair comparison: mixed frozen and joint training_mode within '+family)
        for row in group[1:]:
            for key in ('training_contract','constants_sha256'):
                if row.get(key)!=first.get(key):raise ValueError('Unfair comparison: mismatched '+key)
            if _regime(row)['training_mode']=='joint' and _regime(row)['initialization']!=_regime(first)['initialization']:
                raise ValueError('Unfair comparison: mismatched initialization')
            for key in ('hidden_dim','ode_substeps','condition_information','climode_attention','climode_step_hours','velocity_iterations'):
                if row['config'].get(key)!=first['config'].get(key):raise ValueError('Unfair comparison: mismatched '+key)
        latent = [row for row in group if row['config']['bridge']=='latent']
        if latent and _regime(first)['training_mode']=='joint':
            for row in latent:
                if row.get('representation_config') is None:
                    raise ValueError('Joint latent comparison requires representation_config')
                if row['representation_config']!=latent[0]['representation_config']:
                    raise ValueError('Unfair comparison: mismatched representation_config')
                for key in ('conditioning','total_parameters','trainable_parameters'):
                    if row.get(key)!=latent[0].get(key):
                        raise ValueError('Unfair comparison: mismatched '+key)
                if row.get('regularization') not in ('none','full'):
                    raise ValueError('Joint latent comparison requires regularization=none or full')
    rows=[]
    for report,contract in zip(data,contracts):
        cfg=report['config'];aggregate=report['scores']['aggregate'] or {}
        rows.append(dict(model=cfg['model'],bridge=cfg['bridge'],anchor=cfg['anchor'],seed=report['seed'],
            representation=contract['representation'],prediction_space=contract['prediction_space'],
            **_regime(report), representation_sha256=report.get('representation_sha256'),
            objective_weights=report.get('objective_weights'),
            normalized_rmse=aggregate.get('normalized_rmse'),wind_speed_rmse_mps=aggregate.get('wind_speed_rmse_mps'),
            finite_forecast_fraction=report['finite_forecast_fraction'],trainable_parameters=report['trainable_parameters'],
            total_parameters=report['total_parameters'],inference_seconds=report['inference_seconds'],
            training_seconds=report['training_seconds'],conditioning=report['conditioning']))
    groups={}
    for row in rows:
        key='/'.join(row[k] for k in ('model','representation','bridge','anchor','training_mode','regularization','initialization'))
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
    paired=[];paired_summary={}
    if ranking_allowed and contracts[0]['suite']=='primary':
        indexed={(r['model'],r['seed'],r['representation'],r['training_mode'],r['regularization']):r for r in rows}
        for row in rows:
            if row['representation']!='climate_manifold':continue
            mode=row['training_mode']
            if mode=='joint':
                # Full and forecast-only models share E/F/D architecture and both
                # learn their representation from future prediction supervision.
                if row['regularization']!='full':continue
                controls=[('forecast_only','climate_manifold','none')]
                controls += [(name,name,'none') for name in ('raw','plain_ae')]
            else:
                controls=[(name,name,row['regularization']) for name in ('raw','plain_ae')]
            for control,representation,regularization in controls:
                baseline=indexed.get((row['model'],row['seed'],representation,mode,regularization))
                if baseline is None:continue
                error=baseline['normalized_rmse'];ours=row['normalized_rmse']
                if error is None or ours is None:continue
                paired.append({'model':row['model'],'seed':row['seed'],'control':control,
                               'training_mode':mode,
                               'interpretation':('combined_physical_information_regularization'
                                   if control=='forecast_only' else 'whole_model_comparison'),
                               'rmse_reduction':error-ours,
                               'relative_rmse_reduction':1-ours/error if error>1e-15 else None})
        for row in paired:
            paired_summary.setdefault(row['model']+'/vs_'+row['control'],[]).append(row)
        paired_summary={key:{'seeds':[r['seed'] for r in group],
                             'mean_rmse_reduction':float(np.mean([r['rmse_reduction'] for r in group])),
                             'sample_std_rmse_reduction':float(np.std([r['rmse_reduction'] for r in group],ddof=1)) if len(group)>1 else None}
                        for key,group in paired_summary.items()}
    result={'format':'climate_manifold.comparison.v1','rows':rows,'seed_summary':summary,
        'experiment_suite':contracts[0]['suite'],'paired_effects':paired,'paired_summary':paired_summary,
        'same_successful_origins':same_cases,'ranking_allowed':ranking_allowed,
        'notes':['Inspect per-variable and per-lead physical scores in the original reports.',
                 'Different failure subsets cannot be ranked as an equal-case comparison.',
                 'Joint runs relearn encoder, predictor and decoder per seed; their spread includes all three components.',
                 'Legacy frozen runs share fixed representations when representation hashes match; forecast seeds do not measure representation pretraining variance.',
                 'Positive paired RMSE reduction favors the full Climate Manifold model; pairs share the training seed.',
                 'Joint forecast_only/full pairs isolate the combined added regularization under matched architecture, inputs, initialization and training budget; they do not isolate PINN alone.',
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
    with output.with_suffix('.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader()
        for row in rows:writer.writerow({**row,'conditioning':json.dumps(row['conditioning'],sort_keys=True),
                                          'objective_weights':json.dumps(row['objective_weights'],sort_keys=True)})
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reports',nargs='+',required=True);p.add_argument('--output',required=True)
    p.add_argument('--climode-reference-reports',nargs='+',help='Raw ClimODE reports matched by forecast seed; separate field benchmark')
    r=compare(**vars(p.parse_args(argv)));print(json.dumps(r['seed_summary'],indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
