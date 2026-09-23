"""Validate comparison contracts and produce a JSON/CSV benchmark table."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from ..train import write_json
from .protocol import experiment_contract,validate_experiment


def compare(reports,output):
    output=Path(output)
    if output.exists() or output.with_suffix('.csv').exists():raise FileExistsError('Choose a new comparison path')
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
    # Within each model family compare bridges under identical predictor settings.
    for family in {row['config']['model'] for row in data}:
        group=[row for row in data if row['config']['model']==family]
        first=group[0]
        for row in group[1:]:
            for key in ('training_contract','constants_sha256'):
                if row.get(key)!=first.get(key):raise ValueError('Unfair comparison: mismatched '+key)
            for key in ('hidden_dim','ode_substeps','condition_information','climode_attention','climode_step_hours','velocity_iterations'):
                if row['config'].get(key)!=first['config'].get(key):raise ValueError('Unfair comparison: mismatched '+key)
    rows=[]
    for report,contract in zip(data,contracts):
        cfg=report['config'];aggregate=report['scores']['aggregate'] or {}
        rows.append(dict(model=cfg['model'],bridge=cfg['bridge'],anchor=cfg['anchor'],seed=report['seed'],
            representation=contract['representation'],prediction_space=contract['prediction_space'],
            representation_sha256=report.get('representation_sha256'),
            normalized_rmse=aggregate.get('normalized_rmse'),wind_speed_rmse_mps=aggregate.get('wind_speed_rmse_mps'),
            finite_forecast_fraction=report['finite_forecast_fraction'],trainable_parameters=report['trainable_parameters'],
            total_parameters=report['total_parameters'],inference_seconds=report['inference_seconds'],
            training_seconds=report['training_seconds'],conditioning=report['conditioning']))
    groups={}
    for row in rows:
        key='/'.join(row[k] for k in ('model','representation','bridge','anchor'))
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
        indexed={(r['model'],r['seed'],r['representation']):r for r in rows}
        for row in rows:
            if row['representation']!='climate_manifold':continue
            for control in ('raw','plain_ae'):
                baseline=indexed.get((row['model'],row['seed'],control))
                if baseline is None:continue
                error=baseline['normalized_rmse'];ours=row['normalized_rmse']
                paired.append({'model':row['model'],'seed':row['seed'],'control':control,
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
                 'Repeated seeds share one fixed A checkpoint; this does not measure A pretraining variance.',
                 'Check representation hashes: primary runner fixes both A and AE across forecast seeds; externally mixed AE checkpoints also include AE pretraining variation.',
                 'Positive paired RMSE reduction favors Climate Manifold; pairing uses the same forecast seed.',
                 'Latent coordinate errors are within-representation diagnostics, never cross-encoder rankings.',
                 'The AE control changes the representation training objective and budget; it does not isolate PINN alone.',
                 'Enriched A supplies extra dynamic information to decoded ClimODE; use surface A to isolate representation alone.']}
    write_json(output,result)
    with output.with_suffix('.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader()
        for row in rows:writer.writerow({**row,'conditioning':json.dumps(row['conditioning'],sort_keys=True)})
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reports',nargs='+',required=True);p.add_argument('--output',required=True)
    r=compare(**vars(p.parse_args(argv)));print(json.dumps(r['seed_summary'],indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
