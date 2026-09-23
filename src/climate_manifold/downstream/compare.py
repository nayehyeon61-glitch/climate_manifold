"""Validate comparison contracts and produce a JSON/CSV benchmark table."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from ..train import write_json


def compare(reports,output):
    output=Path(output)
    if output.exists() or output.with_suffix('.csv').exists():raise FileExistsError('Choose a new comparison path')
    data=[json.loads(Path(path).read_text()) for path in reports]
    if not data:raise ValueError('At least one evaluation report is required')
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
    for report in data:
        cfg=report['config'];aggregate=report['scores']['aggregate'] or {}
        rows.append(dict(model=cfg['model'],bridge=cfg['bridge'],anchor=cfg['anchor'],seed=report['seed'],
            normalized_rmse=aggregate.get('normalized_rmse'),wind_speed_rmse_mps=aggregate.get('wind_speed_rmse_mps'),
            finite_forecast_fraction=report['finite_forecast_fraction'],trainable_parameters=report['trainable_parameters'],
            total_parameters=report['total_parameters'],inference_seconds=report['inference_seconds'],
            training_seconds=report['training_seconds'],conditioning=report['conditioning']))
    groups={}
    for row in rows:
        key='/'.join(row[k] for k in ('model','bridge','anchor'))
        if row['normalized_rmse'] is not None:groups.setdefault(key,[]).append(row)
    summary={}
    for key,group in groups.items():
        if len({x['seed'] for x in group})!=len(group):raise ValueError('Duplicate seed within comparison group '+key)
        scores=[x['normalized_rmse'] for x in group]
        summary[key]={'seeds':[x['seed'] for x in group],'mean_normalized_rmse':float(np.mean(scores)),
                      'sample_std_normalized_rmse':float(np.std(scores,ddof=1)) if len(scores)>1 else None}
    result={'format':'climate_manifold.comparison.v1','rows':rows,'seed_summary':summary,
        'same_successful_origins':same_cases,'ranking_allowed':same_cases and all(r['finite_forecast_fraction']==1 for r in rows),
        'notes':['Inspect per-variable and per-lead physical scores in the original reports.',
                 'Different failure subsets cannot be ranked as an equal-case comparison.',
                 'Repeated seeds share one fixed A checkpoint; this does not measure A pretraining variance.',
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
