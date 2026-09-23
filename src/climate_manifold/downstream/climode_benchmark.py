"""Physical-field comparison against matched raw ClimODE forecasts."""
import argparse
import csv
import json
from pathlib import Path
from ..train import write_json


def _a_hash(report):
    if report['format'] == 'climate_manifold.dynamics_evaluation.v1':
        return report['checkpoint_sha256']
    if report['format'] != 'climate_manifold.downstream_evaluation.v1':
        raise ValueError('Expected downstream or pure A dynamics evaluation report')
    return report['a_sha256']


def _identity(report):
    cfg = report['config']
    pure_a = report['format'] == 'climate_manifold.dynamics_evaluation.v1'
    return dict(model='a_drift' if pure_a else cfg['model'],
                bridge='latent' if pure_a else cfg['bridge'],
                representation=('climate_manifold' if pure_a else
                    'raw' if cfg['bridge']=='raw' else cfg.get('representation','climate_manifold')),
                seed=None if pure_a else report['seed'],
                training_mode='a_only' if pure_a else cfg.get('training_mode','frozen'),
                regularization=report.get('regularization','legacy'),
                initialization=report.get('initialization','pretrained' if pure_a or cfg['bridge']!='raw' else 'fresh'),
                representation_sha256=(report['checkpoint_sha256'] if pure_a else report.get('representation_sha256')))


def benchmark(reports, references=None):
    """No architecture substitution: compares final fields across explicit suites."""
    if not reports:
        raise ValueError('At least one candidate report is required')
    if references is None:
        references = [r for r in reports if r['config'].get('model')=='climode'
                      and r['config'].get('bridge')=='raw' and r['config'].get('anchor')=='none']
    all_reports = reports + references
    first = reports[0]
    # A fixed checkpoint is part of the old frozen protocol, but independently
    # jointly trained candidates need not share a pretrained A. Final field
    # comparisons still require exact data, cases, leads and metric protocol.
    joint = any(r['config'].get('training_mode')=='joint' for r in all_reports)
    for report in all_reports:
        candidate_a = _a_hash(report)  # Also validates the report format in joint comparisons.
        if not joint and candidate_a != _a_hash(first):
            raise ValueError('Unfair ClimODE comparison: mismatched a_sha256')
        for key in ('archive_sha256','information_sha256','split','origin_times','lead_hours'):
            if report[key] != first[key]:
                raise ValueError('Unfair ClimODE comparison: mismatched '+key)
        if not report['origin_times']:
            raise ValueError('No evaluation origins')
        if 'climode' not in report['scores']:
            raise ValueError('Re-evaluate existing checkpoints to obtain scores.climode')
        if report['scores']['climode']['protocol'] != first['scores']['climode']['protocol']:
            raise ValueError('Unfair ClimODE comparison: mismatched metric protocol/climatology')
        if report['scores']['climode']['case_count'] != len(report['successful_origin_times']):
            raise ValueError('Metric count differs from successful origins')
        variables = report['scores']['climode']['per_variable']
        if set(variables) != set(first['scores']['climode']['per_variable']):
            raise ValueError('Unfair ClimODE comparison: mismatched variables')
        for name, value in variables.items():
            if value['units'] != first['scores']['climode']['per_variable'][name]['units']:
                raise ValueError('Unfair ClimODE comparison: mismatched units')
            if [r['lead_hours'] for r in value['by_lead']] != report['lead_hours']:
                raise ValueError('Metric lead hours differ from report')
    ref_by_seed = {}
    for ref in references:
        cfg = ref['config']
        if cfg.get('model')!='climode' or cfg.get('bridge')!='raw' or cfg.get('anchor')!='none':
            raise ValueError('Reference must be raw ClimODE with anchor=none')
        if ref['seed'] in ref_by_seed:
            raise ValueError('Duplicate raw ClimODE reference seed')
        ref_by_seed[ref['seed']] = ref
    def complete(report):
        return (report['finite_forecast_fraction']==1 and
                report['successful_origin_times']==report['origin_times'])
    rows, effects, unmatched = [], [], []
    for report in reports:
        identity = _identity(report)
        values = report['scores']['climode']['per_variable']
        for variable, value in values.items():
            for lead in value['by_lead']:
                rows.append(dict(**identity, variable=variable, units=value['units'], **lead,
                                 finite_forecast_fraction=report['finite_forecast_fraction']))
        if identity['model']=='climode' and identity['bridge']=='raw':
            continue
        refs = references if identity['model']=='a_drift' else ([ref_by_seed[identity['seed']]]
               if identity['seed'] in ref_by_seed else [])
        if not refs:
            unmatched.append(identity)
        for ref in refs:
            allowed = complete(report) and complete(ref)
            for variable, value in values.items():
                baseline = ref['scores']['climode']['per_variable'][variable]
                for row, base in zip(value['by_lead'], baseline['by_lead']):
                    def skill(key):
                        score, denominator = row[key], base[key]
                        full = (row[key+'_valid_cases']==row['case_count'] and
                                base[key+'_valid_cases']==base['case_count'])
                        return 1-score/denominator if (allowed and full and score is not None
                               and denominator is not None and denominator>1e-15) else None
                    acc_full = (row['acc_valid_cases']==row['case_count'] and
                                base['acc_valid_cases']==base['case_count'])
                    effects.append(dict(**identity, reference_seed=ref['seed'], variable=variable,
                        units=value['units'], lead_hours=row['lead_hours'], ranking_allowed=allowed,
                        model_rmse=row['rmse'], reference_rmse=base['rmse'],
                        rmse_skill_vs_climode=skill('rmse'), crps_skill_vs_climode=skill('crps'),
                        acc_difference=(row['acc']-base['acc'] if allowed and acc_full else None)))
    return {'format':'climate_manifold.climode_benchmark.v1',
            'protocol':first['scores']['climode']['protocol'], 'split':first['split'],
            'reference':'raw ClimODE custom-data adaptation', 'reference_seeds':list(ref_by_seed),
            'rows':rows, 'effects':effects, 'unmatched_candidates':unmatched,
            'ranking_allowed':bool(effects) and all(e['ranking_allowed'] for e in effects) and not unmatched,
            'notes':['Primary evidence is per-variable/per-lead physical RMSE and ACC; no mixed-unit scalar rank.',
                     'Positive RMSE/CRPS skill or ACC difference favors candidate; undefined ACC prevents ACC comparison.',
                     'Training seeds are paired; joint candidates relearn encoder/predictor/decoder, while pure A drift is fixed and compared to each baseline seed.',
                     'Joint candidates may have different A initialization hashes; data, splits, cases, leads and metric protocol must match.',
                     'Cross-family field comparisons do not isolate the manifold effect; use matched E/F/D forecast-only versus regularized joint controls for that question.',
                     'References use the same data/splits/origins/leads and train climatology, not published paper scores.']}


def write_table(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', nargs='+', required=True)
    parser.add_argument('--climode-reference-reports', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    paths = [output, output.with_suffix('.scores.csv'), output.with_suffix('.effects.csv')]
    if any(path.exists() for path in paths):
        raise FileExistsError('Choose a new benchmark output path')
    read = lambda paths: [json.loads(Path(path).read_text()) for path in paths]
    result = benchmark(read(args.reports), read(args.climode_reference_reports))
    write_json(output, result)
    write_table(paths[1], result['rows']); write_table(paths[2], result['effects'])
    print(json.dumps({'ranking_allowed':result['ranking_allowed'], 'reference_seeds':result['reference_seeds']}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
