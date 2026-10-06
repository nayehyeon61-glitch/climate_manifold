"""Guide loss x forecast signed on/off x arm matrix from per-run test reports.

usage: python scripts/summarize_guided_fusion_matrix.py --output OUT_PREFIX RUN [RUN ...]
Writes OUT_PREFIX.long.csv (one row per report) and OUT_PREFIX.mean.csv
(seed mean/std of normalized RMSE per model, guide loss, forecast signed, arm).
Raw arms carry no guide loss; they are listed once per forecast-signed setting.
"""
import argparse
import csv
import glob
import json
import os
import statistics


def arm(report):
    bridge = report['config']['bridge']
    if bridge == 'guided':
        return 'guided_zero' if report['config'].get('guide_mode') == 'zero' else 'guided'
    return bridge


def rows(runs):
    seen = set()
    for run in runs:
        for path in sorted(glob.glob(os.path.join(run, '*-seed*.test.json'))):
            report = json.load(open(path))
            fsm = report.get('forecast_signed_measure')
            row = dict(model=report['config']['model'], arm=arm(report), seed=report['seed'],
                       guide_loss=report.get('statistical_loss') or '-',
                       forecast_signed=fsm['weight'] if fsm else 0.,
                       normalized_rmse=report['scores']['aggregate']['normalized_rmse'],
                       wind_speed_rmse_mps=report['scores']['aggregate'].get('wind_speed_rmse_mps'),
                       run=os.path.basename(os.path.normpath(run)))
            key = tuple(row[k] for k in ('model', 'arm', 'seed', 'guide_loss', 'forecast_signed'))
            if key in seen:  # raw arms are linked into several runs
                continue
            seen.add(key)
            yield row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('runs', nargs='+')
    args = parser.parse_args(argv)
    data = list(rows(args.runs))
    with open(args.output+'.long.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader(); writer.writerows(data)
    groups = {}
    for row in data:
        key = tuple(row[k] for k in ('model', 'guide_loss', 'forecast_signed', 'arm'))
        groups.setdefault(key, []).append(row['normalized_rmse'])
    with open(args.output+'.mean.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'guide_loss', 'forecast_signed', 'arm', 'seeds', 'nrmse_mean', 'nrmse_std'])
        for key in sorted(groups):
            values = groups[key]
            writer.writerow([*key, len(values), f'{statistics.mean(values):.5f}',
                             f'{statistics.stdev(values):.5f}' if len(values) > 1 else ''])
    print(args.output+'.mean.csv')


if __name__ == '__main__':
    main()
