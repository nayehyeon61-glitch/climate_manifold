"""Scalar tables, ClimODE cross-check and figures for a joint comparison RUN.

Scores come from each run's validation report (ClimODE casewise protocol over
all validation origins). The single forecast case stored per run is re-scored
with the original Aalto-QuML/ClimODE ``evaluation_rmsd_mm`` / ``evaluation_acc_mm``
to confirm that the in-repo protocol matches the reference implementation.
Run with an interpreter that has matplotlib (e.g. /workspace/.venvs/climode-viz).
"""
import argparse
import csv
import json
import re
import sys
import types
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch

NAME = re.compile(r'(?P<model>mlp|neural_ode|climode|convlstm|simvp)-(?P<variant>[a-z_-]+)-seed(?P<seed>\d+)$')
VARIANT_LABEL = {'raw': 'raw (direct)', 'forecast_only': 'E-F-D forecast only', 'climate_manifold': 'E-F-D + Climate Manifold',
                 'pinn_statistical': 'PINN + W2', 'pinn_statistical-kl_entropy': 'PINN + KL entropy'}
SHORT = {'raw': 'raw', 'forecast_only': 'FO', 'climate_manifold': 'CM', 'pinn_statistical': 'W2', 'pinn_statistical-kl_entropy': 'KL'}
VARIANT_ORDER = ['raw', 'forecast_only', 'climate_manifold', 'pinn_statistical', 'pinn_statistical-kl_entropy']
VARIANT_STYLE = {'raw': ':', 'forecast_only': '--', 'climate_manifold': '-', 'pinn_statistical': '-', 'pinn_statistical-kl_entropy': '--'}
MODEL_COLOR = {'neural_ode': '#1f77b4', 'climode': '#d62728', 'mlp': '#2ca02c', 'convlstm': '#9467bd', 'simvp': '#8c564b'}
MAP_LEADS = (6, 24, 72, 120)
SPLIT = 'validation'


def load_climode(path):
    # ClimODE's utils imports torchcubicspline for training splines only; the
    # evaluation functions used here do not touch it.
    stub = types.ModuleType('torchcubicspline')
    stub.natural_cubic_spline_coeffs = stub.NaturalCubicSpline = None
    sys.modules.setdefault('torchcubicspline', stub)
    sys.path.insert(0, str(path))
    import utils
    return utils


def collect(run, split):
    # validation: <stem>.validation.json + <stem>.forecast.npz (written by run_model_comparison.sh)
    # test:       <stem>.test.json       + <stem>.test.forecast.npz (post-sweep evaluation)
    rows = []
    for report in sorted(run.glob(f'*.{split}.json')):
        stem = report.name[:-len(f'.{split}.json')]
        match = NAME.match(stem)
        if not match:
            continue
        prefix = run / stem
        meta = json.loads((prefix.parent / f'{stem}.metadata.json').read_text())
        forecast = run / (f'{stem}.forecast.npz' if split == 'validation' else f'{stem}.{split}.forecast.npz')
        rows.append(dict(match.groupdict(), seed=int(match['seed']), stem=stem, prefix=prefix, forecast=forecast,
                         report=json.loads(report.read_text()), meta=meta,
                         curve=json.loads((prefix.parent / f'{stem}.metrics.json').read_text())))
    if not rows:
        raise SystemExit(f'No finished runs in {run}')
    return rows


def groups_of(rows):
    order = {v: i for i, v in enumerate(VARIANT_ORDER)}
    keys = sorted({(r['model'], r['variant']) for r in rows}, key=lambda k: (k[0], order.get(k[1], 99), k[1]))
    return {k: [r for r in rows if (r['model'], r['variant']) == k] for k in keys}


def label(key):
    return f"{key[0]} | {VARIANT_LABEL.get(key[1], key[1])}"


def mean_std(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(values):
        return None, None
    return float(values.mean()), float(values.std()) if len(values) > 1 else 0.


def scalar_row(r, variables):
    rep, meta = r['report'], r['meta']
    agg = rep['scores']['aggregate']
    row = dict(model=r['model'], variant=r['variant'], seed=r['seed'],
               best_epoch=meta['best_epoch'], best_selection_state_mse=meta['best_selection_state_mse'],
               training_minutes=rep['training_seconds'] / 60, trainable_parameters=rep['trainable_parameters'],
               normalized_rmse=agg['normalized_rmse'], persistence_normalized_rmse=agg['persistence_normalized_rmse'],
               wind_speed_rmse_mps=agg['wind_speed_rmse_mps'])
    climode = rep['scores']['climode']['per_variable']
    physical = rep['scores']['per_variable']
    for v in variables:
        row[f'{v}_rmse'] = climode[v]['aggregate']['rmse']
        row[f'{v}_acc'] = climode[v]['aggregate']['acc']
        row[f'{v}_skill_vs_persistence'] = physical[v]['aggregate'].get('rmse_skill_vs_persistence')
        row[f'{v}_bias'] = physical[v]['aggregate'].get('bias')
        for lead in (24, 72, 120):
            item = next(x for x in climode[v]['by_lead'] if x['lead_hours'] == lead)
            row[f'{v}_rmse_{lead}h'] = item['rmse']
            row[f'{v}_acc_{lead}h'] = item['acc']
    return row


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tables(rows, groups, variables, out):
    per_run = [scalar_row(r, variables) for r in rows]
    write_csv(out / 'summary_by_run.csv', per_run)
    numeric = [k for k in per_run[0] if k not in ('model', 'variant', 'seed')]
    grouped = []
    for key in groups:
        members = [x for x in per_run if (x['model'], x['variant']) == key]
        row = dict(model=key[0], variant=key[1], seeds=' '.join(str(x['seed']) for x in members))
        for k in numeric:
            row[f'{k}_mean'], row[f'{k}_std'] = mean_std([x[k] for x in members])
        grouped.append(row)
    write_csv(out / 'summary_by_group.csv', grouped)
    by_lead = []
    for r in rows:
        climode = r['report']['scores']['climode']['per_variable']
        for v in variables:
            for item in climode[v]['by_lead']:
                by_lead.append(dict(model=r['model'], variant=r['variant'], seed=r['seed'], variable=v,
                                    units=climode[v]['units'], lead_hours=item['lead_hours'],
                                    rmse=item['rmse'], rmse_case_std=item['rmse_std'],
                                    acc=item['acc'], acc_case_std=item['acc_std']))
    write_csv(out / 'climode_by_lead.csv', by_lead)
    return per_run, grouped


def field(npz, schema):
    grid = (len(schema['variables']),) + tuple(schema['variables'][0]['shape'])
    return (npz['mean'].reshape(len(npz['lead_hours']), *grid),
            npz['truth'].reshape(len(npz['lead_hours']), *grid))


def cross_check(rows, climode_utils, out):
    """Re-score each stored forecast case with the original ClimODE functions."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from climate_manifold.downstream.climode_metrics import ClimODEMetrics
    results = []
    for r in rows:
        npz = np.load(r['forecast'])
        schema = json.loads(str(npz['schema_json']))
        pred, truth = field(npz, schema)
        names = [v['name'] for v in schema['variables']]
        coords = schema['variables'][0]['coords']
        lat, lon = np.asarray(coords['lat']), np.asarray(coords['lon'])
        checkpoint = torch.load(f"{r['prefix']}.pt", map_location='cpu', weights_only=False)
        clim = np.asarray(checkpoint['a_metadata']['mean'], dtype=np.float64).reshape(pred.shape[1:])
        ours = ClimODEMetrics(schema, torch.from_numpy(clim), npz['lead_hours'])
        ours.update(torch.from_numpy(pred[None]), torch.from_numpy(truth[None]))
        ours = ours.result()['per_variable']
        ones, zeros = np.ones(len(names)), np.zeros(len(names))
        worst = {'rmse': 0., 'acc': 0.}
        for j in range(len(npz['lead_hours'])):
            p, t = torch.from_numpy(pred[j]), torch.from_numpy(truth[j])
            ref_rmse = climode_utils.evaluation_rmsd_mm(p, t, lat, lon, ones, zeros, *p.shape[1:], names)
            ref_acc = climode_utils.evaluation_acc_mm(p, t, lat, lon, ones, zeros, *p.shape[1:], names, clim)
            for i, v in enumerate(names):
                worst['rmse'] = max(worst['rmse'], abs(ref_rmse[i] - ours[v]['by_lead'][j]['rmse']) / max(ref_rmse[i], 1e-12))
                worst['acc'] = max(worst['acc'], abs(float(ref_acc[i]) - ours[v]['by_lead'][j]['acc']))
        results.append(dict(run=r['stem'], origin_time=str(npz['origin_time']),
                            max_relative_rmse_difference=worst['rmse'], max_abs_acc_difference=worst['acc']))
    summary = dict(reference='Aalto-QuML/ClimODE utils.evaluation_rmsd_mm / evaluation_acc_mm',
                   compared_against='climate_manifold.downstream.climode_metrics.ClimODEMetrics',
                   note='One stored validation case per run; aggregate scores cover all validation origins.',
                   runs=results,
                   passed=all(x['max_relative_rmse_difference'] < 1e-6 and x['max_abs_acc_difference'] < 1e-6 for x in results))
    (out / 'climode_crosscheck.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def plot_leads(rows, groups, variables, units, out):
    fig, axes = plt.subplots(2, len(variables), figsize=(4.2 * len(variables), 7.5), constrained_layout=True)
    for key, members in groups.items():
        for c, v in enumerate(variables):
            for r_idx, metric in enumerate(('rmse', 'acc')):
                series = np.array([[x[metric] for x in m['report']['scores']['climode']['per_variable'][v]['by_lead']]
                                   for m in members], dtype=float)
                leads = [x['lead_hours'] for x in members[0]['report']['scores']['climode']['per_variable'][v]['by_lead']]
                mu, sd = series.mean(0), series.std(0)
                ax = axes[r_idx, c]
                ax.plot(leads, mu, VARIANT_STYLE.get(key[1], '-'), color=MODEL_COLOR.get(key[0]), lw=1.8,
                        label=f'{label(key)} (n={len(members)})')
                if len(members) > 1:
                    ax.fill_between(leads, mu - sd, mu + sd, color=MODEL_COLOR.get(key[0]), alpha=.12)
    for c, v in enumerate(variables):
        axes[0, c].set_title(f'{v} RMSE [{units[v]}]')
        axes[1, c].set_title(f'{v} ACC')
        for ax in axes[:, c]:
            ax.set_xlabel('lead time [h]')
            ax.set_xticks([0, 24, 48, 72, 96, 120])
            ax.grid(alpha=.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(.5, -.07 - .025 * (len(labels) // 3)), frameon=False)
    fig.suptitle(f'ClimODE-protocol {SPLIT} scores by lead (line: seed mean, band: seed std)')
    fig.savefig(out / 'climode_lead_rmse_acc.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_summary(per_run, groups, variables, out):
    keys = list(groups)
    panels = [('normalized_rmse', 'normalized RMSE (all variables)')] + \
             [(f'{v}_skill_vs_persistence', f'{v} RMSE skill vs persistence') for v in variables]
    fig, axes = plt.subplots(1, len(panels), figsize=(max(3.6, .3 * len(keys) + 1.2) * len(panels), 4.8), constrained_layout=True)
    x = np.arange(len(keys))
    for ax, (metric, title) in zip(axes, panels):
        for i, key in enumerate(keys):
            values = [r[metric] for r in per_run if (r['model'], r['variant']) == key and r[metric] is not None]
            ax.bar(i, np.mean(values), color=MODEL_COLOR.get(key[0]), alpha=.35 + .3 * (key[1] != 'raw'),
                   hatch={'raw': '..', 'forecast_only': '//', 'pinn_statistical-kl_entropy': '//'}.get(key[1], ''), edgecolor='k', lw=.5)
            ax.scatter(np.full(len(values), i), values, color='k', s=12, zorder=3)
        if metric == 'normalized_rmse':
            persistence = np.mean([r['persistence_normalized_rmse'] for r in per_run])
            ax.axhline(persistence, color='gray', ls='--', lw=1)
            ax.text(len(keys) - .5, persistence, 'persistence', ha='right', va='bottom', color='gray', fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x, [f'{k[0]} {SHORT.get(k[1], k[1])}' for k in keys], rotation=90, fontsize=7)
        ax.grid(axis='y', alpha=.3)
    fig.suptitle(f'Aggregate {SPLIT} scalars (bar: seed mean, dots: individual seeds)')
    fig.savefig(out / 'aggregate_scalars.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_curves(groups, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for key, members in groups.items():
        for ax, split in zip(axes, ('train', 'selection')):
            curves = np.array([[e[split]['state_mse'] for e in m['curve']] for m in members])
            epochs = np.arange(1, curves.shape[1] + 1)
            ax.plot(epochs, curves.mean(0), VARIANT_STYLE.get(key[1], '-'), color=MODEL_COLOR.get(key[0]), label=label(key))
            if len(members) > 1:
                ax.fill_between(epochs, curves.min(0), curves.max(0), color=MODEL_COLOR.get(key[0]), alpha=.1)
    for ax, split in zip(axes, ('train', 'selection (calibration)')):
        ax.set_title(f'{split} state MSE (normalized)')
        ax.set_xlabel('epoch')
        ax.grid(alpha=.3)
    axes[1].legend(fontsize=7)
    fig.savefig(out / 'learning_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def pretty(units):
    return units.replace('m s**-1', 'm/s')


def representative(members):
    return min(members, key=lambda m: m['report']['scores']['aggregate']['normalized_rmse'])


def plot_maps(groups, out):
    reps = {key: representative(m) for key, m in groups.items()}
    first = np.load(next(iter(reps.values()))['forecast'])
    schema = json.loads(str(first['schema_json']))
    names = [v['name'] for v in schema['variables']]
    coords = schema['variables'][0]['coords']
    extent = [coords['lon'][0], coords['lon'][-1], coords['lat'][0], coords['lat'][-1]]
    leads = list(first['lead_hours'])
    columns = [leads.index(h) for h in MAP_LEADS if h in leads]
    fields = {key: field(np.load(r['forecast']), schema) for key, r in reps.items()}
    truth = next(iter(fields.values()))[1]
    for i, v in enumerate(names):
        rows = len(fields) + 1
        # Two trailing narrow columns hold the value and error colorbars.
        fig, grid = plt.subplots(rows, 2 * len(columns) + 2, figsize=(3.1 * 2 * len(columns) + 1.2, 1.9 * rows),
                                 constrained_layout=True, squeeze=False,
                                 gridspec_kw={'width_ratios': [1] * (2 * len(columns)) + [.06, .06]})
        axes = grid[:, :-2]
        for cax in grid[1:, -2:].flat:
            cax.remove()
        vmin, vmax = np.percentile(truth[:, i], [1, 99])
        err = max(np.percentile(np.abs(f[0][:, i] - truth[:, i]), 99) for f in fields.values())
        for c, j in enumerate(columns):
            axes[0, 2 * c].imshow(truth[j, i], origin='lower', extent=extent, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
            axes[0, 2 * c].set_title(f'truth +{leads[j]:.0f}h', fontsize=8)
            axes[0, 2 * c + 1].axis('off')
            for r_idx, (key, (pred, _)) in enumerate(fields.items(), start=1):
                a = axes[r_idx, 2 * c].imshow(pred[j, i], origin='lower', extent=extent, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
                b = axes[r_idx, 2 * c + 1].imshow(pred[j, i] - truth[j, i], origin='lower', extent=extent, cmap='RdBu_r',
                                                  vmin=-err, vmax=err, aspect='auto')
                axes[r_idx, 2 * c].set_title(f'pred +{leads[j]:.0f}h', fontsize=8)
                axes[r_idx, 2 * c + 1].set_title(f'error +{leads[j]:.0f}h', fontsize=8)
                if c == 0:
                    axes[r_idx, 0].set_ylabel(f"{key[0]}\n{key[1]}\nseed {reps[key]['seed']}", fontsize=7)
        for ax in axes.flat:
            ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(a, cax=grid[0, -2], label=f"{v} [{pretty(schema['variables'][i]['attrs'].get('units', ''))}]")
        fig.colorbar(b, cax=grid[0, -1], label='pred - truth')
        fig.suptitle(f"{v} | {SPLIT} origin {str(first['origin_time'])[:16]} | best seed per group", fontsize=10)
        fig.savefig(out / f'maps_{v}.png', dpi=130, bbox_inches='tight')
        plt.close(fig)
    return reps, fields, truth, names, extent, leads, schema


def animate(reps, fields, truth, names, extent, leads, schema, variable, out):
    i = names.index(variable)
    keys = list(fields)
    fig, axes = plt.subplots(2, len(keys) + 1, figsize=(3.2 * (len(keys) + 1), 4.6), constrained_layout=True, squeeze=False)
    vmin, vmax = np.percentile(truth[:, i], [1, 99])
    err = max(np.percentile(np.abs(fields[k][0][:, i] - truth[:, i]), 99) for k in keys)
    images = [axes[0, 0].imshow(truth[0, i], origin='lower', extent=extent, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')]
    axes[0, 0].set_title('truth', fontsize=8)
    axes[1, 0].axis('off')
    for c, key in enumerate(keys, start=1):
        images.append(axes[0, c].imshow(fields[key][0][0, i], origin='lower', extent=extent, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto'))
        images.append(axes[1, c].imshow(fields[key][0][0, i] - truth[0, i], origin='lower', extent=extent, cmap='RdBu_r', vmin=-err, vmax=err, aspect='auto'))
        axes[0, c].set_title(f'{key[0]}\n{key[1]}', fontsize=8)
        axes[1, c].set_title('error', fontsize=8)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    title = fig.suptitle('')

    def update(j):
        images[0].set_data(truth[j, i])
        for c, key in enumerate(keys):
            images[1 + 2 * c].set_data(fields[key][0][j, i])
            images[2 + 2 * c].set_data(fields[key][0][j, i] - truth[j, i])
        title.set_text(f'{variable} forecast, lead +{leads[j]:.0f}h')
        return images
    FuncAnimation(fig, update, frames=len(leads)).save(out / f'forecast_{variable}.gif', writer=PillowWriter(fps=3), dpi=90)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='Comparison RUN directory')
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--output-dir', help='Default: <run>/report/<split>')
    parser.add_argument('--climode-repo', default='/workspace/external/climode')
    parser.add_argument('--gif-variables', nargs='*', default=['t2m', 'msl'])
    args = parser.parse_args(argv)
    global SPLIT
    SPLIT = args.split
    run = Path(args.run)
    out = Path(args.output_dir or run / 'report' / args.split)
    out.mkdir(parents=True, exist_ok=True)
    rows = collect(run, args.split)
    groups = groups_of(rows)
    per_variable = rows[0]['report']['scores']['climode']['per_variable']
    variables = list(per_variable)
    units = {v: pretty(per_variable[v]['units']) for v in variables}
    per_run, grouped = tables(rows, groups, variables, out)
    check = cross_check(rows, load_climode(args.climode_repo), out)
    plot_leads(rows, groups, variables, units, out)
    plot_summary(per_run, groups, variables, out)
    plot_curves(groups, out)
    maps = plot_maps(groups, out)
    for v in args.gif_variables:
        if v in maps[3]:
            animate(*maps, v, out)
    comparison = run / 'comparison.json'
    (out / 'report_index.json').write_text(json.dumps(dict(
        run=str(run), split=args.split, finished_runs=[r['stem'] for r in rows], groups=[list(k) for k in groups],
        sweep_complete=comparison.exists(), climode_crosscheck_passed=check['passed'],
        files=sorted(p.name for p in out.iterdir())), indent=2) + '\n')
    print(json.dumps(dict(split=args.split, runs=len(rows), groups=len(groups), crosscheck_passed=check['passed'], output=str(out))))


if __name__ == '__main__':
    main()
