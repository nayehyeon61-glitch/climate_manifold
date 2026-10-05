"""Plot Raw / E->M->D / guided fusion / zero-guide results from per-run test JSONs.

Reads ``<model>-<arm>-seed<seed>.test.json`` directly, because the
comparison's ``climode_benchmark`` rows do not carry ``guide_mode`` and so
cannot separate the learned guide from the zero-guide control.
All effects are paired by seed: 100 * (1 - candidate / raw); positive is better.
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

MODEL_ORDER = ('transformer', 'mlp', 'neural_ode', 'climode', 'convlstm', 'simvp', 'fourcastnet', 'climax')
MODEL_NAMES = dict(zip(MODEL_ORDER, ('Transformer', 'NN (spatial)', 'NeuralODE', 'ClimODE adapt.',
                                  'ConvLSTM', 'SimVP', 'FourCastNet adapt.', 'ClimaX adapt.')))
ARMS = ('raw', 'latent', 'guided', 'guided_zero')
ARM_NAMES = {'latent': 'E → M → D', 'guided': 'Fusion (learned guide)', 'guided_zero': 'Fusion (zero guide)'}
# Validated reference categorical slots 1-3, fixed order.
COLORS = {'latent': '#2a78d6', 'guided': '#eb6834', 'guided_zero': '#1baf7a'}
INK, MUTED, GRID, BASE = '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7'
PATTERN = re.compile(r'^(?P<model>.+)-(?P<arm>raw|latent|guided|guided_zero)-seed(?P<seed>\d+)\.test\.json$')


def load(run):
    runs = {}
    for path in sorted(Path(run).glob('*.test.json')):
        match = PATTERN.match(path.name)
        if not match:
            continue
        data = json.loads(path.read_text())
        if data.get('finite_forecast_fraction') != 1 or data.get('failed_origins'):
            raise ValueError(f'{path.name}: incomplete forecasts')
        mode = data['config'].get('guide_mode')
        if match['arm'].startswith('guided') and mode != ('zero' if match['arm'] == 'guided_zero' else 'learned'):
            raise ValueError(f'{path.name}: guide_mode {mode!r} does not match file name')
        scores = data['scores']
        runs[match['model'], match['arm'], int(match['seed'])] = dict(
            normalized_rmse=scores['aggregate']['normalized_rmse'],
            variables={v: s['aggregate']['rmse'] for v, s in scores['climode']['per_variable'].items()})
    seeds = sorted({k[2] for k in runs})
    models = [m for m in MODEL_ORDER if any(k[0] == m for k in runs)]
    missing = [(m, a, s) for m in models for a in ARMS for s in seeds if (m, a, s) not in runs]
    if missing:
        raise ValueError(f'Missing runs: {missing}')
    return runs, models, seeds


def skill(candidate, reference):
    return 100 * (1 - np.asarray(candidate) / np.asarray(reference))


def plot(run, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D

    runs, models, seeds = load(run)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    variables = sorted(runs[models[0], 'raw', seeds[0]]['variables'])
    rows = []
    for model in models:
        raw = [runs[model, 'raw', s]['normalized_rmse'] for s in seeds]
        for arm in ARMS:
            values = [runs[model, arm, s]['normalized_rmse'] for s in seeds]
            effect = skill(values, raw)
            rows.append(dict(model=model, arm=arm, metric='normalized_rmse', variable='all',
                             mean=np.mean(values), sample_std=np.std(values, ddof=1),
                             skill_vs_raw_mean=effect.mean(), skill_vs_raw_std=np.std(effect, ddof=1)))
        for variable in variables:
            zero = [runs[model, 'guided_zero', s]['variables'][variable] for s in seeds]
            effect = skill([runs[model, 'guided', s]['variables'][variable] for s in seeds], zero)
            rows.append(dict(model=model, arm='guided', metric='rmse_vs_zero_guide', variable=variable,
                             mean=effect.mean(), sample_std=np.std(effect, ddof=1),
                             skill_vs_raw_mean='', skill_vs_raw_std=''))

    style = {'font.family': 'DejaVu Sans', 'font.size': 9, 'text.color': INK, 'axes.labelcolor': MUTED,
             'xtick.color': MUTED, 'ytick.color': INK, 'axes.edgecolor': BASE,
             'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False}
    note = f'Seeds {", ".join(map(str, seeds))}: dot = mean of same-seed effects, ticks = individual seeds. Held-out test split.'
    with plt.rc_context(style):
        # 1. Paired normalized-RMSE change against Raw, per model and arm.
        fig, ax = plt.subplots(figsize=(8.2, 5.6))
        fig.subplots_adjust(left=.2, right=.97, top=.82, bottom=.15)
        offsets = dict(zip(('latent', 'guided', 'guided_zero'), (-.24, 0, .24)))
        for i, model in enumerate(models):
            raw = np.array([runs[model, 'raw', s]['normalized_rmse'] for s in seeds])
            for arm, dy in offsets.items():
                effect = skill([runs[model, arm, s]['normalized_rmse'] for s in seeds], raw)
                y = i + dy
                ax.plot([effect.min(), effect.max()], [y, y], color=COLORS[arm], lw=2, solid_capstyle='round', alpha=.45)
                ax.scatter(effect, [y]*len(effect), marker='|', s=60, color=COLORS[arm], lw=1.5)
                ax.scatter(effect.mean(), y, s=48, color=COLORS[arm], edgecolor='white', lw=1.5, zorder=3)
        ax.axvline(0, color=MUTED, lw=1)
        ax.text(0, -.75, ' Raw', color=MUTED, va='center', fontsize=8)
        ax.set_yticks(range(len(models)), [MODEL_NAMES[m] for m in models])
        ax.set_ylim(len(models)-.5, -.9)
        ax.tick_params(axis='y', length=0)
        ax.grid(axis='x', color=GRID, lw=.8); ax.set_axisbelow(True)
        ax.set_xlabel('Normalized RMSE reduction vs Raw (%)  ·  right is better')
        fig.suptitle('Guided fusion vs Raw and E → M → D, per forecast family', x=.2, ha='left', fontsize=13)
        fig.legend([Line2D([], [], marker='o', ls='', color=COLORS[a], markersize=7) for a in offsets],
                   [ARM_NAMES[a] for a in offsets], loc='upper left', bbox_to_anchor=(.19, .93), ncol=3, frameon=False)
        fig.text(.2, .03, note, fontsize=8, color=MUTED)
        fig.savefig(output/'skill_vs_raw.png', dpi=170, bbox_inches='tight')
        plt.close(fig)

        # 2. Guide information effect: learned guide vs zero guide, per variable.
        array = np.array([[next(r['mean'] for r in rows if r['model'] == m and r['variable'] == v) for v in variables]
                          for m in models])
        limit = max(1., np.abs(array).max())
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        fig.subplots_adjust(left=.27, right=.95, top=.84, bottom=.1)
        im = ax.imshow(array, cmap='RdBu', norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect='auto')
        for i in range(len(models)):
            for j in range(len(variables)):
                v = array[i, j]
                ax.text(j, i, f'{v + 0:+.2f}%'.replace('-0.00', '0.00').replace('+0.00', '0.00'), ha='center', va='center', fontsize=9,
                        color='white' if abs(v) > .6*limit else INK)
        ax.set_xticks(range(len(variables)), variables)
        ax.set_yticks(range(len(models)), [MODEL_NAMES[m] for m in models])
        ax.tick_params(length=0); ax.xaxis.tick_top()
        for spine in ax.spines.values(): spine.set_visible(False)
        fig.colorbar(im, ax=ax, shrink=.8, label='RMSE reduction vs zero guide (%)')
        fig.suptitle('Learned guide vs zero guide (guide information effect)', x=.05, ha='left', fontsize=12)
        fig.text(.05, .01, f'Mean of same-seed effects, seeds {", ".join(map(str, seeds))}; physical RMSE pooled over leads.\nBlue = learned guide beats zero guide.',
                 fontsize=7.5, color=MUTED, wrap=True)
        fig.savefig(output/'guide_vs_zero_heatmap.png', dpi=170, bbox_inches='tight')
        plt.close(fig)

    with (output/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='guided fusion run directory')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    print(plot(args.run, args.output))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
