"""Plot validated Raw -> M / E -> M -> D field comparisons, with seed spread."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from .compare import _arm


MODEL_ORDER = ('transformer', 'mlp', 'neural_ode', 'climode', 'convlstm', 'simvp', 'fourcastnet', 'climax')
MODEL_NAMES = dict(zip(MODEL_ORDER, ('Transformer', 'NN (spatial)', 'NeuralODE', 'ClimODE adaptation',
                                  'ConvLSTM', 'SimVP', 'FourCastNet adaptation', 'ClimaX adaptation')))
METRIC_LABELS = {'rmse': 'RMSE', 'acc': 'ACC', 'rmse_skill_percent': 'RMSE reduction vs Raw (%)'}


def _identity(row):
    return row['model'], row['bridge'], _arm(row)


def _label(row):
    if row['bridge'] == 'raw':
        return 'Raw'
    loss = {'w2': 'W2', 'kl_entropy': 'KL-entropy', 'signed_measure': 'signed measure'}.get(row.get('statistical_loss'))
    if loss:
        prefix = 'PINN + ' if row.get('constraint_pair') == 'pinn_statistical' else ''
        return 'Latent / '+prefix+loss
    return 'Latent / '+row.get('regularization', 'full')


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def summarize_comparison(data):
    """Aggregate seeds only; retain variables, physical units and lead times.

    A missing/undefined seed makes the whole plotted point undefined. The
    sample standard deviation is not a confidence interval or forecast spread.
    Paired skill averages per-seed ratios, not ratios of aggregate RMSEs.
    """
    if data.get('format') != 'climate_manifold.comparison.v1':
        raise ValueError('Expected a validated comparison JSON')
    if not data.get('ranking_allowed') or not data.get('same_successful_origins'):
        raise ValueError('Incomplete forecasts or unequal successful origins; inspect evaluation reports before plotting')
    groups, labels = {}, {}
    for row in data['rows']:
        if row['bridge'] not in ('raw', 'latent') or row.get('statistical_flow_config') or row.get('conditional_flow_config'):
            raise ValueError('This plotter supports the two-route matrix with both flow objectives disabled')
        key = _identity(row)
        if row['seed'] in groups.setdefault(key, set()):
            raise ValueError('Duplicate model/route/seed in comparison')
        groups[key].add(row['seed'])
        labels[key] = _label(row)
    if not groups or len({tuple(sorted(v)) for v in groups.values()}) != 1:
        raise ValueError('Use the same seeds in every model/route comparison')
    source = data.get('climode_benchmark', {}).get('rows', [])
    if not source:
        raise ValueError('Missing physical per-variable/per-lead metrics; re-evaluate checkpoints')
    bins, units, axes = {}, {}, set()
    for row in source:
        identity = _identity(row)
        if identity not in groups or row['seed'] not in groups[identity]:
            raise ValueError('Physical score has an unknown model/route/seed')
        variable, lead = row['variable'], row['lead_hours']
        if not _finite(lead) or lead <= 0:
            raise ValueError('Lead hours must be positive and finite')
        if variable in units and units[variable] != row['units']:
            raise ValueError('Inconsistent physical units for '+variable)
        units[variable] = row['units']
        axes.add((variable, lead))
        for metric in ('rmse', 'acc'):
            key = (*identity, variable, lead, metric)
            values = bins.setdefault(key, {})
            if row['seed'] in values:
                raise ValueError('Duplicate physical score')
            full = (row.get('finite_forecast_fraction') == 1 and row.get('case_count', 0) > 0
                    and row.get(metric+'_valid_cases') == row['case_count'])
            value = row.get(metric)
            values[row['seed']] = value if full and _finite(value) else None
    # Enforce the same field/lead/seed coverage as the validated table.
    for identity, seeds in groups.items():
        for variable, lead in axes:
            for metric in ('rmse', 'acc'):
                if set(bins.get((*identity, variable, lead, metric), {})) != seeds:
                    raise ValueError('Missing physical model/route/seed/variable/lead score')
    for row in data.get('direct_comparison', {}).get('effects', []):
        identity = (row['model'], 'latent', row['candidate_arm'])
        if identity not in groups or row['seed'] not in groups[identity]:
            raise ValueError('Raw effect has an unknown model/route/seed')
        key = (*identity, row['variable'], row['lead_hours'], 'rmse_skill_percent')
        if (row['variable'], row['lead_hours']) not in axes:
            raise ValueError('Raw effect has an unknown variable/lead')
        values = bins.setdefault(key, {})
        if row['seed'] in values:
            raise ValueError('Duplicate paired Raw effect')
        value = row.get('rmse_skill_vs_raw')
        values[row['seed']] = 100*value if row.get('ranking_allowed') and _finite(value) else None
    for identity in groups:
        if identity[1] == 'latent':
            for variable, lead in axes:
                bins.setdefault((*identity, variable, lead, 'rmse_skill_percent'), {})
    summary = []
    for key, values in sorted(bins.items()):
        model, bridge, arm, variable, lead, metric = key
        identity = key[:3]
        valid = [v for v in values.values() if v is not None]
        complete = set(values) == groups[identity] and len(valid) == len(groups[identity])
        summary.append(dict(model=model, bridge=bridge, arm=arm, label=labels[identity],
            variable=variable, units=units[variable] if metric == 'rmse' else '%' if metric == 'rmse_skill_percent' else '1',
            lead_hours=lead, metric=metric, seeds=sorted(groups[identity]),
            expected_seeds=len(groups[identity]), valid_seeds=len(valid),
            mean=float(np.mean(valid)) if complete else None,
            sample_std=float(np.std(valid, ddof=1)) if complete and len(valid) > 1 else None))
    return summary


def plot_comparison(comparison, output):
    path, output = Path(comparison), Path(output)
    raw = path.read_bytes()
    data = json.loads(raw)
    summary = summarize_comparison(data)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose a new or empty plots directory')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D

    output.mkdir(parents=True, exist_ok=True)
    models = sorted({r['model'] for r in summary}, key=lambda m: (MODEL_ORDER.index(m) if m in MODEL_ORDER else 100, m))
    arms = sorted({r['arm'] for r in summary}, key=lambda a: (a != 'raw', a))
    labels = {r['arm']: r['label'] for r in summary}
    colors = dict(zip(arms, ['#64748b', '#2563eb', '#ea580c', '#7c3aed', '#0891b2', '#b91c1c', '#059669']))
    for i, arm in enumerate(arms):
        colors.setdefault(arm, plt.get_cmap('tab20')(i % 20))
    variables = sorted({r['variable'] for r in summary})
    split = data.get('climode_benchmark', {}).get('split', 'evaluation')
    files = []
    seed_count = summary[0]['expected_seeds']
    notes = (f'{seed_count} training seeds: mean ± sample SD; not a confidence interval. Undefined scores are gaps.'
             if seed_count>1 else 'One training seed; sample SD unavailable. Undefined scores are gaps.')

    def save(fig, stem):
        for suffix in ('png', 'pdf'):
            name = stem+'.'+suffix
            fig.savefig(output/name, dpi=170, bbox_inches='tight')
            files.append(name)
        plt.close(fig)

    def safe(name):
        return re.sub(r'[^A-Za-z0-9_-]', '_', name)

    def decorate(fig, title, selected):
        fig.suptitle(title, fontsize=14, y=.995)
        fig.legend([Line2D([0], [0], color=colors[a], lw=2) for a in selected],
                   [labels[a] for a in selected], loc='upper center', bbox_to_anchor=(.5,.95),
                   ncol=min(3,len(selected)), frameon=False)
        fig.text(.5,.015,notes,ha='center',fontsize=8,color='#475569')

    with plt.rc_context({'font.family':'DejaVu Sans', 'font.size':9,
                         'axes.spines.top':False, 'axes.spines.right':False}):
        for variable in variables:
            for metric in METRIC_LABELS:
                rows = [r for r in summary if r['variable']==variable and r['metric']==metric]
                selected = [a for a in arms if any(r['arm']==a for r in rows)]
                if not selected:
                    continue
                cols = min(4,len(models)); nrows = math.ceil(len(models)/cols)
                fig, axs = plt.subplots(nrows,cols,figsize=(4.2*cols,3*nrows+1.3),squeeze=False,sharey=True)
                fig.subplots_adjust(top=.78 if nrows==1 else .84,bottom=.17 if nrows==1 else .12,hspace=.4,wspace=.35)
                unit = rows[0]['units']
                ylabel = METRIC_LABELS[metric]+(' ('+unit+')' if metric=='rmse' else '')
                for ax, model in zip(axs.flat,models):
                    for arm in selected:
                        line = sorted((r for r in rows if r['model']==model and r['arm']==arm), key=lambda r:r['lead_hours'])
                        x = np.array([r['lead_hours']/24 for r in line])
                        y = np.array([r['mean'] if r['mean'] is not None else np.nan for r in line])
                        sd = np.array([r['sample_std'] if r['sample_std'] is not None else 0. for r in line])
                        ax.plot(x,y,'o-',color=colors[arm],lw=1.6,ms=3)
                        ax.fill_between(x,y-sd,y+sd,color=colors[arm],alpha=.13)
                    ax.set(title=MODEL_NAMES.get(model,model),xlabel='Lead time (days)',ylabel=ylabel)
                    ax.grid(alpha=.2)
                    if metric=='rmse_skill_percent': ax.axhline(0,color='#334155',lw=.8,ls='--')
                    if not any(r['mean'] is not None for r in rows if r['model']==model):
                        ax.text(.5,.5,'Undefined for these cases',ha='center',transform=ax.transAxes)
                for ax in list(axs.flat)[len(models):]: ax.set_visible(False)
                decorate(fig,f'{split.upper()} · {variable} · {METRIC_LABELS[metric]}',selected)
                save(fig,metric+'_by_lead_'+safe(variable))

            rows = [r for r in summary if r['variable']==variable and r['metric']=='rmse']
            last = max(r['lead_hours'] for r in rows)
            lookup = {(r['model'],r['arm']):r for r in rows if r['lead_hours']==last}
            fig, ax = plt.subplots(figsize=(max(8,len(models)*1.55),5.1))
            fig.subplots_adjust(top=.76,bottom=.23)
            width=.78/len(arms); x=np.arange(len(models))
            for i, arm in enumerate(arms):
                entries = [lookup.get((m,arm),{}) for m in models]
                y = [r.get('mean') if r.get('mean') is not None else np.nan for r in entries]
                sd = [r.get('sample_std') or 0. for r in entries]
                ax.bar(x+(i-(len(arms)-1)/2)*width,y,width,yerr=sd,color=colors[arm],capsize=2)
            ax.set_xticks(x,[MODEL_NAMES.get(m,m) for m in models],rotation=20,ha='right')
            ax.set_ylabel('RMSE ('+rows[0]['units']+')'); ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            decorate(fig,f'{split.upper()} · {variable} · day {last/24:g} RMSE (lower is better)',arms)
            save(fig,'last_lead_rmse_'+safe(variable))

        latent = [a for a in arms if a!='raw']
        if latent:
            heat_cols=min(2,len(variables))
            fig, axs = plt.subplots(math.ceil(len(variables)/2),heat_cols,
                figsize=(max(10,6+2.6*len(latent)),4*math.ceil(len(variables)/2)+1),squeeze=False)
            fig.subplots_adjust(top=.9,bottom=.14,wspace=.20,hspace=.55)
            heat_rows=[r for r in summary if r['metric']=='rmse_skill_percent']
            last=max(r['lead_hours'] for r in heat_rows)
            lookup={(r['variable'],r['model'],r['arm']):r for r in heat_rows if r['lead_hours']==last}
            finite=[abs(r['mean']) for r in lookup.values() if r['mean'] is not None]
            limit=max([1.,*finite]); norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
            cmap=plt.get_cmap('RdBu').copy();cmap.set_bad('#e2e8f0')
            for index,(ax,variable) in enumerate(zip(axs.flat,variables)):
                array=np.array([[lookup.get((variable,m,a),{}).get('mean') for a in latent] for m in models],dtype=float)
                im=ax.imshow(array,cmap=cmap,norm=norm,aspect='auto')
                ax.set_xticks(range(len(latent)),[labels[a] for a in latent],rotation=18,ha='right')
                ax.set_yticks(range(len(models)),[MODEL_NAMES.get(m,m) for m in models]);ax.set_title(variable)
                ax.tick_params(axis='y',labelleft=index%heat_cols==0)
                for i in range(len(models)):
                    for j in range(len(latent)):
                        v=array[i,j];label=f'{v:+.1f}%' if np.isfinite(v) else 'N/A'
                        ax.text(j,i,label,ha='center',va='center',color='white' if np.isfinite(v) and abs(v)>.6*limit else '#0f172a',fontsize=9)
            for ax in list(axs.flat)[len(variables):]: ax.set_visible(False)
            fig.colorbar(im,ax=list(axs.flat),shrink=.7,label='Paired RMSE reduction vs Raw (%)')
            fig.suptitle(f'{split.upper()} · day {last/24:g} · Raw-relative improvement',fontsize=14)
            fig.text(.5,.015,'Mean of same-seed improvements; positive is better. Gray = undefined.\nWhole-model comparison; parameter counts can differ.',ha='center',fontsize=9)
            save(fig,'last_lead_improvement_heatmap')

    csv_path=output/'plot_summary.csv'
    with csv_path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader()
        for row in summary: writer.writerow({**row,'seeds':json.dumps(row['seeds'])})
    manifest={'format':'climate_manifold.comparison_plots.v1','comparison':str(path.resolve()),
              'comparison_sha256':hashlib.sha256(raw).hexdigest(),'split':split,
              'figures':files,'summary':'plot_summary.csv','notes':notes,
              'interpretation':'Raw/latent contrasts compare whole models; seed SD is not predictive uncertainty.'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    result=plot_comparison(args.comparison,args.output)
    print(json.dumps({'output':args.output,'figures':len(result['figures']),'split':result['split']}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
