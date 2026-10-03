"""Physical scores and same-seed ablations for Raw, latent and manifold plots."""
from copy import deepcopy
import json
import math

import pytest

from test_raw_comparison import report, run_compare
from climate_manifold.downstream.plot_comparison import summarize_comparison, plot_comparison


def fixture_comparison(tmp_path):
    reports=[]
    for seed, raw, latent in ((7,4.,2.),(19,2.,1.8)):
        for arm, error in (('raw',raw),('climate_manifold',latent)):
            row=report(arm,seed)
            for score in row['scores']['climode']['per_variable']['t2m']['by_lead']:
                score['rmse']=error
            reports.append(row)
    return run_compare(tmp_path,reports)


def test_seed_sd_and_mean_of_paired_skill_not_ratio_of_means(tmp_path):
    data=fixture_comparison(tmp_path)
    rows=summarize_comparison(data)
    raw=next(r for r in rows if r['arm']=='raw' and r['metric']=='rmse')
    assert raw['mean']==3. and raw['sample_std']==pytest.approx(math.sqrt(2))
    skill=next(r for r in rows if r['metric']=='rmse_skill_percent')
    assert skill['mean']==pytest.approx(30.)  # mean(50%, 10%), not 1-1.9/3
    assert skill['sample_std']==pytest.approx(math.sqrt(800))
    assert raw['units']=='K' and skill['units']=='%'
    assert skill['valid_seeds']==skill['expected_seeds']==2


def test_undefined_metric_is_a_gap_never_partial_seed_average(tmp_path):
    data=fixture_comparison(tmp_path)
    score=data['climode_benchmark']['rows'][0]
    score['acc']=None; score['acc_valid_cases']=0
    rows=summarize_comparison(data)
    item=next(r for r in rows if r['arm']=='raw' and r['metric']=='acc' and r['lead_hours']==6)
    assert item['mean'] is None and item['sample_std'] is None
    assert item['valid_seeds']==1 and item['expected_seeds']==2


@pytest.mark.parametrize('corruption', ['failed', 'missing', 'units', 'duplicate'])
def test_unfair_or_corrupt_inputs_do_not_produce_plots(tmp_path,corruption):
    data=fixture_comparison(tmp_path)
    if corruption=='failed': data['ranking_allowed']=False
    elif corruption=='missing': data['climode_benchmark']['rows'].pop()
    elif corruption=='units': data['climode_benchmark']['rows'][0]['units']='Pa'
    else: data['climode_benchmark']['rows'].append(deepcopy(data['climode_benchmark']['rows'][0]))
    with pytest.raises(ValueError): summarize_comparison(data)


def test_png_pdf_csv_manifest_and_no_overwrite(tmp_path):
    pytest.importorskip('matplotlib')
    fixture_comparison(tmp_path)
    output=tmp_path/'plots'
    result=plot_comparison(tmp_path/'comparison.json',output)
    assert len(result['figures'])==10  # three curves, last-lead bars, heatmap
    for name in result['figures']:
        raw=(output/name).read_bytes()
        assert raw.startswith(b'\x89PNG\r\n\x1a\n' if name.endswith('.png') else b'%PDF')
        assert len(raw)>5000
    assert (output/'plot_summary.csv').is_file()
    assert json.loads((output/'manifest.json').read_text())==result
    with pytest.raises(FileExistsError): plot_comparison(tmp_path/'comparison.json',output)


def fusion_comparison(tmp_path, *, raw=True, fusion=True):
    from test_guided_comparison import guide_report, transformer_raw, forecast_only_guide
    from climate_manifold.downstream.pipeline import PredictorConfig
    from climate_manifold.downstream.train import guide_contract

    reports = []
    for seed, errors in ((7, (4., 2.4, 2., 3., 2.5)), (19, (2., 1.6, 1.5, 1.8, 1.8))):
        rows = [transformer_raw(seed), guide_report('latent', seed=seed),
                guide_report(seed=seed), guide_report(mode='zero', seed=seed),
                forecast_only_guide(seed)]
        for index, (row, error) in enumerate(zip(rows, errors)):
            if fusion:
                row['config']['model'] = 'mlp'
                row['implementation'] = 'raw_matched_mlp' if index == 0 else 'latent_mlp'
                if row['config']['bridge'] == 'guided':
                    row['config'].update(guide_architecture='fusion', guide_fusion_depth=2)
                    row['guide_contract'] = guide_contract(PredictorConfig(**row['config']))
                    row['implementation'] = 'guide_fusion_v1+raw_matched_mlp'
            row['representation_sha256'] = None if index == 0 else f'{seed}-arm-{index}'
            for score in row['scores']['climode']['per_variable']['t2m']['by_lead']:
                score['rmse'] = error
            if raw or index:
                reports.append(row)
    return run_compare(tmp_path, reports)


def test_fusion_raw_and_ablation_effects_resolve_actual_candidate_route(tmp_path):
    data = fusion_comparison(tmp_path)
    summary = summarize_comparison(data)
    arm = 'guided_fusion:learned:statistical'
    rows = {r['metric']:r for r in summary if r['arm'] == arm and r['lead_hours'] == 6}
    assert rows['rmse']['label'] == 'Manifold / W2'
    assert rows['rmse']['units'] == 'K'
    assert rows['rmse']['mean'] == pytest.approx(1.75)
    assert rows['rmse_skill_percent']['mean'] == pytest.approx((50+25)/2)
    assert rows['guide_gain_percent']['mean'] == pytest.approx(100*((1-2/3)+(1-1.5/1.8))/2)
    assert rows['constraint_gain_percent']['mean'] == pytest.approx(100*((1-2/2.5)+(1-1.5/1.8))/2)
    assert all(r['bridge'] == 'guided' for r in rows.values())
    assert any(r['label'] == 'Manifold / Zero guide / W2' for r in summary)
    assert any(r['label'] == 'Manifold / No observed constraints' for r in summary)
    assert any(r['label'] == 'Latent / W2' for r in summary)


def test_legacy_guided_labels_do_not_claim_fusion_manifold(tmp_path):
    summary = summarize_comparison(fusion_comparison(tmp_path, fusion=False))
    assert any(r['label'] == 'Joint guided predictor / W2' for r in summary)
    assert not any(r['label'].startswith('Manifold') for r in summary)


def test_legacy_physical_tables_resolve_fingerprints_and_reject_ambiguity(tmp_path):
    data = fusion_comparison(tmp_path)
    expected = summarize_comparison(data)
    for row in data['climode_benchmark']['rows']:
        row.pop('guide_mode', None)
        row.pop('guide_contract', None)
    assert summarize_comparison(data) == expected
    for row in data['climode_benchmark']['rows']:
        row.pop('representation_sha256', None)
    with pytest.raises(ValueError, match='Ambiguous.*legacy guided'):
        summarize_comparison(data)


def test_missing_ablation_seed_or_lead_produces_explicit_gap(tmp_path):
    data = fusion_comparison(tmp_path)
    data['guide_effects'] = [row for row in data['guide_effects']
                            if row['interpretation'] != 'change_guide_input'
                            or (row['seed'] == 7 and row['lead_hours'] == 6)]
    gains = [row for row in summarize_comparison(data) if row['metric'] == 'guide_gain_percent']
    assert len(gains) == 2
    assert {row['lead_hours']:row['valid_seeds'] for row in gains} == {6.:1, 12.:0}
    assert all(row['mean'] is None and row['sample_std'] is None for row in gains)


@pytest.mark.parametrize('mutation', ['unknown_arm', 'units', 'reversed_guide', 'duplicate'])
def test_corrupted_guide_effects_are_not_plotted(tmp_path, mutation):
    data = fusion_comparison(tmp_path)
    effect = next(row for row in data['guide_effects'] if row['interpretation'] == 'change_guide_input')
    if mutation == 'unknown_arm':
        effect['control'] = 'unknown'
    elif mutation == 'units':
        effect['units'] = 'Pa'
    elif mutation == 'reversed_guide':
        effect['baseline_guide_mode'] = 'learned'
    else:
        data['guide_effects'].append(deepcopy(effect))
    with pytest.raises(ValueError):
        summarize_comparison(data)


@pytest.mark.parametrize('raw', [True, False])
def test_fusion_ablation_figures_render_with_or_without_raw(tmp_path, raw):
    pytest.importorskip('matplotlib')
    fusion_comparison(tmp_path, raw=raw)
    output = tmp_path/'plots'
    result = plot_comparison(tmp_path/'comparison.json', output)
    expected = {'guide_gain_percent_by_lead_t2m.png', 'constraint_gain_percent_by_lead_t2m.png',
                'rmse_by_lead_t2m.pdf', 'acc_by_lead_t2m.pdf'}
    assert expected.issubset(result['figures'])
    assert ('last_lead_improvement_heatmap.png' in result['figures']) == raw
    assert ('rmse_skill_percent_by_lead_t2m.png' in result['figures']) == raw
    assert all((output/name).stat().st_size > 5000 for name in result['figures'])
    assert 'not predictive uncertainty' in result['interpretation']
