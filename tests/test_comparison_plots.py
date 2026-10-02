"""Numerical aggregation and headless rendering for raw/latent result plots."""
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
