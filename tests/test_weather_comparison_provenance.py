"""Published-core names must not hide changed upstream code or model capacity."""
from copy import deepcopy

import pytest

from climate_manifold.downstream.compare import _validate_weather_provenance
from test_raw_comparison import report, run_compare


def weather_report(family='fourcastnet', seed=7, bridge='raw'):
    row = report('raw' if bridge == 'raw' else 'forecast_only', seed, family)
    row['config'].update(weather_depth=4, weather_patch_size=2, hidden_dim=128)
    row['implementation'] = family + '_official_core_adaptation_v1'
    row['predictor_provenance'] = {
        'family': family,
        'implementation': row['implementation'],
        'upstream_source': 'https://github.com/owner/' + family,
        'upstream_commit': 'a' * 40,
        'implementation_variant': 'official_core_local_context_v1',
        'depth': 4, 'patch_size': 2, 'hidden_dim': 128, 'pretrained': False,
        'input_variables': ['t2m', 'msl', 'u10', 'v10'],
        'state_shape': [4, 16, 32],
    }
    if bridge != 'raw':
        row['predictor_provenance'].update(input_variables=['latent_0', 'latent_1'],
                                           state_shape=[2, 8, 16])
    return row


@pytest.mark.parametrize('family', ['fourcastnet', 'climax'])
def test_matching_cores_can_compare_different_raw_and_latent_channels(family):
    rows = [weather_report(family, seed, bridge)
            for seed in (7, 19) for bridge in ('raw', 'latent')]
    _validate_weather_provenance(rows)


@pytest.mark.parametrize('field,value', [
    ('upstream_source', 'https://github.com/different/source'),
    ('upstream_commit', 'b' * 40),
    ('implementation_variant', 'different_adaptation'),
    ('pretrained', True),
])
def test_same_family_seed_pool_rejects_different_source_or_training_initialization(tmp_path, field, value):
    rows = [weather_report(seed=7), weather_report(seed=19)]
    rows[1]['predictor_provenance'][field] = value
    with pytest.raises(ValueError, match='mismatched predictor_provenance.' + field):
        run_compare(tmp_path, rows)


@pytest.mark.parametrize('field,config_field,value', [
    ('depth', 'weather_depth', 8),
    ('patch_size', 'weather_patch_size', 4),
    ('hidden_dim', 'hidden_dim', 256),
])
def test_raw_latent_pair_requires_common_core_hyperparameters(field, config_field, value):
    rows = [weather_report(), weather_report(bridge='latent')]
    rows[1]['predictor_provenance'][field] = value
    rows[1]['config'][config_field] = value
    with pytest.raises(ValueError, match='mismatched predictor_provenance.' + field):
        _validate_weather_provenance(rows)


@pytest.mark.parametrize('field,value,message', [
    ('upstream_commit', 'main', 'full commit SHA'),
    ('upstream_source', '', 'upstream_source'),
    ('family', 'climax', 'family differs'),
    ('implementation', 'another_model', 'implementation differs'),
    ('depth', 8, 'differs from weather_depth'),
    ('depth', True, 'Invalid predictor_provenance.depth'),
    ('patch_size', 0, 'Invalid predictor_provenance.patch_size'),
    ('pretrained', 'false', 'Invalid predictor_provenance.pretrained'),
])
def test_provenance_must_be_present_pinned_and_consistent_with_config(field, value, message):
    row = weather_report()
    row['predictor_provenance'][field] = value
    with pytest.raises(ValueError, match=message):
        _validate_weather_provenance([row])


def test_weather_reports_cannot_omit_provenance_to_bypass_guards(tmp_path):
    row = weather_report()
    del row['predictor_provenance']
    with pytest.raises(ValueError, match='require predictor_provenance'):
        run_compare(tmp_path, [row])


def test_valid_pinned_seeds_remain_comparable(tmp_path):
    result = run_compare(tmp_path, [weather_report(seed=7), weather_report(seed=19)])
    assert len(result['seed_summary']) == 1
    assert next(iter(result['seed_summary'].values()))['seeds'] == [7, 19]


def test_historical_model_reports_without_weather_provenance_still_compare(tmp_path):
    rows = [report('raw', seed, 'neural_ode') for seed in (7, 19)]
    for row in rows:
        row.pop('predictor_provenance', None)
    unchanged = deepcopy(rows)
    result = run_compare(tmp_path, rows)
    assert result['ranking_allowed']
    assert rows == unchanged
