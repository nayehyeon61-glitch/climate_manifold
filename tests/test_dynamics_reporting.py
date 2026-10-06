"""Dynamics on/off/zero are different experiments, never extra training seeds."""
import json

import numpy as np
import pytest

from test_guided_comparison import guide_report, transformer_raw
from test_raw_comparison import run_compare
from test_pinn_training import pinn_prepared
from test_spatial_training import _args
from climate_manifold.downstream.compare import _arm, compare
from climate_manifold.downstream.dynamics_objective import dynamics_contract
from climate_manifold.downstream.evaluate import evaluate
from climate_manifold.downstream.pipeline import PredictorConfig
from climate_manifold.downstream.plot_comparison import summarize_comparison
from climate_manifold.downstream.protocol import validate_experiment
from climate_manifold.downstream.statistical_objective import make_statistical_config
from climate_manifold.downstream.train import guide_contract, train


def dynamics_report(bridge='guided', mode='learned', seed=7, encoder='conv_gru'):
    row = guide_report(bridge, seed=seed)
    row['config'].update(model='mlp', guide_architecture='fusion' if bridge == 'guided' else 'joint', guide_fusion_depth=2,
                         dynamics_encoder=encoder, dynamics_mode=mode,
                         dynamics_hidden_dim=8, dynamics_channels=3)
    cfg = PredictorConfig(**row['config'])
    row['guide_contract'] = guide_contract(cfg)
    row['dynamics_contract'] = dynamics_contract(cfg)
    row['dynamics_auxiliary_config'] = None if encoder == 'off' else {
        'version': 'climate_manifold.dynamics_auxiliary.v1',
        'statistical_loss_config': make_statistical_config('w2'),
        'reconstruction_weight': .1, 'statistical_weight': .1, 'tendency_weight': 0.,
    }
    if encoder != 'off':
        row['forecast_parameters'] += 15
        row['constraint_parameters'] += 15
        row['total_parameters'] += 30
        row['trainable_parameters'] += 30
    row['representation_sha256'] = f'{bridge}-{encoder}-{mode}-{seed}'
    return row


def raw_report(seed):
    row = transformer_raw(seed)
    row['config']['model'] = 'mlp'
    return row


@pytest.mark.parametrize('bridge', ['latent', 'guided'])
def test_protocol_describes_independent_dynamics_encoder_and_actual_route(bridge):
    cfg = dynamics_report(bridge)['config']
    result = validate_experiment(cfg, 'primary')
    assert result['dynamics_encoder'] == 'conv_gru'
    assert 'dynamics encoder' in result['path']
    assert result['prediction_space'] == ('latent' if bridge == 'latent' else 'field')
    for updates in ({'bridge': 'raw'}, {'training_mode': 'frozen'},
                    {'latent_layout': 'global'}, {'representation': 'plain_ae'}):
        with pytest.raises(ValueError):
            validate_experiment({**cfg, **updates}, 'primary')


def test_on_off_zero_and_two_routes_remain_separate_in_metrics_and_plots(tmp_path):
    reports = []
    for seed in (7, 19):
        reports.append(raw_report(seed))
        for bridge in ('latent', 'guided'):
            reports.extend([dynamics_report(bridge, seed=seed, encoder='off'),
                            dynamics_report(bridge, seed=seed, mode='zero'),
                            dynamics_report(bridge, seed=seed)])
    result = run_compare(tmp_path, reports)
    assert result['ranking_allowed'] and len(result['seed_summary']) == 7
    assert all(row['seeds'] == [7, 19] for row in result['seed_summary'].values())
    identities = {(row['model'], row['bridge'], _arm(row)) for row in result['rows']}
    assert identities == {(row['model'], row['bridge'], _arm(row))
                          for row in result['climode_benchmark']['rows']}
    assert len(result['dynamics_effects']) == 16  # two routes, two controls, two seeds/leads
    assert {row['interpretation'] for row in result['dynamics_effects']} == {
        'add_dynamics_encoder', 'change_dynamics_input'}
    assert not result['statistical_loss_effects']
    assert not result['guide_effects']  # Statistical guide is unchanged.
    assert (tmp_path/'comparison.dynamics-effects.csv').is_file()
    summary = summarize_comparison(result)
    assert any('Dynamics (conv_gru)' in row['label'] for row in summary)
    assert any('Zero dynamics (conv_gru)' in row['label'] for row in summary)
    assert {row['metric'] for row in summary} >= {
        'dynamics_gain_percent', 'dynamics_input_gain_percent', 'rmse_skill_percent'}
    assert all(row['valid_seeds'] == 2 for row in summary)


@pytest.mark.parametrize('mutation,match', [
    ('contract', 'dynamics_contract'), ('capacity', 'dynamics_hidden_dim'),
    ('architecture', 'dynamics_encoder'), ('objective', 'dynamics_auxiliary_config'),
    ('negative', 'nonnegative'), ('off_metadata', 'Disabled dynamics'),
])
def test_unfair_dynamics_or_mislabelled_metadata_is_rejected(tmp_path, mutation, match):
    first, second = dynamics_report(seed=7), dynamics_report(seed=19)
    if mutation == 'contract':
        second['dynamics_contract']['future_targets_input'] = True
    elif mutation in ('capacity', 'architecture'):
        second['config'].update({'dynamics_hidden_dim': 16} if mutation == 'capacity'
                                else {'dynamics_encoder': 'temporal_conv'})
        cfg = PredictorConfig(**second['config'])
        second.update(dynamics_contract=dynamics_contract(cfg), guide_contract=guide_contract(cfg))
    elif mutation == 'objective':
        second['dynamics_auxiliary_config']['statistical_weight'] = .2
    elif mutation == 'negative':
        second['dynamics_auxiliary_config']['statistical_weight'] = -.1
    else:
        second['config']['dynamics_encoder'] = 'off'
        second['guide_contract'] = guide_contract(PredictorConfig(**second['config']))
    with pytest.raises(ValueError, match=match):
        run_compare(tmp_path, [first, second])


@pytest.mark.parametrize('bridge', ['latent', 'guided'])
def test_training_evaluation_exports_dynamics_features_and_reload_contract(
        pinn_prepared, tmp_path, bridge):
    _, _, _, archive = pinn_prepared
    info = tmp_path/'pinn-information.npz'
    reports = []
    for encoder, mode in (('off', 'learned'), ('conv_gru', 'learned'), ('conv_gru', 'zero')):
        name = f'{bridge}-{encoder}-{mode}'
        args = _args(archive, tmp_path/(name+'.pt'),
            '--model', 'mlp', '--bridge', bridge, '--raw-backend', 'matched',
            '--guide-architecture', 'fusion' if bridge == 'guided' else 'joint', '--guide-fusion-depth', '1',
            '--transformer-heads', '2', '--weather-depth', '1', '--history-steps', '2',
            '--information', str(info), '--mode', 'enriched', '--constraint-pair', 'statistical',
            '--constraint-decoder', 'separate_surface_and_information',
            '--statistical-weight', '.1', '--regularization', 'full',
            '--dynamics-encoder', encoder, '--dynamics-mode', mode,
            '--dynamics-hidden-dim', '4', '--dynamics-channels', '2')
        checkpoint = train(args)
        path = tmp_path/(name+'.json')
        forecast = tmp_path/(name+'.npz')
        row = evaluate(checkpoint, archive, path, information=info, max_cases=2,
                       forecast_output=forecast)
        reports.append(path)
        assert row['finite_forecast_fraction'] == 1
        with np.load(forecast) as values:
            if encoder == 'off':
                assert row['dynamics_contract'] is None
                assert 'dynamics_history_latent' not in values
            else:
                assert json.loads(str(values['dynamics_contract_json'])) == row['dynamics_contract']
                assert json.loads(str(values['dynamics_auxiliary_config_json'])) == row['dynamics_auxiliary_config']
                assert np.isfinite(values['dynamics_history_latent']).all()
                assert values['dynamics_latent_shape'].tolist() == row['dynamics_latent_shape']
                assert values['dynamics_history_latent'].shape[-1] == np.prod(row['dynamics_latent_shape'])
                if bridge == 'latent':
                    semantics = json.loads(str(values['latent_diagnostic_semantics_json']))
                    assert semantics == row['latent_diagnostic_semantics']
                    assert semantics['reference_encoder'] == 'statistical_encoder_E_s'
                    assert 'not a direct E_d trajectory score' in semantics['interpretation']
                else:
                    assert row['latent_diagnostic_semantics'] is None
                assert ('forecast_history_latent' in values) == (bridge == 'latent')
                assert ('fused_history' in values) == (bridge == 'guided')
    result = compare(reports, tmp_path/'comparison.json')
    assert result['ranking_allowed'] and len(result['seed_summary']) == 3
    assert {row['interpretation'] for row in result['dynamics_effects']} == {
        'add_dynamics_encoder', 'change_dynamics_input'}


@pytest.mark.parametrize('family', ['fourcastnet', 'climax'])
@pytest.mark.parametrize('bridge', ['latent', 'guided'])
def test_weather_core_provenance_survives_dynamics_wrappers(
        pinn_prepared, tmp_path, family, bridge):
    _, _, _, archive = pinn_prepared
    info = tmp_path/'pinn-information.npz'
    reports = []
    for encoder in ('off', 'conv_gru'):
        name = family+'-'+bridge+'-'+encoder
        args = _args(archive, tmp_path/(name+'.pt'),
            '--model', family, '--bridge', bridge, '--raw-backend', 'matched',
            '--guide-architecture', 'fusion' if bridge == 'guided' else 'joint',
            '--guide-fusion-depth', '1', '--transformer-heads', '2',
            '--weather-depth', '1', '--history-steps', '2',
            '--information', str(info), '--mode', 'enriched',
            '--constraint-pair', 'statistical', '--regularization', 'full',
            '--constraint-decoder', 'separate_surface_and_information',
            '--dynamics-encoder', encoder, '--dynamics-hidden-dim', '4', '--dynamics-channels', '2')
        checkpoint = train(args)
        path = tmp_path/(name+'.json')
        report = evaluate(checkpoint, archive, path, information=info, max_cases=2)
        reports.append(path)
        assert report['predictor_provenance']['family'] == family
        if encoder != 'off':
            prefix = 'dual_dynamics_v1+' + ('guide_fusion_v1+' if bridge == 'guided' else '')
            assert report['implementation'] == prefix + report['predictor_provenance']['implementation']
    result = compare(reports, tmp_path/'comparison.json')
    assert result['ranking_allowed'] and len(result['seed_summary']) == 2
    assert {row['interpretation'] for row in result['dynamics_effects']} == {'add_dynamics_encoder'}
