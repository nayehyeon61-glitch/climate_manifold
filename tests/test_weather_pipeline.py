"""Weather architecture adaptations preserve the joint/raw experiment contract."""
from dataclasses import asdict
import json

import pytest
import torch
from torch.utils.data import DataLoader

from test_pinn_training import pinn_prepared
from test_spatial_training import _args, _capture, _assert_finite_gradient
from climate_manifold.downstream.compare import compare
from climate_manifold.downstream.evaluate import evaluate
from climate_manifold.downstream.pipeline import PredictorConfig
from climate_manifold.downstream.protocol import validate_experiment
from climate_manifold.downstream.train import initialize_manifold, load_predictor, train, windows


@pytest.mark.parametrize('family', ['fourcastnet', 'climax'])
@pytest.mark.parametrize('changes', [
    {'training_mode': 'frozen'}, {'latent_layout': 'global'}, {'bridge': 'decoded'},
    {'bridge': 'raw', 'raw_backend': 'legacy'}, {'representation': 'plain_ae'},
    {'anchor': 'origin'},
])
def test_weather_predictor_rejects_unsupported_contracts(family, changes):
    config = dict(model=family, bridge='latent', training_mode='joint',
                  latent_layout='spatial', hidden_dim=16)
    config.update(changes)
    with pytest.raises(ValueError):
        PredictorConfig(**config)
    # A report dictionary must not bypass the constructor's route restrictions.
    config.setdefault('anchor', 'none')
    with pytest.raises(ValueError):
        validate_experiment(config, 'auxiliary')


@pytest.mark.parametrize('family', ['fourcastnet', 'climax'])
def test_weather_raw_and_joint_train_reload_evaluate_compare(
        pinn_prepared, tmp_path, monkeypatch, family):
    """One real CPU optimization step verifies the entire runnable comparison.

    This small synthetic case tests implementation contracts and gradients, not
    the forecast skill or published benchmark performance of either model.
    """
    _, _, _, archive = pinn_prepared
    information = tmp_path / 'pinn-information.npz'
    captured = _capture(monkeypatch)
    reports = []
    for bridge in ('raw', 'latent'):
        extra = ('--regularization', 'none') if bridge == 'raw' else (
            '--constraint-pair', 'pinn_statistical',
            '--constraint-decoder', 'separate_surface_and_information')
        args = _args(archive, tmp_path / f'{family}-{bridge}.pt',
                     '--model', family, '--bridge', bridge,
                     '--information', str(information), '--history-steps', '3',
                     '--history-stride', '4', '--batch-size', '16',
                     '--hidden-dim', '16', '--weather-depth', '2',
                     '--weather-patch-size', '2', *extra)
        before = len(captured)
        checkpoint = train(args)
        trained, initial = captured[before]
        restored, payload = load_predictor(checkpoint)
        config = restored.config
        contract = validate_experiment(asdict(config), 'primary')
        assert contract['prediction_space'] == ('field' if bridge == 'raw' else 'latent')
        assert config.model == family
        assert config.weather_depth == 2 and config.weather_patch_size == 2
        assert payload['training_contract']['batch_size'] == 16
        assert payload['conditioning']['observed_information_available']
        assert payload['conditioning']['direct_origin_information'] == (bridge == 'raw')
        assert trained.predictor.history_dt_hours == 24.
        provenance = payload['predictor_provenance']
        assert provenance['family'] == family
        assert provenance['upstream_source'].startswith('https://github.com/')
        assert len(provenance['upstream_commit']) == 40
        assert provenance['uncertainty'] == 'deterministic'
        assert 'adaptation' in payload['implementation']
        _assert_finite_gradient(trained.predictor)
        assert any(not torch.equal(value, trained.state_dict()[name])
                   for name, value in initial.items() if name.startswith('predictor.'))
        assert trained.conditional_flow is restored.conditional_flow is None
        assert payload['statistical_flow_config'] is None
        assert payload['conditional_flow_config'] is None
        if bridge == 'raw':
            assert trained.bridge.manifold is restored.bridge.manifold is None
            assert all(name.startswith('predictor.') for name in payload['model'])
            assert all(weight == 0 for weight in payload['objective_weights'].values())
        else:
            manifold = trained.bridge.manifold
            assert not payload['a_frozen']
            assert payload['constraint_pair'] == 'pinn_statistical'
            assert payload['constraint_path'] == 'observed_reconstruction'
            assert trained.reconstruction_decoder is not None
            for module in (manifold.core.manifold.encoder, manifold.core.manifold.decoder,
                           manifold.information, manifold.info_head, manifold.pinn,
                           trained.reconstruction_decoder):
                _assert_finite_gradient(module)
            for prefix in ('bridge.manifold.core.manifold.encoder.',
                           'bridge.manifold.core.manifold.decoder.'):
                assert any(not torch.equal(value, trained.state_dict()[name])
                           for name, value in initial.items() if name.startswith(prefix))
            metrics = json.loads(checkpoint.with_suffix('.metrics.json').read_text())[0]['train']
            assert metrics['pinn_tendency'] == metrics['static'] == 0
            assert metrics['pinn_total'] > 0 and metrics['information_spatial_quantile'] > 0
        _, _, data = initialize_manifold(args)
        batch = next(iter(DataLoader(windows(data, restored.a_config, 'validation', max_windows=2),
                                     batch_size=2)))
        inputs = (batch['history'], batch['information'], batch['origin_time_ns'],
                  torch.tensor([6., 12.]))
        trained.eval()
        with torch.no_grad():
            expected, actual = trained(*inputs), restored(*inputs)
            assert expected['std'] is actual['std'] is None
            torch.testing.assert_close(actual['mean'], expected['mean'], rtol=0, atol=0)
            assert actual['mean'].shape == (2, 2, restored.a_config.state_dim)
            assert torch.isfinite(actual['mean']).all()
        if bridge == 'raw':
            # The matched control retains the same causal origin information
            # that reaches E in the manifold arm; it is not silently discarded.
            causal_information = batch['information'].clone().requires_grad_()
            predicted = restored(batch['history'], causal_information,
                                 batch['origin_time_ns'], inputs[-1])['mean']
            gradient, = torch.autograd.grad(predicted.square().mean(), causal_information)
            assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        report_path = tmp_path / f'{family}-{bridge}.evaluation.json'
        report = evaluate(checkpoint, archive, report_path, information=information, max_cases=2)
        assert report['finite_forecast_fraction'] == 1.
        assert not report['failed_origins']
        assert report['predictor_provenance'] == provenance
        for scores in report['scores']['climode']['per_variable'].values():
            assert scores['aggregate']['crps'] is None
        reports.append(report_path)
    comparison = compare(reports, tmp_path / f'{family}-comparison.json')
    assert len(comparison['seed_summary']) == 2
    assert comparison['direct_comparison']['ranking_allowed']
    assert len(comparison['paired_effects']) == 1
    assert comparison['paired_effects'][0]['interpretation'] == 'whole_model_comparison'
