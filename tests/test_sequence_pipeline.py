"""Sequence predictors share raw/latent data contracts and joint training paths."""
from dataclasses import asdict
import json

import pytest
import torch
from torch.utils.data import DataLoader

from test_pinn_training import pinn_prepared
from test_spatial_training import _args, _capture, _assert_finite_gradient
from climate_manifold.downstream.compare import compare
from climate_manifold.downstream.evaluate import evaluate
from climate_manifold.downstream.pipeline import PredictorConfig, SEQUENCE_IMPLEMENTATIONS
from climate_manifold.downstream.protocol import validate_experiment
from climate_manifold.downstream.train import initialize_manifold, load_predictor, train, windows


@pytest.mark.parametrize('family', ['convlstm','simvp'])
@pytest.mark.parametrize('changes', [
    {'training_mode':'frozen'}, {'latent_layout':'global'}, {'bridge':'decoded'},
    {'bridge':'raw','raw_backend':'legacy'}, {'representation':'plain_ae'}, {'anchor':'origin'},
])
def test_sequence_configuration_rejects_unsupported_paths(family,changes):
    config = dict(model=family,bridge='latent',training_mode='joint',latent_layout='spatial')
    config.update(changes)
    with pytest.raises(ValueError):
        PredictorConfig(**config)
    # Reports loaded as dictionaries must not bypass construction-time guards.
    config.setdefault('anchor','none')
    with pytest.raises(ValueError):
        validate_experiment(config,'auxiliary')


@pytest.mark.parametrize('family', ['convlstm','simvp'])
@pytest.mark.parametrize('bridge', ['raw','latent'])
def test_sequence_configuration_is_primary_only_for_supported_contracts(family,bridge):
    config = PredictorConfig(model=family,bridge=bridge,training_mode='joint',
                             latent_layout='spatial',raw_backend='matched')
    contract = validate_experiment(asdict(config),'primary')
    assert contract['suite'] == 'primary'
    assert not contract['representation_frozen']
    assert contract['prediction_space'] == ('field' if bridge == 'raw' else 'latent')


@pytest.mark.parametrize('family', ['convlstm','simvp'])
def test_sequence_raw_and_pinn_statistical_training_roundtrip_and_comparison(
        pinn_prepared,tmp_path,monkeypatch,family):
    _, _, _, archive = pinn_prepared
    information = tmp_path/'pinn-information.npz'
    captured = _capture(monkeypatch)
    reports = []
    for bridge in ('raw','latent'):
        extra = ('--regularization','none') if bridge == 'raw' else (
            '--constraint-pair','pinn_statistical')
        args = _args(archive,tmp_path/f'{family}-{bridge}.pt',
            '--model',family,'--bridge',bridge,'--information',str(information),
            '--history-stride','4','--batch-size','16',*extra)
        before = len(captured)
        checkpoint = train(args)
        trained, initial = captured[before]
        restored,payload = load_predictor(checkpoint)
        assert payload['implementation'] == SEQUENCE_IMPLEMENTATIONS[family]
        assert payload['training_contract']['batch_size'] == 16
        assert payload['experiment']['suite'] == 'primary'
        assert payload['conditioning']['observed_information_available']
        assert payload['conditioning']['direct_origin_information'] == (bridge == 'raw')
        assert trained.predictor.history_dt_hours == 24.
        provenance = payload['predictor_provenance']
        assert provenance['family'] == family
        assert provenance['history_steps'] == 6 and provenance['history_dt_hours'] == 24.
        assert provenance['uncertainty'] == 'deterministic'
        _assert_finite_gradient(trained.predictor)
        assert any(not torch.equal(value,trained.state_dict()[name])
                   for name,value in initial.items() if name.startswith('predictor.'))
        if bridge == 'raw':
            assert trained.bridge.manifold is restored.bridge.manifold is None
            assert all(name.startswith('predictor.') for name in payload['model'])
            assert all(weight == 0 for weight in payload['objective_weights'].values())
        else:
            manifold = trained.bridge.manifold
            assert payload['constraint_pair'] == 'pinn_statistical'
            assert payload['constraint_path'] == 'observed_reconstruction'
            for module in (manifold.core.manifold.encoder,manifold.core.manifold.decoder,
                           manifold.information,manifold.info_head,manifold.pinn):
                _assert_finite_gradient(module)
            metrics = json.loads(checkpoint.with_suffix('.metrics.json').read_text())[0]['train']
            assert metrics['pinn_tendency'] == metrics['static'] == 0
            assert metrics['pinn_total'] > 0 and metrics['information_spatial_quantile'] > 0
        _, _, data = initialize_manifold(args)
        batch = next(iter(DataLoader(windows(data,restored.a_config,'validation',max_windows=2),batch_size=2)))
        inputs = (batch['history'],batch['information'],batch['origin_time_ns'],torch.tensor([6.,12.]))
        trained.eval()
        with torch.no_grad():
            expected,actual = trained(*inputs),restored(*inputs)
            assert expected['std'] is actual['std'] is None
            torch.testing.assert_close(actual['mean'],expected['mean'],rtol=0,atol=0)
            assert actual['mean'].shape == (2,2,restored.a_config.state_dim)
        if bridge == 'raw':
            causal_information = batch['information'].clone().requires_grad_()
            predicted = restored(batch['history'],causal_information,batch['origin_time_ns'],inputs[-1])['mean']
            gradient, = torch.autograd.grad(predicted.square().mean(),causal_information)
            assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        report_path = tmp_path/f'{family}-{bridge}.evaluation.json'
        report = evaluate(checkpoint,archive,report_path,information=information,max_cases=2)
        assert report['finite_forecast_fraction'] == 1.
        assert report['predictor_provenance'] == provenance
        reports.append(report_path)
    result = compare(reports,tmp_path/f'{family}-comparison.json')
    assert len(result['seed_summary']) == 2
    assert result['direct_comparison']['ranking_allowed']
    assert len(result['paired_effects']) == 1
    assert result['paired_effects'][0]['interpretation'] == 'whole_model_comparison'
