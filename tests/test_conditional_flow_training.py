"""Observed-pair conditional Flow Matching stays independent of forecast outputs."""
import importlib
import json

import pytest
import torch
from torch.utils.data import DataLoader

from test_pinn_training import pinn_prepared
from test_spatial_training import _args, _capture, _assert_finite_gradient
from test_split_training import _split_args
from climate_manifold.downstream.evaluate import evaluate
from climate_manifold.downstream.train import (
    digest, initialize_manifold, load_predictor, prepare_constraint_pair, train, windows,
)


def _cfm_args(archive, tmp_path, *extra):
    return _split_args(
        archive, tmp_path, 'pinn_statistical',
        '--conditional-flow-weight', '.2', '--conditional-flow-quantiles', '8',
        '--conditional-flow-hidden-dim', '12', '--conditional-flow-noise-scale', '.15',
        *extra)


@pytest.mark.parametrize('kind', ['w2', 'kl_entropy'])
@pytest.mark.parametrize('family', ['neural_ode', 'climode'])
def test_conditional_flow_train_restore_evaluate_and_causal_forecast(
        pinn_prepared, tmp_path, monkeypatch, kind, family):
    _, _, _, archive = pinn_prepared
    args = _cfm_args(archive, tmp_path, '--model', family, '--statistical-loss', kind)
    captured = _capture(monkeypatch)
    checkpoint = train(args)
    trained, _ = captured[0]
    restored, payload = load_predictor(checkpoint)
    config = payload['conditional_flow_config']
    assert config['weight'] == .2 and config['quantiles'] == 8
    assert config['hidden_dim'] == 12 and config['noise_scale'] == .15
    assert payload['statistical_loss'] == kind
    assert payload['statistical_flow_config'] is None
    assert payload['constraint_path'] == 'observed_reconstruction'
    assert restored.conditional_flow is not None
    for component in (trained.predictor, trained.conditional_flow,
                      trained.reconstruction_decoder, trained.bridge.manifold.info_head,
                      trained.bridge.manifold.core.manifold.encoder):
        _assert_finite_gradient(component)
    for key, value in trained.conditional_flow.state_dict().items():
        torch.testing.assert_close(value, restored.conditional_flow.state_dict()[key],
                                   rtol=0, atol=0)
    assert payload['forecast_parameters'] + payload['constraint_parameters'] == payload['trainable_parameters']
    auxiliary_modules = (trained.conditional_flow, trained.reconstruction_decoder,
                         trained.bridge.manifold.info_head, trained.bridge.manifold.pinn)
    assert payload['constraint_parameters'] == sum(
        parameter.numel() for module in auxiliary_modules if module is not None
        for parameter in module.parameters() if parameter.requires_grad)
    metrics = json.loads(checkpoint.with_suffix('.metrics.json').read_text())[0]
    for split in ('train', 'selection'):
        scores = metrics[split]
        assert scores['statistical_total'] > 0
        assert scores['conditional_flow'] > 0
        assert scores['conditional_flow_regularization'] == pytest.approx(
            .2 * scores['conditional_flow'])
        if kind == 'kl_entropy':
            assert scores['statistical_total'] == scores['statistical_kl_entropy']
            assert scores['information_spatial_quantile'] == 0
        else:
            assert scores['statistical_total'] == scores['information_spatial_quantile']
    _, _, data = initialize_manifold(args)
    batch = next(iter(DataLoader(windows(
        data, restored.a_config, 'validation', max_windows=2,
        reconstruction_constraints=True, statistical_flow_targets=True), batch_size=2)))
    assert 'information_targets' not in batch
    inputs = (batch['history'], batch['information'], batch['origin_time_ns'],
              torch.tensor([6., 12.]))
    trained.eval()
    with torch.no_grad():
        expected, actual = trained(*inputs), restored(*inputs)
        for key in ('mean', 'predicted_latent', 'origin_latent'):
            torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
        # Both kinds of future labels are supervision only; ordinary forecasting
        # does not need observed auxiliary labels or run the auxiliary flow head.
        batch['targets'].fill_(float('nan'))
        batch['statistical_flow_information_targets'].fill_(float('nan'))
        batch['constraint_states'].fill_(float('nan'))
        batch['constraint_information'].fill_(float('nan'))
        unchanged = restored(*inputs)
        for key in ('mean', 'predicted_latent', 'origin_latent'):
            torch.testing.assert_close(actual[key], unchanged[key], rtol=0, atol=0)
    report = evaluate(checkpoint, archive, tmp_path/'conditional-flow.evaluation.json',
                      information=tmp_path/'pinn-information.npz', max_cases=2)
    assert report['conditional_flow_config'] == config
    assert report['statistical_loss_config'] == payload['statistical_loss_config']
    assert report['finite_forecast_fraction'] == 1.


def test_conditional_flow_alone_updates_observed_representation_not_forecaster(
        pinn_prepared, tmp_path, monkeypatch):
    """Isolate the new term instead of inferring its gradients from total loss."""
    _, _, _, archive = pinn_prepared
    args = _cfm_args(archive, tmp_path)
    trainer = importlib.import_module('climate_manifold.downstream.train')
    reconstruction = importlib.import_module('climate_manifold.downstream.reconstruction_objective')
    original_fit = trainer.forecast_loss
    original_aux = reconstruction.reconstruction_constraint_losses

    def no_forecast_supervision(*args, **kwargs):
        result = original_fit(*args, **kwargs)
        result['loss'] = result['loss'].detach().new_zeros(())
        return result

    def no_observed_supervision(*args, **kwargs):
        result = original_aux(*args, **kwargs)
        result['regularization'] = result['regularization'].detach().new_zeros(())
        return result

    monkeypatch.setattr(trainer, 'forecast_loss', no_forecast_supervision)
    monkeypatch.setattr(reconstruction, 'reconstruction_constraint_losses', no_observed_supervision)
    captured = _capture(monkeypatch)
    checkpoint = train(args)
    model, initial = captured[0]
    manifold = model.bridge.manifold
    for prefix, component in (
            ('conditional_flow.', model.conditional_flow),
            ('reconstruction_decoder.', model.reconstruction_decoder),
            ('bridge.manifold.info_head.', manifold.info_head),
            ('bridge.manifold.core.manifold.encoder.', manifold.core.manifold.encoder)):
        _assert_finite_gradient(component)
        assert any(not torch.equal(value, model.state_dict()[key])
                   for key, value in initial.items() if key.startswith(prefix)), prefix
    for prefix, component in (
            ('predictor.', model.predictor),
            ('bridge.manifold.core.manifold.decoder.', manifold.core.manifold.decoder),
            ('bridge.manifold.pinn.', manifold.pinn)):
        assert all(parameter.grad is None for parameter in component.parameters())
        for key, value in initial.items():
            if key.startswith(prefix):
                torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
    scores = json.loads(checkpoint.with_suffix('.metrics.json').read_text())[0]['train']
    assert scores['loss'] == scores['conditional_flow_regularization'] > 0


def test_conditional_flow_default_off_and_historical_checkpoint_restore(
        pinn_prepared, tmp_path):
    _, _, _, archive = pinn_prepared
    checkpoint = train(_split_args(archive, tmp_path))
    restored, payload = load_predictor(checkpoint)
    assert restored.conditional_flow is None
    assert payload.get('conditional_flow_config') is None
    assert not any(key.startswith('conditional_flow.') for key in restored.state_dict())
    # A checkpoint from before this feature has no CFM metadata or weights.
    payload.pop('conditional_flow_config', None)
    historical = tmp_path/'historical-without-conditional-flow.pt'
    torch.save(payload, historical)
    historical.with_suffix('.manifest.json').write_text(json.dumps(
        {'checkpoint_sha256': digest(historical)}))
    legacy, _ = load_predictor(historical)
    assert legacy.conditional_flow is None
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, legacy.state_dict()[key], rtol=0, atol=0)


def test_conditional_flow_training_seed_reproduces_auxiliary_loss_and_parameters(
        pinn_prepared, tmp_path):
    _, _, _, archive = pinn_prepared
    args = _cfm_args(archive, tmp_path)
    first_path = train(args)
    first, _ = load_predictor(first_path)
    first_metrics = json.loads(first_path.with_suffix('.metrics.json').read_text())
    args.output = str(tmp_path/'repeated.pt')
    second_path = train(args)
    second, _ = load_predictor(second_path)
    second_metrics = json.loads(second_path.with_suffix('.metrics.json').read_text())
    for split in ('train', 'selection'):
        for key in ('loss', 'conditional_flow', 'state_mse'):
            assert first_metrics[0][split][key] == second_metrics[0][split][key]
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)


def test_enabling_conditional_flow_preserves_existing_model_initialization(
        pinn_prepared, tmp_path, monkeypatch):
    """The new auxiliary head cannot silently change a matched forecast baseline."""
    _, _, _, archive = pinn_prepared
    captured = _capture(monkeypatch)
    off_args = _split_args(archive, tmp_path)
    off_args.output = str(tmp_path/'cfm-off.pt')
    train(off_args)
    on_args = _cfm_args(archive, tmp_path)
    on_args.output = str(tmp_path/'cfm-on.pt')
    train(on_args)
    initial_off, initial_on = captured[0][1], captured[1][1]
    assert not any(key.startswith('conditional_flow.') for key in initial_off)
    assert any(key.startswith('conditional_flow.') for key in initial_on)
    assert set(initial_off) == {key for key in initial_on
                               if not key.startswith('conditional_flow.')}
    for key, value in initial_off.items():
        torch.testing.assert_close(value, initial_on[key], rtol=0, atol=0)


@pytest.mark.parametrize('extra', [
    ('--constraint-pair', 'pinn_static', '--conditional-flow-weight', '.1'),
    ('--bridge', 'raw', '--conditional-flow-weight', '.1'),
    ('--conditional-flow-weight', '.1'),
    ('--constraint-pair', 'pinn_statistical', '--conditional-flow-weight', '-.1'),
    ('--constraint-pair', 'pinn_statistical', '--conditional-flow-weight', 'nan'),
    ('--constraint-pair', 'pinn_statistical', '--conditional-flow-weight', '.1',
     '--conditional-flow-quantiles', '0'),
    ('--constraint-pair', 'pinn_statistical', '--conditional-flow-weight', '.1',
     '--conditional-flow-hidden-dim', '0'),
    ('--constraint-pair', 'pinn_statistical', '--conditional-flow-weight', '.1',
     '--conditional-flow-noise-scale', '-1'),
    ('--constraint-pair', 'pinn_statistical', '--conditional-flow-weight', '.1',
     '--statistical-flow-weight', '.1'),
    ('--constraint-pair', 'pinn_statistical', '--conditional-flow-weight', '.1',
     '--constraint-decoder', 'surface_and_information'),
])
def test_conditional_flow_incompatible_options_rejected_before_loading(tmp_path, extra):
    args = _args(tmp_path/'absent.npz', tmp_path/'unused.pt', *extra)
    with pytest.raises(ValueError):
        prepare_constraint_pair(args)
