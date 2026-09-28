"""Temporal distribution supervision trains the forecast without future inputs."""
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
    initialize_manifold, load_predictor, prepare_constraint_pair, train, windows,
)


@pytest.mark.parametrize('kind', ['w2', 'kl_entropy'])
@pytest.mark.parametrize('family', ['neural_ode', 'climode'])
def test_flow_train_checkpoint_evaluation_and_causal_inference(
        pinn_prepared, tmp_path, monkeypatch, kind, family):
    _, _, _, archive = pinn_prepared
    args = _split_args(archive, tmp_path, 'pinn_statistical',
        '--model', family, '--statistical-loss', kind,
        '--statistical-flow-weight', '.2', '--statistical-flow-quantiles', '12')
    captured = _capture(monkeypatch)
    checkpoint = train(args)
    trained, _ = captured[0]
    restored, payload = load_predictor(checkpoint)
    assert payload['statistical_loss'] == kind
    config = payload['statistical_flow_config']
    assert config['weight'] == .2 and config['quantiles'] == 12
    assert payload['constraint_path'] == 'observed_reconstruction'
    for component in (trained.predictor, trained.reconstruction_decoder,
                      trained.bridge.manifold.core.manifold.encoder,
                      trained.bridge.manifold.info_head):
        _assert_finite_gradient(component)
    metrics = json.loads(checkpoint.with_suffix('.metrics.json').read_text())[0]
    for split in ('train', 'selection'):
        scores = metrics[split]
        assert scores['statistical_total'] > 0
        assert scores['statistical_flow'] > 0
        assert scores['statistical_flow_surface'] > 0
        assert scores['statistical_flow_information'] > 0
        assert scores['statistical_flow_regularization'] == pytest.approx(
            .2 * scores['statistical_flow'])
        if kind == 'kl_entropy':
            assert scores['statistical_total'] == scores['statistical_kl_entropy']
            assert scores['information_spatial_quantile'] == 0
        else:
            assert scores['statistical_total'] == scores['information_spatial_quantile']

    _, _, data = initialize_manifold(args)
    batch = next(iter(DataLoader(windows(data, restored.a_config, 'validation',
        max_windows=2, reconstruction_constraints=True,
        statistical_flow_targets=True), batch_size=2)))
    assert 'information_targets' not in batch
    assert 'statistical_flow_information_targets' in batch
    inputs = (batch['history'], batch['information'], batch['origin_time_ns'],
              torch.tensor([6., 12.]))
    trained.eval()
    with torch.no_grad():
        expected = trained(*inputs)
        actual = restored(*inputs)
        for key in ('mean', 'predicted_latent', 'origin_latent'):
            torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
        # Future target fields/information are supervision only. Inference
        # remains bitwise identical with poisoned future labels.
        batch['targets'].fill_(float('nan'))
        batch['statistical_flow_information_targets'].fill_(float('nan'))
        unchanged = restored(*inputs)
        for key in ('mean', 'predicted_latent', 'origin_latent'):
            torch.testing.assert_close(actual[key], unchanged[key], rtol=0, atol=0)
    report = evaluate(checkpoint, archive, tmp_path/'flow.evaluation.json',
        information=tmp_path/'pinn-information.npz', max_cases=2)
    assert report['statistical_flow_config'] == config
    assert report['statistical_loss_config'] == payload['statistical_loss_config']
    assert report['finite_forecast_fraction'] == 1.


def test_training_flow_alone_reaches_predictor_and_auxiliary_heads(
        pinn_prepared, tmp_path, monkeypatch):
    """Prove the new term contributes useful gradients beyond existing losses."""
    _, _, _, archive = pinn_prepared
    args = _split_args(archive, tmp_path, 'pinn_statistical',
        '--statistical-flow-weight', '.2')
    trainer = importlib.import_module('climate_manifold.downstream.train')
    reconstruction = importlib.import_module(
        'climate_manifold.downstream.reconstruction_objective')
    original_fit, original_aux = trainer.forecast_loss, reconstruction.reconstruction_constraint_losses

    def no_forecast_supervision(*args, **kwargs):
        result = original_fit(*args, **kwargs)
        result['loss'] = result['loss'].detach().new_zeros(())
        return result

    def no_reconstruction_supervision(*args, **kwargs):
        result = original_aux(*args, **kwargs)
        result['regularization'] = result['regularization'].detach().new_zeros(())
        return result

    monkeypatch.setattr(trainer, 'forecast_loss', no_forecast_supervision)
    monkeypatch.setattr(reconstruction, 'reconstruction_constraint_losses',
                        no_reconstruction_supervision)
    captured = _capture(monkeypatch)
    checkpoint = train(args)
    model, initial = captured[0]
    manifold = model.bridge.manifold
    for prefix, component in (
            ('predictor.', model.predictor),
            ('reconstruction_decoder.', model.reconstruction_decoder),
            ('bridge.manifold.info_head.', manifold.info_head),
            ('bridge.manifold.core.manifold.encoder.', manifold.core.manifold.encoder)):
        _assert_finite_gradient(component)
        assert any(not torch.equal(value, model.state_dict()[key])
                   for key, value in initial.items() if key.startswith(prefix)), prefix
    # The dedicated auxiliary flow does not directly train forecast D or PINN.
    for component in (manifold.core.manifold.decoder, manifold.pinn):
        assert all(parameter.grad is None for parameter in component.parameters())
    metrics = json.loads(checkpoint.with_suffix('.metrics.json').read_text())[0]
    assert metrics['train']['loss'] == metrics['train']['statistical_flow_regularization'] > 0


def test_flow_target_loading_is_opt_in_and_keeps_observed_pair_unchanged(
        pinn_prepared, tmp_path):
    _, _, _, archive = pinn_prepared
    args = _split_args(archive, tmp_path)
    model, _, data = initialize_manifold(args)
    old = windows(data, model.config, 'train', max_windows=1,
                  reconstruction_constraints=True)
    with_flow = windows(data, model.config, 'train', max_windows=1,
                       reconstruction_constraints=True, statistical_flow_targets=True)
    old_row, new_row = old[0], with_flow[0]
    assert 'information_targets' not in old_row
    assert 'statistical_flow_information_targets' not in old_row
    assert set(new_row) - set(old_row) == {'statistical_flow_information_targets'}
    for key in old_row:
        torch.testing.assert_close(old_row[key], new_row[key], rtol=0, atol=0)
    origin = with_flow.starts[0] + model.config.history_span_steps - 1
    expected = torch.from_numpy(data['information'][
        origin+1:origin+model.config.horizon_steps+1].copy())
    torch.testing.assert_close(new_row['statistical_flow_information_targets'], expected,
                               rtol=0, atol=0)
    with pytest.raises(ValueError):
        windows(data, model.config, 'train', max_windows=1,
                reconstruction_constraints=True, information_targets=True)
    with pytest.raises(ValueError):
        windows(data, model.config, 'train', max_windows=1,
                statistical_flow_targets=True)


@pytest.mark.parametrize('extra', [
    ('--constraint-pair', 'pinn_static', '--statistical-flow-weight', '.1'),
    ('--bridge', 'raw', '--statistical-flow-weight', '.1'),
    ('--statistical-flow-weight', '.1'),
    ('--constraint-pair', 'pinn_statistical', '--statistical-flow-weight', '-.1'),
    ('--constraint-pair', 'pinn_statistical', '--statistical-flow-weight', 'nan'),
    ('--constraint-pair', 'pinn_statistical', '--statistical-flow-weight', '.1',
     '--statistical-flow-quantiles', '0'),
])
def test_flow_invalid_or_inactive_training_config_rejected(tmp_path, extra):
    args = _args(tmp_path/'absent.npz', tmp_path/'unused.pt', *extra)
    with pytest.raises(ValueError):
        prepare_constraint_pair(args)
