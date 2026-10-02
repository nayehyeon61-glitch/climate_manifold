"""Guide fusion: E -> Fusion Transformer(raw, guide) -> any matched-raw family."""
import numpy as np
import pytest
import torch

from test_pinn_training import pinn_prepared
from test_spatial_training import _args, _capture, _assert_finite_gradient
from test_guided_training import _batch, _inputs
from climate_manifold.downstream.evaluate import evaluate
from climate_manifold.downstream.guided_fusion import GuidedFusion
from climate_manifold.downstream.pipeline import ForecastPipeline, PredictorConfig
from climate_manifold.downstream.protocol import validate_experiment
from climate_manifold.downstream.train import initialize_manifold, load_predictor, train

FAMILIES = ['mlp', 'neural_ode', 'climode', 'convlstm', 'simvp', 'fourcastnet', 'climax', 'transformer']


def fusion_args(archive, root, model, *extra):
    return _args(archive, root/f'{model}-fusion.pt', '--model', model, '--bridge', 'guided',
                 '--guide-architecture', 'fusion', '--guide-fusion-depth', '1',
                 '--raw-backend', 'matched', '--guide-direct-information',
                 '--information', str(root/'pinn-information.npz'),
                 '--constraint-pair', 'statistical', '--statistical-loss', 'signed_measure',
                 '--spatial-variable-conditioning', '--history-steps', '2',
                 '--weather-depth', '1', '--weather-patch-size', '2',
                 '--transformer-heads', '2', '--tendency-weight', '0', *extra)


def test_fusion_starts_as_identity_and_zero_guide_ignores_guide():
    torch.manual_seed(0)
    fusion = GuidedFusion((3, 5, 7), (2, 3, 4), history_steps=2, hidden=8, depth=1, heads=2)
    history, guide = torch.randn(2, 2, 105), torch.randn(2, 2, 24)
    torch.testing.assert_close(fusion(history, guide), history, rtol=0, atol=0)
    with pytest.raises(ValueError, match='Learned guide'):
        fusion(history)
    torch.nn.init.normal_(fusion.head[-1].weight)
    assert not torch.equal(fusion(history, guide), fusion(history, guide + 1.))
    fusion.guide_mode = 'zero'
    torch.testing.assert_close(fusion(history, guide), fusion(history, guide + 1.), rtol=0, atol=0)


def test_fusion_config_contract():
    with pytest.raises(ValueError, match='Joint guided bridge requires the Transformer'):
        PredictorConfig(model='mlp', bridge='guided', training_mode='joint',
                        latent_layout='spatial', raw_backend='matched')
    with pytest.raises(ValueError, match='Guide fusion requires bridge=guided'):
        PredictorConfig(model='mlp', bridge='raw', training_mode='joint', latent_layout='spatial',
                        raw_backend='matched', guide_architecture='fusion')
    with pytest.raises(ValueError, match='matched-raw'):
        PredictorConfig(model='mlp', bridge='guided', training_mode='joint', latent_layout='spatial',
                        guide_architecture='fusion')
    for model in FAMILIES:
        config = PredictorConfig(model=model, bridge='guided', training_mode='joint',
                                 latent_layout='spatial', raw_backend='matched',
                                 guide_architecture='fusion', hidden_dim=8, transformer_heads=2)
        contract = validate_experiment(config, 'primary')
        assert contract['path'] == 'encoder -> fusion transformer(raw, guide) -> raw-grid predictor -> fields'
        assert contract['predictor_variant'] == 'guide_fusion_raw_spatial_' + model


@pytest.mark.parametrize('model', FAMILIES)
def test_fusion_predictor_initializes_like_raw_arm(pinn_prepared, tmp_path, model):
    _, _, _, archive = pinn_prepared
    manifold, _, data = initialize_manifold(fusion_args(archive, tmp_path, model))
    common = dict(model=model, training_mode='joint', latent_layout='spatial', raw_backend='matched',
                  hidden_dim=8, weather_depth=1, weather_patch_size=2, transformer_heads=2)
    torch.manual_seed(3)
    raw = ForecastPipeline(manifold, PredictorConfig(bridge='raw', **common), schema=data['schema'])
    torch.manual_seed(3)
    guided = ForecastPipeline(manifold, PredictorConfig(bridge='guided', guide_architecture='fusion',
                                                        guide_fusion_depth=1, guide_direct_information=True,
                                                        **common), schema=data['schema'])
    raw_state, guided_state = raw.predictor.state_dict(), guided.predictor.state_dict()
    assert raw_state.keys() == guided_state.keys()
    for key, value in raw_state.items():
        torch.testing.assert_close(guided_state[key], value, rtol=0, atol=0)
    # Zero-initialized fusion: the guided arm forecasts exactly as raw at step 0.
    batch = _batch(data, manifold.config)
    raw.eval(), guided.eval()
    with torch.no_grad():
        torch.testing.assert_close(guided(*_inputs(batch))['mean'], raw(*_inputs(batch))['mean'],
                                   rtol=0, atol=0)


@pytest.mark.parametrize('model', FAMILIES)
def test_fusion_trains_reloads_and_evaluates_every_family(pinn_prepared, tmp_path, model, monkeypatch):
    _, _, _, archive = pinn_prepared
    args = fusion_args(archive, tmp_path, model)
    captured = _capture(monkeypatch)
    checkpoint = train(args)
    trained, _ = captured[0]
    manifold = trained.bridge.manifold
    for component in (manifold.core.manifold.encoder, trained.guide_fusion, trained.predictor):
        _assert_finite_gradient(component)
    assert all(not p.requires_grad for p in manifold.core.manifold.decoder.parameters())

    restored, payload = load_predictor(checkpoint)
    assert payload['guide_contract']['version'] == 'climate_manifold.statistical_guide_fusion.v1'
    assert payload['guide_contract']['downstream_model'] == model
    assert payload['implementation'].startswith('guide_fusion_v1+')
    assert payload['conditioning']['direct_origin_information'] is True
    report = evaluate(checkpoint, archive, tmp_path/f'{model}.evaluation.json',
                      information=tmp_path/'pinn-information.npz', max_cases=2,
                      forecast_output=tmp_path/f'{model}.forecast.npz')
    assert report['finite_forecast_fraction'] == 1. and not report['failed_origins']
    with np.load(tmp_path/f'{model}.forecast.npz') as values:
        assert 'guide_history_latent' in values and np.isfinite(values['mean']).all()


def test_zero_guide_fusion_blocks_forecast_gradient_to_encoder(pinn_prepared, tmp_path):
    _, _, _, archive = pinn_prepared
    manifold, _, data = initialize_manifold(fusion_args(archive, tmp_path, 'mlp'))
    pipe = ForecastPipeline(manifold, PredictorConfig(
        model='mlp', bridge='guided', training_mode='joint', latent_layout='spatial',
        raw_backend='matched', guide_architecture='fusion', guide_mode='zero', hidden_dim=8,
        transformer_heads=2), schema=data['schema'])
    torch.nn.init.normal_(pipe.guide_fusion.head[-1].weight, std=.01)
    batch = _batch(data, manifold.config)
    (pipe(*_inputs(batch), reconstruct_origin=False)['mean']
     - batch['targets'][:, :2]).square().mean().backward()
    _assert_finite_gradient(pipe.guide_fusion)
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0
               for p in manifold.core.manifold.encoder.parameters())
