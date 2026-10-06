"""Joint training and honest held-out interventions for the optional guide T1."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from test_pinn_training import pinn_prepared
from test_dual_dynamics_training import dual_args, _objective_config, _has_no_gradient
from test_guided_training import _batch, _inputs
from test_spatial_training import _capture, _assert_finite_gradient
from climate_manifold.downstream.dynamics_objective import dynamics_auxiliary_losses
from climate_manifold.downstream.evaluate import evaluate
from climate_manifold.downstream.pipeline import ForecastPipeline, PredictorConfig
from climate_manifold.downstream.train import initialize_manifold, load_predictor, train
from climate_manifold.physical_information import digest


def _args(archive, root, encoder='conv_gru', *extra):
    return dual_args(archive, root, 'guided', encoder,
                     '--guide-interaction', 'transformer', '--guide-interaction-depth', '1',
                     '--max-windows', '4', *extra)


@pytest.mark.parametrize('encoder', ['conv_gru', 'temporal_conv'])
def test_two_stage_train_reload_and_same_checkpoint_bypass(pinn_prepared, tmp_path, monkeypatch, encoder):
    _, _, _, archive = pinn_prepared
    args = _args(archive, tmp_path, encoder)
    captured = _capture(monkeypatch)
    checkpoint = train(args)
    trained, initial = captured[0]
    for module in (trained.guide_fusion.guide_interaction, trained.guide_fusion.transformer,
                   trained.dynamics_encoder, trained.predictor):
        _assert_finite_gradient(module)
    assert any(not torch.equal(value, trained.state_dict()[name])
               for name, value in initial.items() if name.startswith('guide_fusion.guide_interaction.'))
    restored, payload = load_predictor(checkpoint)
    assert payload['guide_interaction_contract']['raw_input'] is False
    assert payload['guide_contract']['guide_interaction'] == payload['guide_interaction_contract']
    assert payload['implementation'].startswith('two_stage_guide_v1+dual_dynamics_v1+')
    assert payload['forecast_parameters'] + payload['constraint_parameters'] == payload['trainable_parameters']
    _, _, data = initialize_manifold(args)
    batch = _batch(data, restored.a_config)
    trained.eval()
    with torch.no_grad():
        expected = trained(*_inputs(batch))['mean']
        actual = restored(*_inputs(batch))['mean']
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for name in ('targets', 'constraint_states', 'constraint_information'):
            batch[name].fill_(float('nan'))
        torch.testing.assert_close(restored(*_inputs(batch))['mean'], actual, rtol=0, atol=0)
    for override in ('checkpoint', 'off'):
        forecast = tmp_path/f'{override}.npz'
        report = evaluate(checkpoint, archive, tmp_path/f'{override}.json',
                          information=tmp_path/'pinn-information.npz', max_cases=2,
                          forecast_output=forecast, guide_interaction_override=override)
        bypassed = override == 'off'
        expected_meta = {'requested': override, 'trained': 'transformer',
                         'effective': 'off' if bypassed else 'transformer', 'bypassed': bypassed}
        assert report['guide_interaction_evaluation'] == expected_meta
        assert report['config']['guide_interaction'] == 'transformer'
        assert report['checkpoint_sha256'] == digest(checkpoint)
        assert report['finite_forecast_fraction'] == 1.
        with np.load(forecast, allow_pickle=False) as arrays:
            assert json.loads(str(arrays['guide_interaction_evaluation_json'])) == expected_meta
            assert json.loads(str(arrays['guide_interaction_contract_json'])) == payload['guide_interaction_contract']
        if bypassed:
            assert 'not an independently trained single-stage baseline' in report['limits']
    # Evaluation interventions do not mutate saved config or checkpoint weights.
    after, _ = load_predictor(checkpoint)
    assert after.guide_fusion.guide_interaction_bypass is False


@pytest.mark.parametrize('encoder', ['off', 'conv_gru'])
def test_old_single_stage_reload_and_reject_bypass(pinn_prepared, tmp_path, encoder):
    _, _, _, archive = pinn_prepared
    args = dual_args(archive, tmp_path, 'guided', encoder)
    checkpoint = train(args)
    reference, payload = load_predictor(checkpoint)
    assert reference.guide_fusion.guide_interaction is None
    historical = deepcopy(payload)
    historical['config'].pop('guide_interaction')
    historical['config'].pop('guide_interaction_depth')
    _save_with_manifest(historical, checkpoint)
    restored, _ = load_predictor(checkpoint)
    _, _, data = initialize_manifold(args)
    batch = _batch(data, restored.a_config)
    with torch.no_grad():
        torch.testing.assert_close(restored(*_inputs(batch))['mean'], reference(*_inputs(batch))['mean'], rtol=0, atol=0)
    with pytest.raises(ValueError, match='bypass requires'):
        evaluate(checkpoint, archive, tmp_path/'bad.json', information=tmp_path/'pinn-information.npz',
                 max_cases=1, guide_interaction_override='off')
    report = evaluate(checkpoint, archive, tmp_path/'old.json', information=tmp_path/'pinn-information.npz', max_cases=1)
    assert 'guide_interaction_evaluation' not in report


def _save_with_manifest(payload, checkpoint):
    torch.save(payload, checkpoint)
    manifest = checkpoint.with_suffix('.manifest.json')
    values = json.loads(manifest.read_text())
    values['checkpoint_sha256'] = digest(checkpoint)
    manifest.write_text(json.dumps(values))


def test_interaction_contract_tampering_rejected(pinn_prepared, tmp_path):
    _, _, _, archive = pinn_prepared
    checkpoint = train(_args(archive, tmp_path))
    _, payload = load_predictor(checkpoint)
    payload['guide_interaction_contract']['depth'] += 1
    _save_with_manifest(payload, checkpoint)
    with pytest.raises(ValueError, match='guide_interaction_contract'):
        load_predictor(checkpoint)


def _pipeline(prepared, root, family):
    _, _, _, archive = prepared
    a, _, data = initialize_manifold(_args(archive, root))
    config = PredictorConfig(model=family, bridge='guided', training_mode='joint',
        latent_layout='spatial', raw_backend='matched', hidden_dim=8, weather_depth=1,
        transformer_heads=2, climode_step_hours=6., guide_architecture='fusion',
        guide_fusion_depth=1, guide_direct_information=True, dynamics_encoder='conv_gru',
        dynamics_hidden_dim=4, dynamics_channels=2, guide_interaction='transformer')
    model = ForecastPipeline(a, config, schema=data['schema'], separate_reconstruction_decoder=True)
    return model, _batch(data, a.config)


@pytest.mark.parametrize('family', ['mlp', 'neural_ode', 'climode', 'convlstm', 'simvp', 'fourcastnet', 'climax', 'transformer'])
def test_all_eight_predictors_accept_two_stage_and_learn(pinn_prepared, tmp_path, family):
    model, batch = _pipeline(pinn_prepared, tmp_path, family)
    # Open the deliberately zero-initialized T2 residual for an informative
    # forecast gradient test, rather than mistaking initialization for a detach.
    torch.nn.init.normal_(model.guide_fusion.head[-1].weight, std=.03)
    prediction = model(*_inputs(batch), reconstruct_origin=False)
    assert prediction['mean'].shape == batch['targets'][:, :2].shape
    assert torch.isfinite(prediction['mean']).all()
    (prediction['mean'] - batch['targets'][:, :2]).square().mean().backward()
    for module in (model.guide_fusion.guide_interaction, model.guide_fusion.transformer,
                   model.predictor, model.dynamics_encoder,
                   model.bridge.manifold.core.manifold.encoder):
        _assert_finite_gradient(module)
    assert _has_no_gradient(model.dynamics_auxiliary_head)


def test_auxiliary_objective_does_not_train_either_transformer(pinn_prepared, tmp_path):
    model, batch = _pipeline(pinn_prepared, tmp_path, 'mlp')
    prediction = model(*_inputs(batch), reconstruct_origin=False)
    aux = dynamics_auxiliary_losses(model, prediction, batch, _inputs(batch)[-1], _objective_config())
    aux['dynamics_regularization'].backward()
    _assert_finite_gradient(model.dynamics_encoder)
    _assert_finite_gradient(model.dynamics_auxiliary_head)
    for module in (model.guide_fusion, model.predictor, model.bridge.manifold.core.manifold.encoder):
        assert _has_no_gradient(module)
