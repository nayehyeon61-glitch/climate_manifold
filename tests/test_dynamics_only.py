"""Dynamics-only routes: P(E_d) -> M -> D_s and T2(Raw, E_d) -> M.

E_s is removed from the forecast input but keeps its observed objectives.
"""
import json
from pathlib import Path

import pytest
import torch

from test_pinn_training import pinn_prepared
from test_spatial_training import _assert_finite_gradient
from test_guided_training import _inputs
from test_dual_dynamics_training import _pipeline, _has_no_gradient, dual_args
from test_dynamics_consistency_runner import fits, evaluations
from test_dynamics_vector_runner import setup_vector
from test_fusion_runner import arg, run_script
from climate_manifold.downstream.compare import _arm
from climate_manifold.downstream.dynamics_guide import LatentDynamicsAdapter
from climate_manifold.downstream.evaluate import evaluate
from climate_manifold.downstream.guided_fusion import GuidedFusion
from climate_manifold.downstream.pipeline import PredictorConfig
from climate_manifold.downstream.train import load_predictor, parser, train


def test_adapter_only_returns_projected_dynamics_without_spatial_latent():
    adapter = LatentDynamicsAdapter((3, 4, 6), (2, 4, 6))
    torch.nn.init.normal_(adapter.projection.weight)
    spatial, dynamics = torch.randn(2, 5, 72), torch.randn(2, 5, 48)
    only = adapter(spatial, dynamics, mode='only')
    torch.testing.assert_close(only, adapter(spatial, dynamics) - spatial)
    torch.testing.assert_close(only, adapter(torch.randn_like(spatial), dynamics, mode='only'))


def test_fusion_only_drops_spatial_guide_tokens():
    fusion = GuidedFusion((2, 4, 6), (3, 2, 3), 3, hidden=8, depth=1, heads=2,
                          dynamics_grid=(2, 2, 3), dynamics_mode='only')
    torch.nn.init.normal_(fusion.head[-1].weight)
    def forbidden(*args):
        raise AssertionError('Dynamics-only fusion must not embed E_s tokens')
    fusion.guide_embed.register_forward_pre_hook(forbidden)
    history, dynamics = torch.randn(2, 3, 48), torch.randn(2, 3, 12)
    first = fusion(history, torch.randn(2, 3, 18), dynamics_history=dynamics)
    torch.testing.assert_close(first, fusion(history, torch.randn(2, 3, 18), dynamics_history=dynamics))
    assert not torch.allclose(first, fusion(history, torch.randn(2, 3, 18),
                                            dynamics_history=torch.randn(2, 3, 12)))


@pytest.mark.parametrize('extra,match', [
    ({'dynamics_encoder': 'off'}, 'requires an enabled dynamics encoder'),
    ({'guide_mode': 'zero', 'bridge': 'guided', 'guide_architecture': 'fusion'}, 'guide_mode=learned'),
    ({'guide_interaction': 'transformer', 'bridge': 'guided', 'guide_architecture': 'fusion'},
     'guide_interaction=off'),
])
def test_only_config_rejects_ambiguous_controls(extra, match):
    base = dict(model='mlp', bridge='latent', training_mode='joint', latent_layout='spatial',
                raw_backend='matched', dynamics_encoder='conv_gru', dynamics_mode='only')
    with pytest.raises(ValueError, match=match):
        PredictorConfig(**{**base, **extra})


@pytest.mark.parametrize('bridge', ['latent', 'guided'])
def test_forecast_gradient_reaches_E_d_but_not_E_s(pinn_prepared, tmp_path, bridge):
    pipeline, batch, _ = _pipeline(pinn_prepared, tmp_path, bridge, mode='only')
    projection = (pipeline.dynamics_adapter.projection if bridge == 'latent'
                  else pipeline.guide_fusion.head[-1])
    torch.nn.init.normal_(projection.weight, std=.05)
    prediction = pipeline(*_inputs(batch), reconstruct_origin=False)
    (prediction['mean'] - batch['targets'][:, :2]).square().mean().backward()
    _assert_finite_gradient(pipeline.predictor)
    _assert_finite_gradient(pipeline.dynamics_encoder)
    assert _has_no_gradient(pipeline.bridge.manifold.core.manifold.encoder)


@pytest.mark.parametrize('bridge', ['latent', 'guided'])
def test_only_train_reload_evaluate_and_distinct_arm(pinn_prepared, tmp_path, bridge):
    _, _, _, archive = pinn_prepared
    args = dual_args(archive, tmp_path, bridge, 'conv_gru', '--dynamics-mode', 'only')
    checkpoint = train(args)
    restored, payload = load_predictor(checkpoint)
    assert payload['config']['dynamics_mode'] == 'only'
    assert payload['dynamics_contract']['spatial_forecast_input'] is False
    assert payload['dynamics_contract']['forecast_gradient_to_dynamics_encoder'] is True
    if bridge == 'guided':
        assert payload['guide_contract']['spatial_guide_tokens'] is False
    report = evaluate(checkpoint, archive, tmp_path/'only.validation.json',
                      information=tmp_path/'pinn-information.npz', max_cases=2)
    assert report['finite_forecast_fraction'] == 1. and not report['failed_origins']
    assert _arm(report).endswith(':dynamics=conv_gru:only')


def test_runner_dynamics_only_suite_trains_exactly_the_six_new_routes(tmp_path):
    env, log = setup_vector(tmp_path)
    result, rows = run_script(env, log, INCLUDE_DYNAMICS_ONLY='1', DYNAMICS_ONLY_SUITE='1',
                              INCLUDE_TWO_STAGE='1')
    assert result.returncode == 0, result.stderr
    configs = [parser().parse_args(r[2:]) for r in fits(rows)]
    assert len(configs) == 6 and 'Fits: 6' in result.stdout
    assert all(a.dynamics_mode == 'only' and a.guide_interaction == 'off' for a in configs)
    assert all(a.dynamics_auxiliary_architecture == 'latent_rollout' for a in configs)
    assert [(a.bridge, a.dynamics_supervision) for a in configs] == [
        ('latent', 'field'), ('latent', 'field_latent'), ('guided', 'field'),
        ('guided', 'field_latent'), ('latent', 'vector_only'), ('guided', 'vector_only')]
    assert all('--dynamics-diagnostics' in r for r in evaluations(rows))
    settings = json.loads((Path(env['RUN'])/'run-manifest.json').read_text())['settings']
    assert settings['include_dynamics_only'] == '1' and settings['dynamics_only_suite'] == '1'


def test_runner_without_flag_keeps_original_arms_and_settings(tmp_path):
    env, log = setup_vector(tmp_path)
    result, rows = run_script(env, log)
    assert result.returncode == 0, result.stderr
    assert not any(arg(r, '--dynamics-mode') == 'only' for r in fits(rows))
    settings = json.loads((Path(env['RUN'])/'run-manifest.json').read_text())['settings']
    assert 'include_dynamics_only' not in settings


def test_runner_rejects_dynamics_only_without_consistency(tmp_path):
    env, log = setup_vector(tmp_path)
    result, rows = run_script(env, log, INCLUDE_DYNAMICS_ONLY='1', INCLUDE_VECTOR_ONLY='0',
                              INCLUDE_DYNAMICS_CONSISTENCY='0')
    assert result.returncode != 0 and 'requires INCLUDE_DYNAMICS=1' in result.stderr
    assert not fits(rows)
