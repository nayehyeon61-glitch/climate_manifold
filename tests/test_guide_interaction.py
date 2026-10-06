"""Two-stage attention gates: compatibility, information isolation and gradients."""
from copy import deepcopy
from dataclasses import asdict, replace

import pytest
import torch

from test_pinn_training import pinn_prepared
from test_dual_dynamics_training import dual_args
from test_guided_training import _batch, _inputs
from test_spatial_training import _assert_finite_gradient
from climate_manifold.downstream.guide_interaction import GuideInteraction
from climate_manifold.downstream.guided_fusion import GuidedFusion
from climate_manifold.downstream.pipeline import ForecastPipeline, PredictorConfig
from climate_manifold.downstream.train import initialize_manifold


def _fusion(**extra):
    return GuidedFusion((2, 5, 7), (3, 3, 4), 3, hidden=8, depth=1, heads=2,
                        patch_size=2, dynamics_grid=(2, 2, 3), **extra)


def _histories():
    return torch.randn(2, 3, 70), torch.randn(2, 3, 36), torch.randn(2, 3, 12)


def _config(**extra):
    return replace(PredictorConfig(
        model='mlp', bridge='guided', training_mode='joint', latent_layout='spatial',
        raw_backend='matched', hidden_dim=8, weather_depth=1, transformer_heads=2,
        guide_architecture='fusion', guide_fusion_depth=1, guide_direct_information=True,
        dynamics_encoder='conv_gru', dynamics_hidden_dim=4, dynamics_channels=2), **extra)


def _no_gradient(module):
    return all(p.grad is None or not bool(torch.count_nonzero(p.grad))
               for p in module.parameters())


def test_off_retains_legacy_keys_rng_and_strict_checkpoint_contract():
    torch.manual_seed(12)
    historical = _fusion()
    state, rng = deepcopy(historical.state_dict()), torch.random.get_rng_state().clone()
    torch.manual_seed(12)
    off = _fusion(guide_interaction='off', guide_interaction_depth=3)
    assert off.guide_interaction is None
    assert not any('guide_interaction' in key for key in off.state_dict())
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    off.load_state_dict(state, strict=True)
    for key, value in state.items():
        torch.testing.assert_close(value, off.state_dict()[key], rtol=0, atol=0)
    # Historical configs simply omit the new fields.
    old_config = asdict(_config())
    old_config.pop('guide_interaction')
    old_config.pop('guide_interaction_depth')
    assert PredictorConfig(**old_config) == _config()
    # Compare active, nontrivial T2 forecasts, not just the zero residual head.
    torch.nn.init.normal_(historical.head[-1].weight, std=.1)
    off.load_state_dict(historical.state_dict(), strict=True)
    inputs = _histories()
    torch.testing.assert_close(off(*inputs), historical(*inputs), rtol=0, atol=0)


def test_on_preserves_shared_initialization_and_bypass_matches_one_stage():
    torch.manual_seed(5)
    one_stage = _fusion()
    rng = torch.random.get_rng_state().clone()
    torch.manual_seed(5)
    two_stage = _fusion(guide_interaction='transformer')
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    for key, value in one_stage.state_dict().items():
        torch.testing.assert_close(two_stage.state_dict()[key], value, rtol=0, atol=0)
    assert not ({id(p) for p in two_stage.transformer.parameters()}
                & {id(p) for p in two_stage.guide_interaction.parameters()})
    inputs = _histories()
    torch.testing.assert_close(two_stage(*inputs), inputs[0], rtol=0, atol=0)
    torch.nn.init.normal_(two_stage.head[-1].weight, std=.1)
    one_stage.load_state_dict({k: v for k, v in two_stage.state_dict().items()
                              if not k.startswith('guide_interaction.')}, strict=True)
    before = deepcopy(two_stage.state_dict())
    active = two_stage(*inputs)
    two_stage.guide_interaction_bypass = True
    bypassed = two_stage(*inputs)
    torch.testing.assert_close(bypassed, one_stage(*inputs), rtol=0, atol=0)
    assert not torch.allclose(active, bypassed)
    assert all(torch.equal(before[k], v) for k, v in two_stage.state_dict().items())


def test_t1_receives_only_guide_tokens_and_t2_receives_refined_banks():
    fusion = _fusion(guide_interaction='transformer')
    history, spatial, dynamics = _histories()
    t1_inputs, t1_outputs, t2_inputs = [], [], []
    handles = [
        fusion.guide_interaction.register_forward_pre_hook(
            lambda module, args: t1_inputs.append(tuple(t.detach().clone() for t in args))),
        fusion.guide_interaction.register_forward_hook(
            lambda module, args, out: t1_outputs.append(tuple(t.detach().clone() for t in out))),
        fusion.transformer.register_forward_pre_hook(
            lambda module, args: t2_inputs.append(args[0].detach().clone())),
    ]
    try:
        fusion(history, spatial, dynamics)
        fusion(history + 3., spatial, dynamics)
    finally:
        for handle in handles:
            handle.remove()
    for first, changed_raw in zip(t1_inputs[0], t1_inputs[1]):
        torch.testing.assert_close(first, changed_raw, rtol=0, atol=0)
    for first, changed_raw in zip(t1_outputs[0], t1_outputs[1]):
        torch.testing.assert_close(first, changed_raw, rtol=0, atol=0)
    raw_length = fusion.history_steps * fusion.raw_patch_count
    assert not torch.equal(t2_inputs[0][:, :raw_length], t2_inputs[1][:, :raw_length])
    torch.testing.assert_close(t2_inputs[0][:, raw_length:], torch.cat(t1_outputs[0], 1),
                               rtol=0, atol=0)
    assert [v.shape[1] for v in t1_outputs[0]] == [12, 6]


def test_t1_attention_connects_both_guide_banks_across_observed_times():
    torch.manual_seed(8)
    attention = GuideInteraction(8, depth=1, heads=2)
    spatial = torch.randn(2, 9, 8, requires_grad=True)
    dynamics = torch.randn(2, 6, 8, requires_grad=True)
    spatial_out, dynamics_out = attention(spatial, dynamics)
    assert spatial_out.shape == spatial.shape and dynamics_out.shape == dynamics.shape
    # A single spatial token depends on dynamics and distant observed spatial
    # tokens. All tokens precede the forecast origin; no causal future mask is
    # needed within this complete observed-window representation.
    spatial_out[:, 0, 0].sum().backward()
    assert spatial.grad[:, -1].abs().sum() > 0
    assert dynamics.grad[:, -1].abs().sum() > 0
    _assert_finite_gradient(attention)


@pytest.mark.parametrize('zero_bank', ['spatial', 'dynamics'])
def test_zero_controls_remove_source_before_t1(zero_bank):
    fusion = _fusion(guide_interaction='transformer',
                     guide_mode='zero' if zero_bank == 'spatial' else 'learned',
                     dynamics_mode='zero' if zero_bank == 'dynamics' else 'learned')
    torch.nn.init.normal_(fusion.head[-1].weight, std=.1)
    history, spatial, dynamics = (x.requires_grad_() for x in _histories())
    output = fusion(history, spatial, dynamics)
    changed = [history, spatial + 2. if zero_bank == 'spatial' else spatial,
               dynamics + 2. if zero_bank == 'dynamics' else dynamics]
    torch.testing.assert_close(output, fusion(*changed), rtol=0, atol=0)
    output.square().mean().backward()
    ignored, used = (spatial, dynamics) if zero_bank == 'spatial' else (dynamics, spatial)
    assert ignored.grad is None and used.grad.abs().sum() > 0
    _assert_finite_gradient(fusion.guide_interaction)


def test_pipeline_on_off_preserve_all_shared_modules_and_initial_forecasts(pinn_prepared, tmp_path):
    _, _, _, archive = pinn_prepared
    manifold, _, data = initialize_manifold(dual_args(archive, tmp_path))
    torch.manual_seed(14)
    off = ForecastPipeline(deepcopy(manifold), _config(), schema=data['schema'],
                           separate_reconstruction_decoder=True)
    rng = torch.random.get_rng_state().clone()
    torch.manual_seed(14)
    on = ForecastPipeline(deepcopy(manifold), _config(guide_interaction='transformer'),
                          schema=data['schema'], separate_reconstruction_decoder=True)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    for key, value in off.state_dict().items():
        torch.testing.assert_close(on.state_dict()[key], value, rtol=0, atol=0)
    batch = _batch(data, manifold.config)
    torch.testing.assert_close(on(*_inputs(batch))['mean'], off(*_inputs(batch))['mean'],
                               rtol=0, atol=0)


@pytest.mark.parametrize('guide_mode,dynamics_mode', [
    ('learned', 'learned'), ('zero', 'learned'), ('learned', 'zero')])
def test_pipeline_forecast_gradients_reach_expected_encoders_only(
        pinn_prepared, tmp_path, guide_mode, dynamics_mode):
    _, _, _, archive = pinn_prepared
    manifold, _, data = initialize_manifold(dual_args(archive, tmp_path))
    pipeline = ForecastPipeline(manifold, _config(
        guide_interaction='transformer', guide_mode=guide_mode, dynamics_mode=dynamics_mode),
        schema=data['schema'], separate_reconstruction_decoder=True)
    torch.nn.init.normal_(pipeline.guide_fusion.head[-1].weight, std=.1)
    batch = _batch(data, manifold.config)
    output = pipeline(*_inputs(batch), reconstruct_origin=False)
    (output['mean'] - batch['targets'][:, :2]).square().mean().backward()
    for module in (pipeline.guide_fusion.guide_interaction, pipeline.guide_fusion.transformer,
                   pipeline.predictor):
        _assert_finite_gradient(module)
    for module, mode in ((manifold.core.manifold.encoder, guide_mode),
                         (pipeline.dynamics_encoder, dynamics_mode)):
        if mode == 'learned':
            _assert_finite_gradient(module)
        else:
            assert _no_gradient(module)
    for module in (pipeline.dynamics_auxiliary_head, pipeline.reconstruction_decoder,
                   manifold.info_head, manifold.core.manifold.decoder):
        assert _no_gradient(module)
    # Targets stay loss-only even with the extra attention stage.
    for key in ('targets', 'constraint_states', 'constraint_information'):
        batch[key].fill_(float('nan'))
    with torch.no_grad():
        torch.testing.assert_close(pipeline(*_inputs(batch))['mean'], output['mean'], rtol=0, atol=0)


@pytest.mark.parametrize('changes', [
    {'guide_interaction': 'invalid'}, {'guide_interaction_depth': 0},
    {'guide_interaction_depth': True}, {'guide_interaction_depth': 1.5},
    {'guide_interaction': 'transformer', 'dynamics_encoder': 'off'},
    {'guide_interaction': 'transformer', 'bridge': 'latent', 'guide_architecture': 'joint'},
    {'guide_interaction': 'transformer', 'guide_architecture': 'joint'},
])
def test_config_rejects_invalid_or_unsupported_interaction(changes):
    with pytest.raises(ValueError, match='Guide interaction'):
        _config(**changes)


def test_interaction_validates_module_arguments_and_tokens():
    with pytest.raises(ValueError, match='enabled dynamics'):
        GuidedFusion((2, 4, 4), (2, 2, 2), 2, hidden=8, heads=2,
                     guide_interaction='transformer')
    for options in ({'hidden': 0}, {'hidden': 8, 'depth': True}, {'hidden': 7, 'heads': 2}):
        with pytest.raises(ValueError, match='Guide interaction'):
            GuideInteraction(**options)
    interaction = GuideInteraction(8, heads=2)
    good = torch.randn(2, 3, 8)
    for bad in (torch.empty(2, 0, 8), torch.randn(2, 3, 7), good * float('nan')):
        with pytest.raises(ValueError, match='finite observed'):
            interaction(good, bad)
    with pytest.raises(ValueError, match='batch size'):
        interaction(good, good[:1])
    with pytest.raises(ValueError, match='dtype and device'):
        interaction(good, good.double())
