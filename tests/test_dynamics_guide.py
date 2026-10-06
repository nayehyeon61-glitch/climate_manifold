"""Observed-only dynamics representations and both downstream integration paths."""
import pytest
import torch

from climate_manifold.downstream.dynamics_guide import (
    DynamicsAuxiliaryHead, DynamicsGuide, LatentDynamicsAdapter,
)
from climate_manifold.downstream.guided_fusion import GuidedFusion


def _has_gradient(module):
    gradients = [p.grad for p in module.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum().item() for g in gradients) > 0


@pytest.mark.parametrize('kind', ['conv_gru', 'temporal_conv'])
def test_dynamics_is_causal_and_sensitive_to_observed_order(kind):
    torch.manual_seed(8)
    encoder = DynamicsGuide((2, 4, 6), (3, 2, 3), 4, hidden=8, kind=kind)
    history = torch.randn(2, 4, 48)
    encoded = encoder(history)
    assert encoded.shape == (2, 4, 18) and torch.isfinite(encoded).all()
    changed_future = history.clone()
    changed_future[:, 2:] += 3.
    torch.testing.assert_close(encoder(changed_future)[:, :2], encoded[:, :2], rtol=0, atol=0)
    # Same last observation and same set of earlier fields; only their order
    # changes, so a per-frame encoder without temporal memory would fail here.
    reordered = history[:, [1, 0, 2, 3]]
    assert not torch.allclose(encoder(reordered)[:, -1], encoded[:, -1])
    differentiable = history.detach().requires_grad_()
    encoder(differentiable)[:, 1].sum().backward()
    assert torch.count_nonzero(differentiable.grad[:, 2:]) == 0
    assert torch.count_nonzero(differentiable.grad[:, :2]) > 0


@pytest.mark.parametrize('kind', ['conv_gru', 'temporal_conv'])
def test_dynamics_auxiliary_future_supervision_reaches_independent_encoder(kind):
    torch.manual_seed(2)
    encoder = DynamicsGuide((2, 5, 7), (3, 2, 3), 4, hidden=8, kind=kind)
    auxiliary = DynamicsAuxiliaryHead((3, 2, 3), (2, 5, 7), hidden=8)
    history = torch.randn(2, 4, 70)
    prediction = auxiliary(encoder(history), torch.tensor([24., 48., 96.]))
    assert prediction.shape == (2, 3, 70)
    assert not torch.allclose(prediction[:, 0], prediction[:, 1])
    (prediction - torch.randn_like(prediction)).square().mean().backward()
    _has_gradient(encoder)
    _has_gradient(auxiliary)
    # Each lead is a query, not a future observation or rollout input.
    with torch.no_grad():
        direct = auxiliary(encoder(history), torch.tensor([96.]))
    torch.testing.assert_close(direct[:, 0], prediction[:, 2], rtol=1e-5, atol=1e-6)


def test_latent_adapter_identity_and_parameter_matched_zero_control():
    adapter = LatentDynamicsAdapter((2, 3, 4), (3, 2, 3))
    spatial = torch.randn(2, 4, 24, requires_grad=True)
    dynamics = torch.randn(2, 4, 18, requires_grad=True)
    torch.testing.assert_close(adapter(spatial, dynamics), spatial, rtol=0, atol=0)
    with torch.no_grad():
        adapter.projection.weight.fill_(.2)
    learned = adapter(spatial, dynamics)
    assert learned.shape == spatial.shape and not torch.equal(learned, spatial)
    assert not torch.equal(learned, adapter(spatial, dynamics + 1.))
    learned.square().mean().backward()
    assert dynamics.grad.abs().sum() > 0 and spatial.grad.abs().sum() > 0
    _has_gradient(adapter)
    adapter.zero_grad(set_to_none=True)
    dynamics.grad = None
    zero = adapter(spatial, dynamics, mode='zero')
    torch.testing.assert_close(zero, adapter(spatial, dynamics + 10., mode='zero'), rtol=0, atol=0)
    zero.square().mean().backward()
    assert dynamics.grad is None
    assert adapter.projection.bias.grad is not None


def _fusion(**extra):
    return GuidedFusion((2, 5, 7), (3, 3, 4), 4, hidden=8, depth=1,
                        heads=2, patch_size=2, **extra)


def test_fusion_dynamics_off_has_legacy_checkpoint_and_rng_contract():
    torch.manual_seed(7)
    original = _fusion()
    original_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(7)
    disabled = _fusion(dynamics_grid=None, dynamics_mode='zero')
    torch.testing.assert_close(torch.random.get_rng_state(), original_rng, rtol=0, atol=0)
    assert not any('dynamics' in name for name in disabled.state_dict())
    assert disabled.type_embed.shape == (2, 8)
    disabled.load_state_dict(original.state_dict(), strict=True)
    for key, value in disabled.state_dict().items():
        torch.testing.assert_close(value, original.state_dict()[key], rtol=0, atol=0)
    torch.manual_seed(7)
    enabled = _fusion(dynamics_grid=(2, 2, 3))
    torch.testing.assert_close(torch.random.get_rng_state(), original_rng, rtol=0, atol=0)
    for key, value in original.state_dict().items():
        actual = enabled.state_dict()[key]
        if key == 'type_embed':
            actual = actual[:2]
        torch.testing.assert_close(actual, value, rtol=0, atol=0)


def test_fusion_dynamics_identity_information_and_zero_controls():
    torch.manual_seed(5)
    fusion = _fusion(dynamics_grid=(2, 2, 3))
    history = torch.randn(2, 4, 70)
    guide = torch.randn(2, 4, 36, requires_grad=True)
    dynamics = torch.randn(2, 4, 12, requires_grad=True)
    torch.testing.assert_close(fusion(history, guide, dynamics), history, rtol=0, atol=0)
    with torch.no_grad():
        torch.nn.init.normal_(fusion.head[-1].weight, std=.05)
    output = fusion(history, guide, dynamics)
    assert not torch.equal(output, fusion(history, guide, dynamics + 2.))
    output.square().mean().backward()
    assert guide.grad.abs().sum() > 0 and dynamics.grad.abs().sum() > 0
    fusion.zero_grad(set_to_none=True)
    guide.grad = dynamics.grad = None
    fusion.dynamics_mode = 'zero'
    zero = fusion(history, guide, dynamics)
    torch.testing.assert_close(zero, fusion(history, guide, dynamics + 2.), rtol=0, atol=0)
    assert not torch.equal(zero, fusion(history, guide + 2., dynamics))
    zero.square().mean().backward()
    assert dynamics.grad is None and guide.grad.abs().sum() > 0
    fusion.guide_mode = 'zero'
    fusion.dynamics_mode = 'learned'
    assert not torch.equal(fusion(history, None, dynamics), fusion(history, None, dynamics + 2.))


def test_dynamics_physical_history_spacing_and_input_validation():
    torch.manual_seed(3)
    daily = DynamicsGuide((1, 2, 3), (2, 2, 3), 3, hidden=4, history_dt_hours=24.)
    half_day = DynamicsGuide((1, 2, 3), (2, 2, 3), 3, hidden=4, history_dt_hours=12.)
    half_day.load_state_dict(daily.state_dict())
    history = torch.arange(3.).view(1, 3, 1).expand(1, 3, 6)
    assert not torch.allclose(daily(history), half_day(history))
    for dt in (0, -1, float('nan'), float('inf'), True):
        with pytest.raises(ValueError, match='spacing'):
            DynamicsGuide((1, 2, 3), (2, 2, 3), 3, history_dt_hours=dt)
    with pytest.raises(ValueError, match='Dynamics kind'):
        DynamicsGuide((1, 2, 3), (2, 2, 3), 3, kind='future')
    with pytest.raises(ValueError, match='Raw history'):
        daily(history[:, :2])
    invalid = history.clone()
    invalid[0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='Raw history'):
        daily(invalid)
    fusion = _fusion(dynamics_grid=(2, 2, 3))
    with pytest.raises(ValueError, match='Learned dynamics'):
        fusion(torch.randn(2, 4, 70), torch.randn(2, 4, 36))
    with pytest.raises(ValueError, match='Dynamics-disabled'):
        _fusion()(torch.randn(2, 4, 70), torch.randn(2, 4, 36), torch.randn(2, 4, 12))
    head = DynamicsAuxiliaryHead((2, 2, 3), (1, 2, 3), hidden=4)
    for leads in ([], [0.], [24., 12.], [24., float('nan')]):
        with pytest.raises(ValueError, match='Lead hours'):
            head(daily(history), torch.tensor(leads))
