"""Variable conditioning remains opt-in and preserves spatial information paths."""
from dataclasses import asdict, replace

import pytest
import torch

from test_pinn_training import pinn_prepared
from climate_manifold.architecture import ManifoldConfig
from climate_manifold.spatial import (
    SpatialClimateManifold, SpatialEncoder, SpatialDecoder,
    VariableSpatialEncoder, VariableSpatialDecoder,
)


def representation(previous, data, enabled=False, config=None):
    if config is None:
        config = replace(previous.config, representation_kind='spatial',
                         latent_channels=4, spatial_hidden_dim=6,
                         spatial_variable_conditioning=enabled)
    return SpatialClimateManifold(config, data['schema'], data['mean'], data['scale'],
                                  data['statistics'], data['information_metadata'])


def test_legacy_config_and_checkpoint_keep_additive_information_path(pinn_prepared):
    old, batch, data, _ = pinn_prepared
    original = representation(old, data)
    # Legacy payloads contain no conditioning flag and load strictly with the
    # pre-existing SpatialEncoder/Decoder parameter names.
    payload_config = asdict(original.config)
    payload_config.pop('spatial_variable_conditioning')
    restored = representation(old, data, config=ManifoldConfig(**payload_config))
    restored.load_state_dict(original.state_dict(), strict=True)
    assert isinstance(restored.core.manifold.encoder, SpatialEncoder)
    assert isinstance(restored.info_head, SpatialDecoder)
    assert restored.latent_fusion is None
    assert not any('variable_' in key or 'latent_fusion' in key for key in restored.state_dict())
    expected = (restored.core.manifold.encoder(batch['origin'])
                + restored.information(batch['information']))
    torch.testing.assert_close(restored.raw_encode(batch['origin'], batch['information']), expected,
                               rtol=0, atol=0)
    torch.testing.assert_close(original.raw_encode(batch['origin'], batch['information']), expected,
                               rtol=0, atol=0)


def test_conditioned_information_loss_reaches_all_encoder_and_decoder_branches(pinn_prepared):
    old, batch, data, _ = pinn_prepared
    model = representation(old, data, enabled=True)
    assert isinstance(model.core.manifold.encoder, VariableSpatialEncoder)
    assert isinstance(model.core.manifold.decoder, VariableSpatialDecoder)
    assert isinstance(model.information, VariableSpatialEncoder)
    assert isinstance(model.info_head, VariableSpatialDecoder)
    latent = model.raw_encode(batch['origin'], batch['information'])
    assert latent.shape == (len(batch['origin']), model.config.manifold_dim)
    decoded_info = model.info_head(latent)
    assert decoded_info.shape == batch['information'].shape
    # Only the information route is supervised here: this verifies that its
    # gradients constrain the shared representation, including surface inputs.
    (decoded_info - batch['information']).square().mean().backward()
    branches = [*model.core.manifold.encoder.variable_encoders,
                model.core.manifold.encoder.fusion,
                *model.information.variable_encoders, model.information.fusion,
                model.latent_fusion, *model.info_head.variable_decoders]
    for branch in branches:
        gradients = [parameter.grad for parameter in branch.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients) > 0
    # Forecast decoding is separate and is not trained by this information-only
    # objective. Joint forecast loss will supervise it through its own route.
    assert all(parameter.grad is None for parameter in model.core.manifold.decoder.parameters())
    clone = representation(old, data, config=ManifoldConfig(**asdict(model.config)))
    clone.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(clone.raw_encode(batch['origin'], batch['information']), latent)


def test_variable_heads_support_history_axes_and_preserve_variable_order():
    torch.set_num_threads(1)
    encoder = VariableSpatialEncoder((3, 5, 7), (4, 3, 4), 6, 2, True)
    decoder = VariableSpatialDecoder((4, 3, 4), (3, 5, 7), 6, 2, True)
    states = torch.randn(2, 3, 3 * 5 * 7)
    latent = encoder(states)
    assert latent.shape == (2, 3, 4 * 3 * 4)
    assert decoder(latent).shape == states.shape
    output = decoder(latent).reshape(2, 3, 3, 5 * 7)
    for index, head in enumerate(decoder.variable_decoders):
        torch.testing.assert_close(output[:, :, index], head(latent))
    assert decoder(encoder(states[0, 0])).shape == states.shape[-1:]


def test_conditioning_supports_surface_only_and_does_not_sample_noise(pinn_prepared):
    old, batch, data, _ = pinn_prepared
    config = replace(old.config, representation_kind='spatial', latent_channels=4,
                     spatial_hidden_dim=6, spatial_variable_conditioning=True)
    model = SpatialClimateManifold(config, data['schema'], data['mean'], data['scale'],
                                   data['statistics'])
    before = torch.random.get_rng_state()
    first = model.raw_encode(batch['history'])
    second = model.raw_encode(batch['history'])
    assert torch.equal(before, torch.random.get_rng_state())
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert model.information is model.info_head is model.latent_fusion is None
    assert len(model.core.manifold.decoder.variable_decoders) == config.grid[0]
    assert model.core.decode(first).shape == batch['history'].shape


def test_conditioning_flag_rejects_invalid_or_global_configuration():
    with pytest.raises(ValueError, match='boolean'):
        ManifoldConfig(state_dim=128, grid=(4, 4, 8), spatial_variable_conditioning='false')
    with pytest.raises(ValueError, match='spatial representation'):
        ManifoldConfig(state_dim=128, grid=(4, 4, 8), spatial_variable_conditioning=True)
