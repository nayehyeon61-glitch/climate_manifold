"""Independent observed-field decoder: raw coordinates and gradient isolation."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from test_pinn_training import pinn_prepared
from climate_manifold.downstream.pipeline import ForecastPipeline, PredictorConfig
from climate_manifold.spatial import SpatialClimateManifold


def _pipeline(prepared, layout, *, separate=True):
    original, batch, data, _ = prepared
    if layout == 'global':
        manifold = deepcopy(original)
    else:
        config = replace(original.config, representation_kind='spatial',
                         latent_channels=3, spatial_downsample=2, spatial_hidden_dim=8)
        manifold = SpatialClimateManifold(
            config, data['schema'], data['mean'], data['scale'], data['statistics'],
            data['information_metadata'])
    config = PredictorConfig(model='neural_ode', bridge='latent', training_mode='joint',
                             latent_layout=layout, hidden_dim=8)
    return ForecastPipeline(manifold, config, schema=data['schema'],
                            separate_reconstruction_decoder=separate), batch


@pytest.mark.parametrize('layout', ['global', 'spatial'])
def test_observed_head_clones_only_decoder_and_decodes_raw_coordinates(pinn_prepared, layout):
    pipe, _ = _pipeline(pinn_prepared, layout)
    manifold, head = pipe.bridge.manifold, pipe.reconstruction_decoder
    ae = manifold.core.manifold
    raw = torch.randn(2, 3, manifold.config.manifold_dim)
    # An accidental core.decode copy would rescale these raw coordinates twice.
    manifold.core.latent_mean.fill_(2.)
    manifold.core.latent_scale.fill_(3.)
    torch.testing.assert_close(head(raw), ae.decode(raw), rtol=0, atol=0)
    assert not torch.equal(head(raw), manifold.core.decode(raw))
    assert not hasattr(head, 'encoder') and not hasattr(head, 'info_head')
    assert not hasattr(head, 'latent_drift') and not hasattr(head, 'pinn')
    forecast_parameters = dict(ae.decoder.named_parameters())
    assert len(list(head.parameters())) == len(forecast_parameters)
    for name, parameter in head.decoder.named_parameters():
        assert parameter.requires_grad
        assert parameter is not forecast_parameters[name]
        assert parameter.data_ptr() != forecast_parameters[name].data_ptr()
        torch.testing.assert_close(parameter, forecast_parameters[name], rtol=0, atol=0)
    if layout == 'global':
        assert head.spatial_dct is not ae.spatial_dct
        assert not list(head.spatial_dct.parameters())
        for original, copied in zip(ae.spatial_dct.buffers(), head.spatial_dct.buffers()):
            assert original.data_ptr() != copied.data_ptr()
            torch.testing.assert_close(original, copied, rtol=0, atol=0)
        # A global head returning coefficients instead of fields is incorrect.
        assert not torch.allclose(head(raw), head.decoder(raw))
    else:
        assert head.spatial_dct is None


@pytest.mark.parametrize('layout', ['global', 'spatial'])
def test_enabling_observed_head_preserves_forecast_initialization_and_rng(pinn_prepared, layout):
    torch.manual_seed(43)
    plain, batch = _pipeline(pinn_prepared, layout, separate=False)
    original_rng = torch.get_rng_state()
    torch.manual_seed(43)
    separate, _ = _pipeline(pinn_prepared, layout)
    assert torch.equal(original_rng, torch.get_rng_state())
    assert plain.reconstruction_decoder is None
    assert not any(k.startswith('reconstruction_decoder.') for k in plain.state_dict())
    for name, value in plain.state_dict().items():
        torch.testing.assert_close(value, separate.state_dict()[name], rtol=0, atol=0)
    inputs = (batch['history'], batch['information'], batch['origin_time_ns'], torch.tensor([6., 12.]))
    torch.testing.assert_close(plain(*inputs)['mean'], separate(*inputs)['mean'], rtol=0, atol=0)


@pytest.mark.parametrize('layout', ['global', 'spatial'])
def test_forecast_never_calls_or_updates_observed_head(pinn_prepared, monkeypatch, layout):
    pipe, batch = _pipeline(pinn_prepared, layout)

    def forbidden(*args, **kwargs):
        raise AssertionError('Observed reconstruction head must not enter forecasting')

    monkeypatch.setattr(pipe.reconstruction_decoder, 'forward', forbidden)
    inputs = (batch['history'], batch['information'], batch['origin_time_ns'], torch.tensor([6., 12.]))
    before = deepcopy(pipe.reconstruction_decoder.state_dict())
    optimizer = torch.optim.SGD(pipe.parameters(), lr=.01)
    prediction = pipe(*inputs, reconstruct_origin=False)
    (prediction['mean'] - batch['targets'][:, :2]).square().mean().backward()
    assert prediction['reconstructed_origin'] is None
    assert all(p.grad is None for p in pipe.reconstruction_decoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in pipe.bridge.manifold.core.manifold.decoder.parameters())
    optimizer.step()
    for name, value in before.items():
        torch.testing.assert_close(value, pipe.reconstruction_decoder.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize('layout', ['global', 'spatial'])
def test_observed_reconstruction_trains_its_own_head_and_encoder_only(pinn_prepared, layout):
    pipe, batch = _pipeline(pinn_prepared, layout)
    manifold = pipe.bridge.manifold
    forecast_before = deepcopy(manifold.core.manifold.decoder.state_dict())
    observed_before = deepcopy(pipe.reconstruction_decoder.state_dict())
    optimizer = torch.optim.SGD(pipe.parameters(), lr=.01)
    raw = manifold.raw_encode(batch['history'][:, -1], batch['information'])
    (pipe.reconstruction_decoder(raw) - batch['history'][:, -1]).square().mean().backward()
    for module in (pipe.reconstruction_decoder, manifold.core.manifold.encoder, manifold.information):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    for module in (manifold.core.manifold.decoder, pipe.predictor, manifold.info_head):
        assert all(p.grad is None for p in module.parameters())
    optimizer.step()
    for name, value in forecast_before.items():
        torch.testing.assert_close(value, manifold.core.manifold.decoder.state_dict()[name], rtol=0, atol=0)
    assert any(not torch.equal(value, pipe.reconstruction_decoder.state_dict()[name])
               for name, value in observed_before.items())


@pytest.mark.parametrize('changes', [
    {'bridge': 'raw'}, {'training_mode': 'frozen'},
    {'representation': 'plain_ae', 'model': 'mlp'},
])
def test_observed_head_rejects_incompatible_forecast_routes(pinn_prepared, changes):
    manifold, _, data, _ = pinn_prepared
    config = PredictorConfig(model='neural_ode', training_mode='joint', **{
        key: value for key, value in changes.items() if key not in ('model', 'training_mode')})
    config = replace(config, **changes)
    with pytest.raises(ValueError, match='Separate reconstruction decoder requires'):
        ForecastPipeline(manifold, config, schema=data['schema'], separate_reconstruction_decoder=True)
