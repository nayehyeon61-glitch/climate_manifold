"""Reconstruction-control gates: fresh weights, causal data, frozen decoding."""
from dataclasses import asdict
import json

import numpy as np
import pytest
import torch

from test_pinn_training import pinn_prepared
from climate_manifold.downstream.bridge import ManifoldBridge
from climate_manifold.downstream.plain_ae import (
    FORMAT, PlainAutoencoder, ReconstructionWindows, load_plain_ae, new_plain_ae,
    parser, reconstruction_loss, train,
)
from climate_manifold.physical_information import digest
from climate_manifold.temporal_supervision import area_weights
from climate_manifold.train import write_json


def metadata(a, data):
    return {'config': asdict(a.config), 'information_metadata': data['information_metadata']}


def test_plain_ae_matches_representation_capacity_but_has_no_trained_a_weights(pinn_prepared):
    a, batch, data, _ = pinn_prepared
    torch.manual_seed(123)
    model = new_plain_ae(metadata(a, data))
    for name in ('encoder', 'decoder'):
        original = dict(getattr(a.core.manifold, name).named_parameters())
        control = dict(getattr(model.core.manifold, name).named_parameters())
        assert {key: value.shape for key, value in original.items()} == {
            key: value.shape for key, value in control.items()}
        assert any(not torch.equal(original[key], control[key]) for key in original)
    assert model.information[0].in_features == a.information[0].in_features
    assert model.config.manifold_dim == a.config.manifold_dim
    assert not hasattr(model.core.manifold, 'latent_drift')
    assert not hasattr(model, 'pinn') and not hasattr(model, 'info_head')
    assert not hasattr(model, 'a_sampler')
    history = batch['history']
    info = batch['information'][:, None].expand(-1, history.shape[1], -1)
    assert model.reconstruct(history, info).shape == history.shape
    with pytest.raises(ValueError, match='origin information'):
        model.encode(history, info[..., :-1])
    with pytest.raises(ValueError, match='origin information'):
        model.encode(history)
    surface = PlainAutoencoder(a.config)
    assert surface.encode(history).shape[-1] == a.config.manifold_dim
    with pytest.raises(ValueError, match='Surface-only'):
        surface.encode(history, info)


def test_reconstruction_only_training_and_seal_preserve_decode_gradient(pinn_prepared):
    a, batch, data, _ = pinn_prepared
    torch.manual_seed(2)
    model = new_plain_ae(metadata(a, data))
    history = batch['history']
    info = batch['information'][:, None].expand(-1, history.shape[1], -1)
    metric = torch.tensor(np.tile(area_weights(data['schema']).reshape(-1), a.config.grid[0]) / a.config.grid[0],
                          dtype=torch.float32)
    initial = float(reconstruction_loss(model, history, info, metric).detach())
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    for _ in range(25):
        optimizer.zero_grad()
        reconstruction_loss(model, history, info, metric).backward()
        optimizer.step()
    assert float(reconstruction_loss(model, history, info, metric).detach()) < .8*initial
    before = model.reconstruct(history, info).detach()
    raw = model.raw_encode(history, info).detach().reshape(-1, model.config.manifold_dim)
    assert model.seal_batches([(history[:1], info[:1]), (history[1:], info[1:])]) == history.shape[0]*history.shape[1]
    torch.testing.assert_close(model.core.latent_mean, raw.mean(0))
    torch.testing.assert_close(model.core.latent_scale, raw.std(0, unbiased=False).clamp_min(.05))
    torch.testing.assert_close(model.core.decode(model.encode(history, info)), before)
    with pytest.raises(ValueError, match='once'):
        model.seal(history, info)
    model.zero_grad(set_to_none=True)
    frozen = {key: value.clone() for key, value in model.state_dict().items()}
    bridge = ManifoldBridge(model, mode='latent')
    bridge.train()
    latent = bridge.encode_history(history, batch['information']).detach().requires_grad_()
    bridge.decode(latent).square().mean().backward()
    assert latent.grad is not None and latent.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not model.training
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in frozen.items())


def test_reconstruction_loader_uses_history_and_origin_information_only(pinn_prepared):
    a, _, data, _ = pinn_prepared
    start = data['split']['train'][0]
    origin = start + a.config.history_span_steps-1
    class HistoricalStates:
        def __getitem__(self, index):
            assert index == slice(start, origin+1, a.config.history_stride)
            return data['states'][index]
    class OriginInformation:
        def __getitem__(self, index):
            assert index == origin
            return data['information'][index]
    causal = {**data, 'states': HistoricalStates(), 'information': OriginInformation()}
    sample = ReconstructionWindows(causal, a.config, 'train', max_windows=1)[0]
    assert set(sample) == {'history', 'information'}
    torch.testing.assert_close(sample['information'], torch.tensor(data['information'][origin]))


def test_plain_ae_cli_checkpoint_is_sealed_standalone_and_hash_checked(pinn_prepared, tmp_path):
    a, batch, data, archive = pinn_prepared
    info = tmp_path/'pinn-information.npz'
    a.seal(batch['origin'], batch['information'])
    parent = {
        'format': 'climate_manifold.a.v1', 'config': asdict(a.config),
        'schema': data['schema'], 'mean': data['mean'], 'scale': data['scale'],
        'statistics': data['statistics'], 'information_metadata': data['information_metadata'],
        'information_mean': data['information_mean'], 'information_scale': data['information_scale'],
        'information_tendency_scale': data['information_tendency_scale'],
        'pinn_config': asdict(a.pinn.config), 'model': a.state_dict(), 'stage': 'A',
        'mode': 'enriched', 'archive_sha256': digest(archive), 'information_sha256': digest(info),
        'split': data['split'],
    }
    a_path, output = tmp_path/'a.pt', tmp_path/'plain.pt'
    torch.save(parent, a_path)
    write_json(a_path.with_suffix('.manifest.json'), {'checkpoint_sha256': digest(a_path)})
    a_hash = digest(a_path)
    args = parser().parse_args([
        '--a-checkpoint', str(a_path), '--archive', str(archive), '--information', str(info),
        '--output', str(output), '--epochs', '4', '--batch-size', '2', '--max-windows', '2',
        '--window-stride', '1', '--learning-rate', '.01'])
    assert train(args) == output
    restored, payload = load_plain_ae(output)
    assert bool(restored.core.manifold_ready)
    assert payload['format'] == FORMAT
    assert payload['a_sha256'] == a_hash == digest(a_path)
    assert 'model' not in payload['a_metadata']
    assert payload['selection_split'] == 'expert_validation'
    assert payload['seal_split'] == 'train'
    assert payload['seal_state_information_pairs'] == 2*a.config.history_steps
    assert payload['objective'] == 'area_weighted_normalized_state_reconstruction_mse'
    assert payload['initialization'] == 'fresh_random_no_A_weights'
    assert all(not any(token in key for token in ('pinn', 'latent_drift', 'a_sampler', 'info_head'))
               for key in payload['model'])
    fresh = new_plain_ae(payload['a_metadata'])
    fresh.load_state_dict(payload['model'], strict=True)
    torch.testing.assert_close(restored.encode(batch['origin'], batch['information']),
                               fresh.encode(batch['origin'], batch['information']), rtol=0, atol=0)
    rows = json.loads(output.with_suffix('.metrics.json').read_text())
    assert rows[-1]['train_reconstruction_mse'] < rows[0]['train_reconstruction_mse']
    with pytest.raises(FileExistsError):
        train(args)
    with output.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_plain_ae(output)
