"""Fresh reconstruction-only AE control with Climate Manifold's encoder capacity.

The data normalization, splits, latent width and information input match A.
Only reconstruction of observed historical states is trained; A weights and
its dynamics, distribution, geometry and PINN objectives are never reused.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..architecture import ManifoldConfig
from ..nn import FieldDCT, mlp
from ..physical_information import digest
from ..temporal_supervision import area_weights
from ..train import data_contract, load_checkpoint, source_commit, write_json

FORMAT = 'climate_manifold.plain_ae.v1'


class _ReconstructionAE(nn.Module):
    """Exactly A's state encoder/decoder architecture, without latent drift."""
    def __init__(self, config):
        super().__init__()
        self.spatial_dct = FieldDCT(config.grid)
        self.encoder = mlp(config.state_dim, config.hidden_dim, config.manifold_dim)
        self.decoder = nn.Sequential(
            nn.Linear(config.manifold_dim, config.hidden_dim), nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim), nn.SiLU(),
            nn.Linear(config.hidden_dim, config.state_dim))

    def encode(self, state):
        return self.encoder(self.spatial_dct(state))

    def decode(self, latent):
        return self.spatial_dct(self.decoder(latent), inverse=True)


class _PlainCore(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.manifold = _ReconstructionAE(config)
        self.register_buffer('latent_mean', torch.zeros(config.manifold_dim))
        self.register_buffer('latent_scale', torch.ones(config.manifold_dim))
        self.register_buffer('manifold_ready', torch.tensor(False))

    def decode(self, q):
        return self.manifold.decode(q * self.latent_scale + self.latent_mean)


class PlainAutoencoder(nn.Module):
    """Bridge-compatible AE; fresh initialization and reconstruction-only loss."""
    def __init__(self, config, info_metadata=None):
        super().__init__()
        self.core = _PlainCore(config)
        self.info_metadata = info_metadata
        self.information_dim = math.prod(info_metadata['shape']) if info_metadata else 0
        self.information = (mlp(self.information_dim, config.hidden_dim, config.manifold_dim)
                            if self.information_dim else None)

    @property
    def config(self):
        return self.core.config

    def raw_encode(self, state, information=None):
        if state.shape[-1] != self.config.state_dim:
            raise ValueError('State width does not match the A data contract')
        encoded = self.core.manifold.encode(state)
        if self.information is None:
            if information is not None:
                raise ValueError('Surface-only plain AE does not accept enriched information')
        else:
            if (information is None or information.shape[:-1] != state.shape[:-1]
                    or information.shape[-1] != self.information_dim):
                raise ValueError('Enriched plain AE requires matching origin information')
            encoded = encoded + self.information(information)
        return encoded

    def encode(self, state, information=None):
        return (self.raw_encode(state, information) - self.core.latent_mean) / self.core.latent_scale

    def reconstruct(self, state, information=None):
        return self.core.manifold.decode(self.raw_encode(state, information))

    @torch.no_grad()
    def seal_batches(self, batches):
        """Streaming float64 moments of training-only (state, information) pairs."""
        if bool(self.core.manifold_ready):
            raise ValueError('Seal only once, after selecting the best plain AE')
        total = 0
        mean = second = None
        device = next(self.parameters()).device
        for states, information in batches:
            states = states.to(device)
            information = None if information is None else information.to(device)
            raw = self.raw_encode(states, information).reshape(-1, self.config.manifold_dim).double()
            if not raw.numel() or not torch.isfinite(raw).all():
                raise ValueError('Latent sealing requires nonempty finite training states')
            size = len(raw)
            batch_mean = raw.mean(0)
            batch_second = ((raw - batch_mean) ** 2).sum(0)
            if total == 0:
                mean, second = batch_mean, batch_second
            else:
                delta = batch_mean - mean
                second = second + batch_second + delta.square() * total * size / (total + size)
                mean = mean + delta * size / (total + size)
            total += size
        if total == 0:
            raise ValueError('Cannot seal plain AE without training states')
        self.core.latent_mean.copy_(mean)
        self.core.latent_scale.copy_((second / total).sqrt().clamp_min(.05))
        self.core.manifold_ready.fill_(True)
        return total

    @torch.no_grad()
    def seal(self, states, information=None):
        return self.seal_batches((states[i:i+256], None if information is None else information[i:i+256])
                                for i in range(0, len(states), 256))


def new_plain_ae(a_metadata):
    """Construct fresh weights from architecture metadata, never load A weights."""
    return PlainAutoencoder(ManifoldConfig(**a_metadata['config']), a_metadata.get('information_metadata'))


def load_plain_ae(path, device='cpu'):
    manifest = Path(path).with_suffix('.manifest.json')
    if not manifest.exists() or json.loads(manifest.read_text()).get('checkpoint_sha256') != digest(path):
        raise ValueError('Plain AE checkpoint manifest/hash mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('format') != FORMAT or payload.get('selection_split') != 'expert_validation':
        raise ValueError('Not a standalone reconstruction-only plain AE checkpoint')
    if payload['config'] != payload['a_metadata']['config']:
        raise ValueError('Plain AE architecture does not match its A data contract')
    model = new_plain_ae(payload['a_metadata'])
    model.load_state_dict(payload['model'], strict=True)
    if not bool(model.core.manifold_ready):
        raise ValueError('Plain AE checkpoint must have training-only sealed coordinates')
    return model.to(device).eval(), payload


class ReconstructionWindows(Dataset):
    """Observed history only, conditioned on the same origin information as inference.

    Neither future states nor future information are accessed. Starts come from
    the original A split; calibration, validation and test are not used here.
    """
    def __init__(self, data, config, split, stride=1, max_windows=0):
        self.data, self.config = data, config
        self.starts = list(data['split'][split][::stride])
        if max_windows:
            self.starts = self.starts[:max_windows]
        if not self.starts:
            raise ValueError('Plain AE requires nonempty training/selection windows')

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        data, config = self.data, self.config
        start = self.starts[index]
        origin = start + config.history_span_steps - 1
        history = data['states'][start:origin+1:config.history_stride]
        row = {'history': torch.as_tensor((history-data['mean'])/data['scale'], dtype=torch.float32)}
        if data['information'] is not None:
            row['information'] = torch.as_tensor(data['information'][origin].copy(), dtype=torch.float32)
        return row


def _pairs(batch):
    history = batch['history']
    info = batch.get('information')
    if info is not None:
        info = info[:, None].expand(-1, history.shape[1], -1)
    return history, info


def reconstruction_loss(model, history, information, metric):
    """Area-weighted normalized-state MSE, with equal total weight per variable."""
    reconstructed = model.reconstruct(history, information)
    loss = ((reconstructed-history).square() * metric).sum(-1).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite plain AE reconstruction loss')
    return loss


def train(args):
    output = Path(args.output)
    if output.suffix != '.pt':
        raise ValueError('Plain AE output must end in .pt')
    if any(output.with_suffix(s).exists() for s in ('.pt', '.manifest.json', '.metrics.json', '.metadata.json')):
        raise FileExistsError('Choose a new plain AE checkpoint path')
    if min(args.epochs, args.batch_size, args.window_stride) < 1 or args.max_windows < 0:
        raise ValueError('Invalid plain AE training counts')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('Learning rate must be finite and positive')
    a, parent = load_checkpoint(args.a_checkpoint)
    if not bool(a.core.manifold_ready):
        raise ValueError('The reference A checkpoint must be sealed')
    metadata = {key: value for key, value in parent.items() if key != 'model'}
    data = data_contract(args.archive, args.information, parent['mode'], a.config, parent)
    del a, parent
    # Reset after loading A so its constructor cannot affect the control seed.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = new_plain_ae(metadata).to(args.device)
    loaders = [DataLoader(ReconstructionWindows(data, model.config, split, args.window_stride, args.max_windows),
                          batch_size=args.batch_size, shuffle=(split == 'train'),
                          generator=torch.Generator().manual_seed(args.seed))
               for split in ('train', 'expert_validation')]
    weights = np.tile(area_weights(data['schema']).reshape(-1), model.config.grid[0]) / model.config.grid[0]
    metric = torch.tensor(weights, dtype=torch.float32, device=args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    started = time.perf_counter()
    best, best_epoch, best_state, rows = float('inf'), 0, None, []
    for epoch in range(1, args.epochs+1):
        row = {'epoch': epoch}
        for training, loader in zip((True, False), loaders):
            model.train(training)
            total, count = 0., 0
            with torch.set_grad_enabled(training):
                for batch in loader:
                    batch = {key: value.to(args.device) for key, value in batch.items()}
                    history, information = _pairs(batch)
                    loss = reconstruction_loss(model, history, information, metric)
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                        optimizer.step()
                    total += float(loss.detach()) * len(history)
                    count += len(history)
            row['train_reconstruction_mse' if training else 'selection_reconstruction_mse'] = total/count
        score = row['selection_reconstruction_mse']
        if score < best:
            best, best_epoch = score, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        rows.append(row)
        print(json.dumps({'representation': 'plain_ae', **row}), flush=True)
    model.load_state_dict(best_state, strict=True)
    model.eval()
    seal_loader = DataLoader(loaders[0].dataset, batch_size=args.batch_size, shuffle=False)
    seal_count = model.seal_batches(_pairs(batch) for batch in seal_loader)
    payload = {
        'format': FORMAT, 'a_metadata': metadata, 'a_sha256': digest(args.a_checkpoint),
        'config': asdict(model.config),
        'model': {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        'options': vars(args), 'best_epoch': best_epoch, 'best_selection_reconstruction_mse': best,
        'selection_split': 'expert_validation', 'training_seconds': time.perf_counter()-started,
        'source_commit': source_commit(), 'objective': 'area_weighted_normalized_state_reconstruction_mse',
        'initialization': 'fresh_random_no_A_weights', 'origin_information_policy': 'fixed_across_observed_history',
        'seal_split': 'train', 'seal_state_information_pairs': seal_count,
        'training_contract': {
            **{key: getattr(args, key) for key in ('epochs', 'batch_size', 'learning_rate', 'window_stride', 'max_windows', 'seed')},
            'train_starts_sha256': hashlib.sha256(json.dumps(loaders[0].dataset.starts).encode()).hexdigest(),
            'selection_starts_sha256': hashlib.sha256(json.dumps(loaders[1].dataset.starts).encode()).hexdigest(),
        },
        'trainable_parameters': sum(parameter.numel() for parameter in model.parameters()),
        'resume': 'optimizer/RNG resume not implemented',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    write_json(output.with_suffix('.manifest.json'), {'format': FORMAT, 'checkpoint_sha256': digest(output)})
    write_json(output.with_suffix('.metrics.json'), rows)
    write_json(output.with_suffix('.metadata.json'),
               {key: value for key, value in payload.items() if key not in ('model', 'a_metadata')})
    return output


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for key in ('a-checkpoint', 'archive', 'output'):
        result.add_argument('--'+key, required=True)
    result.add_argument('--information')
    for key, value in dict(epochs=20, batch_size=2, seed=7, window_stride=4, max_windows=0).items():
        result.add_argument('--'+key.replace('_', '-'), type=int, default=value)
    result.add_argument('--learning-rate', type=float, default=1e-3)
    result.add_argument('--device', default='cpu')
    return result


def main(argv=None):
    print(train(parser().parse_args(argv)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
