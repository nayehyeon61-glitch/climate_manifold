"""Joint or frozen representation middleware using only observed history.

Evaluation may separately encode future targets for labelled diagnostics.
"""
import torch
from torch import nn
from contextlib import nullcontext


class ManifoldBridge(nn.Module):
    MODES = ('raw', 'latent', 'decoded')

    def __init__(self, manifold, mode='latent', anchor='none', training_mode='frozen'):
        super().__init__()
        if mode not in self.MODES or anchor not in ('none', 'origin'):
            raise ValueError('Expected raw/latent/decoded bridge and none/origin anchor')
        if training_mode not in ('joint', 'frozen'):
            raise ValueError('Training mode must be joint or frozen')
        if training_mode == 'joint' and anchor != 'none':
            raise ValueError('Joint training requires anchor=none')
        if training_mode == 'frozen' and not bool(manifold.core.manifold_ready):
            raise ValueError('Downstream experiments require a sealed, trained representation checkpoint')
        self.mode, self.anchor, self.training_mode = mode, anchor, training_mode
        self.dimension = manifold.config.manifold_dim if mode == 'latent' else manifold.config.state_dim
        # Raw baselines do not instantiate unused A parameters.
        self.manifold = manifold if mode != 'raw' else None
        if self.manifold is not None:
            self.manifold.requires_grad_(False)
            if training_mode == 'joint':
                # The downstream predictor supplies the dynamics. A's sampler,
                # context and intrinsic drift are not part of this forecast.
                ae = manifold.core.manifold
                modules = [ae.encoder, ae.decoder]
                modules += [getattr(manifold, name, None)
                            for name in ('information', 'info_head', 'pinn')]
                for module in modules:
                    if module is not None:
                        module.requires_grad_(True)
            else:
                self.manifold.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.manifold is not None and self.training_mode == 'frozen':
            self.manifold.eval()
        return self

    def encode_history(self, history, information=None):
        if self.mode == 'raw':
            return history
        # This is A's original origin-fixed information contract, including history.
        info = None if information is None else information[:, None].expand(-1, history.shape[1], -1)
        # Never re-enable gradients inside evaluation's no_grad context. Frozen
        # history encoding remains detached for backward-compatible checkpoints.
        context = torch.no_grad() if self.training_mode == 'frozen' else nullcontext()
        with context:
            q = self.manifold.encode(history, info)
            return q if self.mode == 'latent' else self.manifold.core.decode(q)

    def decode(self, features):
        # In frozen mode weights stay fixed but gradients still reach the predictor.
        return self.manifold.core.decode(features) if self.mode == 'latent' else features

    def to_fields(self, prediction, origin, encoded_origin):
        fields = self.decode(prediction)
        if self.anchor == 'origin':
            fields = fields + (origin - self.decode(encoded_origin))[:, None]
        return fields
