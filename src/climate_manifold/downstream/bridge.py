"""Frozen representation middleware; forecasting supplies only observed history.

Evaluation may separately encode future targets for labelled diagnostics.
"""
import torch
from torch import nn


class ManifoldBridge(nn.Module):
    MODES = ('raw', 'latent', 'decoded')

    def __init__(self, manifold, mode='latent', anchor='none'):
        super().__init__()
        if mode not in self.MODES or anchor not in ('none', 'origin'):
            raise ValueError('Expected raw/latent/decoded bridge and none/origin anchor')
        if not bool(manifold.core.manifold_ready):
            raise ValueError('Downstream experiments require a sealed, trained representation checkpoint')
        self.mode, self.anchor = mode, anchor
        self.dimension = manifold.config.manifold_dim if mode == 'latent' else manifold.config.state_dim
        # Raw baselines do not instantiate unused A parameters.
        self.manifold = manifold.requires_grad_(False).eval() if mode != 'raw' else None

    def train(self, mode=True):
        super().train(mode)
        if self.manifold is not None:
            self.manifold.eval()
        return self

    @torch.no_grad()
    def encode_history(self, history, information=None):
        if self.mode == 'raw':
            return history
        # This is A's original origin-fixed information contract, including history.
        info = None if information is None else information[:, None].expand(-1, history.shape[1], -1)
        q = self.manifold.encode(history, info)
        return q if self.mode == 'latent' else self.manifold.core.decode(q)

    def decode(self, features):
        # Frozen decoder weights still allow d(loss)/d(predicted latent).
        return self.manifold.core.decode(features) if self.mode == 'latent' else features

    def to_fields(self, prediction, origin, encoded_origin):
        fields = self.decode(prediction)
        if self.anchor == 'origin':
            fields = fields + (origin - self.decode(encoded_origin))[:, None]
        return fields
