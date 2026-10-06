"""Optional guide-only attention before raw/guide fusion.

The two inputs contain spatial-guide and dynamics-guide tokens for the complete
observed history, including their existing position, time and type embeddings.
T1 attends across these two banks only. It does not receive raw-field tokens,
future fields or lead queries, and preserves the bank lengths for T2.
"""
import torch
from torch import nn


class GuideInteraction(nn.Module):
    implementation_variant = 'observed_spatial_dynamics_guide_attention_v1'

    def __init__(self, hidden, depth=1, heads=4):
        super().__init__()
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
               for v in (hidden, depth, heads)):
            raise ValueError('Guide interaction width, depth and heads must be positive integers')
        if hidden % heads:
            raise ValueError('Guide interaction width must be divisible by attention heads')
        self.hidden, self.depth, self.num_heads = hidden, depth, heads
        layer = nn.TransformerEncoderLayer(
            hidden, heads, dim_feedforward=4 * hidden, dropout=0.,
            activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(
            layer, depth, norm=nn.LayerNorm(hidden), enable_nested_tensor=False)

    def forward(self, spatial_tokens, dynamics_tokens):
        for tokens in (spatial_tokens, dynamics_tokens):
            if (tokens.ndim != 3 or not tokens.shape[0] or not tokens.shape[1]
                    or tokens.shape[2] != self.hidden or not torch.isfinite(tokens).all()):
                raise ValueError('Guide interaction requires nonempty finite observed guide token banks')
        if spatial_tokens.shape[0] != dynamics_tokens.shape[0]:
            raise ValueError('Guide interaction token banks must have the same batch size')
        if (spatial_tokens.device != dynamics_tokens.device
                or spatial_tokens.dtype != dynamics_tokens.dtype):
            raise ValueError('Guide interaction token banks must share dtype and device')
        split = spatial_tokens.shape[1]
        # Full attention over the observed window is intentional: all times
        # are <= the forecast origin. No future labels can enter this module.
        refined = self.transformer(torch.cat((spatial_tokens, dynamics_tokens), dim=1))
        return refined[:, :split], refined[:, split:]
