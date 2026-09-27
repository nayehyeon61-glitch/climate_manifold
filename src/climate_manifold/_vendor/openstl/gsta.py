"""Modified OpenSTL gSTA components (Apache-2.0).

Original source: chengtan9907/OpenSTL at eecf8a3078f0a178dbc7b28723da20f94ce36985.
See NOTICE.md for source files, credits, and this adaptation's changes.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class PaddedConv2d(nn.Conv2d):
    """Convolution with replicated latitude and optional cyclic longitude."""

    def __init__(self, *args, padding=0, periodic_lon=False, **kwargs):
        super().__init__(*args, padding=0, **kwargs)
        self.edge_padding = padding
        self.periodic_lon = bool(periodic_lon)

    def forward(self, x):
        padding = self.edge_padding
        if padding:
            if self.periodic_lon:
                # Unlike F.pad(circular), this supports padding > grid width.
                index = torch.arange(-padding, x.shape[-1] + padding, device=x.device)
                x = x.index_select(-1, index.remainder(x.shape[-1]))
            else:
                x = F.pad(x, (padding, padding, 0, 0), mode='replicate')
            x = F.pad(x, (0, 0, padding, padding), mode='replicate')
        return super().forward(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, *, stride=1, upsampling=False,
                 periodic_lon=False):
        super().__init__()
        conv = PaddedConv2d(in_channels, out_channels * (4 if upsampling else 1),
                            kernel_size=3, stride=1 if upsampling else stride,
                            padding=1, periodic_lon=periodic_lon)
        self.conv = nn.Sequential(conv, nn.PixelShuffle(2)) if upsampling else conv
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU()
        nn.init.trunc_normal_(conv.weight, std=.02)
        nn.init.zeros_(conv.bias)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Encoder(nn.Module):
    """OpenSTL spatial encoder specialized to N_S=2."""

    def __init__(self, input_channels, hidden, periodic_lon=False):
        super().__init__()
        self.enc = nn.ModuleList([
            BasicConv2d(input_channels, hidden, periodic_lon=periodic_lon),
            BasicConv2d(hidden, hidden, stride=2, periodic_lon=periodic_lon),
        ])

    def forward(self, x):
        skip = self.enc[0](x)
        return self.enc[1](skip), skip


class Decoder(nn.Module):
    """OpenSTL N_S=2 PixelShuffle decoder and spatial skip connection."""

    def __init__(self, hidden, output_channels, periodic_lon=False):
        super().__init__()
        self.dec = nn.ModuleList([
            BasicConv2d(hidden, hidden, upsampling=True, periodic_lon=periodic_lon),
            BasicConv2d(hidden, hidden, periodic_lon=periodic_lon),
        ])
        self.readout = nn.Conv2d(hidden, output_channels, 1)

    def forward(self, hidden, skip):
        return self.readout(self.dec[1](self.dec[0](hidden) + skip))


class MixMlp(nn.Module):
    """Convolutional FFN from the VAN component included in OpenSTL."""

    def __init__(self, channels, hidden, periodic_lon=False):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.dwconv = PaddedConv2d(hidden, hidden, 3, padding=1,
                                   groups=hidden, periodic_lon=periodic_lon)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        return self.fc2(self.act(self.dwconv(self.fc1(x))))


class AttentionModule(nn.Module):
    """OpenSTL large-kernel gSTA attention with gated feature channels."""

    def __init__(self, channels, kernel_size=21, dilation=3, periodic_lon=False):
        super().__init__()
        first_kernel = 2 * dilation - 1
        second_kernel = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        self.conv0 = PaddedConv2d(channels, channels, first_kernel,
                                  padding=(first_kernel - 1) // 2,
                                  groups=channels, periodic_lon=periodic_lon)
        self.conv_spatial = PaddedConv2d(channels, channels, second_kernel,
                                         padding=dilation * (second_kernel - 1) // 2,
                                         groups=channels, dilation=dilation,
                                         periodic_lon=periodic_lon)
        self.conv1 = nn.Conv2d(channels, 2 * channels, 1)

    def forward(self, x):
        features, gate = self.conv1(self.conv_spatial(self.conv0(x))).chunk(2, dim=1)
        return gate.sigmoid() * features


class SpatialAttention(nn.Module):
    def __init__(self, channels, periodic_lon=False):
        super().__init__()
        self.proj_1 = nn.Conv2d(channels, channels, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(channels, periodic_lon=periodic_lon)
        self.proj_2 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        return x + self.proj_2(self.spatial_gating_unit(self.activation(self.proj_1(x))))


class GASubBlock(nn.Module):
    """OpenSTL gSTA residual block; GroupNorm replaces BatchNorm here."""

    def __init__(self, channels, mlp_ratio=4, periodic_lon=False):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.attn = SpatialAttention(channels, periodic_lon=periodic_lon)
        self.norm2 = nn.GroupNorm(1, channels)
        self.mlp = MixMlp(channels, int(channels * mlp_ratio), periodic_lon)
        self.layer_scale_1 = nn.Parameter(.01 * torch.ones(channels))
        self.layer_scale_2 = nn.Parameter(.01 * torch.ones(channels))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Conv2d):
            fan_out = math.prod(module.kernel_size) * module.out_channels // module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2. / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = x + self.layer_scale_1[None, :, None, None] * self.attn(self.norm1(x))
        return x + self.layer_scale_2[None, :, None, None] * self.mlp(self.norm2(x))


class MetaBlock(nn.Module):
    def __init__(self, input_channels, output_channels, periodic_lon=False):
        super().__init__()
        self.block = GASubBlock(input_channels, periodic_lon=periodic_lon)
        self.reduction = (nn.Conv2d(input_channels, output_channels, 1)
                          if input_channels != output_channels else nn.Identity())

    def forward(self, x):
        return self.reduction(self.block(x))


class MidMetaNet(nn.Module):
    """Mix history along channels, retaining its spatial grid and time count."""

    def __init__(self, channel_in, hidden, blocks=4, periodic_lon=False):
        super().__init__()
        if blocks < 2:
            raise ValueError('The gSTA translator requires at least two blocks')
        dimensions = [channel_in] + [hidden] * (blocks - 1) + [channel_in]
        self.enc = nn.Sequential(*[
            MetaBlock(a, b, periodic_lon) for a, b in zip(dimensions[:-1], dimensions[1:])
        ])

    def forward(self, x):
        batch, time, channels, height, width = x.shape
        return self.enc(x.reshape(batch, time * channels, height, width)).reshape_as(x)
