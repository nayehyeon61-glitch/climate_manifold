"""Independent field reconstruction head for observed-state constraints.

The forecast decoder remains part of E--F--D. This head maps the auxiliary
route's *raw* encoder coordinates to normalized physical fields, without
sharing trainable parameters with D or the information decoder.
"""
from copy import deepcopy

from torch import nn


class ObservedFieldDecoder(nn.Module):
    """Clone only the field decoder and any fixed inverse spatial transform.

    Matching initial weights keep forecast initialization independent of this
    auxiliary head. Deep copies give the two heads independent parameters and
    buffers thereafter. No encoder, latent drift, or physical closure is copied.
    """

    def __init__(self, manifold):
        super().__init__()
        ae = manifold.core.manifold
        layout = getattr(manifold.config, 'representation_kind', 'global')
        if layout not in ('global', 'spatial'):
            raise ValueError('Observed field decoder requires a global or spatial manifold')
        self.decoder = deepcopy(ae.decoder)
        self.decoder.requires_grad_(True)
        # The legacy global decoder emits DCT coefficients; its fixed inverse
        # transform is essential for physical-field reconstruction. Spatial
        # decoders already emit fields on the original latitude/longitude grid.
        self.spatial_dct = deepcopy(ae.spatial_dct) if layout == 'global' else None

    def forward(self, raw_latent):
        fields = self.decoder(raw_latent)
        if self.spatial_dct is not None:
            fields = self.spatial_dct(fields, inverse=True)
        return fields
