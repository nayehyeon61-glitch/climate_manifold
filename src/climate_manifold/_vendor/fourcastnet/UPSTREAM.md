# FourCastNet AFNO provenance

- Repository: https://github.com/NVlabs/FourCastNet
- Pinned commit: `93360c1720a9f97aabf970689f21c9fad8737788`
- Source: `networks/afnonet.py`
- Original source Git blob: `05e35cff8d5ae91a0b2e1e509d8e9af91faf9875`
- License: BSD-3-Clause, reproduced in LICENSE; AUTHORS also retained.

`afnonet.py` retains the upstream Mlp, AFNO2D, Block, AFNONet and
PatchEmbed architecture, Fourier mode selection, double skip, trainable
position embeddings and output head. Minimal dependency adaptations:

1. Removed unused imports, unused precipitation-specific PrecipNet and the
   upstream standalone demonstration.
2. Replaced timm's truncated-normal initializer with torch.nn.init.trunc_normal_.
3. Replaced timm DropPath with an equivalent per-sample Torch implementation;
   the downstream adapter uses drop_path_rate=0.
4. Replaced the einops output rearrangement with equivalent reshape/permute.

The downstream FourCastNetPredictor configures this core for the available
variables, grid, patch size, width and depth and trains from scratch. Its extra
conditioning channels contain the fixed observed history, actual observation
offsets, target calendar, forecast lead, fixed forecast step and origin-only
auxiliary information. The predicted field is fed into the next AFNO step.
These conditioning and small-grid changes are experimental adaptations, not
the published 0.25-degree pretrained model or a reproduction of its benchmark.

The two-dimensional spectral transform keeps the upstream topology: it is
periodic along both patch-grid axes. The adapter does not reinterpret this as
a spherical or nonperiodic-latitude operator, and periodic_lon does not alter
that fact. Input grid dimensions must be divisible by the patch size.
