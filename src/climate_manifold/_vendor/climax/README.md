# Official ClimaX core

Source: <https://github.com/microsoft/ClimaX>

Pinned commit: `6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c` (2023-09-30).

`arch.py` and `parallelpatchembed.py` are copied from upstream
`src/climax/arch.py` and `src/climax/parallelpatchembed.py`. The Microsoft MIT
license and source copyright headers are preserved. The original variable tokenizers, learned variable
query, cross-attention aggregation, positional and lead-time embeddings,
Transformer stack, and patch prediction head are used.

`pos_embed.py` is a local independent implementation of the elementary
sine/cosine initialization formula, not a copy of the upstream MAE-derived
helper. It preserves frequency values, sine-before-cosine ordering,
column-before-row grid features, row-major patch ordering and the optional
zero class-token prefix. No copied positional interpolation code is included.

Local changes to the copied core:

- Relative imports use this namespace and the bundled timm primitive subset.
- `torch.nn.init.trunc_normal_` replaces timm's equivalent initializer.
- Positional initialization uses the local formula helper described above.
- `aggregate_variables` uses `squeeze(1)`, fixing the upstream singleton
  batch/single-patch case while preserving its non-singleton computation.

`timm_layers.py` contains only the required unmodified class/function bodies
from timm 0.6.12, with imports combined and `torch._assert` used directly.
This is the timm version pinned by ClimaX `docker/environment.yml`.
Source: <https://github.com/huggingface/pytorch-image-models/tree/ce4d3485b690837ba4e1cb4e0e6c4ed415e36cea>
(Apache-2.0, `LICENSE.timm`, Copyright Ross Wightman).

The project wrapper in `downstream/climax.py` configures this official backbone
for the archive's channel count/grid or a learned spatial latent grid, defaults
to width 128 and depth 4, and trains from scratch. It consumes the **last**
observed field, following upstream ClimaX, with each target lead divided by
100 as in upstream `pretrain/dataset.py`. History length/spacing and timestamp
arguments do not add temporal-context or calendar features. Origin-only
auxiliary channels are tokenized as additional variables and only prognostic
channels are returned. Latent channels are learned coordinates, not the
physical variables used in the original pretraining. No pretrained weights,
original benchmark scores, or pretrained-checkpoint compatibility are claimed.

Defaults retain upstream dropout/stochastic-depth rates of 0.1 and decoder
depth 2. Patch size defaults to 2 and requires divisible grid dimensions. The
periodic-longitude flag is domain metadata; no circular convolution/padding is
added to the original absolute-position Transformer.
