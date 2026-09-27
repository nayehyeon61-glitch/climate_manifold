# OpenSTL gSTA subset

Upstream: https://github.com/chengtan9907/OpenSTL

Commit: `eecf8a3078f0a178dbc7b28723da20f94ce36985`

License: Apache-2.0, reproduced in `LICENSE`.

`gsta.py` adapts the following upstream files by the OpenSTL authors:

- `openstl/models/simvp_model.py`: Encoder, Decoder, MetaBlock, MidMetaNet.
- `openstl/modules/simvp_modules.py`: BasicConv2d, ConvSC, AttentionModule,
  SpatialAttention, GASubBlock.
- `openstl/modules/layers/van.py`: MixMlp. Upstream attributes this component
  to https://github.com/Visual-Attention-Network/VAN-Classification.

Changes in this repository:

- Keep only the gSTA architecture, removing OpenSTL/timm/training-framework
  dependencies; use `torch.nn.init.trunc_normal_` and no stochastic depth.
- Fix the spatial encoder/decoder depth at two stages and use four translator
  blocks. The model width remains configurable; this is not a reproduction of
  the original weather benchmark's eight-block, 256-channel configuration.
- Replace translator BatchNorm with GroupNorm(1) to work with singleton batches
  and tiny latent maps. Keep spatial encoder GroupNorm(2).
- Use replicated latitude and regional longitude padding, optionally periodic
  longitude. Modular indexing permits large kernels on tiny longitude grids.
- The adapter in `downstream/simvp.py` adds observed-time/calendar and raw
  origin-information conditioning; it pads odd spatial dimensions and crops
  decoded outputs.
- Replace OpenSTL's equal-length output/recursive block rollout with learned
  lead-query mixing of translated history and skip features plus feature-wise
  conditioning. It directly predicts each requested physical lead, including
  24-hourly history to 6-hourly output, without future ground-truth inputs.

This implementation is named `openstl_gsta_direct_lead_v1`. It is an adapted
SimVP-gSTA predictor, not an unchanged official checkpoint or benchmark result.

References: Gao et al., *SimVP: Simpler yet Better Video Prediction* (CVPR 2022);
Tan et al., *SimVP: Towards Simple yet Powerful Spatiotemporal Predictive
Learning* (arXiv:2211.12509); Tan et al., *OpenSTL* (NeurIPS 2023 Datasets and
Benchmarks).
