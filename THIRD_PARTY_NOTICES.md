# Third-party code

## ClimODE — MIT

Copyright (c) 2023 Aalto-QuML.

Source: https://github.com/Aalto-QuML/ClimODE

Pinned commit: `e729d23e8799ce0e075699e76d60227d848d8d0c`.

`src/climate_manifold/_vendor/climode/model_function.py` and `model_utils.py`
contain the upstream model definitions. The only edits inside these vendored
files are a package-relative `model_utils` import, removal of unused wildcard
`utils` imports, and trailing whitespace normalization. The full MIT license
is included in that directory and in the installed Python package.

`src/climate_manifold/downstream/climode.py` adapts these definitions to the
current four-variable archive, frozen-manifold middleware and common evaluation.
The differences from the paper's original experiment are listed in
[docs/downstream.md](docs/downstream.md). No upstream pretrained weights are bundled.

Please cite the original work when reporting ClimODE experiments:

```bibtex
@inproceedings{verma2024climode,
  title={ClimODE: Climate and Weather Forecasting with Physics-informed Neural ODEs},
  author={Yogesh Verma and Markus Heinonen and Vikas Garg},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=xuY33XhEGR}
}
```

## OpenSTL SimVP-gSTA — Apache-2.0

Source: https://github.com/chengtan9907/OpenSTL

Pinned commit: `eecf8a3078f0a178dbc7b28723da20f94ce36985`.

`src/climate_manifold/_vendor/openstl/gsta.py` is a modified, torch-only subset
of OpenSTL's spatial encoder/decoder, gSTA translator, and VAN-derived MixMlp.
The full Apache-2.0 license and detailed source/change provenance are included
in that directory as `LICENSE` and `NOTICE.md`, including in installed packages.

`downstream/simvp.py` adds explicit observation spacing, calendar and origin-only
information conditioning, and a direct lead-query output adapter. It does not
use OpenSTL's equal-length recursive block forecast. Group normalization,
geographic padding, and reduced configurable widths also differ from the
original benchmark. Report this variant as **adapted SimVP-gSTA**, identifier
`openstl_gsta_direct_lead_v1`, not the unchanged paper model. No pretrained
weights are bundled. Cite Gao et al., *SimVP* (CVPR 2022), Tan et al., *SimVP:
Towards Simple yet Powerful Spatiotemporal Predictive Learning*
(arXiv:2211.12509), and Tan et al., *OpenSTL* (NeurIPS 2023 Datasets and
Benchmarks).
