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
