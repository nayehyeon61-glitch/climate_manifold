# External weather models and the present data contract

These models are useful scientific references, but are not interchangeable with
the freshly trained, four-variable, 16 x 32 surface-field comparisons in this
repository. Listing a source below does not mean that its model is implemented
or runs in the pairwise training script.

## Pangu-Weather

Primary source: [official repository, pinned README](https://github.com/198808xc/Pangu-Weather/blob/72bdd99096721e1a1f8912c37a9a3aff9ff0a4f2/README.md),
commit `72bdd99096721e1a1f8912c37a9a3aff9ff0a4f2`.

The official release provides pretrained ONNX models, inference scripts and
architecture pseudocode. It does not provide a directly runnable PyTorch
training entry point. The README also describes a reduced training recipe, but
that description is not a trainable model implementation in this repository.

The published ONNX interface requires both:

| Input | Exact required dimensions | Variables, in order |
| --- | --- | --- |
| Surface | `(4, 721, 1440)` | MSLP, U10, V10, T2M |
| Upper air | `(5, 13, 721, 1440)` | Z, Q, T, U, V |

The pressure levels are 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200,
150, 100 and 50 hPa. Latitude runs from 90 to -90 degrees and longitude from
0 to 359.75 degrees, at 0.25-degree spacing. Z is geopotential, not
geopotential height.

Upsampling our surface archive would not supply the missing upper-air fields.
The inference-only ONNX interface also does not provide the differentiable
predictor required for the present joint PyTorch E -> model -> D experiment.
An official pretrained Pangu evaluation therefore requires a separate data and
execution path; it must not silently use a generic transformer under this name.

The official README licenses trained parameters under CC BY-NC-SA 4.0 and
forbids commercial use. No standalone code LICENSE was present in the pinned
repository tree. This repository does not redistribute Pangu code or weights.

## NowcastNet

Primary sources: [Nature paper](https://www.nature.com/articles/s41586-023-06184-4)
and its [official Code Ocean release v1](https://doi.org/10.24433/CO.0832447.v1)
([capsule](https://codeocean.com/capsule/3935105/tree/v1)).

NowcastNet predicts precipitation from radar observations, with forecasts up to
three hours. This is a different observation and target contract from our
surface temperature, pressure and wind archive. That archive does not contain
the required radar precipitation history or targets. Reinterpreting those
surface variables as precipitation inputs would not be a NowcastNet benchmark.

The paper states that code and pretrained weights are released on Code Ocean.
The capsule's file contents, training entry points and license could not be
verified during this integration because direct retrieval returned HTTP 403.
Consequently, this document makes no assertion that official training code is
absent and does not assign an unverified software license. Third-party GitHub
rewrites are not treated as the authors' original implementation.

## Interpreting the comparisons that are runnable here

- A pinned official neural-network core adapted to our input variables, context,
  resolution and fresh training protocol is an **architecture adaptation**. It
  is not a reproduction of published pretrained-model performance.
- Compare raw -> predictor and E -> predictor -> D within each family using the
  same observed data, splits, forecast origins, leads, seeds and training budget.
  The latent route changes state channels and spatial resolution and adds
  encoder/decoder capacity. Its raw contrast is a **whole-model comparison**.
- Record parameter counts, upstream source and commit, local implementation
  variant, depth, patch size, hidden width and pretrained status. Do not pool
  runs with changed upstream versions or model hyperparameters as random seeds.
- The current constraint-pair runner does not include a forecast-only E/F/D
  control. A raw-versus-constrained-latent improvement alone does not isolate
  the contribution of PINN or statistical regularization.
- Pretrained external references require a separate comparison panel disclosing
  their training data, initialization and input information. They are not
  same-data, fresh-initialization ablations of our manifold.
