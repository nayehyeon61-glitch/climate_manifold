# Official GraphCast external raw baseline

This runner **executes Google's actual JAX/Haiku `GraphCast` class** and its
`InputsAndResiduals` normalization wrapper. It does not replace GraphCast with a
local PyTorch graph network.

The original `google-deepmind/graphcast` repository now redirects to
[google-deepmind/weathernext](https://github.com/google-deepmind/weathernext).
The implementation is pinned to commit
[`f2f2c5117d2f864e2d5e7c2f9f220db5e1049dd3`](https://github.com/google-deepmind/weathernext/tree/f2f2c5117d2f864e2d5e7c2f9f220db5e1049dd3),
using
[`weathernext/weathernext1_graph/graphcast.py`](https://github.com/google-deepmind/weathernext/blob/f2f2c5117d2f864e2d5e7c2f9f220db5e1049dd3/weathernext/weathernext1_graph/graphcast.py)
and the upstream Apache-2.0 license. The runner refuses a different revision or
modified tracked upstream code. No upstream model code is copied into this repo.

## What is and is not compared

| Item | This baseline |
|---|---|
| Path | Raw surface fields → official GraphCast → forecast fields |
| Trainable model | Official grid-to-mesh, multi-mesh processor, mesh-to-grid architecture |
| Variables | `msl`, `t2m`, `u10`, `v10` with official variable names |
| Grid | The original archive's global latitude/longitude grid; no interpolation |
| Observed input | Exactly `t−6h`, `t` from the dense archive |
| Known-future input | Deterministic sine/cosine year and local-day clock features |
| Extra atmospheric information | No information sidecar, pressure levels, precipitation, terrain, or land/sea mask |
| Default size | Mesh refinements 2, latent width 128, 4 processor message-passing steps |
| Weights | Trained from scratch; no pretrained GraphCast weights |
| Objective | Area-weighted normalized field MSE, differentiated through the autoregressive rollout |
| Model selection | Existing `calibration` split only |
| Final metrics | The same `ForecastMetrics`, including physical-unit per-variable/lead ClimODE-style RMSE/ACC |
| Forecast horizon | 6–120 h by default; every future input field comes from the previous prediction |
| PyTorch E → GraphCast → D | Not implemented; GraphCast is a separate raw JAX baseline |

This is a **surface-only, reduced-size, custom-data GraphCast experiment**, not a
reproduction of the published operational model or its reported skill. Its
observed history and extra-information availability differ from the joint
manifold arms (`6 × 24h` history and an information sidecar). Cross-family scores
are useful descriptive comparisons, but do not isolate a manifold effect.
Reports deliberately use `external_baseline_evaluation.v1` and
`ranking_allowed: false`; they are not silently admitted to the paired
Raw/E→Model→D comparator.

## Install and train

The JAX environment is separate from the existing PyTorch environment. The
optional GraphCast runtime requires **Python 3.11 or newer** (verified with 3.12).
The installer pins a verified CPU runtime; training larger configurations on a GPU
requires installing the matching JAX GPU runtime for your machine in this same
environment. The project Torch dependency serves archive/metrics code only.

```bash
bash scripts/install_graphcast_official.sh

ARCHIVE=/absolute/path/states.npz \
RUN=runs/graphcast_official_surface_bs16 \
GRAPHCAST_PYTHON=.venv-graphcast/bin/python \
SEEDS="7 19 43" EPOCHS=20 BATCH_SIZE=16 \
HISTORY_STEPS=6 HISTORY_STRIDE=4 HORIZON_STEPS=20 \
WINDOW_STRIDE=4 MESH_SIZE=2 LATENT_SIZE=128 MESSAGE_STEPS=4 \
bash scripts/run_graphcast_comparison.sh
```

`HISTORY_STEPS` and `HISTORY_STRIDE` define the existing five-way split and
forecast origins. They do **not** change GraphCast's exact two-state, 6-hourly
input. The split reserves 20 future steps, matching the existing downstream
trainer even when `HORIZON_STEPS` is reduced for a smoke test. Values above 20 are
rejected. Match the other experiments' archive, history settings, window stride,
origin stride, and lead horizon when comparing results.

A small executable smoke uses the real upstream architecture:

```bash
ARCHIVE=/absolute/path/states.npz RUN=runs/graphcast_smoke \
SEEDS=7 EPOCHS=1 BATCH_SIZE=1 MAX_WINDOWS=2 MAX_CASES=2 \
HORIZON_STEPS=2 MESH_SIZE=0 LATENT_SIZE=8 MESSAGE_STEPS=1 \
bash scripts/run_graphcast_comparison.sh
```

Each seed directory contains `checkpoint.npz`, `metadata.json`, `evaluation.json`
and `forecast.npz`. Checkpoints use NumPy arrays plus JSON, with pickle disabled.
The forecast file contains physical fields, truth, origin/valid times and schema
for the first finite evaluation origin, matching the existing evaluator's
visualization-export convention. Metrics cover **all** selected origins, with
nonfinite failures reported separately.

Evaluate the selected checkpoint on the untouched test split:

```bash
.venv-graphcast/bin/python -m climate_manifold.downstream.graphcast_official evaluate \
  --checkpoint runs/graphcast_official_surface_bs16/graphcast_official_raw_seed7/checkpoint.npz \
  --archive /absolute/path/states.npz \
  --split test \
  --output runs/graphcast_official_surface_bs16/graphcast_official_raw_seed7/test.json \
  --forecast-output runs/graphcast_official_surface_bs16/graphcast_official_raw_seed7/test.npz
```

Normalization moments and residual scales are fitted only to the training
observations. Future truth is only a training loss target or held-out scoring
reference. Forecast templates contain zeros and supply shape/coordinates only.
Known-future clock features are computed from timestamps and longitude, never
from future weather data.

## Verification

Basic archive/clock/checkpoint tests do not require JAX:

```bash
python -m pytest tests/test_graphcast_official.py -q
```

The opt-in test runs an actual two-step upstream GraphCast rollout and checks
finite, nonzero parameter gradients:

```bash
GRAPHCAST_OFFICIAL_SMOKE=1 .venv-graphcast/bin/python -m pytest tests/test_graphcast_official.py -q
```

Install `pytest` in the isolated environment if using this optional test command.
