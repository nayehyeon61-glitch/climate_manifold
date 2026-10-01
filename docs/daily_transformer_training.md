# Daily ERA5 Transformer comparison

Branch: `feature/daily-statistical-guide-transformer` (extends `15714dd`).

The source has one 00 UTC-labelled value per day, not four 6-hourly snapshots.
This route uses 24-hour archive steps, observed constraints at origin−24h/origin,
and physically correct 24/48/72/96/120-hour leads. It does not invent intraday
states. A 00 UTC label alone does not establish whether values are instantaneous
or daily means; source `cell_methods` and field attributes are retained.

## What is read and written

The supplied root contains `daily/YYYYMMDD.nc` and
`era5_orography_0p25.nc`. The default date range is 1979-01-01–2025-12-31.
Every selected date must exist, have exactly one matching timestamp, and use the
same source grid as orography. NetCDFs are opened one at a time; only selected
fields and pressure levels are spatially averaged using the existing block
pooling implementation. Missing pooled cells are rejected, not filled.

- Surface targets: `mslp` → canonical `msl`, `t2m`, `u10`, `v10`.
- Origin information: Z850/Z500/Z250, U850/V850, T850/T500, terrain height/slope.
- Geopotential is converted to metres with `g=9.80665`; Pa/hPa conversions use
  declared units. Missing field units require explicit `SOURCE_UNITS_JSON`.
- `lsm`, humidity and other surface variables are currently unused; having files
  on disk does not imply those variables are model inputs.
- Original files stay at their original paths and are only opened for reading.
- Derived archives, month caches, code, virtual environment, logs and outputs
  are placed under a work subdirectory of the original root.

The initial experiment uses **16×32 coarse fields**, not full 721×1440 prediction.
Coarse arrays are assembled in memory after streaming the high-resolution files;
the raw 47-year dataset is never assembled in memory. `preparation_plan.json`
records raw relative paths, sizes and modification times, not full raw-content
hashes. Prepared outputs are SHA-256 verified. Completed month caches support
resuming preparation with the same work directory. A changed input plan is
rejected; choose a new derived output instead of mixing preparations.

## Complete server command

Run from an allocated GPU session. The command does not change GPU visibility or
override a scheduler's allocation. PyTorch 2.8.0 with CUDA 12.8 is explicitly
installed in the new environment; the runner executes a CUDA matrix kernel
before preprocessing to detect a driver/build mismatch.

```bash
bash <<'BASH'
set -euo pipefail
export ERA5_ROOT=/lustre/home/mahmed/ERA5_0p25_DAILY
cd "$ERA5_ROOT"
export DAILY_WORK="$ERA5_ROOT/climate_manifold_daily_transformer_$(date +%Y%m%d_%H%M%S)"
mkdir "$DAILY_WORK"
mkdir -p "$DAILY_WORK"/{tmp,cache,logs}
export TMPDIR="$DAILY_WORK/tmp"
export XDG_CACHE_HOME="$DAILY_WORK/cache"
export PIP_CACHE_DIR="$DAILY_WORK/cache/pip"
export TORCH_HOME="$DAILY_WORK/cache/torch"
export HF_HOME="$DAILY_WORK/cache/huggingface"
export CUDA_CACHE_PATH="$DAILY_WORK/cache/cuda"
export TRITON_CACHE_DIR="$DAILY_WORK/cache/triton"
export PYTHONPYCACHEPREFIX="$DAILY_WORK/cache/pycache"
export PYTHONNOUSERSITE=1

git clone --single-branch --branch feature/daily-statistical-guide-transformer \
  https://github.com/nayehyeon61-glitch/climate_manifold.git "$DAILY_WORK/code"
cd "$DAILY_WORK/code"
python3 -m venv "$DAILY_WORK/venv"
source "$DAILY_WORK/venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'xarray==2024.11.0' -e '.[forecast,era5]'
export PYTHON="$DAILY_WORK/venv/bin/python"
export DEVICE=cuda BATCH_SIZE=16 EPOCHS=20 SEEDS="7 19 43"
export STATISTICAL_LOSS=signed_measure INCLUDE_ZERO_GUIDE=1
export START_DATE=1979-01-01 END_DATE=2025-12-31
export TARGET_LAT_POINTS=16 TARGET_LON_POINTS=32
export MAX_WINDOWS=0 MAX_CASES=0
export OMP_NUM_THREADS=4
git log -1 --oneline
printf 'Work directory: %s\n' "$DAILY_WORK"
bash scripts/run_daily_guided_transformer.sh
BASH
```

The runner trains **12 fits**, four routes × seeds 7/19/43:

| Route | Forecast input/output |
| --- | --- |
| raw | raw observations + available origin information → Transformer → fields |
| latent | encoder → Transformer → forecast decoder |
| guided | raw observations + variable-conditioned latent guide → Transformer → fields |
| guided_zero | same guided architecture with guide values zeroed |

All use six daily observations (t−5d through t), daily training-window stride,
five future days, batch 16 and 20 epochs. Encoder and predictor train jointly.
Constrained arms retain independent observed surface/information decoders and
signed-measure supervision. PINN, static penalties and both flow objectives are
off. Static terrain remains available as input context. Guide sampling remains
deterministic. Changing `STATISTICAL_LOSS` to `w2` or `kl_entropy` is supported.
Raw/latent/guided total parameter counts differ and are recorded in reports.

The full date range supplies the existing five chronological train/held-out
splits; it is not all used for fitting. Calibration selects checkpoints, and
the comparison evaluates the held-out validation split. The test split remains
reserved. No maximum-window/case truncation is used in the full command.

Daily support is deliberately scoped to fresh, jointly trained spatial
Transformers with observed statistical constraints or forecast-only losses.
Other forecasters, daily PINN and daily flow losses are rejected until validated.
Existing 6-hour experiments retain their settings and checkpoint semantics.

## Results and restart

- `$DAILY_WORK/logs/prepare-*.log`: month-by-month preprocessing progress.
- `$DAILY_WORK/prepared/surface.npz`, `information.npz`: derived training inputs.
- `$DAILY_WORK/runs/comparison-*/`: checkpoints, per-run metrics, validation
  reports, forecasts and `comparison.json`/CSV summaries.

To resume preprocessing, set `ERA5_ROOT`, `DAILY_WORK`, `PYTHON`, and the same
preparation settings to their previous values and rerun the runner from its
`code` directory. It reuses verified complete archives or completed month caches.
It starts a **new training comparison**; this is not optimizer-state resumption.
For a quick integration run use `SEEDS=7 EPOCHS=1 MAX_WINDOWS=8 MAX_CASES=2`
with the same prepared data. New comparison directories never overwrite prior
checkpoints. Loss settings can change without reprocessing the same data.

Validation in the development environment uses synthetic daily NetCDFs, including
mslp in hPa, geopotential conversion, descending latitudes, missing dates,
incorrect timestamps, read-only source hashes, preparation reuse, four-route
training/checkpoint roundtrip/evaluation, and rejecting a 6h constraint label in
a daily report. Real-server ERA5 training has not been executed here.
