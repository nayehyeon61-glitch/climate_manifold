#!/usr/bin/env bash
# Run inside the existing ERA5 root. Raw data is read in place.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${ERA5_ROOT:?Set the existing ERA5 root with daily/YYYYMMDD.nc}"
: "${DAILY_WORK:?Set a work directory below ERA5_ROOT}"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
export ERA5_ROOT DAILY_WORK
"$PYTHON" - <<'PY'
import os
from pathlib import Path
root=Path(os.environ['ERA5_ROOT']).resolve(strict=True)
work=Path(os.environ['DAILY_WORK']).resolve()
repo=Path.cwd().resolve()
if work==root or not work.is_relative_to(root) or not repo.is_relative_to(root):
    raise SystemExit('Repository and work directory must both be inside ERA5_ROOT')
if work.is_relative_to(root/'daily'):
    raise SystemExit('Use a separate derived-work directory, outside raw daily files')
work.mkdir(parents=True,exist_ok=True)
PY

export TMPDIR="$DAILY_WORK/tmp"
export XDG_CACHE_HOME="$DAILY_WORK/cache"
export PIP_CACHE_DIR="$DAILY_WORK/cache/pip"
export TORCH_HOME="$DAILY_WORK/cache/torch"
export HF_HOME="$DAILY_WORK/cache/huggingface"
export CUDA_CACHE_PATH="$DAILY_WORK/cache/cuda"
export TRITON_CACHE_DIR="$DAILY_WORK/cache/triton"
export PYTHONPYCACHEPREFIX="$DAILY_WORK/cache/pycache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$DAILY_WORK/logs"
export DEVICE="${DEVICE:-cuda}"
"$PYTHON" - <<'PY'
import os, torch
if os.environ['DEVICE'].startswith('cuda'):
    if not torch.cuda.is_available(): raise SystemExit('CUDA PyTorch is unavailable; install the server-compatible CUDA build first')
    # Verify a real kernel, including Blackwell compatibility, before reading 47 years.
    device=os.environ['DEVICE']
    x=torch.randn(32,32,device=device)
    (x@x).sum().item()
    print('GPU:',torch.cuda.get_device_name(torch.device(device)),flush=True)
PY

prepare=(--root "$ERA5_ROOT" --output "$DAILY_WORK/prepared"
  --start "${START_DATE:-1979-01-01}" --end "${END_DATE:-2025-12-31}"
  --target-lat-points "${TARGET_LAT_POINTS:-16}" --target-lon-points "${TARGET_LON_POINTS:-32}")
if [[ -n "${OROGRAPHY:-}" ]]; then prepare+=(--orography "$OROGRAPHY"); fi
if [[ -n "${SOURCE_UNITS_JSON:-}" ]]; then prepare+=(--units-json "$SOURCE_UNITS_JSON"); fi
"$PYTHON" -u -m climate_manifold.daily_era5 "${prepare[@]}" \
  2>&1 | tee "$DAILY_WORK/logs/prepare-$(date +%Y%m%d-%H%M%S).log"

export ARCHIVE="$DAILY_WORK/prepared/surface.npz"
export INFO="$DAILY_WORK/prepared/information.npz"
export RUN="$DAILY_WORK/runs/comparison-$(date +%Y%m%d-%H%M%S)"
export BATCH_SIZE="${BATCH_SIZE:-16}" EPOCHS="${EPOCHS:-20}" SEEDS="${SEEDS:-7 19 43}"
export HISTORY_STEPS="${HISTORY_STEPS:-6}" HISTORY_STRIDE=1
export HORIZON_STEPS="${HORIZON_STEPS:-5}" WINDOW_STRIDE=1
export VARIABLE_CONDITIONING=1 INCLUDE_ZERO_GUIDE="${INCLUDE_ZERO_GUIDE:-1}"
export STATISTICAL_LOSS="${STATISTICAL_LOSS:-signed_measure}"
export STATISTICAL_FLOW_WEIGHT=0 CONDITIONAL_FLOW_WEIGHT=0
export PYTHONUNBUFFERED=1
# The existing runner refuses overwrite and evaluates all held-out validation
# origins by default. 6 observed daily fields -> leads 24,48,72,96,120 hours.
bash scripts/run_guided_transformer_comparison.sh \
  2>&1 | tee "$DAILY_WORK/logs/train-$(date +%Y%m%d-%H%M%S).log"
printf '\nCompleted comparison: %s\n' "$RUN/comparison.json"
