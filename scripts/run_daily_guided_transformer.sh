#!/usr/bin/env bash
# Managed writes stay under /lustre/home/yehyeon; ERA5 is read in place.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${ERA5_ROOT:?Set the existing ERA5 root with daily/YYYYMMDD.nc}"
: "${DAILY_WORK:?Set a work directory below /lustre/home/yehyeon}"
SUITE="${SUITE:-guided_transformer}"
case "$SUITE" in guided_transformer|manifold_fusion) ;; *) echo 'Unknown SUITE' >&2; exit 2;; esac
if [[ "$SUITE" == manifold_fusion ]]; then
  RESUME="${RESUME:-0}"
  [[ "$RESUME" == 0 || "$RESUME" == 1 ]] || { echo 'RESUME must be 0 or 1' >&2; exit 2; }
  [[ "$RESUME" == 0 || -n "${RUN:-}" ]] || { echo 'RESUME=1 requires an explicit RUN' >&2; exit 2; }
fi
PYTHON="${PYTHON:-python}"
export WORK_ROOT=/lustre/home/yehyeon
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
export ERA5_ROOT DAILY_WORK
# -B avoids bytecode writes before the path policy has been validated.
"$PYTHON" -B -m climate_manifold.workspace
if [[ "$SUITE" == manifold_fusion && -n "${RUN:-}" ]]; then
  # An explicit resume/new-run destination obeys the same write boundary.
  export RUN
  "$PYTHON" -B - <<'PY'
import os
from pathlib import Path
root = (Path(os.environ['DAILY_WORK'])/'runs').resolve()
run = Path(os.environ['RUN']).resolve()
if run == root or not run.is_relative_to(root):
    raise SystemExit('RUN must be a subdirectory of DAILY_WORK/runs')
PY
fi

export TMPDIR="$DAILY_WORK/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export XDG_CACHE_HOME="$DAILY_WORK/cache"
export PIP_CACHE_DIR="$DAILY_WORK/cache/pip"
export TORCH_HOME="$DAILY_WORK/cache/torch"
export HF_HOME="$DAILY_WORK/cache/huggingface"
export CUDA_CACHE_PATH="$DAILY_WORK/cache/cuda"
export TRITON_CACHE_DIR="$DAILY_WORK/cache/triton"
export PYTHONPYCACHEPREFIX="$DAILY_WORK/cache/pycache"
export MPLCONFIGDIR="$DAILY_WORK/cache/matplotlib"
export NUMBA_CACHE_DIR="$DAILY_WORK/cache/numba"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$DAILY_WORK/logs"
export DEVICE="${DEVICE:-cuda}"
export GPU_LOCK_ROOT="$WORK_ROOT/climate_manifold_gpu_locks"
export GPU_MAX_UTILIZATION="${GPU_MAX_UTILIZATION:-10}"
export GPU_MAX_MEMORY_PERCENT="${GPU_MAX_MEMORY_PERCENT:-10}" GPU_MIN_FREE_GIB="${GPU_MIN_FREE_GIB:-8}"
case "$DEVICE" in
  cuda*)
    export GPU_GUARD=1 DEVICE=cuda:0
    # Verify actual CUDA execution only after admission, before reading 47 years.
    "$PYTHON" -m climate_manifold.gpu_guard --lock-root "$GPU_LOCK_ROOT" \
      --max-utilization "$GPU_MAX_UTILIZATION" --max-memory-percent "$GPU_MAX_MEMORY_PERCENT" \
      --min-free-gib "$GPU_MIN_FREE_GIB" -- "$PYTHON" -c \
      'import torch; x=torch.randn(32,32,device="cuda:0"); print("CUDA kernel OK:",(x@x).sum().item(),torch.cuda.get_device_name(0))' \
      2>&1 | tee "$DAILY_WORK/logs/gpu-preflight-$(date +%Y%m%d-%H%M%S).log"
    ;;
  cpu) export GPU_GUARD=0 ;;
  *) echo 'DEVICE must be cuda or cpu' >&2; exit 2 ;;
esac

RUN_PREFLIGHT_TESTS="${RUN_PREFLIGHT_TESTS:-1}"
[[ "$RUN_PREFLIGHT_TESTS" == 0 || "$RUN_PREFLIGHT_TESTS" == 1 ]] || { echo 'RUN_PREFLIGHT_TESTS must be 0 or 1' >&2; exit 2; }
if [[ "$RUN_PREFLIGHT_TESTS" == 1 ]]; then
  # CPU-only software tests; temporary fixtures/caches also stay in DAILY_WORK.
  # mktemp supplies a fresh basetemp because pytest clears that directory.
  test_tmp=$(mktemp -d "$TMPDIR/preflight-XXXXXXXX")
  preflight=(tests/test_guided_training.py tests/test_guided_comparison.py tests/test_daily_era5.py
    tests/test_runtime_guards.py)
  if [[ "$SUITE" == manifold_fusion ]]; then
    preflight+=(tests/test_guided_fusion.py tests/test_fusion_runner.py tests/test_comparison_plots.py)
  fi
  CUDA_VISIBLE_DEVICES="" GPU_GUARD=0 OMP_NUM_THREADS=1 "$PYTHON" -m pytest -q \
    "${preflight[@]}" --basetemp "$test_tmp" -o "cache_dir=$DAILY_WORK/cache/pytest" \
    2>&1 | tee "$DAILY_WORK/logs/tests-$(date +%Y%m%d-%H%M%S).log"
fi

prepare=(--root "$ERA5_ROOT" --write-root "$WORK_ROOT" --output "$DAILY_WORK/prepared"
  --start "${START_DATE:-1979-01-01}" --end "${END_DATE:-2025-12-31}"
  --target-lat-points "${TARGET_LAT_POINTS:-16}" --target-lon-points "${TARGET_LON_POINTS:-32}")
if [[ -n "${OROGRAPHY:-}" ]]; then prepare+=(--orography "$OROGRAPHY"); fi
if [[ -n "${SOURCE_UNITS_JSON:-}" ]]; then prepare+=(--units-json "$SOURCE_UNITS_JSON"); fi
"$PYTHON" -u -m climate_manifold.daily_era5 "${prepare[@]}" \
  2>&1 | tee "$DAILY_WORK/logs/prepare-$(date +%Y%m%d-%H%M%S).log"

export ARCHIVE="$DAILY_WORK/prepared/surface.npz"
export INFO="$DAILY_WORK/prepared/information.npz"
if [[ "$SUITE" == manifold_fusion ]]; then
  export RUN="${RUN:-$DAILY_WORK/runs/manifold-fusion-$(date +%Y%m%d-%H%M%S)}"
else
  export RUN="$DAILY_WORK/runs/comparison-$(date +%Y%m%d-%H%M%S)"
fi
export BATCH_SIZE="${BATCH_SIZE:-16}" EPOCHS="${EPOCHS:-20}" SEEDS="${SEEDS:-7 19 43}"
export HISTORY_STEPS="${HISTORY_STEPS:-6}" HISTORY_STRIDE=1
export HORIZON_STEPS="${HORIZON_STEPS:-5}" WINDOW_STRIDE=1
if [[ "$SUITE" == manifold_fusion ]]; then
  export VARIABLE_CONDITIONING="${VARIABLE_CONDITIONING:-0}"
else
  export VARIABLE_CONDITIONING=1
fi
export INCLUDE_ZERO_GUIDE="${INCLUDE_ZERO_GUIDE:-1}"
if [[ "$SUITE" == manifold_fusion ]]; then default_loss=w2; else default_loss=signed_measure; fi
export STATISTICAL_LOSS="${STATISTICAL_LOSS:-$default_loss}"
export STATISTICAL_FLOW_WEIGHT=0 CONDITIONAL_FLOW_WEIGHT=0
export EVALUATE_TEST="${EVALUATE_TEST:-1}"
# Recheck GPU admission before each train/eval command. No parallel fits.
# 6 observed daily fields -> leads 24,48,72,96,120 hours.
if [[ "$SUITE" == manifold_fusion ]]; then runner=scripts/run_guided_fusion_comparison.sh
else runner=scripts/run_guided_transformer_comparison.sh; fi
bash "$runner" \
  2>&1 | tee "$DAILY_WORK/logs/train-$(date +%Y%m%d-%H%M%S).log"
printf '\nCompleted comparison: %s\n' "$RUN/comparison.json"
if [[ "$EVALUATE_TEST" == 1 ]]; then printf 'Held-out test comparison: %s\n' "$RUN/comparison.test.json"; fi
