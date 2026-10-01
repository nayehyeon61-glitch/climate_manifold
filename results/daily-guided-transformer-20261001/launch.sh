#!/usr/bin/env bash
set -euo pipefail
export ERA5_ROOT=/lustre/home/mahmed/ERA5_0p25_DAILY
export DAILY_WORK=/lustre/home/yehyeon/climate_manifold_transformer_20261001_140940_mrVHVMnV
export TMPDIR="$DAILY_WORK/tmp" TMP="$DAILY_WORK/tmp" TEMP="$DAILY_WORK/tmp"
export XDG_CACHE_HOME="$DAILY_WORK/cache" PIP_CACHE_DIR="$DAILY_WORK/cache/pip"
export TORCH_HOME="$DAILY_WORK/cache/torch" HF_HOME="$DAILY_WORK/cache/huggingface"
export CUDA_CACHE_PATH="$DAILY_WORK/cache/cuda" TRITON_CACHE_DIR="$DAILY_WORK/cache/triton"
export PYTHONPYCACHEPREFIX="$DAILY_WORK/cache/pycache" MPLCONFIGDIR="$DAILY_WORK/cache/matplotlib"
export NUMBA_CACHE_DIR="$DAILY_WORK/cache/numba" PYTHONNOUSERSITE=1
cd "$DAILY_WORK/code"
source "$DAILY_WORK/venv/bin/activate"
export PYTHON="$DAILY_WORK/venv/bin/python"
unset A_CHECKPOINT
export DEVICE=cuda BATCH_SIZE=16 EPOCHS=20 SEEDS="7 19 43"
export HISTORY_STEPS=6 HORIZON_STEPS=5
export STATISTICAL_LOSS=signed_measure CONSTRAINT_DECODER=separate_surface_and_information
export INCLUDE_ZERO_GUIDE=1
export START_DATE=1979-01-01 END_DATE=2025-12-31
export TARGET_LAT_POINTS=16 TARGET_LON_POINTS=32
export MAX_WINDOWS=0 MAX_CASES=0 ORIGIN_STRIDE=1 OMP_NUM_THREADS=4
export RUN_PREFLIGHT_TESTS=1 EVALUATE_TEST=1
export GPU_MAX_UTILIZATION=10 GPU_MAX_MEMORY_PERCENT=10 GPU_MIN_FREE_GIB=8
export GPU_WAIT_SECONDS=300
printf '작업·결과 저장 위치: %s\n' "$DAILY_WORK"
bash scripts/run_daily_guided_transformer.sh
echo "EXIT: $?"
