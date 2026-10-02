#!/usr/bin/env bash
# Raw->M vs E->M->D. Transformer is one choice of M, never an added stage.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${DAILY_WORK:?Set the existing work directory below /lustre/home/yehyeon}"
export ERA5_ROOT="${ERA5_ROOT:-/lustre/home/mahmed/ERA5_0p25_DAILY}"
export DAILY_WORK
PYTHON="${PYTHON:-python}"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -B -m climate_manifold.workspace
export TMPDIR="$DAILY_WORK/tmp" XDG_CACHE_HOME="$DAILY_WORK/cache"
export TMP="$TMPDIR" TEMP="$TMPDIR" PIP_CACHE_DIR="$DAILY_WORK/cache/pip"
export TORCH_HOME="$DAILY_WORK/cache/torch" HF_HOME="$DAILY_WORK/cache/huggingface"
export CUDA_CACHE_PATH="$DAILY_WORK/cache/cuda" TRITON_CACHE_DIR="$DAILY_WORK/cache/triton"
export PYTHONPYCACHEPREFIX="$DAILY_WORK/cache/pycache" MPLCONFIGDIR="$DAILY_WORK/cache/matplotlib"
export NUMBA_CACHE_DIR="$DAILY_WORK/cache/numba" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$DAILY_WORK/cache" "$DAILY_WORK/runs" "$DAILY_WORK/logs"
read -r -a models <<< "${MODELS:-transformer mlp neural_ode climode convlstm simvp fourcastnet climax}"
read -r -a seeds <<< "${SEEDS:-7 19 43}"
read -r -a losses <<< "${STATISTICAL_LOSSES:-w2 signed_measure}"
read -r -a pairs <<< "${CONSTRAINT_PAIRS:-statistical}"
[[ ${#models[@]} -gt 0 && ${#seeds[@]} -gt 0 && ${#losses[@]} -gt 0 && ${#pairs[@]} -gt 0 ]] || exit 2
declare -A seen=()
for family in "${models[@]}"; do
  case "$family" in transformer|mlp|neural_ode|climode|convlstm|simvp|fourcastnet|climax) ;;
    *) echo "Unsupported downstream M: $family" >&2; exit 2;; esac
  [[ ! -v "seen[model:$family]" ]] || { echo 'Duplicate model' >&2; exit 2; }
  seen[model:$family]=1
done
for loss in "${losses[@]}"; do
  case "$loss" in w2|signed_measure|kl_entropy) ;; *) echo "Unknown loss: $loss" >&2; exit 2;; esac
  [[ ! -v "seen[loss:$loss]" ]] || { echo 'Duplicate loss' >&2; exit 2; }
  seen[loss:$loss]=1
done
for pair in "${pairs[@]}"; do
  case "$pair" in statistical|pinn_statistical) ;; *) echo "Unknown constraint pair: $pair" >&2; exit 2;; esac
  [[ ! -v "seen[pair:$pair]" ]] || { echo 'Duplicate pair' >&2; exit 2; }
  seen[pair:$pair]=1
done
if [[ -z "${ARCHIVE:-}" && -v 'seen[pair:pinn_statistical]' ]]; then
  echo 'Daily PINN is unsupported. Supply a complete prepared 6-hour ARCHIVE and INFO for pinn_statistical.' >&2
  exit 2
fi
for seed in "${seeds[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ && ! -v "seen[seed:$seed]" ]] || { echo 'Invalid/duplicate seed' >&2; exit 2; }
  seen[seed:$seed]=1
done
for name in STATISTICAL_FLOW_WEIGHT CONDITIONAL_FLOW_WEIGHT; do
  [[ "${!name:-0}" == 0 ]] || { echo "$name must remain zero" >&2; exit 2; }
done
[[ -z "${A_CHECKPOINT:-}" ]] || { echo 'This comparison uses fresh joint training; unset A_CHECKPOINT' >&2; exit 2; }
if [[ -n "${ARCHIVE:-}" && -z "${INFO:-}" || -z "${ARCHIVE:-}" && -n "${INFO:-}" ]]; then
  echo 'Supply both ARCHIVE and INFO, or neither to prepare daily data' >&2; exit 2
fi
EVALUATE_TEST="${EVALUATE_TEST:-1}"
[[ "$EVALUATE_TEST" == 0 || "$EVALUATE_TEST" == 1 ]] || { echo 'EVALUATE_TEST must be 0 or 1' >&2; exit 2; }
RUN_PREFLIGHT_TESTS="${RUN_PREFLIGHT_TESTS:-1}"
[[ "$RUN_PREFLIGHT_TESTS" == 0 || "$RUN_PREFLIGHT_TESTS" == 1 ]] || { echo 'RUN_PREFLIGHT_TESTS must be 0 or 1' >&2; exit 2; }
if [[ "$RUN_PREFLIGHT_TESTS" == 1 ]]; then
  test_tmp=$(mktemp -d "$TMPDIR/preflight-XXXXXXXX")
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PYTHON" -m pytest -q \
    tests/test_daily_model_comparison.py tests/test_raw_latent_runner.py tests/test_runtime_guards.py \
    --basetemp "$test_tmp" -o "cache_dir=$DAILY_WORK/cache/pytest" \
    2>&1 | tee "$DAILY_WORK/logs/tests-$(date +%Y%m%d-%H%M%S).log"
fi
DEVICE="${DEVICE:-cuda}"
case "$DEVICE" in cuda|cuda:0|cpu) ;; *) echo 'Use DEVICE=cuda or cpu' >&2; exit 2;; esac
guard=("$PYTHON" -m climate_manifold.gpu_guard --lock-root /lustre/home/yehyeon/climate_manifold_gpu_locks
  --max-utilization "${GPU_MAX_UTILIZATION:-10}" --max-memory-percent "${GPU_MAX_MEMORY_PERCENT:-10}"
  --min-free-gib "${GPU_MIN_FREE_GIB:-8}")
run_model() {
  if [[ "$DEVICE" == cpu ]]; then "$PYTHON" -m "$@" --device cpu
  else "${guard[@]}" -- "$PYTHON" -m "$@" --device cuda:0; fi
}
if [[ "$DEVICE" != cpu ]]; then
  "${guard[@]}" -- "$PYTHON" -c 'import torch; x=torch.randn(32,32,device="cuda:0"); print("CUDA kernel OK", (x@x).sum().item())'
fi
if [[ -z "${ARCHIVE:-}" && -z "${INFO:-}" ]]; then
  prepare=(--root "$ERA5_ROOT" --write-root /lustre/home/yehyeon --output "$DAILY_WORK/prepared"
    --start "${START_DATE:-1979-01-01}" --end "${END_DATE:-2025-12-31}"
    --target-lat-points "${TARGET_LAT_POINTS:-16}" --target-lon-points "${TARGET_LON_POINTS:-32}")
  if [[ -n "${OROGRAPHY:-}" ]]; then prepare+=(--orography "$OROGRAPHY"); fi
  if [[ -n "${SOURCE_UNITS_JSON:-}" ]]; then prepare+=(--units-json "$SOURCE_UNITS_JSON"); fi
  "$PYTHON" -m climate_manifold.daily_era5 "${prepare[@]}" \
    2>&1 | tee "$DAILY_WORK/logs/prepare-$(date +%Y%m%d-%H%M%S).log"
  ARCHIVE="$DAILY_WORK/prepared/surface.npz"; INFO="$DAILY_WORK/prepared/information.npz"
fi
: "${ARCHIVE:?Supply both ARCHIVE and INFO, or neither to prepare daily data}"
: "${INFO:?Supply both ARCHIVE and INFO, or neither to prepare daily data}"
export ARCHIVE INFO
export ABLATION_PAIRS="${pairs[*]}"
step=$("$PYTHON" - <<'PY'
import json, os
from pathlib import Path
import numpy as np
archive, info = Path(os.environ['ARCHIVE']), Path(os.environ['INFO'])
if not archive.is_file() or not info.exists(): raise SystemExit('Missing prepared ARCHIVE/INFO')
schema = json.loads(archive.with_suffix('.schema.json').read_text())
step = schema['forecast_step_hours']
if step not in (6,24): raise SystemExit('Use a 6-hour or 24-hour archive')
if 'pinn_statistical' in os.environ['ABLATION_PAIRS'].split():
    if step == 24: raise SystemExit('Daily PINN is unsupported; use statistical-only or a complete 6-hour PINN archive')
    if info.is_dir(): meta=json.loads((info/'metadata.json').read_text())
    else:
        with np.load(info,allow_pickle=False) as data: meta=json.loads(str(data['metadata_json']))
    needed={f'{v}{p}' for v in 'uvtzw' for p in (500,850)}|{'sp','terrain_height','terrain_slope'}
    missing=needed-{v['name'] for v in meta['variables']}
    if missing: raise SystemExit('PINN variables missing: '+', '.join(sorted(missing)))
print(step)
PY
)
if [[ "$step" == 24 ]]; then stride=1; horizon=5; window=1
else stride=4; horizon=20; window=4; fi
RUN=$(mktemp -d "$DAILY_WORK/runs/raw-latent-XXXXXXXX")
exec > >(tee "$RUN/run.log") 2>&1
git rev-parse HEAD > "$RUN/source-commit.txt"
printf 'Run: %s\nFits: %s\n' "$RUN" "$(( ${#models[@]} * ${#seeds[@]} * (1 + ${#losses[@]} * ${#pairs[@]}) ))"
common=(--archive "$ARCHIVE" --information "$INFO" --training-mode joint --initialization fresh
  --mode enriched --representation climate_manifold --latent-layout spatial --raw-backend matched
  --experiment primary --anchor none --spatial-variable-conditioning
  --latent-channels "${LATENT_CHANNELS:-32}" --spatial-downsample "${SPATIAL_DOWNSAMPLE:-2}"
  --spatial-hidden-dim "${SPATIAL_HIDDEN_DIM:-64}" --hidden-dim "${HIDDEN_DIM:-128}"
  --climode-step-hours "${CLIMODE_STEP_HOURS:-1}" --ode-substeps "${ODE_SUBSTEPS:-2}"
  --weather-depth "${WEATHER_DEPTH:-4}" --weather-patch-size "${WEATHER_PATCH_SIZE:-2}"
  --transformer-heads "${TRANSFORMER_HEADS:-4}"
  --history-steps "${HISTORY_STEPS:-6}" --history-stride "$stride"
  --horizon-steps "${HORIZON_STEPS:-$horizon}" --window-stride "$window"
  --epochs "${EPOCHS:-20}" --batch-size "${BATCH_SIZE:-16}" --learning-rate "${LEARNING_RATE:-0.001}"
  --reconstruction-weight "${RECONSTRUCTION_WEIGHT:-0.1}" --tendency-weight "${TENDENCY_WEIGHT:-0.1}"
  --statistical-flow-weight 0 --conditional-flow-weight 0 --static-weight 0
  --max-windows "${MAX_WINDOWS:-0}")
prefixes=(); validation_reports=()
fit() {
  local prefix="$1"; shift
  run_model climate_manifold.downstream.train "${common[@]}" "$@" --output "$prefix.pt"
  run_model climate_manifold.downstream.evaluate --checkpoint "$prefix.pt" \
    --archive "$ARCHIVE" --information "$INFO" --split validation --output "$prefix.validation.json" \
    --max-cases "${MAX_CASES:-0}" --origin-stride "${ORIGIN_STRIDE:-1}"
  prefixes+=("$prefix"); validation_reports+=("$prefix.validation.json")
}
for seed in "${seeds[@]}"; do
  for family in "${models[@]}"; do
    fit "$RUN/$family-raw-seed$seed" --model "$family" --seed "$seed" --bridge raw --regularization none
    for pair in "${pairs[@]}"; do
      pinn=(); [[ "$pair" != pinn_statistical ]] || pinn=(--pinn --pinn-levels 500 850 --pinn-weight "${PINN_WEIGHT:-0.1}")
      for loss in "${losses[@]}"; do
        fit "$RUN/$family-$pair-$loss-latent-seed$seed" \
          --model "$family" --seed "$seed" --bridge latent --regularization full \
          --constraint-pair "$pair" --constraint-decoder separate_surface_and_information \
          --statistical-loss "$loss" --statistical-weight "${STATISTICAL_WEIGHT:-0.1}" \
          "${pinn[@]}"
      done
    done
  done
done
"$PYTHON" -m climate_manifold.downstream.compare --reports "${validation_reports[@]}" --output "$RUN/comparison.validation.json"
# Test is read only after every fit/checkpoint is fixed.
if [[ "$EVALUATE_TEST" == 0 ]]; then
  printf '\nCompleted validation: %s\n' "$RUN/comparison.validation.json"
  exit 0
fi
test_reports=()
for prefix in "${prefixes[@]}"; do
  run_model climate_manifold.downstream.evaluate --checkpoint "$prefix.pt" \
    --archive "$ARCHIVE" --information "$INFO" --split test --output "$prefix.test.json" \
    --max-cases "${MAX_CASES:-0}" --origin-stride "${ORIGIN_STRIDE:-1}"
  test_reports+=("$prefix.test.json")
done
"$PYTHON" -m climate_manifold.downstream.compare --reports "${test_reports[@]}" --output "$RUN/comparison.test.json"
printf '\nCompleted: %s\n' "$RUN/comparison.test.raw-effects.csv"
