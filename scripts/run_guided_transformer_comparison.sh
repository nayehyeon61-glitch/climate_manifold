#!/usr/bin/env bash
# Matched-backbone raw / E-F-D / raw+statistical-guide experiments.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${ARCHIVE:?Set the canonical surface archive}"
: "${INFO:?Set the aligned physical-information archive}"
: "${RUN:?Set a new experiment directory}"
[[ ! -e "$RUN" ]] || { echo 'Choose a new RUN' >&2; exit 2; }
[[ -z "${A_CHECKPOINT:-}" ]] || { echo 'This runner initializes fresh representations; unset A_CHECKPOINT' >&2; exit 2; }
PYTHON="${PYTHON:-python}"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
read -r -a seeds <<< "${SEEDS:-7 19 43}"
[[ ${#seeds[@]} -gt 0 ]] || { echo 'SEEDS must be nonempty' >&2; exit 2; }
declare -A seen=()
for seed in "${seeds[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo 'SEEDS must be nonnegative integers' >&2; exit 2; }
  [[ ! -v "seen[$seed]" ]] || { echo "Duplicate seed: $seed" >&2; exit 2; }
  seen[$seed]=1
done
STATISTICAL_LOSS="${STATISTICAL_LOSS:-w2}"
case "$STATISTICAL_LOSS" in w2|kl_entropy|signed_measure) ;; *) echo 'Unknown STATISTICAL_LOSS' >&2; exit 2;; esac
CONSTRAINT_DECODER="${CONSTRAINT_DECODER:-separate_surface_and_information}"
case "$CONSTRAINT_DECODER" in
  separate_surface_and_information|information_only) ;;
  *) echo 'Guided forecasts require an independent information/reconstruction decoder' >&2; exit 2;;
esac
INCLUDE_ZERO_GUIDE="${INCLUDE_ZERO_GUIDE:-0}"
[[ "$INCLUDE_ZERO_GUIDE" == 0 || "$INCLUDE_ZERO_GUIDE" == 1 ]] || { echo 'INCLUDE_ZERO_GUIDE must be 0 or 1' >&2; exit 2; }
GUIDE_DIRECT_INFORMATION="${GUIDE_DIRECT_INFORMATION:-0}"
[[ "$GUIDE_DIRECT_INFORMATION" == 0 || "$GUIDE_DIRECT_INFORMATION" == 1 ]] || { echo 'GUIDE_DIRECT_INFORMATION must be 0 or 1' >&2; exit 2; }
VARIABLE_CONDITIONING="${VARIABLE_CONDITIONING:-0}"
[[ "$VARIABLE_CONDITIONING" == 0 || "$VARIABLE_CONDITIONING" == 1 ]] || { echo 'VARIABLE_CONDITIONING must be 0 or 1' >&2; exit 2; }
EVALUATE_TEST="${EVALUATE_TEST:-0}"
GPU_GUARD="${GPU_GUARD:-0}"
for name in EVALUATE_TEST GPU_GUARD; do
  [[ "${!name}" == 0 || "${!name}" == 1 ]] || { echo "$name must be 0 or 1" >&2; exit 2; }
done
if [[ "$GPU_GUARD" == 1 ]]; then
  : "${GPU_LOCK_ROOT:?GPU_GUARD requires a GPU_LOCK_ROOT below /lustre/home/yehyeon}"
fi
run_model() {
  if [[ "$GPU_GUARD" == 1 ]]; then
    "$PYTHON" -m climate_manifold.gpu_guard --lock-root "$GPU_LOCK_ROOT" \
      --max-utilization "${GPU_MAX_UTILIZATION:-10}" \
      --max-memory-percent "${GPU_MAX_MEMORY_PERCENT:-10}" --min-free-gib "${GPU_MIN_FREE_GIB:-8}" \
      -- "$PYTHON" -m "$@" --device cuda:0
  else
    "$PYTHON" -m "$@" --device "${DEVICE:-cpu}"
  fi
}
# These mechanisms remain explicitly disabled in this experiment.
for name in STATISTICAL_FLOW_WEIGHT CONDITIONAL_FLOW_WEIGHT; do
  [[ "${!name:-0}" == 0 ]] || { echo "$name must remain 0 in this runner" >&2; exit 2; }
done
setup=(--model transformer --training-mode joint --initialization fresh
  --mode enriched --representation climate_manifold --latent-layout spatial
  --experiment primary --anchor none --raw-backend matched
  --manifold-dim "${MANIFOLD_DIM:-64}" --manifold-hidden-dim "${MANIFOLD_HIDDEN_DIM:-512}"
  --latent-channels "${LATENT_CHANNELS:-32}" --spatial-downsample "${SPATIAL_DOWNSAMPLE:-2}"
  --spatial-hidden-dim "${SPATIAL_HIDDEN_DIM:-64}" --context-dim "${CONTEXT_DIM:-64}"
  --history-steps "${HISTORY_STEPS:-6}" --history-stride "${HISTORY_STRIDE:-4}"
  --hidden-dim "${HIDDEN_DIM:-128}" --weather-depth "${WEATHER_DEPTH:-4}"
  --weather-patch-size "${WEATHER_PATCH_SIZE:-2}" --transformer-heads "${TRANSFORMER_HEADS:-4}"
  --epochs "${EPOCHS:-20}" --batch-size "${BATCH_SIZE:-16}"
  --learning-rate "${LEARNING_RATE:-0.001}" --horizon-steps "${HORIZON_STEPS:-20}"
  --tendency-weight "${TENDENCY_WEIGHT:-0.1}" --reconstruction-weight "${RECONSTRUCTION_WEIGHT:-0.1}"
  --window-stride "${WINDOW_STRIDE:-4}" --max-windows "${MAX_WINDOWS:-0}")
if [[ "$VARIABLE_CONDITIONING" == 1 ]]; then setup+=(--spatial-variable-conditioning); fi
statistical=(--constraint-pair statistical --regularization full
  --constraint-decoder "$CONSTRAINT_DECODER" --statistical-loss "$STATISTICAL_LOSS"
  --statistical-weight "${STATISTICAL_WEIGHT:-0.1}"
  --statistical-flow-weight 0 --conditional-flow-weight 0)
if [[ "$STATISTICAL_LOSS" == kl_entropy ]]; then
  statistical+=(--kl-bins "${KL_BINS:-64}" --kl-range "${KL_RANGE:-6}" --kl-bandwidth "${KL_BANDWIDTH:-0.2}")
fi
# Optionally give guided arms the same direct origin-information tokens as raw.
guide_information=()
if [[ "$GUIDE_DIRECT_INFORMATION" == 1 ]]; then guide_information=(--guide-direct-information); fi
arms=(raw latent guided)
if [[ "$INCLUDE_ZERO_GUIDE" == 1 ]]; then arms+=(guided_zero); fi
mkdir -p "$RUN"
reports=()
test_reports=()
for seed in "${seeds[@]}"; do
  for arm in "${arms[@]}"; do
    prefix="$RUN/transformer-$arm-seed$seed"
    case "$arm" in
      raw) route=(--bridge raw --regularization none) ;;
      latent) route=(--bridge latent "${statistical[@]}") ;;
      guided) route=(--bridge guided --guide-mode learned "${guide_information[@]}" "${statistical[@]}") ;;
      guided_zero) route=(--bridge guided --guide-mode zero "${guide_information[@]}" "${statistical[@]}") ;;
    esac
    run_model climate_manifold.downstream.train "${setup[@]}" "${route[@]}" \
      --archive "$ARCHIVE" --information "$INFO" --output "$prefix.pt" \
      --seed "$seed"
    run_model climate_manifold.downstream.evaluate --checkpoint "$prefix.pt" \
      --archive "$ARCHIVE" --information "$INFO" --split validation \
      --output "$prefix.validation.json" --forecast-output "$prefix.forecast.npz" \
      --max-cases "${MAX_CASES:-0}" --origin-stride "${ORIGIN_STRIDE:-1}"
    reports+=("$prefix.validation.json")
  done
done
"$PYTHON" -m climate_manifold.downstream.compare --reports "${reports[@]}" --output "$RUN/comparison.json"
if [[ "$EVALUATE_TEST" == 1 ]]; then
  # Every fit/checkpoint is fixed before test evaluation. Test scores never
  # enter optimization, checkpoint selection, or validation comparisons.
  for seed in "${seeds[@]}"; do
    for arm in "${arms[@]}"; do
      prefix="$RUN/transformer-$arm-seed$seed"
      run_model climate_manifold.downstream.evaluate --checkpoint "$prefix.pt" \
        --archive "$ARCHIVE" --information "$INFO" --split test \
        --output "$prefix.test.json" --forecast-output "$prefix.test.forecast.npz" \
        --max-cases "${MAX_CASES:-0}" --origin-stride "${ORIGIN_STRIDE:-1}"
      test_reports+=("$prefix.test.json")
    done
  done
  "$PYTHON" -m climate_manifold.downstream.compare --reports "${test_reports[@]}" --output "$RUN/comparison.test.json"
fi
