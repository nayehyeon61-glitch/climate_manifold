#!/usr/bin/env bash
# Default: spatial E -> latent ClimODE -> D. Explicit raw/decoded are auxiliary.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ "${CLIMODE_BRIDGES:-latent}" == latent ]]; then
  [[ "${TRAINING_MODE:-joint}" == joint ]] || {
    echo 'Latent ClimODE is a joint spatial experiment; use raw/decoded for legacy controls' >&2; exit 2;
  }
  export MODELS=climode
  exec bash scripts/run_model_comparison.sh
fi
: "${ARCHIVE:?Set the original canonical surface archive}"
: "${CONSTANTS:?ClimODE needs aligned real orography and land-sea mask NPZ}"
: "${RUN:?Set a new auxiliary comparison experiment directory}"
[[ ! -e "$RUN" ]] || { echo 'Choose a new RUN' >&2; exit 2; }
PYTHON="${PYTHON:-python}"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
read -r -a seeds <<< "${SEEDS:-7 19 43}"
read -r -a bridges <<< "$CLIMODE_BRIDGES"
[[ ${#seeds[@]} -gt 0 ]] || { echo 'SEEDS must be nonempty' >&2; exit 2; }
[[ ${#bridges[@]} -gt 0 ]] || { echo 'CLIMODE_BRIDGES must be nonempty' >&2; exit 2; }
for mode in "${bridges[@]}"; do
  [[ "$mode" == raw || "$mode" == decoded ]] || {
    echo 'CLIMODE_BRIDGES must be latent alone, or an auxiliary selection of raw and/or decoded' >&2; exit 2;
  }
done
info=(); [[ -z "${INFO:-}" ]] || info=(--information "$INFO")
training_mode="${TRAINING_MODE:-joint}"
initialization="${INITIALIZATION:-fresh}"
if [[ "$training_mode" == frozen ]]; then initialization=pretrained; fi
if [[ "$initialization" == pretrained && -z "${A_CHECKPOINT:-}" ]]; then
  echo 'Pretrained/frozen experiments require A_CHECKPOINT' >&2; exit 2
fi
setup=(--training-mode "$training_mode" --initialization "$initialization"
  --manifold-dim "${MANIFOLD_DIM:-64}" --manifold-hidden-dim "${MANIFOLD_HIDDEN_DIM:-512}"
  --latent-channels "${LATENT_CHANNELS:-32}" --spatial-downsample "${SPATIAL_DOWNSAMPLE:-2}"
  --spatial-hidden-dim "${SPATIAL_HIDDEN_DIM:-64}"
  --context-dim "${CONTEXT_DIM:-64}" --history-steps "${HISTORY_STEPS:-6}" --history-stride "${HISTORY_STRIDE:-4}")
if [[ -n "${LATENT_LAYOUT:-}" ]]; then
  setup+=(--latent-layout "$LATENT_LAYOUT")
elif [[ "$initialization" == fresh ]]; then
  setup+=(--latent-layout spatial)
fi
[[ -z "${A_CHECKPOINT:-}" ]] || setup+=(--a-checkpoint "$A_CHECKPOINT")
[[ -z "${MODE:-}" ]] || setup+=(--mode "$MODE")
# Keep metadata/construction matched to a fresh primary PINN experiment. Joint decoded
# ClimODE has no future latent path, so no future information/PINN loss is applied.
if [[ "${PINN:-0}" == 1 ]]; then
  read -r -a levels <<< "${PINN_LEVELS:-500 850}"
  setup+=(--pinn --pinn-levels "${levels[@]}")
fi
[[ "${CLIMODE_ATTENTION:-1}" != 0 ]] || setup+=(--no-climode-attention)
mkdir -p "$RUN"
reports=()
for seed in "${seeds[@]}"; do
  for mode in "${bridges[@]}"; do
    prefix="$RUN/climode-${mode}-seed${seed}"
    "$PYTHON" -m climate_manifold.downstream.train "${setup[@]}" \
      --regularization none --reconstruction-weight "${RECONSTRUCTION_WEIGHT:-0.1}" \
      --archive "$ARCHIVE" "${info[@]}" --constants "$CONSTANTS" \
      --experiment auxiliary --model climode --bridge "$mode" --representation climate_manifold \
      --output "$prefix.pt" --epochs "${EPOCHS:-20}" --batch-size "${BATCH_SIZE:-2}" \
      --learning-rate "${LEARNING_RATE:-0.001}" --tendency-weight "${TENDENCY_WEIGHT:-0.1}" \
      --horizon-steps "${HORIZON_STEPS:-20}" --window-stride "${WINDOW_STRIDE:-4}" \
      --max-windows "${MAX_WINDOWS:-0}" --seed "$seed" --device "${DEVICE:-cpu}" --anchor none \
      --climode-step-hours "${CLIMODE_STEP_HOURS:-1}" --velocity-iterations "${VELOCITY_ITERATIONS:-20}"
    "$PYTHON" -m climate_manifold.downstream.evaluate --checkpoint "$prefix.pt" \
      --archive "$ARCHIVE" "${info[@]}" --split validation --output "$prefix.validation.json" \
      --forecast-output "$prefix.forecast.npz" --max-cases "${MAX_CASES:-0}" \
      --origin-stride "${ORIGIN_STRIDE:-1}" --device "${DEVICE:-cpu}"
    reports+=("$prefix.validation.json")
  done
done
"$PYTHON" -m climate_manifold.downstream.compare --reports "${reports[@]}" --output "$RUN/comparison.json"
