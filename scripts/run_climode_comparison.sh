#!/usr/bin/env bash
# Auxiliary experiment: reconstructed grid -> ClimODE, separate from latent prediction.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${A_CHECKPOINT:?Set the trained Climate Manifold checkpoint}"
: "${ARCHIVE:?Set the original canonical surface archive}"
: "${CONSTANTS:?ClimODE needs aligned real orography and land-sea mask NPZ}"
: "${RUN:?Set a new auxiliary comparison experiment directory}"
[[ ! -e "$RUN" ]] || { echo 'Choose a new RUN' >&2; exit 2; }
PYTHON="${PYTHON:-python}"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
read -r -a seeds <<< "${SEEDS:-7 19 43}"
[[ ${#seeds[@]} -gt 0 ]] || { echo 'SEEDS must be nonempty' >&2; exit 2; }
info=(); [[ -z "${INFO:-}" ]] || info=(--information "$INFO")
mkdir -p "$RUN"
reports=()
for seed in "${seeds[@]}"; do
  for mode in raw decoded; do
    prefix="$RUN/climode-${mode}-seed${seed}"
    "$PYTHON" -m climate_manifold.downstream.train --a-checkpoint "$A_CHECKPOINT" \
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
