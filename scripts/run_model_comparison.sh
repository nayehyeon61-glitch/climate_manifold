#!/usr/bin/env bash
# Matched direct/manifold experiments; A remains frozen in every variant.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${A_CHECKPOINT:?Set the trained Climate Manifold checkpoint}"
: "${ARCHIVE:?Set the original canonical surface archive}"
: "${RUN:?Set a new comparison experiment directory}"
[[ ! -e "$RUN" ]] || { echo 'Choose a new RUN' >&2;exit 2; }
PYTHON="${PYTHON:-python}"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
read -r -a families <<< "${MODELS:-mlp neural_ode climode}"
read -r -a seeds <<< "${SEEDS:-7 19 43}"
for family in "${families[@]}"; do
  case "$family" in mlp|neural_ode) ;; climode) : "${CONSTANTS:?ClimODE needs aligned orography and land-sea mask NPZ}" ;;
    *) echo "Unknown model: $family" >&2;exit 2;; esac
done
info=();[[ -z "${INFO:-}" ]] || info=(--information "$INFO")
mkdir -p "$RUN"
reports=()
for seed in "${seeds[@]}"; do
  for family in "${families[@]}"; do
    bridge=latent;extra=()
    if [[ "$family" == climode ]]; then bridge=decoded;extra=(--constants "$CONSTANTS");fi
    for mode in raw "$bridge"; do
      prefix="$RUN/${family}-${mode}-seed${seed}"
      "$PYTHON" -m climate_manifold.downstream.train --a-checkpoint "$A_CHECKPOINT" \
        --archive "$ARCHIVE" "${info[@]}" "${extra[@]}" --model "$family" --bridge "$mode" \
        --output "$prefix.pt" --epochs "${EPOCHS:-20}" --batch-size "${BATCH_SIZE:-2}" \
        --hidden-dim "${HIDDEN_DIM:-128}" --horizon-steps "${HORIZON_STEPS:-20}" \
        --window-stride "${WINDOW_STRIDE:-4}" --max-windows "${MAX_WINDOWS:-0}" \
        --seed "$seed" --device "${DEVICE:-cpu}" --anchor "${ANCHOR:-none}"
      "$PYTHON" -m climate_manifold.downstream.evaluate --checkpoint "$prefix.pt" \
        --archive "$ARCHIVE" "${info[@]}" --split validation --output "$prefix.validation.json" \
        --forecast-output "$prefix.forecast.npz" --max-cases "${MAX_CASES:-0}" \
        --origin-stride "${ORIGIN_STRIDE:-1}" --device "${DEVICE:-cpu}"
      reports+=("$prefix.validation.json")
    done
  done
done
"$PYTHON" -m climate_manifold.downstream.compare --reports "${reports[@]}" --output "$RUN/comparison.json"
