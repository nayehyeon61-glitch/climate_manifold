#!/usr/bin/env bash
# Joint experiment: match E-F-D capacity and change only representation regularization.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ "${TRAINING_MODE:-joint}" == frozen ]]; then
  exec bash scripts/run_frozen_model_comparison.sh
fi
[[ "${TRAINING_MODE:-joint}" == joint ]] || { echo 'TRAINING_MODE must be joint or frozen' >&2; exit 2; }
: "${ARCHIVE:?Set the original canonical surface archive}"
: "${RUN:?Set a new comparison experiment directory}"
[[ ! -e "$RUN" ]] || { echo 'Choose a new RUN' >&2; exit 2; }
[[ "${ANCHOR:-none}" == none ]] || { echo 'Primary experiments require ANCHOR=none (pure decoder output)' >&2; exit 2; }
PYTHON="${PYTHON:-python}"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
read -r -a families <<< "${MODELS:-neural_ode climode}"
read -r -a seeds <<< "${SEEDS:-7 19 43}"
[[ ${#families[@]} -gt 0 && ${#seeds[@]} -gt 0 ]] || { echo 'MODELS and SEEDS must be nonempty' >&2; exit 2; }
for family in "${families[@]}"; do
  case "$family" in
    mlp|neural_ode|climode) ;;
    *) echo "Unsupported primary model: $family" >&2; exit 2;;
  esac
done
initialization="${INITIALIZATION:-fresh}"
[[ "$initialization" == fresh || "$initialization" == pretrained ]] || { echo 'INITIALIZATION must be fresh or pretrained' >&2; exit 2; }
if [[ "$initialization" == pretrained && -z "${A_CHECKPOINT:-}" ]]; then
  echo 'INITIALIZATION=pretrained requires A_CHECKPOINT' >&2; exit 2
fi
info=(); [[ -z "${INFO:-}" ]] || info=(--information "$INFO")
setup=(--training-mode joint --initialization "$initialization"
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
if [[ "${PINN:-0}" == 1 ]]; then
  read -r -a levels <<< "${PINN_LEVELS:-500 850}"
  setup+=(--pinn --pinn-levels "${levels[@]}")
fi
[[ -z "${PINN_WEIGHT:-}" ]] || setup+=(--pinn-weight "$PINN_WEIGHT")
variants=(forecast_only climate_manifold)
[[ "${INCLUDE_RAW:-0}" != 1 ]] || variants+=(raw)
if [[ "${INCLUDE_RAW:-0}" == 1 && " ${families[*]} " == *" climode "* ]]; then
  echo 'Raw ClimODE is skipped here; run run_climode_benchmark.sh with CONSTANTS for that reference.' >&2
fi
mkdir -p "$RUN"
reports=()
for seed in "${seeds[@]}"; do
  for family in "${families[@]}"; do
    for variant in "${variants[@]}"; do
      # Raw ClimODE has a different grid/data contract and belongs to the external
      # reference runner, not to this same-representation loss ablation.
      if [[ "$family" == climode && "$variant" == raw ]]; then continue; fi
      bridge=latent; regularization=full
      [[ "$variant" != forecast_only ]] || regularization=none
      if [[ "$variant" == raw ]]; then bridge=raw; regularization=none; fi
      prefix="$RUN/${family}-${variant}-seed${seed}"
      "$PYTHON" -m climate_manifold.downstream.train "${setup[@]}" \
        --archive "$ARCHIVE" "${info[@]}" --experiment primary --model "$family" \
        --bridge "$bridge" --representation climate_manifold --regularization "$regularization" --anchor none \
        --output "$prefix.pt" --epochs "${EPOCHS:-20}" --batch-size "${BATCH_SIZE:-2}" \
        --learning-rate "${LEARNING_RATE:-0.001}" --tendency-weight "${TENDENCY_WEIGHT:-0.1}" \
        --reconstruction-weight "${RECONSTRUCTION_WEIGHT:-0.1}" --information-weight "${INFORMATION_WEIGHT:-0.1}" \
        --distribution-weight "${DISTRIBUTION_WEIGHT:-0}" --static-weight "${STATIC_WEIGHT:-0.05}" \
        --physics-weight "${PHYSICS_WEIGHT:-0.01}" \
        --climode-step-hours "${CLIMODE_STEP_HOURS:-1}" \
        --latent-max-speed "${LATENT_MAX_SPEED:-2}" --latent-max-acceleration "${LATENT_MAX_ACCELERATION:-1}" \
        --hidden-dim "${HIDDEN_DIM:-128}" --horizon-steps "${HORIZON_STEPS:-20}" \
        --window-stride "${WINDOW_STRIDE:-4}" --max-windows "${MAX_WINDOWS:-0}" \
        --seed "$seed" --device "${DEVICE:-cpu}"
      "$PYTHON" -m climate_manifold.downstream.evaluate --checkpoint "$prefix.pt" \
        --archive "$ARCHIVE" "${info[@]}" --split validation --output "$prefix.validation.json" \
        --forecast-output "$prefix.forecast.npz" --max-cases "${MAX_CASES:-0}" \
        --origin-stride "${ORIGIN_STRIDE:-1}" --device "${DEVICE:-cpu}"
      reports+=("$prefix.validation.json")
    done
  done
done
reference=()
if [[ -n "${CLIMODE_REFERENCE_DIR:-}" ]]; then
  reference=(--climode-reference-reports)
  for seed in "${seeds[@]}"; do
    reference+=("$CLIMODE_REFERENCE_DIR/climode-raw-seed${seed}.validation.json")
  done
fi
"$PYTHON" -m climate_manifold.downstream.compare --reports "${reports[@]}" "${reference[@]}" --output "$RUN/comparison.json"
