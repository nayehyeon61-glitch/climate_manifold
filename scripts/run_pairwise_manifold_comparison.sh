#!/usr/bin/env bash
# Shared E: forecasting through E-F-D; observed constraints default to E-D_I.
# Include one data-to-forecast raw control per model/seed, shared across pairs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${ARCHIVE:?Set the canonical surface archive}"
: "${INFO:?Set the aligned dynamic and static information archive}"
: "${RUN:?Set a new comparison experiment directory}"
[[ ! -e "$RUN" ]] || { echo 'Choose a new RUN' >&2; exit 2; }
[[ "${TRAINING_MODE:-joint}" == joint ]] || { echo 'Pairwise constraints require TRAINING_MODE=joint' >&2; exit 2; }
[[ "${INITIALIZATION:-fresh}" == fresh ]] || { echo 'Pairwise runner requires INITIALIZATION=fresh' >&2; exit 2; }
[[ "${ANCHOR:-none}" == none ]] || { echo 'Pairwise experiments require ANCHOR=none' >&2; exit 2; }
[[ "${LATENT_LAYOUT:-spatial}" == spatial ]] || { echo 'Pairwise runner requires LATENT_LAYOUT=spatial' >&2; exit 2; }
[[ -z "${A_CHECKPOINT:-}" ]] || { echo 'Pairwise runner initializes fresh E/F/D; unset A_CHECKPOINT' >&2; exit 2; }
INCLUDE_RAW="${INCLUDE_RAW:-1}"
[[ "$INCLUDE_RAW" == 0 || "$INCLUDE_RAW" == 1 ]] || { echo 'INCLUDE_RAW must be 0 or 1' >&2; exit 2; }
CONSTRAINT_DECODER="${CONSTRAINT_DECODER:-information_only}"
case "$CONSTRAINT_DECODER" in
  information_only|surface_and_information) ;;
  *) echo 'CONSTRAINT_DECODER must be information_only or surface_and_information' >&2; exit 2;;
esac
PYTHON="${PYTHON:-python}"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
read -r -a families <<< "${MODELS:-neural_ode climode}"
read -r -a seeds <<< "${SEEDS:-7 19 43}"
read -r -a pairs <<< "${PAIRS:-pinn_statistical pinn_static statistical_static}"
read -r -a levels <<< "${PINN_LEVELS:-500 850}"
[[ ${#families[@]} -gt 0 && ${#seeds[@]} -gt 0 && ${#pairs[@]} -gt 0 ]] || {
  echo 'MODELS, SEEDS and PAIRS must be nonempty' >&2; exit 2;
}
for family in "${families[@]}"; do
  case "$family" in
    neural_ode|climode|mlp|convlstm|simvp) ;;
    *) echo "Unsupported pairwise model: $family" >&2; exit 2;;
  esac
done
for pair in "${pairs[@]}"; do
  case "$pair" in
    pinn_statistical|pinn_static|statistical_static) ;;
    *) echo "Unsupported constraint pair: $pair" >&2; exit 2;;
  esac
done
arms=("${pairs[@]}")
if [[ "$INCLUDE_RAW" == 1 ]]; then arms+=(raw); fi
# Duplicate identities must fail before starting any expensive training.
declare -A identities=()
for seed in "${seeds[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo 'SEEDS must be nonnegative integers' >&2; exit 2; }
  for family in "${families[@]}"; do
    for arm in "${arms[@]}"; do
      identity="$family-$arm-seed$seed"
      [[ ! -v "identities[$identity]" ]] || { echo "Duplicate experiment: $identity" >&2; exit 2; }
      identities[$identity]=1
    done
  done
done
setup=(--training-mode joint --initialization fresh --latent-layout spatial
  --mode enriched --representation climate_manifold
  --experiment primary --anchor none
  --manifold-dim "${MANIFOLD_DIM:-64}" --manifold-hidden-dim "${MANIFOLD_HIDDEN_DIM:-512}"
  --latent-channels "${LATENT_CHANNELS:-32}" --spatial-downsample "${SPATIAL_DOWNSAMPLE:-2}"
  --spatial-hidden-dim "${SPATIAL_HIDDEN_DIM:-64}" --context-dim "${CONTEXT_DIM:-64}"
  --history-steps "${HISTORY_STEPS:-6}" --history-stride "${HISTORY_STRIDE:-4}")
mkdir -p "$RUN"
reports=()
for seed in "${seeds[@]}"; do
  for family in "${families[@]}"; do
    for arm in "${arms[@]}"; do
      prefix="$RUN/${family}-${arm}-seed${seed}"
      if [[ "$arm" == raw ]]; then
        route=(--bridge raw --raw-backend matched --regularization none)
      else
        route=(--bridge latent --regularization full --constraint-pair "$arm"
          --constraint-decoder "$CONSTRAINT_DECODER"
          --statistical-weight "${STATISTICAL_WEIGHT:-0.1}" --static-weight "${STATIC_WEIGHT:-0.05}")
        if [[ "$arm" == pinn_* ]]; then
          route+=(--pinn --pinn-levels "${levels[@]}" --pinn-weight "${PINN_WEIGHT:-0.1}")
        fi
      fi
      "$PYTHON" -m climate_manifold.downstream.train "${setup[@]}" \
        --archive "$ARCHIVE" --information "$INFO" --model "$family" "${route[@]}" \
        --output "$prefix.pt" --epochs "${EPOCHS:-20}" --batch-size "${BATCH_SIZE:-16}" \
        --learning-rate "${LEARNING_RATE:-0.001}" --tendency-weight "${TENDENCY_WEIGHT:-0.1}" \
        --reconstruction-weight "${RECONSTRUCTION_WEIGHT:-0.1}" \
        --climode-step-hours "${CLIMODE_STEP_HOURS:-1}" \
        --latent-max-speed "${LATENT_MAX_SPEED:-2}" --latent-max-acceleration "${LATENT_MAX_ACCELERATION:-1}" \
        --hidden-dim "${HIDDEN_DIM:-128}" --horizon-steps "${HORIZON_STEPS:-20}" \
        --window-stride "${WINDOW_STRIDE:-4}" --max-windows "${MAX_WINDOWS:-0}" \
        --seed "$seed" --device "${DEVICE:-cpu}"
      "$PYTHON" -m climate_manifold.downstream.evaluate --checkpoint "$prefix.pt" \
        --archive "$ARCHIVE" --information "$INFO" --split validation --output "$prefix.validation.json" \
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
