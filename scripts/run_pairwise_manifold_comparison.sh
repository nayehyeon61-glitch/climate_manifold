#!/usr/bin/env bash
# Shared E: E-F-D forecasts; independent D_rec and D_I reconstruct observations.
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
CONSTRAINT_DECODER="${CONSTRAINT_DECODER:-separate_surface_and_information}"
case "$CONSTRAINT_DECODER" in
  separate_surface_and_information|information_only|surface_and_information) ;;
  *) echo 'CONSTRAINT_DECODER must be separate_surface_and_information, information_only or surface_and_information' >&2; exit 2;;
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
read -r -a statistical_losses <<< "${STATISTICAL_LOSSES:-${STATISTICAL_LOSS:-w2}}"
[[ ${#statistical_losses[@]} -gt 0 ]] || { echo 'STATISTICAL_LOSSES must be nonempty' >&2; exit 2; }
declare -A loss_seen=()
for loss in "${statistical_losses[@]}"; do
  case "$loss" in
    w2|kl_entropy) ;;
    *) echo "Unsupported statistical loss: $loss" >&2; exit 2;;
  esac
  [[ ! -v "loss_seen[$loss]" ]] || { echo "Duplicate statistical loss: $loss" >&2; exit 2; }
  loss_seen[$loss]=1
done
read -r -a flow_weights <<< "${STATISTICAL_FLOW_WEIGHTS:-${STATISTICAL_FLOW_WEIGHT:-0}}"
[[ ${#flow_weights[@]} -gt 0 ]] || { echo 'STATISTICAL_FLOW_WEIGHTS must be nonempty' >&2; exit 2; }
declare -A flow_seen=()
flow_enabled=0
for index in "${!flow_weights[@]}"; do
  weight="${flow_weights[$index]}"
  [[ "$weight" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || {
    echo 'Statistical flow weights must be finite nonnegative decimal numbers' >&2; exit 2;
  }
  # Match comparison arm formatting without invoking the training Python process.
  weight=$(LC_ALL=C awk -v value="$weight" 'BEGIN { printf "%.12g", value + 0 }')
  [[ "$weight" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]] || {
    echo 'Statistical flow weights must be finite nonnegative decimal numbers' >&2; exit 2;
  }
  [[ ! -v "flow_seen[$weight]" ]] || { echo "Duplicate statistical flow weight: $weight" >&2; exit 2; }
  flow_seen[$weight]=1
  flow_weights[$index]="$weight"
  if [[ "$weight" != 0 ]]; then flow_enabled=1; fi
done
if [[ "$flow_enabled" == 1 ]]; then
  [[ " ${pairs[*]} " == *statistical* ]] || {
    echo 'Statistical flow requires a pair containing statistical' >&2; exit 2;
  }
fi
if [[ -n "${STATISTICAL_FLOW_QUANTILES:-}" ]]; then
  [[ "$flow_enabled" == 1 ]] || {
    echo 'STATISTICAL_FLOW_QUANTILES requires a positive statistical flow weight' >&2; exit 2;
  }
  if ! [[ "$STATISTICAL_FLOW_QUANTILES" =~ ^[0-9]+$ ]] ||
      ! LC_ALL=C awk -v count="$STATISTICAL_FLOW_QUANTILES" 'BEGIN { exit !(count >= 1 && count <= 512) }'; then
    echo 'STATISTICAL_FLOW_QUANTILES must be a positive integer between 1 and 512' >&2; exit 2
  fi
fi
read -r -a conditional_weights <<< "${CONDITIONAL_FLOW_WEIGHTS:-${CONDITIONAL_FLOW_WEIGHT:-0}}"
[[ ${#conditional_weights[@]} -gt 0 ]] || { echo 'CONDITIONAL_FLOW_WEIGHTS must be nonempty' >&2; exit 2; }
declare -A conditional_seen=()
conditional_enabled=0
for index in "${!conditional_weights[@]}"; do
  weight="${conditional_weights[$index]}"
  [[ "$weight" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || {
    echo 'Conditional flow weights must be finite nonnegative decimal numbers' >&2; exit 2;
  }
  weight=$(LC_ALL=C awk -v value="$weight" 'BEGIN { printf "%.12g", value + 0 }')
  [[ "$weight" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]] || {
    echo 'Conditional flow weights must be finite nonnegative decimal numbers' >&2; exit 2;
  }
  [[ ! -v "conditional_seen[$weight]" ]] || { echo "Duplicate conditional flow weight: $weight" >&2; exit 2; }
  conditional_seen[$weight]=1
  conditional_weights[$index]="$weight"
  if [[ "$weight" != 0 ]]; then conditional_enabled=1; fi
done
if [[ "$conditional_enabled" == 1 ]]; then
  [[ "$flow_enabled" == 0 ]] || {
    echo 'Conditional flow and statistical flow cannot both have positive weights in one run' >&2; exit 2;
  }
  [[ " ${pairs[*]} " == *statistical* ]] || {
    echo 'Conditional flow requires a pair containing statistical' >&2; exit 2;
  }
  [[ "$CONSTRAINT_DECODER" != surface_and_information ]] || {
    echo 'Conditional flow requires an independent or information-only constraint decoder' >&2; exit 2;
  }
fi
for setting in CONDITIONAL_FLOW_QUANTILES CONDITIONAL_FLOW_HIDDEN_DIM CONDITIONAL_FLOW_NOISE_SCALE; do
  if [[ -n "${!setting:-}" && "$conditional_enabled" == 0 ]]; then
    echo "$setting requires a positive conditional flow weight" >&2; exit 2
  fi
done
if [[ "$conditional_enabled" == 1 ]]; then
  conditional_quantiles="${CONDITIONAL_FLOW_QUANTILES:-32}"
  conditional_hidden="${CONDITIONAL_FLOW_HIDDEN_DIM:-128}"
  conditional_noise="${CONDITIONAL_FLOW_NOISE_SCALE:-0.2}"
  if ! [[ "$conditional_quantiles" =~ ^[0-9]+$ ]] ||
      ! LC_ALL=C awk -v count="$conditional_quantiles" 'BEGIN { exit !(count >= 1 && count <= 512) }'; then
    echo 'CONDITIONAL_FLOW_QUANTILES must be a positive integer between 1 and 512' >&2; exit 2
  fi
  if ! [[ "$conditional_hidden" =~ ^[0-9]+$ ]] ||
      ! LC_ALL=C awk -v count="$conditional_hidden" 'BEGIN { exit !(count >= 1 && count <= 4096) }'; then
    echo 'CONDITIONAL_FLOW_HIDDEN_DIM must be a positive integer between 1 and 4096' >&2; exit 2
  fi
  [[ "$conditional_noise" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]] || {
    echo 'CONDITIONAL_FLOW_NOISE_SCALE must be finite and positive' >&2; exit 2;
  }
  conditional_noise=$(LC_ALL=C awk -v value="$conditional_noise" 'BEGIN { printf "%.12g", value + 0 }')
  if ! [[ "$conditional_noise" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]] ||
      ! LC_ALL=C awk -v value="$conditional_noise" 'BEGIN { exit !(value > 0) }'; then
    echo 'CONDITIONAL_FLOW_NOISE_SCALE must be finite and positive' >&2; exit 2
  fi
fi
arms=()
for pair in "${pairs[@]}"; do
  if [[ "$pair" == *statistical* ]]; then
    for loss in "${statistical_losses[@]}"; do
      base_arm="$pair"
      if [[ "$loss" != w2 ]]; then base_arm+=":kl_entropy"; fi
      if [[ "$conditional_enabled" == 1 ]]; then
        for weight in "${conditional_weights[@]}"; do
          if [[ "$weight" == 0 ]]; then arms+=("$base_arm"); else arms+=("$base_arm:cfm=$weight"); fi
        done
      else
        for weight in "${flow_weights[@]}"; do
          if [[ "$weight" == 0 ]]; then arms+=("$base_arm"); else arms+=("$base_arm:flow=$weight"); fi
        done
      fi
    done
  else
    arms+=("$pair")
  fi
done
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
      arm_label="${arm//:/-}"
      arm_label="${arm_label//flow=/flow}"
      prefix="$RUN/${family}-${arm_label//cfm=/cfm}-seed${seed}"
      pair="${arm%%:*}"
      if [[ "$arm" == raw ]]; then
        route=(--bridge raw --raw-backend matched --regularization none)
      else
        route=(--bridge latent --regularization full --constraint-pair "$pair"
          --constraint-decoder "$CONSTRAINT_DECODER"
          --statistical-weight "${STATISTICAL_WEIGHT:-0.1}" --static-weight "${STATIC_WEIGHT:-0.05}")
        if [[ "$pair" == *statistical* ]]; then
          loss=w2
          [[ ":$arm:" != *:kl_entropy:* ]] || loss=kl_entropy
          route+=(--statistical-loss "$loss")
          if [[ "$loss" == kl_entropy ]]; then
            route+=(--kl-bins "${KL_BINS:-64}" --kl-range "${KL_RANGE:-6}"
              --kl-bandwidth "${KL_BANDWIDTH:-0.2}")
          fi
          if [[ "$arm" == *:flow=* ]]; then
            route+=(--statistical-flow-weight "${arm##*:flow=}"
              --statistical-flow-quantiles "${STATISTICAL_FLOW_QUANTILES:-32}")
          fi
          if [[ "$arm" == *:cfm=* ]]; then
            route+=(--conditional-flow-weight "${arm##*:cfm=}"
              --conditional-flow-quantiles "$conditional_quantiles"
              --conditional-flow-hidden-dim "$conditional_hidden"
              --conditional-flow-noise-scale "$conditional_noise")
          fi
        fi
        if [[ "$pair" == pinn_* ]]; then
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
