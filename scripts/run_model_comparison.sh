#!/usr/bin/env bash
# Primary experiment: predict in a frozen representation, then decode to fields.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${A_CHECKPOINT:?Set the trained Climate Manifold checkpoint}"
: "${ARCHIVE:?Set the original canonical surface archive}"
: "${RUN:?Set a new comparison experiment directory}"
[[ ! -e "$RUN" ]] || { echo 'Choose a new RUN' >&2; exit 2; }
[[ "${ANCHOR:-none}" == none ]] || { echo 'Primary experiments require ANCHOR=none (pure decoder output)' >&2; exit 2; }
PYTHON="${PYTHON:-python}"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
read -r -a families <<< "${MODELS:-mlp neural_ode}"
read -r -a seeds <<< "${SEEDS:-7 19 43}"
[[ ${#families[@]} -gt 0 && ${#seeds[@]} -gt 0 ]] || { echo 'MODELS and SEEDS must be nonempty' >&2; exit 2; }
for family in "${families[@]}"; do
  case "$family" in
    mlp|neural_ode) ;;
    climode) echo 'ClimODE is an auxiliary grid experiment; use scripts/run_climode_comparison.sh' >&2; exit 2;;
    *) echo "Unsupported primary model: $family" >&2; exit 2;;
  esac
done
info=(); [[ -z "${INFO:-}" ]] || info=(--information "$INFO")
mkdir -p "$RUN"
reports=()
ae_checkpoint="${AE_CHECKPOINT:-$RUN/plain-ae.pt}"
if [[ -z "${AE_CHECKPOINT:-}" ]]; then
  # Match the fixed A: one fresh AE shared by all forecast families and seeds.
  "$PYTHON" -m climate_manifold.downstream.plain_ae --a-checkpoint "$A_CHECKPOINT" \
    --archive "$ARCHIVE" "${info[@]}" --output "$ae_checkpoint" \
    --epochs "${AE_EPOCHS:-20}" --batch-size "${BATCH_SIZE:-2}" \
    --learning-rate "${AE_LEARNING_RATE:-0.001}" --window-stride "${WINDOW_STRIDE:-4}" \
    --max-windows "${MAX_WINDOWS:-0}" --seed "${AE_SEED:-7}" --device "${DEVICE:-cpu}"
fi
for seed in "${seeds[@]}"; do
  for family in "${families[@]}"; do
    for variant in raw climate_manifold plain_ae; do
      mode=latent; representation="$variant"; extra=()
      if [[ "$variant" == raw ]]; then mode=raw; representation=climate_manifold; fi
      if [[ "$variant" == plain_ae ]]; then extra=(--ae-checkpoint "$ae_checkpoint"); fi
      prefix="$RUN/${family}-${variant}-seed${seed}"
      "$PYTHON" -m climate_manifold.downstream.train --a-checkpoint "$A_CHECKPOINT" \
        --archive "$ARCHIVE" "${info[@]}" --experiment primary --model "$family" \
        --bridge "$mode" --representation "$representation" "${extra[@]}" --anchor none \
        --output "$prefix.pt" --epochs "${EPOCHS:-20}" --batch-size "${BATCH_SIZE:-2}" \
        --learning-rate "${LEARNING_RATE:-0.001}" --tendency-weight "${TENDENCY_WEIGHT:-0.1}" \
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
