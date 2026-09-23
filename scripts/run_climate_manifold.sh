#!/usr/bin/env bash
# Train/evaluate only Climate Manifold A using an existing surface archive.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${ARCHIVE:?Set the canonical 6h surface .npz archive}"
: "${RUN:?Set an experiment directory}"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
MODE="${MODE:-enriched}"
PINN="${PINN:-0}"
case "$PINN" in 0|1) ;; *) echo 'PINN must be 0 or 1' >&2; exit 2;; esac
info=(); pinn=()
if [[ "$MODE" == enriched ]]; then
  : "${INFO:?Set information .npz or compact-shard directory}"
  info=(--information "$INFO")
fi
if [[ "$PINN" == 1 ]]; then
  read -r -a levels <<< "${PINN_LEVELS:-500 850}"
  pinn=(--pinn --pinn-levels "${levels[@]}")
fi
case "${1:-train}" in
  preflight)
    "$PYTHON" scripts/prepare_temporal_120h.py --archive "$ARCHIVE" --output "$RUN/preflight.json" \
      --history-steps 6 --history-stride "${HISTORY_STRIDE:-4}" ;;
  train|A)
    "$PYTHON" -m climate_manifold.train --archive "$ARCHIVE" "${info[@]}" \
      --output "$RUN/manifold.pt" --stage A --mode "$MODE" --profile "${PROFILE:-process}" \
      --epochs "${EPOCHS:-60}" --batch-size "${BATCH_SIZE:-2}" --members "${MEMBERS:-4}" \
      --tau-steps "${TAU_STEPS:-4}" --history-steps 6 --history-stride "${HISTORY_STRIDE:-4}" \
      --manifold-dim "${MANIFOLD_DIM:-64}" --hidden-dim "${HIDDEN_DIM:-512}" \
      --context-dim "${CONTEXT_DIM:-64}" --curriculum-interval "${CURRICULUM_INTERVAL:-4}" \
      --dynamics-max-steps "${DYNAMICS_MAX_STEPS:-4}" \
      --window-stride "${WINDOW_STRIDE:-4}" --max-windows "${MAX_WINDOWS:-0}" \
      --device "${DEVICE:-cpu}" "${pinn[@]}" ;;
  audit)
    "$PYTHON" -m climate_manifold.audit --checkpoint "$RUN/manifold.pt" --archive "$ARCHIVE" \
      "${info[@]}" --output "$RUN/geometry-audit.json" ;;
  pure-drift-validation)
    "$PYTHON" -m climate_manifold.dynamics_evaluate --checkpoint "$RUN/manifold.pt" \
      --archive "$ARCHIVE" "${info[@]}" --split validation \
      --output "$RUN/pure-drift-validation.json" --forecast-output "$RUN/pure-drift-validation.npz" \
      --steps "${DYNAMICS_EVAL_STEPS:-20}" --max-cases "${MAX_CASES:-0}" \
      --device "${DEVICE:-cpu}" ;;
  selection|validation|test|drift-validation)
    split="$1"; drift=()
    [[ "$split" != selection ]] || split=expert_validation
    if [[ "$split" == drift-validation ]]; then split=validation;drift=(--drift-only);fi
    "$PYTHON" -m climate_manifold.forecast --checkpoint "$RUN/manifold.pt" --archive "$ARCHIVE" \
      "${info[@]}" --split "$split" --output "$RUN/$1.json" --forecast-output "$RUN/$1.npz" \
      --members "${MEMBERS:-4}" --tau-steps "${TAU_STEPS:-4}" --max-cases "${MAX_CASES:-0}" \
      --device "${DEVICE:-cpu}" "${drift[@]}" ;;
  *) echo 'Use preflight | train | audit | selection | validation | drift-validation | pure-drift-validation | test' >&2;exit 2 ;;
esac
