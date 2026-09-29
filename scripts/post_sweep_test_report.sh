#!/usr/bin/env bash
# After a comparison sweep finishes: evaluate every checkpoint once on the
# held-out test split, then build ClimODE-protocol reports for validation and test.
# Usage: RUN=runs/<sweep> [RUNNER_PID=<pid>] bash scripts/post_sweep_test_report.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${RUN:?Set RUN to the comparison directory}"
ARCHIVE="${ARCHIVE:-/workspace/experiments/a64-b512-pinn-20260922/data/surface.npz}"
INFO="${INFO:-/workspace/experiments/a64-b512-pinn-20260922/data/information-pinn.npz}"
VIZ_PYTHON="${VIZ_PYTHON:-/workspace/.venvs/climode-viz/bin/python}"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python}"
GIF_VARIABLES=(${GIF_VARIABLES-t2m msl})

if [[ -n "${RUNNER_PID:-}" ]]; then
  echo "$(date -u +%FT%TZ) waiting for sweep runner $RUNNER_PID"
  while ps -p "$RUNNER_PID" >/dev/null; do sleep 60; done
fi
echo "$(date -u +%FT%TZ) sweep finished: $(ls "$RUN"/*.validation.json | wc -l) validation reports"
[[ -f "$RUN/comparison.json" ]] || echo 'WARNING: comparison.json missing; sweep may have stopped early'

for checkpoint in "$RUN"/*-seed*.pt; do
  prefix="${checkpoint%.pt}"
  if [[ -s "$prefix.test.json" ]]; then echo "skip $prefix (test done)"; continue; fi
  echo "$(date -u +%FT%TZ) test: $prefix"
  "$PYTHON" -m climate_manifold.downstream.evaluate --checkpoint "$checkpoint" \
    --archive "$ARCHIVE" --information "$INFO" --split test --output "$prefix.test.json" \
    --forecast-output "$prefix.test.forecast.npz" --max-cases 0 --origin-stride 1 --device cuda
done

for split in validation test; do
  CUDA_VISIBLE_DEVICES= "$VIZ_PYTHON" scripts/climode_report.py --run "$RUN" --split "$split" --gif-variables "${GIF_VARIABLES[@]}"
done
echo "$(date -u +%FT%TZ) POST_SWEEP_DONE"
