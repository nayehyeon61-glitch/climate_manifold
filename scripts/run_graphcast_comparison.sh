#!/usr/bin/env bash
# Official raw JAX GraphCast only: distinct from paired PyTorch Raw/E-model-D arms.
set -euo pipefail
if [[ ${1:-} == --help ]]; then
  cat <<'HELP'
Required: ARCHIVE=/path/states.npz RUN=/new/output/path
Optional: GRAPHCAST_PYTHON=.venv-graphcast/bin/python SEEDS="7 19 43"
EPOCHS=20 BATCH_SIZE=16 HISTORY_STEPS=6 HISTORY_STRIDE=4 HORIZON_STEPS=20
WINDOW_STRIDE=4 MAX_WINDOWS=0 MAX_CASES=0 ORIGIN_STRIDE=1 SPLIT=validation
MESH_SIZE=2 LATENT_SIZE=128 MESSAGE_STEPS=4 LEARNING_RATE=0.001
History settings control shared split origins; actual GraphCast input is t-6h,t.
Install first: bash scripts/install_graphcast_official.sh
No information sidecar is consumed. This is an external raw baseline, not a
matched E/F/D pair or a reproduction of operational GraphCast benchmark skill.
HELP
  exit 0
fi
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${ARCHIVE:?Set ARCHIVE to the canonical 6-hourly four-field archive}"
: "${RUN:?Set RUN to a new output directory}"
PYTHON=${GRAPHCAST_PYTHON:-"$ROOT/.venv-graphcast/bin/python"}
[[ ! -e "$RUN" ]] || { echo "Choose a new RUN directory: $RUN" >&2; exit 1; }
mkdir -p "$RUN"
for seed in ${SEEDS:-7 19 43}; do
  "$PYTHON" -m climate_manifold.downstream.graphcast_official train \
    --archive "$ARCHIVE" --output "$RUN/graphcast_official_raw_seed${seed}" \
    --seed "$seed" --epochs "${EPOCHS:-20}" --batch-size "${BATCH_SIZE:-16}" \
    --history-steps "${HISTORY_STEPS:-6}" --history-stride "${HISTORY_STRIDE:-4}" \
    --horizon-steps "${HORIZON_STEPS:-20}" --window-stride "${WINDOW_STRIDE:-4}" \
    --max-windows "${MAX_WINDOWS:-0}" --max-cases "${MAX_CASES:-0}" \
    --origin-stride "${ORIGIN_STRIDE:-1}" --split "${SPLIT:-validation}" \
    --mesh-size "${MESH_SIZE:-2}" --latent-size "${LATENT_SIZE:-128}" \
    --message-steps "${MESSAGE_STEPS:-4}" --learning-rate "${LEARNING_RATE:-0.001}" \
    2>&1 | tee "$RUN/graphcast_official_raw_seed${seed}.log"
done
