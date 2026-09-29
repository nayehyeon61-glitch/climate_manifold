#!/usr/bin/env bash
# Wait for WAIT_PID (post-sweep test/report pipeline) to exit, then run the
# PINN+statistical (W2, KL entropy) vs raw comparison with every Flow term off.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
echo "$(date -u +%FT%TZ) waiting for PID ${WAIT_PID:?}"
while ps -p "$WAIT_PID" >/dev/null; do sleep 60; done
D=/workspace/experiments/a64-b512-pinn-20260922/data
export ARCHIVE="$D/surface.npz" INFO="$D/information-pinn.npz"
test -f "$ARCHIVE"; test -e "$INFO"
source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))"
export PYTHON="$PWD/.venv/bin/python" PYTHONUNBUFFERED=1 DEVICE=cuda
export MODELS="mlp neural_ode climode convlstm simvp"
export PAIRS="pinn_statistical" STATISTICAL_LOSSES="w2 kl_entropy" INCLUDE_RAW=1 SEEDS="7 19 43"
unset A_CHECKPOINT
export TRAINING_MODE=joint INITIALIZATION=fresh ANCHOR=none LATENT_LAYOUT=spatial
export CONSTRAINT_DECODER=separate_surface_and_information
unset STATISTICAL_FLOW_QUANTILES CONDITIONAL_FLOW_QUANTILES CONDITIONAL_FLOW_HIDDEN_DIM CONDITIONAL_FLOW_NOISE_SCALE
export STATISTICAL_FLOW_WEIGHT=0 STATISTICAL_FLOW_WEIGHTS=0 CONDITIONAL_FLOW_WEIGHT=0 CONDITIONAL_FLOW_WEIGHTS=0
export BATCH_SIZE=16 EPOCHS=20 LEARNING_RATE=0.001
export HISTORY_STEPS=6 HISTORY_STRIDE=4 HORIZON_STEPS=20 WINDOW_STRIDE=4
export LATENT_CHANNELS=32 SPATIAL_DOWNSAMPLE=2 SPATIAL_HIDDEN_DIM=64 CONTEXT_DIM=64
export HIDDEN_DIM=128 MANIFOLD_DIM=64 MANIFOLD_HIDDEN_DIM=512
export TENDENCY_WEIGHT=0.1 RECONSTRUCTION_WEIGHT=0.1 STATISTICAL_WEIGHT=0.1
export PINN_WEIGHT=0.1 PINN_LEVELS="500 850" STATIC_WEIGHT=0
export KL_BINS=64 KL_RANGE=6 KL_BANDWIDTH=0.2
export MAX_WINDOWS=0 MAX_CASES=0 ORIGIN_STRIDE=1
export RUN="$PWD/runs/pinn_statistical_flow_off_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$(dirname "$RUN")"
echo "$(date -u +%FT%TZ) starting $RUN (commit $(git rev-parse --short HEAD))"
bash scripts/run_pairwise_manifold_comparison.sh 2>&1 | tee "${RUN}.log"
echo "$(date -u +%FT%TZ) FLOW_OFF_DONE $RUN"
