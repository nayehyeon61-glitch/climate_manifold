#!/usr/bin/env bash
# Safe daily ERA5 entry point for the shared-manifold / forecast-model matrix.
# Preparation, caches, logs, checkpoints and figures remain below DAILY_WORK.
set -euo pipefail
export SUITE=manifold_fusion
export ERA5_ROOT="${ERA5_ROOT:-/lustre/home/mahmed/ERA5_0p25_DAILY}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_daily_guided_transformer.sh" "$@"
