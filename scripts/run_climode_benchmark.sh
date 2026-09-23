#!/usr/bin/env bash
# Raw ClimODE reference + matched raw/manifold/plain-AE forecast experiments.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${RUN:?Set a new benchmark directory}"
: "${A_CHECKPOINT:?Set the trained Climate Manifold checkpoint}"
: "${ARCHIVE:?Set the original canonical surface archive}"
: "${CONSTANTS:?Set aligned real orography and land-sea mask NPZ}"
[[ ! -e "$RUN" ]] || { echo 'Choose a new RUN' >&2; exit 2; }
benchmark_run="$RUN"
RUN="$benchmark_run/reference" CLIMODE_BRIDGES=raw bash scripts/run_climode_comparison.sh
RUN="$benchmark_run/latent" CLIMODE_REFERENCE_DIR="$benchmark_run/reference" bash scripts/run_model_comparison.sh
