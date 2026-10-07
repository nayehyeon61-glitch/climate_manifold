#!/usr/bin/env bash
# Matched autoregressive dynamics: future fields vs fields + latent consistency.
# Existing direct dynamics and baseline runners keep their historical defaults.
set -euo pipefail
export INCLUDE_DYNAMICS="${INCLUDE_DYNAMICS:-1}"
export INCLUDE_DYNAMICS_CONSISTENCY="${INCLUDE_DYNAMICS_CONSISTENCY:-1}"
export CONSISTENCY_ONLY="${CONSISTENCY_ONLY:-1}"
export INCLUDE_ZERO_DYNAMICS="${INCLUDE_ZERO_DYNAMICS:-0}"
export INCLUDE_TWO_STAGE="${INCLUDE_TWO_STAGE:-0}"
export INCLUDE_TWO_STAGE_BYPASS="${INCLUDE_TWO_STAGE_BYPASS:-0}"
export DYNAMICS_LATENT_WEIGHT="${DYNAMICS_LATENT_WEIGHT:-0.1}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
script_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_root/run_dynamics_manifold_comparison.sh" "$@"
