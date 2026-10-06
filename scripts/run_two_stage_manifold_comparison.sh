#!/usr/bin/env bash
# Add a guide-only Transformer before the established Raw/guide Fusion.
# A bypass evaluation reuses the trained checkpoint; it is not another fit.
set -euo pipefail
export INCLUDE_DYNAMICS="${INCLUDE_DYNAMICS:-1}"
export INCLUDE_ZERO_DYNAMICS="${INCLUDE_ZERO_DYNAMICS:-1}"
export INCLUDE_TWO_STAGE="${INCLUDE_TWO_STAGE:-1}"
export INCLUDE_TWO_STAGE_BYPASS="${INCLUDE_TWO_STAGE_BYPASS:-$INCLUDE_TWO_STAGE}"
export GUIDE_INTERACTION_DEPTH="${GUIDE_INTERACTION_DEPTH:-1}"
script_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_root/run_dynamics_manifold_comparison.sh" "$@"
