#!/usr/bin/env bash
# Optional dynamics encoder with separate latent and guided comparisons.
# With prepared ARCHIVE/INFO/RUN, use the general comparison runner.
# Otherwise retain the server's daily ERA5/workspace/GPU policy.
set -euo pipefail
export INCLUDE_DYNAMICS="${INCLUDE_DYNAMICS:-1}"
export INCLUDE_ZERO_DYNAMICS="${INCLUDE_ZERO_DYNAMICS:-1}"
export DYNAMICS_ENCODER="${DYNAMICS_ENCODER:-conv_gru}"
export STATISTICAL_LOSS="${STATISTICAL_LOSS:-signed_measure}"
export DYNAMICS_STATISTICAL_LOSS="${DYNAMICS_STATISTICAL_LOSS:-$STATISTICAL_LOSS}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
script_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -n "${ARCHIVE:-}" || -n "${INFO:-}" ]]; then
  : "${ARCHIVE:?Prepared mode requires ARCHIVE, INFO and RUN}"
  : "${INFO:?Prepared mode requires ARCHIVE, INFO and RUN}"
  : "${RUN:?Prepared mode requires ARCHIVE, INFO and RUN}"
  exec bash "$script_root/run_guided_fusion_comparison.sh" "$@"
fi
exec bash "$script_root/run_daily_manifold_fusion.sh" "$@"
