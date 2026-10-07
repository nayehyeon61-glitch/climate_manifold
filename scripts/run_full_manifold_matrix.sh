#!/usr/bin/env bash
# Full requested matrix: 12 routes/objectives x 3 statistical losses x M x seed.
# Use the daily server entry point so every suite checks workspace/GPU policies.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${DAILY_WORK:?Set a new work directory below /lustre/home/yehyeon}"
export ERA5_ROOT="${ERA5_ROOT:-/lustre/home/mahmed/ERA5_0p25_DAILY}"
PYTHON="${PYTHON:-python}"
export PYTHON
read -r -a losses <<< "${FULL_MATRIX_LOSSES:-w2 kl_entropy signed_measure}"
[[ ${#losses[@]} -gt 0 ]] || { echo 'FULL_MATRIX_LOSSES must be nonempty' >&2; exit 2; }
declare -A seen=()
for loss in "${losses[@]}"; do
  case "$loss" in w2|kl_entropy|signed_measure) ;; *) echo "Unknown statistical loss: $loss" >&2; exit 2;; esac
  [[ ! -v "seen[$loss]" ]] || { echo "Duplicate statistical loss: $loss" >&2; exit 2; }
  seen[$loss]=1
done

# Explicitly select the scientific comparison rather than inherited old arms.
export INCLUDE_DYNAMICS=1 INCLUDE_DYNAMICS_CONSISTENCY=1 INCLUDE_VECTOR_ONLY=1
export CONSISTENCY_ONLY=0 INCLUDE_LEGACY_DYNAMICS=0
export INCLUDE_ZERO_GUIDE=0 INCLUDE_ZERO_DYNAMICS=0 INCLUDE_GUIDED_FORECAST_ONLY=0
export INCLUDE_TWO_STAGE=1 INCLUDE_TWO_STAGE_BYPASS=0
export STATISTICAL_FLOW_WEIGHT=0 CONDITIONAL_FLOW_WEIGHT=0
export MODELS="${MODELS:-mlp neural_ode climode convlstm simvp fourcastnet climax transformer}"
export SEEDS="${SEEDS:-7 19 43}" BATCH_SIZE="${BATCH_SIZE:-16}" EPOCHS="${EPOCHS:-20}"
export DYNAMICS_ENCODER="${DYNAMICS_ENCODER:-conv_gru}"
export DYNAMICS_LATENT_WEIGHT="${DYNAMICS_LATENT_WEIGHT:-0.1}"
export CONSTRAINT_DECODER=separate_surface_and_information
export VARIABLE_CONDITIONING="${VARIABLE_CONDITIONING:-0}" GUIDE_DIRECT_INFORMATION="${GUIDE_DIRECT_INFORMATION:-1}"
export EVALUATE_TEST="${EVALUATE_TEST:-1}" MAKE_PLOTS="${MAKE_PLOTS:-1}"
export RESUME="${RESUME:-0}"
preflight="${RUN_PREFLIGHT_TESTS:-1}"

# Every comparison uses the same preparation plan and the same held-out origins.
# The daily preparer verifies/reuses completed archives on subsequent loss suites.
unset ARCHIVE INFO A_CHECKPOINT
for loss in "${losses[@]}"; do
  export STATISTICAL_LOSS="$loss" DYNAMICS_STATISTICAL_LOSS="$loss"
  export RUN="$DAILY_WORK/runs/full-$loss" RUN_PREFLIGHT_TESTS="$preflight"
  printf '\nStatistical suite: %s\nExperiment directory: %s\n' "$loss" "$RUN"
  bash scripts/run_daily_manifold_fusion.sh
  preflight=0
done

# Keep loss suites separate: changing both E_s and Dynamics field statistics is
# a whole-objective comparison, not an isolated E_s intervention. The flat table
# permits cross-suite inspection without merging physical variables/units.
"$PYTHON" - "$DAILY_WORK" "$EVALUATE_TEST" "${losses[@]}" <<'PY'
import csv, json, sys
from pathlib import Path
root=Path(sys.argv[1])
splits=['validation','test'] if sys.argv[2]=='1' else ['validation']
fields=['statistical_suite','split','model','bridge','candidate_arm','seed',
        'dynamics_encoder','dynamics_supervision','guide_interaction',
        'normalized_rmse','wind_speed_rmse_mps','finite_forecast_fraction',
        'total_parameters','trainable_parameters','training_seconds']
rows=[]
for loss in sys.argv[3:]:
    for split in splits:
        name='comparison.test.json' if split=='test' else 'comparison.json'
        report=json.loads((root/'runs'/('full-'+loss)/name).read_text())
        for row in report.get('rows',[]):
            out={key:row.get(key) for key in fields}
            out.update(statistical_suite=loss,split=split,
                       dynamics_supervision=(row.get('dynamics_auxiliary_config') or {}).get('supervision'))
            rows.append(out)
target=root/'full-matrix-summary.csv'
with target.open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=fields)
    writer.writeheader();writer.writerows(rows)
print('Cross-suite summary:',target)
PY
