#!/usr/bin/env bash
# Per forecast family: raw -> F, E -> F -> D, and
# E -> Fusion Transformer(raw, guide) -> F (guide fusion, plus zero-guide control).
# Optional temporal encoder arms compare E_s+E_d in latent and guided routes.
# RESUME=1 skips completed fits only after checking settings, data and source.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${ARCHIVE:?Set the canonical surface archive}"
: "${INFO:?Set the aligned physical-information archive}"
: "${RUN:?Set a new experiment directory}"
RESUME="${RESUME:-0}"
[[ "$RESUME" == 0 || "$RESUME" == 1 ]] || { echo 'RESUME must be 0 or 1' >&2; exit 2; }
[[ "$RESUME" == 1 || ! -e "$RUN" ]] || { echo 'Choose a new RUN (or RESUME=1)' >&2; exit 2; }
read -r -a models <<< "${MODELS:-transformer mlp neural_ode climode convlstm simvp fourcastnet climax}"
[[ ${#models[@]} -gt 0 ]] || { echo 'MODELS must be nonempty' >&2; exit 2; }
declare -A model_seen=()
for family in "${models[@]}"; do
  case "$family" in transformer|mlp|neural_ode|climode|convlstm|simvp|fourcastnet|climax) ;;
    *) echo "Unknown model: $family" >&2; exit 2;; esac
  [[ ! -v "model_seen[$family]" ]] || { echo "Duplicate model: $family" >&2; exit 2; }
  model_seen[$family]=1
done
[[ -z "${A_CHECKPOINT:-}" ]] || { echo 'This runner initializes fresh representations; unset A_CHECKPOINT' >&2; exit 2; }
PYTHON="${PYTHON:-python}"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
read -r -a seeds <<< "${SEEDS:-7 19 43}"
[[ ${#seeds[@]} -gt 0 ]] || { echo 'SEEDS must be nonempty' >&2; exit 2; }
declare -A seen=()
for seed in "${seeds[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo 'SEEDS must be nonnegative integers' >&2; exit 2; }
  [[ ! -v "seen[$seed]" ]] || { echo "Duplicate seed: $seed" >&2; exit 2; }
  seen[$seed]=1
done
STATISTICAL_LOSS="${STATISTICAL_LOSS:-w2}"
case "$STATISTICAL_LOSS" in w2|kl_entropy|signed_measure) ;; *) echo 'Unknown STATISTICAL_LOSS' >&2; exit 2;; esac
CONSTRAINT_DECODER="${CONSTRAINT_DECODER:-separate_surface_and_information}"
case "$CONSTRAINT_DECODER" in
  separate_surface_and_information|information_only) ;;
  *) echo 'Guided forecasts require an independent information/reconstruction decoder' >&2; exit 2;;
esac
INCLUDE_ZERO_GUIDE="${INCLUDE_ZERO_GUIDE:-1}"
[[ "$INCLUDE_ZERO_GUIDE" == 0 || "$INCLUDE_ZERO_GUIDE" == 1 ]] || { echo 'INCLUDE_ZERO_GUIDE must be 0 or 1' >&2; exit 2; }
INCLUDE_GUIDED_FORECAST_ONLY="${INCLUDE_GUIDED_FORECAST_ONLY:-0}"
[[ "$INCLUDE_GUIDED_FORECAST_ONLY" == 0 || "$INCLUDE_GUIDED_FORECAST_ONLY" == 1 ]] || { echo 'INCLUDE_GUIDED_FORECAST_ONLY must be 0 or 1' >&2; exit 2; }
# Preserve the established four-arm experiment unless explicitly enabled.
INCLUDE_DYNAMICS="${INCLUDE_DYNAMICS:-0}"
INCLUDE_LEGACY_DYNAMICS="${INCLUDE_LEGACY_DYNAMICS:-1}"
INCLUDE_ZERO_DYNAMICS="${INCLUDE_ZERO_DYNAMICS:-1}"
INCLUDE_DYNAMICS_CONSISTENCY="${INCLUDE_DYNAMICS_CONSISTENCY:-0}"
INCLUDE_VECTOR_ONLY="${INCLUDE_VECTOR_ONLY:-0}"
CONSISTENCY_ONLY="${CONSISTENCY_ONLY:-0}"
for name in INCLUDE_DYNAMICS INCLUDE_LEGACY_DYNAMICS INCLUDE_ZERO_DYNAMICS INCLUDE_DYNAMICS_CONSISTENCY INCLUDE_VECTOR_ONLY CONSISTENCY_ONLY; do
  [[ "${!name}" == 0 || "${!name}" == 1 ]] || { echo "$name must be 0 or 1" >&2; exit 2; }
done
INCLUDE_TWO_STAGE="${INCLUDE_TWO_STAGE:-0}"
INCLUDE_TWO_STAGE_BYPASS="${INCLUDE_TWO_STAGE_BYPASS:-0}"
for name in INCLUDE_TWO_STAGE INCLUDE_TWO_STAGE_BYPASS; do
  [[ "${!name}" == 0 || "${!name}" == 1 ]] || { echo "$name must be 0 or 1" >&2; exit 2; }
done
[[ "$INCLUDE_TWO_STAGE" == 0 || "$INCLUDE_DYNAMICS" == 1 ]] || { echo 'INCLUDE_TWO_STAGE=1 requires INCLUDE_DYNAMICS=1' >&2; exit 2; }
# Dynamics-only routes: P(E_d) -> M -> D_s and T2(Raw, E_d) -> M (E_s removed from the
# forecast input, its observed objectives kept). DYNAMICS_ONLY_SUITE=1 trains only these
# arms, as a supplement to a completed suite with identical settings.
INCLUDE_DYNAMICS_ONLY="${INCLUDE_DYNAMICS_ONLY:-0}"
DYNAMICS_ONLY_SUITE="${DYNAMICS_ONLY_SUITE:-0}"
for name in INCLUDE_DYNAMICS_ONLY DYNAMICS_ONLY_SUITE; do
  [[ "${!name}" == 0 || "${!name}" == 1 ]] || { echo "$name must be 0 or 1" >&2; exit 2; }
done
[[ "$DYNAMICS_ONLY_SUITE" == 0 || "$INCLUDE_DYNAMICS_ONLY" == 1 ]] || { echo 'DYNAMICS_ONLY_SUITE=1 requires INCLUDE_DYNAMICS_ONLY=1' >&2; exit 2; }
[[ "$INCLUDE_DYNAMICS_ONLY" == 0 || ( "$INCLUDE_DYNAMICS" == 1 && "$INCLUDE_DYNAMICS_CONSISTENCY" == 1 ) ]] || {
  echo 'INCLUDE_DYNAMICS_ONLY=1 requires INCLUDE_DYNAMICS=1 and INCLUDE_DYNAMICS_CONSISTENCY=1' >&2; exit 2; }
[[ "$INCLUDE_TWO_STAGE_BYPASS" == 0 || "$INCLUDE_TWO_STAGE" == 1 ]] || { echo 'INCLUDE_TWO_STAGE_BYPASS=1 requires INCLUDE_TWO_STAGE=1' >&2; exit 2; }
[[ "$INCLUDE_DYNAMICS_CONSISTENCY" == 0 || "$INCLUDE_DYNAMICS" == 1 ]] || { echo 'INCLUDE_DYNAMICS_CONSISTENCY=1 requires INCLUDE_DYNAMICS=1' >&2; exit 2; }
[[ "$INCLUDE_VECTOR_ONLY" == 0 || "$INCLUDE_DYNAMICS_CONSISTENCY" == 1 ]] || { echo 'INCLUDE_VECTOR_ONLY=1 requires INCLUDE_DYNAMICS_CONSISTENCY=1' >&2; exit 2; }
[[ "$CONSISTENCY_ONLY" == 0 || "$INCLUDE_DYNAMICS_CONSISTENCY" == 1 ]] || { echo 'CONSISTENCY_ONLY=1 requires INCLUDE_DYNAMICS_CONSISTENCY=1' >&2; exit 2; }
[[ "$CONSISTENCY_ONLY" == 0 || "$INCLUDE_TWO_STAGE_BYPASS" == 0 ]] || { echo 'CONSISTENCY_ONLY=1 requires INCLUDE_TWO_STAGE_BYPASS=0' >&2; exit 2; }
[[ "$INCLUDE_LEGACY_DYNAMICS" == 1 || "$INCLUDE_DYNAMICS" == 0 || "$INCLUDE_ZERO_DYNAMICS" == 0 ]] || { echo 'INCLUDE_LEGACY_DYNAMICS=0 requires INCLUDE_ZERO_DYNAMICS=0 when Dynamics is enabled' >&2; exit 2; }
[[ "$INCLUDE_LEGACY_DYNAMICS" == 1 || "$INCLUDE_TWO_STAGE_BYPASS" == 0 ]] || { echo 'INCLUDE_TWO_STAGE_BYPASS=1 requires INCLUDE_LEGACY_DYNAMICS=1' >&2; exit 2; }
[[ "$INCLUDE_LEGACY_DYNAMICS" == 1 || "$INCLUDE_DYNAMICS" == 0 || "$INCLUDE_DYNAMICS_CONSISTENCY" == 1 ]] || { echo 'INCLUDE_LEGACY_DYNAMICS=0 requires INCLUDE_DYNAMICS_CONSISTENCY=1 when Dynamics is enabled' >&2; exit 2; }
GUIDE_INTERACTION_DEPTH="${GUIDE_INTERACTION_DEPTH:-1}"
[[ "$GUIDE_INTERACTION_DEPTH" =~ ^[1-9][0-9]*$ ]] || { echo 'GUIDE_INTERACTION_DEPTH must be a positive integer' >&2; exit 2; }
DYNAMICS_ENCODER="${DYNAMICS_ENCODER:-conv_gru}"
case "$DYNAMICS_ENCODER" in conv_gru|temporal_conv) ;;
  *) echo 'DYNAMICS_ENCODER must be conv_gru or temporal_conv; disable with INCLUDE_DYNAMICS=0' >&2; exit 2;;
esac
DYNAMICS_STATISTICAL_LOSS="${DYNAMICS_STATISTICAL_LOSS:-$STATISTICAL_LOSS}"
case "$DYNAMICS_STATISTICAL_LOSS" in w2|kl_entropy|signed_measure) ;;
  *) echo 'Unknown DYNAMICS_STATISTICAL_LOSS' >&2; exit 2;;
esac
GUIDE_DIRECT_INFORMATION="${GUIDE_DIRECT_INFORMATION:-1}"
[[ "$GUIDE_DIRECT_INFORMATION" == 0 || "$GUIDE_DIRECT_INFORMATION" == 1 ]] || { echo 'GUIDE_DIRECT_INFORMATION must be 0 or 1' >&2; exit 2; }
VARIABLE_CONDITIONING="${VARIABLE_CONDITIONING:-0}"
[[ "$VARIABLE_CONDITIONING" == 0 || "$VARIABLE_CONDITIONING" == 1 ]] || { echo 'VARIABLE_CONDITIONING must be 0 or 1' >&2; exit 2; }
EVALUATE_TEST="${EVALUATE_TEST:-1}"
MAKE_PLOTS="${MAKE_PLOTS:-1}"
GPU_GUARD="${GPU_GUARD:-0}"
for name in EVALUATE_TEST MAKE_PLOTS GPU_GUARD; do
  [[ "${!name}" == 0 || "${!name}" == 1 ]] || { echo "$name must be 0 or 1" >&2; exit 2; }
done
step=$("$PYTHON" -m climate_manifold.fusion_run --archive "$ARCHIVE" --information "$INFO" --step-only)
if [[ "$step" == 24 ]]; then stride=1; horizon=5; window=1
else stride=4; horizon=20; window=4; fi
consistency=()
consistency_latent=()
vector_only=()
consistency_evaluation=()
consistency_settings=()
if [[ "$INCLUDE_DYNAMICS_CONSISTENCY" == 1 ]]; then
  # The new field/hybrid pair differs only by the latent loss. Both retain
  # meaningful physical-field supervision and the same learned rollout module.
  "$PYTHON" - "${DYNAMICS_LATENT_WEIGHT:-0.1}" "${DYNAMICS_RECONSTRUCTION_WEIGHT:-0.1}" \
    "${DYNAMICS_STATISTICAL_WEIGHT:-0.1}" "${AUX_ROLLOUT_STEPS:-}" \
    "${HORIZON_STEPS:-$horizon}" "$CONSISTENCY_ONLY" <<'PY'
import math, sys
for name, value in zip(('DYNAMICS_LATENT_WEIGHT', 'DYNAMICS_RECONSTRUCTION_WEIGHT',
                         'DYNAMICS_STATISTICAL_WEIGHT'), sys.argv[1:4]):
    try:
        positive = math.isfinite(float(value)) and float(value) > 0
    except ValueError:
        positive = False
    if not positive:
        raise SystemExit(f'{name} must be finite and positive for the consistency comparison')
requested, horizon, only = sys.argv[4:]
if requested:
    if not requested.isdecimal() or int(requested) < int(horizon):
        raise SystemExit('AUX_ROLLOUT_STEPS must be an integer >= HORIZON_STEPS')
    if int(requested) > int(horizon) and only != '1':
        raise SystemExit('Extended AUX_ROLLOUT_STEPS requires CONSISTENCY_ONLY=1 for matched evaluation origins')
PY
  consistency=(--dynamics-auxiliary-architecture latent_rollout)
  consistency_latent=(--dynamics-latent-weight "${DYNAMICS_LATENT_WEIGHT:-0.1}")
  if [[ "$INCLUDE_VECTOR_ONLY" == 1 ]]; then
    vector_only=(--dynamics-supervision vector_only "${consistency_latent[@]}"
      --dynamics-reconstruction-weight 0 --dynamics-statistical-weight 0 --dynamics-tendency-weight 0)
  fi
  consistency_evaluation=(--dynamics-diagnostics)
  consistency_settings=(--setting consistency_rollout_steps "${AUX_ROLLOUT_STEPS:-${HORIZON_STEPS:-$horizon}}"
    --setting consistency_train_horizon "${HORIZON_STEPS:-$horizon}" --setting consistency_step_hours "$step")
  if [[ -n "${AUX_ROLLOUT_STEPS:-}" ]]; then
    consistency_evaluation+=(--dynamics-rollout-steps "$AUX_ROLLOUT_STEPS")
  fi
fi
if [[ "$MAKE_PLOTS" == 1 ]]; then
  "$PYTHON" -c 'import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot'
fi
if [[ "$GPU_GUARD" == 1 ]]; then
  : "${GPU_LOCK_ROOT:?GPU_GUARD requires a GPU_LOCK_ROOT below /lustre/home/yehyeon}"
fi
run_model() {
  if [[ "$GPU_GUARD" == 1 ]]; then
    "$PYTHON" -m climate_manifold.gpu_guard --lock-root "$GPU_LOCK_ROOT" \
      --max-utilization "${GPU_MAX_UTILIZATION:-10}" \
      --max-memory-percent "${GPU_MAX_MEMORY_PERCENT:-10}" --min-free-gib "${GPU_MIN_FREE_GIB:-8}" \
      -- "$PYTHON" -m "$@" --device cuda:0
  else
    "$PYTHON" -m "$@" --device "${DEVICE:-cpu}"
  fi
}
# These mechanisms remain explicitly disabled in this experiment.
for name in STATISTICAL_FLOW_WEIGHT CONDITIONAL_FLOW_WEIGHT; do
  [[ "${!name:-0}" == 0 ]] || { echo "$name must remain 0 in this runner" >&2; exit 2; }
done
setup=(--training-mode joint --initialization fresh
  --mode enriched --representation climate_manifold --latent-layout spatial
  --experiment primary --anchor none --raw-backend matched
  --manifold-dim "${MANIFOLD_DIM:-64}" --manifold-hidden-dim "${MANIFOLD_HIDDEN_DIM:-512}"
  --latent-channels "${LATENT_CHANNELS:-32}" --spatial-downsample "${SPATIAL_DOWNSAMPLE:-2}"
  --spatial-hidden-dim "${SPATIAL_HIDDEN_DIM:-64}" --context-dim "${CONTEXT_DIM:-64}"
  --history-steps "${HISTORY_STEPS:-6}" --history-stride "${HISTORY_STRIDE:-$stride}"
  --hidden-dim "${HIDDEN_DIM:-128}" --weather-depth "${WEATHER_DEPTH:-4}"
  --weather-patch-size "${WEATHER_PATCH_SIZE:-2}" --transformer-heads "${TRANSFORMER_HEADS:-4}"
  --epochs "${EPOCHS:-20}" --batch-size "${BATCH_SIZE:-16}"
  --learning-rate "${LEARNING_RATE:-0.001}" --horizon-steps "${HORIZON_STEPS:-$horizon}"
  --tendency-weight "${TENDENCY_WEIGHT:-0.1}" --reconstruction-weight "${RECONSTRUCTION_WEIGHT:-0.1}"
  --window-stride "${WINDOW_STRIDE:-$window}" --max-windows "${MAX_WINDOWS:-0}"
  --climode-step-hours "${CLIMODE_STEP_HOURS:-1}" --ode-substeps "${ODE_SUBSTEPS:-2}"
  --static-weight 0)
if [[ "$VARIABLE_CONDITIONING" == 1 ]]; then setup+=(--spatial-variable-conditioning); fi
statistical=(--constraint-pair statistical --regularization full
  --constraint-decoder "$CONSTRAINT_DECODER" --statistical-loss "$STATISTICAL_LOSS"
  --statistical-weight "${STATISTICAL_WEIGHT:-0.1}"
  --statistical-flow-weight 0 --conditional-flow-weight 0)
if [[ "$STATISTICAL_LOSS" == kl_entropy ]]; then
  statistical+=(--kl-bins "${KL_BINS:-64}" --kl-range "${KL_RANGE:-6}" --kl-bandwidth "${KL_BANDWIDTH:-0.2}")
fi
# Optionally give guided arms the same direct origin-information tokens as raw.
guide_information=(--guide-architecture fusion --guide-fusion-depth "${GUIDE_FUSION_DEPTH:-2}")
if [[ "$GUIDE_DIRECT_INFORMATION" == 1 ]]; then guide_information+=(--guide-direct-information); fi
dynamics=(--dynamics-encoder "$DYNAMICS_ENCODER"
  --dynamics-hidden-dim "${DYNAMICS_HIDDEN_DIM:-32}" --dynamics-channels "${DYNAMICS_CHANNELS:-8}"
  --dynamics-statistical-loss "$DYNAMICS_STATISTICAL_LOSS"
  --dynamics-reconstruction-weight "${DYNAMICS_RECONSTRUCTION_WEIGHT:-0.1}"
  --dynamics-statistical-weight "${DYNAMICS_STATISTICAL_WEIGHT:-0.1}"
  --dynamics-tendency-weight "${DYNAMICS_TENDENCY_WEIGHT:-0.0}")
# Record only effective optional settings; inactive knobs do not alter a run.
effective_dynamics=()
if [[ "$INCLUDE_DYNAMICS" == 1 ]]; then effective_dynamics=("${dynamics[@]}"); fi
two_stage=()
if [[ "$INCLUDE_TWO_STAGE" == 1 ]]; then
  two_stage=(--guide-interaction transformer --guide-interaction-depth "$GUIDE_INTERACTION_DEPTH")
fi
arms=(raw latent guided)
if [[ "$INCLUDE_ZERO_GUIDE" == 1 ]]; then arms+=(guided_zero); fi
if [[ "$INCLUDE_GUIDED_FORECAST_ONLY" == 1 ]]; then arms+=(guided_forecast_only); fi
effective_legacy_dynamics=0
if [[ "$INCLUDE_DYNAMICS" == 1 && "$INCLUDE_LEGACY_DYNAMICS" == 1 && "$CONSISTENCY_ONLY" == 0 ]]; then
  effective_legacy_dynamics=1
  arms+=(latent_dynamics guided_dynamics)
  if [[ "$INCLUDE_ZERO_DYNAMICS" == 1 ]]; then
    arms+=(latent_dynamics_zero guided_dynamics_zero)
  fi
  if [[ "$INCLUDE_TWO_STAGE" == 1 ]]; then arms+=(guided_dynamics_two_stage); fi
fi
if [[ "$CONSISTENCY_ONLY" == 1 ]]; then arms=(); fi
if [[ "$INCLUDE_DYNAMICS_CONSISTENCY" == 1 ]]; then
  arms+=(latent_dynamics_field latent_dynamics_field_latent
         guided_dynamics_field guided_dynamics_field_latent)
  if [[ "$INCLUDE_TWO_STAGE" == 1 ]]; then
    arms+=(guided_dynamics_two_stage_field guided_dynamics_two_stage_field_latent)
  fi
  if [[ "$INCLUDE_VECTOR_ONLY" == 1 ]]; then
    arms+=(latent_dynamics_vector_only guided_dynamics_vector_only)
    if [[ "$INCLUDE_TWO_STAGE" == 1 ]]; then arms+=(guided_dynamics_two_stage_vector_only); fi
  fi
fi
dynamics_only_settings=()
if [[ "$INCLUDE_DYNAMICS_ONLY" == 1 ]]; then
  if [[ "$DYNAMICS_ONLY_SUITE" == 1 ]]; then arms=(); fi
  arms+=(latent_dynamics_only_field latent_dynamics_only_field_latent
         guided_dynamics_only_field guided_dynamics_only_field_latent)
  if [[ "$INCLUDE_VECTOR_ONLY" == 1 ]]; then
    arms+=(latent_dynamics_only_vector_only guided_dynamics_only_vector_only)
  fi
  dynamics_only_settings=(--setting include_dynamics_only 1 --setting dynamics_only_suite "$DYNAMICS_ONLY_SUITE")
fi
# Capture actual effective arguments, not just exported overrides. Fingerprints
# cover prepared inputs and code; original daily ERA5 files are never hashed.
"$PYTHON" -m climate_manifold.fusion_run --archive "$ARCHIVE" --information "$INFO" \
  --run "$RUN" --source-root "$PWD" --resume "$RESUME" \
  --setting models "${models[*]}" --setting seeds "${seeds[*]}" --setting arms "${arms[*]}" \
  --setting evaluate_test "$EVALUATE_TEST" --setting make_plots "$MAKE_PLOTS" \
  --setting include_two_stage "$INCLUDE_TWO_STAGE" --setting include_two_stage_bypass "$INCLUDE_TWO_STAGE_BYPASS" \
  --setting include_dynamics_consistency "$INCLUDE_DYNAMICS_CONSISTENCY" --setting consistency_only "$CONSISTENCY_ONLY" \
  --setting include_vector_only "$INCLUDE_VECTOR_ONLY" \
  --setting include_legacy_dynamics "$effective_legacy_dynamics" \
  "${consistency_settings[@]}" "${dynamics_only_settings[@]}" \
  --setting max_cases "${MAX_CASES:-0}" --setting origin_stride "${ORIGIN_STRIDE:-1}" \
  --setting device "$([[ "$GPU_GUARD" == 1 ]] && echo cuda:0 || echo "${DEVICE:-cpu}")" \
  -- setup "${setup[@]}" statistical "${statistical[@]}" guide "${guide_information[@]}" dynamics "${effective_dynamics[@]}" two_stage "${two_stage[@]}" \
  consistency "${consistency[@]}" hybrid "${consistency_latent[@]}" vector_only "${vector_only[@]}" consistency_evaluation "${consistency_evaluation[@]}"
printf 'Run: %s\nFits: %s\n' "$RUN" "$(( ${#models[@]} * ${#arms[@]} * ${#seeds[@]} ))"
# Analysis is written to fresh staging paths. Publish the comparison JSON last
# and the plot directory only after success, so interrupted analysis is resumable.
compare_reports() {
  local output="$1"; shift
  local stage staged file
  # compare omits CSVs for empty effect tables. Require only populated tables,
  # otherwise a valid run would regenerate analysis on every RESUME.
  if [[ -f "$output" ]] && "$PYTHON" - "$output" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text())
    climode = data.get('climode_benchmark', {})
    tables = {
        '.climode.csv': climode.get('rows'),
        '.climode-effects.csv': climode.get('effects'),
        '.raw-effects.csv': data.get('direct_comparison', {}).get('effects'),
        '.constraint-effects.csv': data.get('constraint_pair_effects'),
        '.statistical-effects.csv': data.get('statistical_loss_effects'),
        '.flow-effects.csv': data.get('statistical_flow_effects'),
        '.conditional-flow-effects.csv': data.get('conditional_flow_effects'),
        '.guide-effects.csv': data.get('guide_effects'),
        '.route-effects.csv': data.get('route_effects'),
        '.dynamics-effects.csv': data.get('dynamics_effects'),
        '.dynamics-consistency-effects.csv': data.get('dynamics_consistency_effects'),
        '.dynamics-vector-effects.csv': data.get('dynamics_vector_effects'),
        '.guide-interaction-effects.csv': data.get('guide_interaction_effects'),
    }
    complete = path.with_suffix('.csv').is_file() and all(
        not rows or path.with_suffix(suffix).is_file() for suffix, rows in tables.items())
except (ValueError, OSError, AttributeError):
    complete = False
raise SystemExit(0 if complete else 1)
PY
  then return 0; fi
  stage=$(mktemp -d "$RUN/.comparison-staging-XXXXXXXX")
  staged="$stage/$(basename "$output")"
  "$PYTHON" -m climate_manifold.downstream.compare --reports "$@" --output "$staged"
  for file in "$stage"/*; do
    [[ "$file" == "$staged" ]] || mv -- "$file" "$RUN/$(basename "$file")"
  done
  mv -- "$staged" "$output"
  rmdir "$stage"
}
make_plots() {
  local comparison="$1" split="$2" target="$RUN/plots/$2" stage backup
  [[ "$MAKE_PLOTS" == 1 && ! -f "$target/manifest.json" ]] || return 0
  mkdir -p "$RUN/plots"
  stage=$(mktemp -d "$RUN/plots/.$split-staging-XXXXXXXX")
  "$PYTHON" -m climate_manifold.downstream.plot_comparison --comparison "$comparison" --output "$stage"
  if [[ -e "$target" ]]; then
    backup=$(mktemp -d "$RUN/plots/$split.partial-XXXXXXXX")
    rmdir "$backup"
    mv -- "$target" "$backup"
    printf 'Preserved incomplete plots: %s\n' "$backup"
  fi
  mv -- "$stage" "$target"
}
reports=()
test_reports=()
evaluation_options() {
  evaluation=()
  case "$1" in
    *_dynamics_field|*_dynamics_field_latent|*_dynamics_two_stage_field|*_dynamics_two_stage_field_latent|*_dynamics_vector_only|*_dynamics_two_stage_vector_only|*_dynamics_only_field|*_dynamics_only_field_latent|*_dynamics_only_vector_only)
      evaluation=("${consistency_evaluation[@]}") ;;
  esac
}
for seed in "${seeds[@]}"; do
 for family in "${models[@]}"; do
  for arm in "${arms[@]}"; do
    prefix="$RUN/$family-$arm-seed$seed"
    reports+=("$prefix.validation.json")
    [[ ! -f "$prefix.validation.json" ]] || { echo "skip $prefix (done)"; continue; }
    case "$arm" in
      raw) route=(--bridge raw --regularization none --dynamics-encoder off) ;;
      latent) route=(--bridge latent --dynamics-encoder off "${statistical[@]}") ;;
      guided) route=(--bridge guided --guide-mode learned --dynamics-encoder off "${guide_information[@]}" "${statistical[@]}") ;;
      guided_zero) route=(--bridge guided --guide-mode zero --dynamics-encoder off "${guide_information[@]}" "${statistical[@]}") ;;
      # Same learned-guide Fusion and M, without either observed auxiliary loss.
      # This is a combined reconstruction+statistics ablation, not statistics alone.
      guided_forecast_only) route=(--bridge guided --guide-mode learned --dynamics-encoder off "${guide_information[@]}" --regularization none) ;;
      latent_dynamics) route=(--bridge latent "${statistical[@]}" "${dynamics[@]}" --dynamics-mode learned) ;;
      guided_dynamics) route=(--bridge guided --guide-mode learned "${guide_information[@]}" "${statistical[@]}" "${dynamics[@]}" --dynamics-mode learned) ;;
      latent_dynamics_zero) route=(--bridge latent "${statistical[@]}" "${dynamics[@]}" --dynamics-mode zero) ;;
      guided_dynamics_zero) route=(--bridge guided --guide-mode learned "${guide_information[@]}" "${statistical[@]}" "${dynamics[@]}" --dynamics-mode zero) ;;
      guided_dynamics_two_stage) route=(--bridge guided --guide-mode learned "${guide_information[@]}" "${statistical[@]}" "${dynamics[@]}" --dynamics-mode learned "${two_stage[@]}") ;;
      latent_dynamics_field|latent_dynamics_field_latent|latent_dynamics_vector_only)
        route=(--bridge latent "${statistical[@]}" "${dynamics[@]}" --dynamics-mode learned "${consistency[@]}") ;;
      guided_dynamics_field|guided_dynamics_field_latent|guided_dynamics_vector_only)
        route=(--bridge guided --guide-mode learned "${guide_information[@]}" "${statistical[@]}" "${dynamics[@]}" --dynamics-mode learned "${consistency[@]}") ;;
      guided_dynamics_two_stage_field|guided_dynamics_two_stage_field_latent|guided_dynamics_two_stage_vector_only)
        route=(--bridge guided --guide-mode learned "${guide_information[@]}" "${statistical[@]}" "${dynamics[@]}" --dynamics-mode learned "${two_stage[@]}" "${consistency[@]}") ;;
      latent_dynamics_only_field|latent_dynamics_only_field_latent|latent_dynamics_only_vector_only)
        route=(--bridge latent "${statistical[@]}" "${dynamics[@]}" --dynamics-mode only "${consistency[@]}") ;;
      guided_dynamics_only_field|guided_dynamics_only_field_latent|guided_dynamics_only_vector_only)
        route=(--bridge guided --guide-mode learned "${guide_information[@]}" "${statistical[@]}" "${dynamics[@]}" --dynamics-mode only "${consistency[@]}") ;;
    esac
    case "$arm" in
      *_field) route+=(--dynamics-supervision field) ;;
      *_field_latent) route+=(--dynamics-supervision field_latent "${consistency_latent[@]}") ;;
      *_vector_only) route+=("${vector_only[@]}") ;;
    esac
    if [[ ! -f "$prefix.manifest.json" ]]; then
      run_model climate_manifold.downstream.train --model "$family" "${setup[@]}" "${route[@]}" \
        --archive "$ARCHIVE" --information "$INFO" --output "$prefix.pt" \
        --seed "$seed"
    fi
    evaluation_options "$arm"
    run_model climate_manifold.downstream.evaluate --checkpoint "$prefix.pt" "${evaluation[@]}" \
      --archive "$ARCHIVE" --information "$INFO" --split validation \
      --output "$prefix.validation.json" --forecast-output "$prefix.forecast.npz" \
      --max-cases "${MAX_CASES:-0}" --origin-stride "${ORIGIN_STRIDE:-1}"
  done
 done
done
# This is a diagnostic intervention on the already trained two-stage checkpoint,
# not another fit or a replacement for the independently trained one-stage arm.
evaluate_two_stage_bypass() {
  local split="$1" seed family checkpoint prefix forecast
  [[ "$INCLUDE_TWO_STAGE_BYPASS" == 1 ]] || return 0
  for seed in "${seeds[@]}"; do
   for family in "${models[@]}"; do
    checkpoint="$RUN/$family-guided_dynamics_two_stage-seed$seed.pt"
    prefix="$RUN/$family-guided_dynamics_two_stage_bypass-seed$seed"
    if [[ "$split" == validation ]]; then
      reports+=("$prefix.validation.json")
      forecast="$prefix.forecast.npz"
    else
      test_reports+=("$prefix.test.json")
      forecast="$prefix.test.forecast.npz"
    fi
    [[ ! -f "$prefix.$split.json" ]] || continue
    run_model climate_manifold.downstream.evaluate --checkpoint "$checkpoint" \
      --guide-interaction-override off --archive "$ARCHIVE" --information "$INFO" --split "$split" \
      --output "$prefix.$split.json" --forecast-output "$forecast" \
      --max-cases "${MAX_CASES:-0}" --origin-stride "${ORIGIN_STRIDE:-1}"
   done
  done
}
evaluate_two_stage_bypass validation
compare_reports "$RUN/comparison.json" "${reports[@]}"
make_plots "$RUN/comparison.json" validation
if [[ "$EVALUATE_TEST" == 1 ]]; then
  # Every fit/checkpoint is fixed before test evaluation. Test scores never
  # enter optimization, checkpoint selection, or validation comparisons.
  for seed in "${seeds[@]}"; do
   for family in "${models[@]}"; do
    for arm in "${arms[@]}"; do
      prefix="$RUN/$family-$arm-seed$seed"
      test_reports+=("$prefix.test.json")
      [[ ! -f "$prefix.test.json" ]] || continue
      evaluation_options "$arm"
      run_model climate_manifold.downstream.evaluate --checkpoint "$prefix.pt" "${evaluation[@]}" \
        --archive "$ARCHIVE" --information "$INFO" --split test \
        --output "$prefix.test.json" --forecast-output "$prefix.test.forecast.npz" \
        --max-cases "${MAX_CASES:-0}" --origin-stride "${ORIGIN_STRIDE:-1}"
    done
   done
  done
  evaluate_two_stage_bypass test
  compare_reports "$RUN/comparison.test.json" "${test_reports[@]}"
  make_plots "$RUN/comparison.test.json" test
fi
