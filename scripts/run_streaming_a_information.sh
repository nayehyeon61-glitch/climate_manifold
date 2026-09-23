#!/usr/bin/env bash
# Start A when its committed input prefix is ready; finish remaining publication.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${ARCHIVE:?Set the canonical 6h surface archive}"
: "${INFO:?Set a new or matching resumable compact-shard directory}"
: "${RUN:?Set a new experiment directory}"
[[ ! -e "$RUN" ]] || { echo 'Choose a new RUN directory' >&2;exit 2; }
PYTHON="${PYTHON:-python}"
export PYTHON INFO ARCHIVE RUN MODE=enriched
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
PINN="${PINN:-0}";export PINN
case "$PINN" in 0|1) ;; *) echo 'PINN must be 0 or 1' >&2;exit 2;; esac
extra=()
if [[ "$PINN" == 1 ]]; then
  read -r -a levels <<< "${PINN_LEVELS:-500 850}"
  extra=(--pinn --pinn-levels "${levels[@]}")
fi
mkdir -p "$RUN"
bash scripts/run_climate_manifold.sh preflight
producer=''
cleanup() {
  if [[ -n "$producer" ]] && kill -0 "$producer" 2>/dev/null; then
    kill "$producer" 2>/dev/null || true
    wait "$producer" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"$PYTHON" scripts/stream_era5_extra.py --archive "$ARCHIVE" --store "$INFO" \
  --download --delete-raw --regrid "${REGRID:-linear}" --days-per-request "${CHUNK_DAYS:-3}" \
  "${extra[@]}" > "$RUN/download.log" 2>&1 &
producer=$!
began=$SECONDS
while true; do
  if "$PYTHON" scripts/stream_era5_extra.py --archive "$ARCHIVE" --store "$INFO" \
      --check-ready A --history-stride "${HISTORY_STRIDE:-4}" "${extra[@]}" \
      > "$RUN/readiness-A.json" 2> "$RUN/readiness-wait.log"; then break;else status=$?;fi
  [[ "$status" == 75 ]] || { cat "$RUN/readiness-wait.log" >&2;exit "$status"; }
  if ! kill -0 "$producer" 2>/dev/null; then
    wait "$producer" || true
    echo 'Producer stopped before readiness; inspect download.log' >&2;exit 1
  fi
  if (( SECONDS-began >= ${WAIT_TIMEOUT_SECONDS:-86400} )); then
    echo 'Readiness timeout; reuse INFO with a new RUN' >&2;exit 1
  fi
  sleep "${POLL_SECONDS:-5}"
done
bash scripts/run_climate_manifold.sh train 2>&1 | tee "$RUN/train.log"
bash scripts/run_climate_manifold.sh audit
wait "$producer";producer=''
"$PYTHON" scripts/stream_era5_extra.py --archive "$ARCHIVE" --store "$INFO" \
  --prune-verified-raw > "$RUN/raw-cleanup.json"
echo 'A and input publication complete. Evaluate with run_climate_manifold.sh validation.'
