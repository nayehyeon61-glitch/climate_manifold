#!/usr/bin/env bash
# Separate JAX environment; default CPU smoke/research runtime. No pretrained download.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${GRAPHCAST_VENV:-"$ROOT/.venv-graphcast"}
SOURCE=${GRAPHCAST_SOURCE:-"$ROOT/external/weathernext"}
UPSTREAM_COMMIT=f2f2c5117d2f864e2d5e7c2f9f220db5e1049dd3
"${PYTHON:-python3}" -c 'import sys; sys.exit("GraphCast requires Python 3.11 or newer") if sys.version_info < (3,11) else None'
"${PYTHON:-python3}" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install 'jax[cpu]==0.8.2' 'dm-haiku==0.0.16' 'optax==0.2.7' \
  'jraph==0.0.6.dev0' 'trimesh==4.12.2' 'rtree==1.4.1' 'xarray==2026.2.0' 'chex==0.1.92' \
  'gdm-xarray-jax @ git+https://github.com/google-deepmind/xarray_jax.git@0291d0c8e91ca04b00f5f01ea29a53c0e575eeee'
if [[ ! -e "$SOURCE" ]]; then
  mkdir -p "$(dirname "$SOURCE")"
  git clone https://github.com/google-deepmind/weathernext.git "$SOURCE"
fi
if [[ -n $(git -C "$SOURCE" status --porcelain) ]]; then
  echo "Refusing to change a modified upstream checkout: $SOURCE" >&2; exit 1
fi
git -C "$SOURCE" fetch origin "$UPSTREAM_COMMIT"
git -C "$SOURCE" checkout --detach "$UPSTREAM_COMMIT"
"$VENV/bin/python" -m pip install --no-deps -e "$SOURCE"
# Torch is used only by the shared archive and metric code, never as a GraphCast replacement.
"$VENV/bin/python" -m pip install -e "$ROOT"
"$VENV/bin/python" -c 'from climate_manifold.downstream.graphcast_official import official_dependencies; official_dependencies(); print("Pinned official GraphCast import verified")'
printf 'GraphCast Python: %s\n' "$VENV/bin/python"
