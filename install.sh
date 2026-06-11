#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
gui_dir_name="hawi_gui"
gui_dir="$script_dir/$gui_dir_name"

if [[ ! -f "$gui_dir/package.json" ]]; then
  echo "Current Hawi GUI directory is missing: $gui_dir" >&2
  exit 1
fi

cd "$gui_dir"

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required to install Hawi GUI." >&2
  exit 1
fi

requires_uv=1
for arg in "$@"; do
  case "$arg" in
    --skip-build|-h|--help)
      requires_uv=0
      ;;
  esac
done

if [[ "$requires_uv" == "1" ]] && ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to build the bundled Hawi engine. Use --skip-build only if the selected GUI shell already has build output." >&2
  exit 1
fi

if [[ "$requires_uv" == "1" ]]; then
  echo "Syncing Hawi Python dependencies..."
  uv sync --all-extras --all-groups
fi

dependencies_ready() {
  [[ -d node_modules ]] && npm ls --depth=0 --silent >/dev/null 2>&1
}

if ! dependencies_ready; then
  echo "Installing or repairing Hawi GUI dependencies..."
  npm install
fi

if [[ "${HAWI_INSTALL_SKIP_PREFLIGHT:-}" != "1" ]]; then
  npm run install:preflight -- "$@"
fi

export HAWI_RELEASE_COMMAND="${HAWI_RELEASE_COMMAND:-$script_dir/install.sh}"
exec npm run release:local -- "$@"
