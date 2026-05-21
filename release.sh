#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
gui_dir="$script_dir/hawi_gui"
cd "$gui_dir"

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required to release Hawi GUI." >&2
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
  echo "uv is required to build the bundled Hawi engine. Use --skip-build only if hawi_gui/release already exists." >&2
  exit 1
fi

if [[ ! -d node_modules ]]; then
  echo "Installing Hawi GUI dependencies..."
  npm install
fi

export HAWI_RELEASE_COMMAND="${HAWI_RELEASE_COMMAND:-$script_dir/release.sh}"
exec npm run release:local -- "$@"
